# fgsm_eval.py
import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------
# COPY the same model code here (or import from your ccn.py)
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -----------------------------
# Utilities
# -----------------------------
def tensor_to_image(tensor, mean, std):
    """Convert normalized tensor C,H,W to uint8 HxWx3"""
    t = tensor.detach().clone().cpu()
    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]
    t = torch.clamp(t, 0.0, 1.0)
    arr = (t.numpy().transpose(1,2,0) * 255).astype(np.uint8)
    return arr

def fgsm_attack(image, epsilon, data_grad, mean, std):
    # image: normalized tensor 1xCxxHxxW
    sign_grad = data_grad.sign()
    perturbed = image + epsilon * sign_grad
    # clamp per-channel to normalized [0,1] range mapped to normalized space
    min_vals = [(0.0 - m)/s for m,s in zip(mean, std)]
    max_vals = [(1.0 - m)/s for m,s in zip(mean, std)]
    for c in range(perturbed.shape[1]):
        perturbed[:,c,:,:].clamp_(min_vals[c], max_vals[c])
    return perturbed

# -----------------------------
# Evaluation that creates adversarial examples and metrics
# -----------------------------
def evaluate_fgsm(model, device, test_loader, mean, std, epsilon=0.03, examples_dir='fgsm_examples', max_examples=20):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total = 0
    correct_clean = 0
    correct_adv = 0
    correct_before = 0
    successful_attacks = 0
    os.makedirs(examples_dir, exist_ok=True)
    saved = 0

    for data, target in tqdm(test_loader, desc=f'FGSM eps={epsilon}'):
        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)
        total += batch_size

        # clean preds
        with torch.no_grad():
            out = model(data)
            pred_clean = out.argmax(dim=1)
        correct_clean += (pred_clean == target).sum().item()

        # compute gradients w.r.t. input
        data.requires_grad = True
        out2 = model(data)
        loss = criterion(out2, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        # create adversarial
        adv_data = fgsm_attack(data.clone(), epsilon, data_grad, mean, std)

        # adv preds
        with torch.no_grad():
            out_adv = model(adv_data)
            pred_adv = out_adv.argmax(dim=1)
        correct_adv += (pred_adv == target).sum().item()

        # attack success counts (only where clean was correct)
        for i in range(batch_size):
            if pred_clean[i] == target[i]:
                correct_before += 1
                if pred_adv[i] != target[i]:
                    successful_attacks += 1

        # save example images
        if saved < max_examples:
            for i in range(batch_size):
                if saved >= max_examples: break
                orig = tensor_to_image(data[i].cpu(), mean, std)
                adv = tensor_to_image(adv_data[i].cpu(), mean, std)
                perturb = (adv.astype(np.int16) - orig.astype(np.int16))
                amp = np.clip((perturb * 10) + 128, 0, 255).astype(np.uint8)

                fig, axes = plt.subplots(1,3, figsize=(9,3))
                axes[0].imshow(orig); axes[0].set_title('orig'); axes[0].axis('off')
                axes[1].imshow(adv); axes[1].set_title('adv'); axes[1].axis('off')
                axes[2].imshow(amp); axes[2].set_title('perturb x10'); axes[2].axis('off')
                out_path = os.path.join(examples_dir, f'ex_{saved:03d}_eps{epsilon}.png')
                plt.savefig(out_path, bbox_inches='tight')
                plt.close(fig)
                saved += 1

    acc_clean = correct_clean / total
    acc_adv = correct_adv / total
    attack_success_rate = successful_attacks / correct_before if correct_before>0 else 0.0

    return {
        'epsilon': epsilon,
        'total': total,
        'clean_acc': acc_clean,
        'adv_acc' : acc_adv,
        'attack_success_rate': attack_success_rate,
        'correct_before': correct_before,
        'successful_attacks': successful_attacks,
        'examples_saved': saved
    }

# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    MODEL_PATH = './results/best_cnn_cifar10.ckpt'  # must match save name above
    # load checkpoint (map first to CPU)
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    mean = ckpt.get('train_mean', (0.5,0.5,0.5))
    std  = ckpt.get('train_std',  (0.5,0.5,0.5))
    num_classes = ckpt.get('num_classes', 10)

    # instantiate and load
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    model.eval()

    # test loader (small batch size on MPS)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

    epsilons = [0.0, 0.01, 0.03, 0.06]
    reports = []
    for eps in epsilons:
        out_dir = f'fgsm_examples_eps{eps}'
        res = evaluate_fgsm(model, device, test_loader, mean, std, epsilon=eps, examples_dir=out_dir, max_examples=20)
        print("Epsilon:", eps, "res:", res)
        reports.append(res)

    # save report
    import json
    with open('fgsm_report.json', 'w') as f:
        json.dump(reports, f, indent=2)
    print("Saved fgsm_report.json")

        # Plot graph: Clean & Adversarial accuracy vs Epsilon
    eps_list = [r['epsilon'] for r in reports]
    clean_acc = [r['clean_acc'] * 100 for r in reports]
    adv_acc = [r['adv_acc'] * 100 for r in reports]

    plt.figure(figsize=(6,4))
    plt.plot(eps_list, clean_acc, marker='o', label='Clean Accuracy')
    plt.plot(eps_list, adv_acc, marker='s', label='Adversarial Accuracy')
    plt.title('FGSM Attack Effect on CIFAR-10 Accuracy')
    plt.xlabel('Epsilon (Attack Strength)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('fgsm_accuracy_vs_epsilon.png')
    plt.show()



#Success Rate :
    success_rate = [r['attack_success_rate'] * 100 for r in reports]
    plt.figure(figsize=(6,4))
    plt.bar([str(e) for e in eps_list], success_rate, color='orange')
    plt.title('FGSM Attack Success Rate vs Epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Attack Success Rate (%)')
    plt.tight_layout()
    plt.savefig('fgsm_attack_success_rate.png')
    plt.show()


#Stats Table:
    print("\n===== FGSM Attack Summary =====")
    print(f"{'Epsilon':<10}{'Clean Acc (%)':<15}{'Adv Acc (%)':<15}{'Attack Success (%)':<20}")
    for r in reports:
        print(f"{r['epsilon']:<10}{r['clean_acc']*100:<15.2f}{r['adv_acc']*100:<15.2f}{r['attack_success_rate']*100:<20.2f}")
    print("================================")


if __name__ == '__main__':
    main()


