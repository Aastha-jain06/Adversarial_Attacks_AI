# check_prediction.py
import argparse
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

# --- re-declare your model (must match training architecture) ---
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

# --- helpers ---
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

def load_image(path, size=(32,32)):
    img = Image.open(path).convert('RGB')
    tf = T.Compose([
        T.Resize(size),
        T.ToTensor()
    ])
    return tf(img).unsqueeze(0)  # [1,3,H,W]

def normalize_tensor(x, mean=CIFAR_MEAN, std=CIFAR_STD):
    mean_t = torch.tensor(mean).view(1,3,1,1).to(x.device)
    std_t  = torch.tensor(std).view(1,3,1,1).to(x.device)
    return (x - mean_t) / std_t

def try_load_checkpoint(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'])
    elif isinstance(state, dict) and all(k in model.state_dict() or k.startswith('module') for k in state.keys()):
        model.load_state_dict(state)
    else:
        # try plain load
        try:
            model.load_state_dict(state)
        except Exception as e:
            raise RuntimeError("Couldn't load checkpoint. Provide .pth or .ckpt with 'model_state'") from e
    model.to(device)
    model.eval()
    # return normalization if available
    mean = state.get('train_mean', CIFAR_MEAN) if isinstance(state, dict) else CIFAR_MEAN
    std  = state.get('train_std', CIFAR_STD) if isinstance(state, dict) else CIFAR_STD
    return mean, std

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default='./results/best_cnn_cifar10.pth', help='model checkpoint (.pth or .ckpt)')
    p.add_argument('--img', type=str, required=True, help='input image path to test (png/jpg)')
    p.add_argument('--target', type=int, default=None, help='(optional) target class index to check success')
    p.add_argument('--topk', type=int, default=5, help='show top-k predictions')
    p.add_argument('--use_mps', action='store_true', help='force mps if available')
    args = p.parse_args()

    # device
    if args.use_mps and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # model load
    model = SimpleCNN(num_classes=10)
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    mean, std = try_load_checkpoint(model, args.ckpt, device)

    # load and preprocess image
    x = load_image(args.img).to(device)        # [1,3,32,32] in [0,1]
    x_norm = normalize_tensor(x, mean, std)

    with torch.no_grad():
        logits = model(x_norm)                # [1,10]
        probs = F.softmax(logits, dim=1)[0]   # [10]

    logits_list = logits[0].cpu().numpy()
    probs_list = probs.cpu().numpy()

    # print top-k
    topk = min(args.topk, logits_list.shape[0])
    topk_idx = probs.topk(topk).indices.cpu().tolist()
    print("\nTop-{} predictions (class : prob)".format(topk))
    for idx in topk_idx:
        print(f"  {idx} : {probs_list[idx]:.6f}  (logit {logits_list[idx]:.4f})")

    # target check
    if args.target is not None:
        t = int(args.target)
        tprob = probs_list[t]
        print(f"\nTarget class {t} probability = {tprob:.6f} (logit {logits_list[t]:.4f})")
        # simple success heuristic
        if tprob > 0.5:
            print("-> SUCCESS: target class predicted with >50% probability")
        elif tprob > 0.1:
            print("-> PARTIAL: target probability >10% but <50%")
        else:
            print("-> FAIL: target probability <10%")

    # also print all logits in case you want exact numbers
    print("\nAll logits:", ["{:.4f}".format(v) for v in logits_list])
    print("All probs :", ["{:.6f}".format(v) for v in probs_list])
    print("\nSaved image tested:", args.img)

if __name__ == '__main__':
    main()
