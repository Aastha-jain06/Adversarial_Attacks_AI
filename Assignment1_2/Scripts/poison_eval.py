# poison_eval.py
import os
import random
import json
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1) Model: copy your SimpleCNN here
# ---------------------------
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
            nn.MaxPool2d(2, 2),   # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 16 -> 8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)    # 8 -> 4
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

# ---------------------------
# 2) Utilities: image save & transforms
# ---------------------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

def tensor_to_uint8(tensor):
    t = tensor.clone().cpu()
    for c in range(3):
        t[c] = t[c] * CIFAR_STD[c] + CIFAR_MEAN[c]
    t = torch.clamp(t, 0.0, 1.0)
    arr = (t.numpy().transpose(1,2,0) * 255).astype(np.uint8)
    return arr

def save_image(arr, path):
    Image.fromarray(arr).save(path)

# ---------------------------
# 3) Poisoning helpers
# ---------------------------
def add_patch_to_pil(img_pil, patch_size=6, color=(255,0,0), location='bottom_right'):
    """Return PIL image with a colored square patch applied (un-normalized pixel domain)."""
    img = img_pil.copy()
    w,h = img.size
    patch_w = patch_size
    patch_h = patch_size
    if location == 'bottom_right':
        x0 = w - patch_w - 1
        y0 = h - patch_h - 1
    elif location == 'top_left':
        x0 = 1
        y0 = 1
    elif location == 'center':
        x0 = (w - patch_w)//2
        y0 = (h - patch_h)//2
    else:
        x0 = w - patch_w - 1; y0 = h - patch_h - 1
    draw = ImageDraw.Draw(img)
    draw.rectangle([x0, y0, x0+patch_w, y0+patch_h], fill=color)
    return img

class PoisonedCIFAR10(Dataset):
    """
    Wrap CIFAR10 train dataset and apply poisoning:
    - label_flip: flip labels for a fraction (flip map provided)
    -
    """
    def __init__(self, root, train=True, transform=None, download=True,
                 poison_type=None, poison_rate=0.1, flip_from=None, flip_to=None,
                 backdoor_target=0, backdoor_patch_size=6, backdoor_color=(255,0,0), backdoor_loc='bottom_right',
                 seed=0):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=download)
        self.transform = transform
        self.poison_type = poison_type
        self.poison_rate = poison_rate
        self.rng = random.Random(seed)
        self.flip_from = flip_from
        self.flip_to = flip_to
        self.backdoor_target = backdoor_target
        self.backdoor_patch_size = backdoor_patch_size
        self.backdoor_color = backdoor_color
        self.backdoor_loc = backdoor_loc

        # Build index list for poisoning
        n = len(self.dataset)
        idxs = list(range(n))
        self.rng.shuffle(idxs)
        k = int(self.poison_rate * n)
        self.poison_idxs = set(idxs[:k])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        orig_img = img  # PIL image
        poisoned = False

        if self.poison_type == 'label_flip':
            # Only flip labels of specified source class , else flip random labels
            if idx in self.poison_idxs:
                if self.flip_from is not None:
                    if label == self.flip_from:
                        label = self.flip_to
                        poisoned = True
                else:
                    # pick a random wrong label
                    choices = list(range(10))
                    choices.remove(label)
                    label = self.rng.choice(choices)
                    poisoned = True

        elif self.poison_type == 'backdoor':
            if idx in self.poison_idxs:
                # add patch and set label to backdoor_target
                img = add_patch_to_pil(img, patch_size=self.backdoor_patch_size,
                                       color=self.backdoor_color, location=self.backdoor_loc)
                label = self.backdoor_target
                poisoned = True

        # apply transform 
        if self.transform:
            img_t = self.transform(img)
        else:
            img_t = transforms.ToTensor()(img)

        return img_t, label, poisoned, np.array(orig_img)  # return poisoned flag and original for possible save

# ---------------------------
# 4) Train & eval helpers (simple)
# ---------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y, _, _ in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return running_loss/total, correct/total

def eval_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    per_class = {i: {'correct':0, 'total':0} for i in range(10)}
    with torch.no_grad():
        for x, y, in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            for i in range(x.size(0)):
                gt = int(y[i].cpu()); p = int(preds[i].cpu())
                per_class[gt]['total'] += 1
                if p == gt: per_class[gt]['correct'] += 1
    acc = correct/total
    per_class_acc = {cls: (per_class[cls]['correct']/max(1,per_class[cls]['total'])) for cls in per_class}
    return acc, per_class_acc

# ---------------------------
# 5) Run experiments function
# ---------------------------
def run_experiment(root='./data', save_dir='./poison_results', poison_type=None, poison_rate=0.1, epochs=50, seed=0,
                   flip_from=None, flip_to=None, backdoor_target=0):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    # create poisoned train dataset
    train_ds = PoisonedCIFAR10(root=root, train=True, transform=train_transform, download=True,
                               poison_type=poison_type, poison_rate=poison_rate,
                               flip_from=flip_from, flip_to=flip_to,
                               backdoor_target=backdoor_target, seed=seed)
    # for eval and backdoor test we want unpoisoned test set and also triggered test set if backdoor
    test_ds_clean = datasets.CIFAR10(root=root, train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_ds_clean, batch_size=64, shuffle=False, num_workers=0)

    # if backdoor: build a separate test set where we add trigger to all test images and set labels to target
    if poison_type == 'backdoor':
        bd_test_images = []
        bd_test_labels = []
        for i in range(len(test_ds_clean)):
            img, label = test_ds_clean[i]
            pil = Image.fromarray((tensor_to_uint8(img)).astype(np.uint8))  # inverse transform approximate
            pil_bd = add_patch_to_pil(pil, patch_size=6, color=(255,0,0), location='bottom_right')
            # apply normalized transform
            pil_bd_t = test_transform(pil_bd)
            bd_test_images.append(pil_bd_t)
            bd_test_labels.append(backdoor_target)
        class TriggerTestDataset(Dataset):
            def __len__(self): return len(bd_test_images)
            def __getitem__(self, idx): return bd_test_images[idx], bd_test_labels[idx], False, None
        bd_test_loader = DataLoader(TriggerTestDataset(), batch_size=64, shuffle=False, num_workers=0)
    else:
        bd_test_loader = None

    # DataLoader for poisoned training (note: our dataset returns (img, label, poisoned_flag, orig_numpy))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)

    # model, optimizer
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # training loop
    best_val = 0.0
    for ep in range(1, epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_acc, per_class = eval_model(model, test_loader, device)
        print(f"Epoch {ep}/{epochs} | train_acc {tr_acc:.4f} | val_acc {val_acc:.4f}")
        # save small checkpoint
        torch.save({'model_state': model.state_dict(), 'poison_type':poison_type, 'poison_rate':poison_rate}, os.path.join(save_dir, f'chk_ep{ep}.pth'))

    # final evaluation
    clean_acc, per_class_acc = eval_model(model, test_loader, device)
    report = {'poison_type':poison_type, 'poison_rate':poison_rate, 'clean_acc':clean_acc, 'per_class_acc':per_class_acc}

    # if backdoor, evaluate backdoor success rate on trigger test set
    if poison_type == 'backdoor' and bd_test_loader is not None:
        model.eval()
        total=0; success=0
        with torch.no_grad():
            for x,y,_,_ in bd_test_loader:
                x,y = x.to(device), torch.tensor(y).to(device)
                out = model(x)
                pred = out.argmax(1)
                success += (pred == y).sum().item()
                total += x.size(0)
        bd_success_rate = success / total
        report['backdoor_success_rate'] = bd_success_rate
        print("Backdoor success rate:", bd_success_rate)

    # save some poisoned training examples (for the report)
    os.makedirs(os.path.join(save_dir, 'examples'), exist_ok=True)
    saved = 0
    for i in range(2000):  # scan first 2000 training examples for poisoned ones
        if i >= len(train_ds): break
        img_t, label, poisoned, orig_np = train_ds[i]
        if poisoned and saved < 20:
            arr = (orig_np)  # orig numpy from dataset (un-normalized)
            save_image(arr, os.path.join(save_dir, 'examples', f'poisoned_{saved}.png'))
            saved += 1

    # save report
    with open(os.path.join(save_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print("Done. Report saved to", os.path.join(save_dir, 'report.json'))
    return report

# ---------------------------
# 6) Example runs 
# ---------------------------
if __name__ == '__main__':
    # Example 1: label-flip 10% random
    print("Running label-flip 10%")
    run_experiment(save_dir='./poison_labelflip_10', poison_type='label_flip', poison_rate=0.10, epochs=30, seed=42)

   