# inversion.py
import argparse
import os
from math import ceil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.utils import save_image
import torch.nn.functional as F
from tqdm import trange
import numpy as np
from copy import deepcopy

# ----- re-define your model class (so we can load ckpt) -----
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

# ----- helpers -----
def tv_loss(x):
    # total variation for a batch: encourage spatial smoothness
    # x shape (B, C, H, W)
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw

def normalize_batch(x, mean, std):
    # x in [0,1], convert to normalized expected by model
    b, c, h, w = x.shape
    mean = torch.tensor(mean, device=x.device).view(1, c, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, c, 1, 1)
    return (x - mean) / std

def clamp_image(x):
    return x.clamp(0.0, 1.0)

# ----- inversion procedure for one target -----
def invert_target(model, device, target, ckpt_normalize_mean, ckpt_normalize_std,
                  img_size=(32,32), steps=2000, lr=0.1,
                  l2=1e-3, tv=1e-3, restarts=3, save_path='./inversions',
                  show_progress=False):
    os.makedirs(save_path, exist_ok=True)
    best_global = None
    best_score = -1e9

    C, H, W = 3, img_size[0], img_size[1]
    for r in range(restarts):
        # start from noise +/- small gaussian or from uniform/random
        init = torch.rand(1, C, H, W, device=device)
        img = init.clone().detach().requires_grad_(True)

        optim_img = optim.Adam([img], lr=lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optim_img, milestones=[int(steps*0.6), int(steps*0.85)], gamma=0.2)

        best_local = img.clone().detach()
        best_local_score = -1e9

        iterator = trange(steps, desc=f"restart {r+1}/{restarts}") if show_progress else range(steps)
        for it in iterator:
            optim_img.zero_grad()
            # normalize as model expects
            inp = normalize_batch(img, ckpt_normalize_mean, ckpt_normalize_std)
            logits = model(inp)
            # maximize logit[target] -> minimize negative
            target_logit = logits[0, target]

            # We minimize total loss: -target_logit + reg
            loss = - target_logit
            # regularizers
            loss = loss + l2 * (img.view(img.size(0), -1).pow(2).mean())
            loss = loss + tv * tv_loss(img)

            loss.backward()
            optim_img.step()

            # clamp to [0,1]
            with torch.no_grad():
                img[:] = clamp_image(img)

            if (it % 50) == 0 or it == steps-1:
                cur_score = target_logit.item()
                if cur_score > best_local_score:
                    best_local_score = cur_score
                    best_local = img.clone().detach()

            scheduler.step()

        # evaluate best_local
        with torch.no_grad():
            inp_best = normalize_batch(best_local, ckpt_normalize_mean, ckpt_normalize_std)
            logits_best = model(inp_best)
            score = logits_best[0, target].item()

        if score > best_score:
            best_score = score
            best_global = best_local.clone().detach()

        # save restart result
        save_image(best_local, os.path.join(save_path, f'inv_target{target}_restart{r+1}.png'))
        print(f"Restart {r+1}: best logit for class {target}: {best_local_score:.4f} (eval {score:.4f})")

    # final save
    save_image(best_global, os.path.join(save_path, f'inv_target{target}_best.png'))
    print(f"Saved best reconstruction for class {target} (logit {best_score:.4f}) -> {os.path.join(save_path, f'inv_target{target}_best.png')}")
    return best_global

# ----- CLI -----
def main_cli():
    parser = argparse.ArgumentParser(description="Model Inversion attack (gradient-based) for SimpleCNN CIFAR-10")
    parser.add_argument('--ckpt', type=str, default='./results/best_cnn_cifar10.pth', help='path to saved model .pth (state_dict)')
    parser.add_argument('--target', type=int, default=0, help='target class index (0-9)')
    parser.add_argument('--steps', type=int, default=1500, help='optimization steps per restart')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate for optimizer on pixels')
    parser.add_argument('--l2', type=float, default=1e-3, help='L2 regularization weight')
    parser.add_argument('--tv', type=float, default=1e-3, help='total variation weight')
    parser.add_argument('--restarts', type=int, default=3, help='random restarts')
    parser.add_argument('--out', type=str, default='./inversions', help='output folder')
    parser.add_argument('--use_mps', action='store_true', help='force mps if available')
    parser.add_argument('--show', action='store_true', help='show progress bars')
    args = parser.parse_args()

    # device logic
    if args.use_mps and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # load model
    model = SimpleCNN(num_classes=10).to(device)
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    state = torch.load(args.ckpt, map_location='cpu')
    # handle either a plain state_dict or your ckpt dict
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'])
        # extract stored normalization if present
        mean = state.get('train_mean', (0.4914, 0.4822, 0.4465))
        std = state.get('train_std', (0.2470, 0.2435, 0.2616))
    elif isinstance(state, dict) and all(k.startswith('module') or k in model.state_dict() for k in state.keys()):
        # plain state dict
        model.load_state_dict(state)
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    else:
        # unknown format fallback
        try:
            model.load_state_dict(state)
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2470, 0.2435, 0.2616)
        except Exception as e:
            raise RuntimeError("Could not load checkpoint. Provide either state_dict or your ckpt with 'model_state' key.") from e

    model.eval()

    invert_target(model, device, args.target,
                  ckpt_normalize_mean=mean, ckpt_normalize_std=std,
                  steps=args.steps, lr=args.lr,
                  l2=args.l2, tv=args.tv, restarts=args.restarts,
                  save_path=args.out, show_progress=args.show)

if __name__ == '__main__':
    main_cli()
