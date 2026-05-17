import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader, Subset
import numpy as np

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),      # brightness/contrast/saturation/ change
            transforms.RandomRotation(15),                    # random rotation
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])

def get_dataloaders(root='./data', batch_size=128, val_size=5000, num_workers=7):
    train_full = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=get_transforms(True))
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=get_transforms(False))

    # split train -> train & val
    train_size = len(train_full) - val_size
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def accuracy(outputs, targets, topk=(1,)):
    preds = outputs.argmax(dim=1)
    correct = preds.eq(targets).sum().item()
    return correct / targets.size(0)
