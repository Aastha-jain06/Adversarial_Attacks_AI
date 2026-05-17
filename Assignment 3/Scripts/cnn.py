import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from utils import get_dataloaders, accuracy

# ------------------------------
# Define a simple CNN for CIFAR-10
# ------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 32 -> 16

            # Conv block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 16 -> 8

            # Conv block 3
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


# ------------------------------
# Training and evaluation
# ------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total


def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return running_loss / total, correct / total


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)

    # Use custom CNN instead of ResNet
    model = SimpleCNN(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_val_acc = 0.0
    for epoch in range(1, 70):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch:03d} | Train acc {train_acc:.4f} loss {train_loss:.4f} | Val acc {val_acc:.4f} loss {val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_cnn_cifar10.pth'))

            # move model to CPU first (avoid MPS memory spike) - Since laptop was gettigng shutdown during saving state.
            model_cpu = model.to('cpu')

            ckpt = {
               'arch': 'SimpleCNN',
                'num_classes': 10,
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'model_state': model_cpu.state_dict(),
                'train_mean': (0.4914, 0.4822, 0.4465),   # <-- CIFAR10 Normalization
                'train_std' : (0.2470, 0.2435, 0.2616)
            }
            torch.save(ckpt, os.path.join(save_dir, 'best_cnn_cifar10.ckpt'))  # note .ckpt extension

            # move model back to device for further training
            model = model.to(device)

    # final test
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_cnn_cifar10.pth')))
    test_loss, test_acc = eval_model(model, test_loader, criterion, device)
    print("Best val acc:", best_val_acc)
    print("Test acc:", test_acc)


if __name__ == '__main__':
    main()
