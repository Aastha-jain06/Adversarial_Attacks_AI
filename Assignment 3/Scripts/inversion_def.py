import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from utils import get_dataloaders, accuracy
import torch.nn.functional as F

# ------------------------------
# Define CNN with Defense Mechanisms
# ------------------------------
class SimpleCNNWithDefense(nn.Module):
    """
    CNN with multiple defense mechanisms against model inversion:
    1. Gradient masking via input noise during training
    2. Temperature scaling on outputs
    3. Top-k softmax (only return top-k predictions)
    4. Confidence thresholding
    """
    def __init__(self, num_classes=10, temperature=3.0, top_k=3, 
                 training_noise_std=0.1, enable_defenses=True):
        super(SimpleCNNWithDefense, self).__init__()
        self.temperature = temperature
        self.top_k = top_k
        self.training_noise_std = training_noise_std
        self.enable_defenses = enable_defenses
        
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv block 3
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

    def forward(self, x, return_raw_logits=False):
        # Defense 1: Add noise during training (gradient masking)
        if self.training and self.enable_defenses and self.training_noise_std > 0:
            noise = torch.randn_like(x) * self.training_noise_std
            x = x + noise
            x = torch.clamp(x, 0, 1)  # Keep in valid range
        
        x = self.features(x)
        logits = self.classifier(x)
        
        # Return raw logits for training
        if return_raw_logits or not self.enable_defenses:
            return logits
        
        # Defense 2: Temperature scaling (smooths the output distribution)
        scaled_logits = logits / self.temperature
        
        # Defense 3: Top-k output (only return top-k predictions)
        if not self.training and self.top_k > 0:
            # Zero out all but top-k logits
            topk_vals, topk_indices = torch.topk(scaled_logits, self.top_k, dim=1)
            mask = torch.zeros_like(scaled_logits)
            mask.scatter_(1, topk_indices, 1.0)
            scaled_logits = scaled_logits * mask + (1 - mask) * (-1e9)
        
        return scaled_logits


# ------------------------------
# Training with Mixup Augmentation (Additional Defense)
# ------------------------------
def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation to reduce model confidence"""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ------------------------------
# Training and evaluation
# ------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=True, mixup_alpha=0.2):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        
        if use_mixup and mixup_alpha > 0:
            x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)
            optimizer.zero_grad()
            logits = model(x, return_raw_logits=True)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            logits = model(x, return_raw_logits=True)
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
            logits = model(x, return_raw_logits=True)
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

    # Create model with defenses enabled
    model = SimpleCNNWithDefense(
        num_classes=10,
        temperature=3.0,          # Higher = more smoothing
        top_k=3,                  # Only return top-3 predictions
        training_noise_std=0.05,  # Input noise during training
        enable_defenses=True
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_val_acc = 0.0
    for epoch in range(1, 20):
        # Train with mixup augmentation (additional defense)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            use_mixup=True, mixup_alpha=0.2
        )
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch:03d} | Train acc {train_acc:.4f} loss {train_loss:.4f} | "
              f"Val acc {val_acc:.4f} loss {val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Move to CPU before saving
            model_cpu = model.to('cpu')
            
            ckpt = {
                'arch': 'SimpleCNNWithDefense',
                'num_classes': 10,
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'model_state': model_cpu.state_dict(),
                'train_mean': (0.4914, 0.4822, 0.4465),
                'train_std': (0.2470, 0.2435, 0.2616),
                # Save defense parameters
                'defense_params': {
                    'temperature': model_cpu.temperature,
                    'top_k': model_cpu.top_k,
                    'training_noise_std': model_cpu.training_noise_std,
                    'enable_defenses': model_cpu.enable_defenses
                }
            }
            torch.save(ckpt, os.path.join(save_dir, 'best_cnn_cifar10_defended.ckpt'))
            
            # Move back to device
            model = model.to(device)

    # Final test
    model.load_state_dict(
        torch.load(os.path.join(save_dir, 'best_cnn_cifar10_defended.ckpt'))['model_state']
    )
    model = model.to(device)
    test_loss, test_acc = eval_model(model, test_loader, criterion, device)
    
    print(f"\n{'='*50}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"{'='*50}")
    print("\nDefense mechanisms enabled:")
    print(f"  - Temperature scaling: {model.temperature}")
    print(f"  - Top-k predictions: {model.top_k}")
    print(f"  - Training noise: {model.training_noise_std}")
    print(f"  - Mixup augmentation: alpha=0.2")


if __name__ == '__main__':
    main()