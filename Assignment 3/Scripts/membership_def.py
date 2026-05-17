import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from utils import get_dataloaders, accuracy
import numpy as np

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

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ------------------------------
# Defense mechanisms
# ------------------------------
def add_gradient_noise(model, noise_scale=0.001):
    """Add noise to gradients (DP-SGD style)"""
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_scale
                param.grad.add_(noise)


def gradient_clipping(model, max_norm=1.0):
    """Clip gradients by norm"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


class MixupDataLoader:
    """Mixup augmentation wrapper"""
    def __init__(self, loader, alpha=1.0):
        self.loader = loader
        self.alpha = alpha
    
    def __iter__(self):
        for x, y in self.loader:
            if self.alpha > 0:
                lam = np.random.beta(self.alpha, self.alpha)
                batch_size = x.size(0)
                index = torch.randperm(batch_size)
                
                mixed_x = lam * x + (1 - lam) * x[index]
                y_a, y_b = y, y[index]
                yield mixed_x, y_a, y_b, lam
            else:
                yield x, y, y, 1.0
    
    def __len__(self):
        return len(self.loader)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss computation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ------------------------------
# Training with defenses
# ------------------------------
def train_one_epoch_with_defenses(model, loader, criterion, optimizer, device, 
                                   defense_config):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    # Wrap loader with mixup if enabled
    if defense_config.get('use_mixup', False):
        loader = MixupDataLoader(loader, alpha=defense_config.get('mixup_alpha', 1.0))
    
    for batch_data in tqdm(loader, leave=False):
        if defense_config.get('use_mixup', False):
            x, y_a, y_b, lam = batch_data
            x = x.to(device)
            y_a = y_a.to(device)
            y_b = y_b.to(device)
        else:
            x, y = batch_data
            x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        
        # Compute loss
        if defense_config.get('use_mixup', False):
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            loss = criterion(logits, y)
        
        loss.backward()
        
        # Apply gradient clipping
        if defense_config.get('use_gradient_clipping', False):
            gradient_clipping(model, max_norm=defense_config.get('clip_norm', 1.0))
        
        # Add gradient noise (DP-SGD style)
        if defense_config.get('use_gradient_noise', False):
            add_gradient_noise(model, noise_scale=defense_config.get('noise_scale', 0.001))
        
        optimizer.step()
        
        running_loss += loss.item() * x.size(0)
        if defense_config.get('use_mixup', False):
            # For mixup, use the primary label for accuracy
            correct += (logits.argmax(dim=1) == y_a).sum().item()
        else:
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


def train_with_defenses(defense_config, save_suffix='defended'):
    """
    Train model with specified defense configurations
    
    defense_config: dict with keys like:
        - use_label_smoothing: bool
        - label_smoothing: float
        - use_mixup: bool
        - mixup_alpha: float
        - use_gradient_clipping: bool
        - clip_norm: float
        - use_gradient_noise: bool
        - noise_scale: float
        - use_dropout: bool
        - dropout_rate: float
        - weight_decay: float
        - early_stopping_patience: int
    """
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)

    model = SimpleCNN(num_classes=10).to(device)

    # Label smoothing
    label_smoothing = defense_config.get('label_smoothing', 0.1) if defense_config.get('use_label_smoothing', True) else 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Weight decay (L2 regularization)
    weight_decay = defense_config.get('weight_decay', 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_val_acc = 0.0
    patience_counter = 0
    patience = defense_config.get('early_stopping_patience', 10)
    
    num_epochs = defense_config.get('num_epochs', 20)
    
    print(f"\n{'='*60}")
    print(f"Training with defenses: {save_suffix}")
    print(f"Defense config: {defense_config}")
    print(f"{'='*60}\n")
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch_with_defenses(
            model, train_loader, criterion, optimizer, device, defense_config
        )
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch:03d} | Train acc {train_acc:.4f} loss {train_loss:.4f} | "
              f"Val acc {val_acc:.4f} loss {val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save model
            model_path = os.path.join(save_dir, f'best_cnn_cifar10_{save_suffix}.pth')
            torch.save(model.state_dict(), model_path)

            # Save checkpoint
            model_cpu = model.to('cpu')
            ckpt = {
                'arch': 'SimpleCNN',
                'num_classes': 10,
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'model_state': model_cpu.state_dict(),
                'train_mean': (0.4914, 0.4822, 0.4465),
                'train_std': (0.2470, 0.2435, 0.2616),
                'defense_config': defense_config
            }
            ckpt_path = os.path.join(save_dir, f'best_cnn_cifar10_{save_suffix}.ckpt')
            torch.save(ckpt, ckpt_path)
            
            model = model.to(device)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Final test
    model.load_state_dict(torch.load(os.path.join(save_dir, f'best_cnn_cifar10_{save_suffix}.pth')))
    test_loss, test_acc = eval_model(model, test_loader, criterion, device)
    print(f"\nBest val acc: {best_val_acc:.4f}")
    print(f"Test acc: {test_acc:.4f}\n")
    
    return {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'checkpoint_path': os.path.join(save_dir, f'best_cnn_cifar10_{save_suffix}.ckpt')
    }


def main():
    # Define different defense strategies
    
    # 1. Baseline (minimal defense - your original model)
    baseline_config = {
        'use_label_smoothing': True,
        'label_smoothing': 0.1,
        'use_mixup': False,
        'use_gradient_clipping': False,
        'use_gradient_noise': False,
        'weight_decay': 1e-4,
        'early_stopping_patience': 15,
        'num_epochs': 20
    }
    
    # 2. Strong regularization
    strong_reg_config = {
        'use_label_smoothing': True,
        'label_smoothing': 0.2,
        'use_mixup': False,
        'use_gradient_clipping': False,
        'use_gradient_noise': False,
        'weight_decay': 5e-4,  # Increased weight decay
        'early_stopping_patience': 15,
        'num_epochs': 20
    }
    
    # 3. Mixup defense
    mixup_config = {
        'use_label_smoothing': True,
        'label_smoothing': 0.1,
        'use_mixup': True,
        'mixup_alpha': 1.0,
        'use_gradient_clipping': False,
        'use_gradient_noise': False,
        'weight_decay': 1e-4,
        'early_stopping_patience': 15,
        'num_epochs': 20
    }
    
    # 4. DP-SGD style (gradient noise + clipping)
    dpsgd_config = {
        'use_label_smoothing': True,
        'label_smoothing': 0.1,
        'use_mixup': False,
        'use_gradient_clipping': True,
        'clip_norm': 1.0,
        'use_gradient_noise': True,
        'noise_scale': 0.01,  # Adjust based on privacy budget
        'weight_decay': 1e-4,
        'early_stopping_patience': 15,
        'num_epochs': 20
    }
    
    # 5. Combined defenses
    combined_config = {
        'use_label_smoothing': True,
        'label_smoothing': 0.2,
        'use_mixup': True,
        'mixup_alpha': 0.8,
        'use_gradient_clipping': True,
        'clip_norm': 1.0,
        'use_gradient_noise': True,
        'noise_scale': 0.005,
        'weight_decay': 5e-4,
        'early_stopping_patience': 15,
        'num_epochs': 20
    }
    
    # Train models with different defenses
    configs = [
        (baseline_config, 'baseline'),
        (strong_reg_config, 'strong_reg'),
        (mixup_config, 'mixup'),
        (dpsgd_config, 'dpsgd'),
        (combined_config, 'combined')
    ]
    
    results = {}
    for config, name in configs:
        print(f"\n{'#'*60}")
        print(f"Training: {name}")
        print(f"{'#'*60}")
        result = train_with_defenses(config, save_suffix=name)
        results[name] = result
    
    print("\n" + "="*60)
    print("SUMMARY OF ALL MODELS")
    print("="*60)
    for name, result in results.items():
        print(f"{name:20s} | Val Acc: {result['best_val_acc']:.4f} | Test Acc: {result['test_acc']:.4f}")


if __name__ == '__main__':
    main()