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
# FGSM Attack Generation
# ------------------------------
def fgsm_attack(model, x, y, criterion, epsilon):
    """
    Generate FGSM adversarial examples
    
    Args:
        model: Neural network model
        x: Input images
        y: True labels
        criterion: Loss function
        epsilon: Perturbation magnitude
    
    Returns:
        x_adv: Adversarial examples
    """
    x_adv = x.clone().detach().requires_grad_(True)
    
    # Forward pass
    logits = model(x_adv)
    loss = criterion(logits, y)
    
    # Backward pass to get gradients
    model.zero_grad()
    loss.backward()
    
    # Generate adversarial example using FGSM
    with torch.no_grad():
        # Get sign of gradients
        grad_sign = x_adv.grad.sign()
        # Add perturbation
        x_adv = x_adv + epsilon * grad_sign
        # Clamp to valid image range [0, 1]
        x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv.detach()


# ------------------------------
# Adversarial Training
# ------------------------------
def train_one_epoch_adversarial(model, loader, criterion, optimizer, device, epsilon=0.03, alpha=0.2):
    """
    Train with adversarial examples using FGSM
    
    Args:
        model: Neural network model
        loader: Data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cpu/cuda/mps)
        epsilon: Perturbation magnitude for FGSM
        alpha: Weight for adversarial loss (0.5 means equal weight)
    """
    model.train()
    running_loss, running_clean_loss, running_adv_loss = 0.0, 0.0, 0.0
    correct_clean, correct_adv, total = 0, 0, 0
    
    for x, y in tqdm(loader, leave=False, desc='Training'):
        x, y = x.to(device), y.to(device)
        
        # ====== Clean examples ======
        optimizer.zero_grad()
        logits_clean = model(x)
        loss_clean = criterion(logits_clean, y)
        
        # ====== Generate adversarial examples ======
        x_adv = fgsm_attack(model, x, y, criterion, epsilon)
        
        # Forward pass on adversarial examples
        logits_adv = model(x_adv)
        loss_adv = criterion(logits_adv, y)
        
        # ====== Combined loss ======
        # Mix clean and adversarial losses
        loss = (1 - alpha) * loss_clean + alpha * loss_adv
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * x.size(0)
        running_clean_loss += loss_clean.item() * x.size(0)
        running_adv_loss += loss_adv.item() * x.size(0)
        
        correct_clean += (logits_clean.argmax(dim=1) == y).sum().item()
        correct_adv += (logits_adv.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    
    return {
        'loss': running_loss / total,
        'clean_loss': running_clean_loss / total,
        'adv_loss': running_adv_loss / total,
        'clean_acc': correct_clean / total,
        'adv_acc': correct_adv / total
    }


def eval_model_with_attack(model, loader, criterion, device, epsilon=0.03):
    """
    Evaluate model on both clean and adversarial examples
    """
    model.eval()
    running_clean_loss, running_adv_loss = 0.0, 0.0
    correct_clean, correct_adv, total = 0, 0, 0
    
    for x, y in tqdm(loader, leave=False, desc='Evaluating'):
        x, y = x.to(device), y.to(device)
        
        with torch.no_grad():
            # Clean examples
            logits_clean = model(x)
            loss_clean = criterion(logits_clean, y)
            correct_clean += (logits_clean.argmax(dim=1) == y).sum().item()
            running_clean_loss += loss_clean.item() * x.size(0)
        
        # Adversarial examples (need gradients)
        x_adv = fgsm_attack(model, x, y, criterion, epsilon)
        
        with torch.no_grad():
            logits_adv = model(x_adv)
            loss_adv = criterion(logits_adv, y)
            correct_adv += (logits_adv.argmax(dim=1) == y).sum().item()
            running_adv_loss += loss_adv.item() * x.size(0)
        
        total += x.size(0)
    
    return {
        'clean_loss': running_clean_loss / total,
        'adv_loss': running_adv_loss / total,
        'clean_acc': correct_clean / total,
        'adv_acc': correct_adv / total
    }


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)

    # Initialize model
    model = SimpleCNN(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Adversarial training parameters
    epsilon = 0.03  # FGSM perturbation magnitude (8/255 ≈ 0.031)
    alpha = 0.2     # Weight for adversarial loss

    best_val_clean_acc = 0.0
    best_val_adv_acc = 0.0
    
    print(f"Starting adversarial training with epsilon={epsilon}, alpha={alpha}")
    print("=" * 80)
    
    for epoch in range(1, 70):
        # Adversarial training
        train_stats = train_one_epoch_adversarial(
            model, train_loader, criterion, optimizer, device, 
            epsilon=epsilon, alpha=alpha
        )
        
        # Evaluation on clean and adversarial examples
        val_stats = eval_model_with_attack(
            model, val_loader, criterion, device, epsilon=epsilon
        )
        
        scheduler.step()
        
        # Print statistics
        print(f"Epoch {epoch:03d}")
        print(f"  Train | Clean acc: {train_stats['clean_acc']:.4f}, "
              f"Adv acc: {train_stats['adv_acc']:.4f}, "
              f"Loss: {train_stats['loss']:.4f}")
        print(f"  Val   | Clean acc: {val_stats['clean_acc']:.4f}, "
              f"Adv acc: {val_stats['adv_acc']:.4f}")
        
        # Save best model based on validation adversarial accuracy
        if val_stats['adv_acc'] > best_val_adv_acc:
            best_val_clean_acc = val_stats['clean_acc']
            best_val_adv_acc = val_stats['adv_acc']
            
            # Move to CPU for saving
            model_cpu = model.to('cpu')
            
            # Save state dict
            torch.save(model_cpu.state_dict(), 
                      os.path.join(save_dir, 'best_cnn_cifar10_adv_trained.pth'))
            
            # Save full checkpoint
            ckpt = {
                'arch': 'SimpleCNN',
                'num_classes': 10,
                'epoch': epoch,
                'best_val_clean_acc': best_val_clean_acc,
                'best_val_adv_acc': best_val_adv_acc,
                'model_state': model_cpu.state_dict(),
                'train_mean': (0.4914, 0.4822, 0.4465),
                'train_std': (0.2470, 0.2435, 0.2616),
                'epsilon': epsilon,
                'alpha': alpha
            }
            torch.save(ckpt, os.path.join(save_dir, 'best_cnn_cifar10_adv_trained.ckpt'))
            
            # Move back to device
            model = model.to(device)
            
            print(f"  ✓ Best model saved (Adv acc: {best_val_adv_acc:.4f})")
        
        print("-" * 80)

    # Final test evaluation
    print("\nFinal Test Evaluation:")
    print("=" * 80)
    
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_cnn_cifar10_adv_trained.pth')))
    model = model.to(device)
    
    test_stats = eval_model_with_attack(model, test_loader, criterion, device, epsilon=epsilon)
    
    print(f"Best validation | Clean acc: {best_val_clean_acc:.4f}, Adv acc: {best_val_adv_acc:.4f}")
    print(f"Test            | Clean acc: {test_stats['clean_acc']:.4f}, Adv acc: {test_stats['adv_acc']:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()