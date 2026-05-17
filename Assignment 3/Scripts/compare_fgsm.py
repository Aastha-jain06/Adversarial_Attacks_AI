import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import get_dataloaders

# ------------------------------
# Define the same CNN architecture
# ------------------------------
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


# ------------------------------
# FGSM Attack
# ------------------------------
def fgsm_attack(model, x, y, criterion, epsilon):
    """Generate FGSM adversarial examples"""
    x_adv = x.clone().detach().requires_grad_(True)
    
    logits = model(x_adv)
    loss = criterion(logits, y)
    
    model.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        grad_sign = x_adv.grad.sign()
        x_adv = x_adv + epsilon * grad_sign
        x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv.detach()


# ------------------------------
# Evaluation Functions
# ------------------------------
def evaluate_model(model, loader, device, epsilon=0.0):
    """
    Evaluate model accuracy on clean or adversarial examples
    
    Args:
        model: Neural network model
        loader: Data loader
        device: Device (cpu/cuda/mps)
        epsilon: If 0, evaluate on clean images; else generate FGSM attacks
    
    Returns:
        accuracy: Classification accuracy
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    
    for x, y in tqdm(loader, leave=False, desc=f'Eval (ε={epsilon:.3f})'):
        x, y = x.to(device), y.to(device)
        
        if epsilon > 0:
            # Generate adversarial examples
            x = fgsm_attack(model, x, y, criterion, epsilon)
        
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    accuracy = correct / total
    return accuracy


def evaluate_robustness_curve(model, loader, device, epsilons):
    """
    Evaluate model across multiple epsilon values
    
    Args:
        model: Neural network model
        loader: Data loader
        device: Device
        epsilons: List of epsilon values to test
    
    Returns:
        accuracies: List of accuracies for each epsilon
    """
    accuracies = []
    
    for eps in epsilons:
        acc = evaluate_model(model, loader, device, epsilon=eps)
        accuracies.append(acc)
        print(f"  ε={eps:.4f}: Accuracy = {acc:.4f} ({acc*100:.2f}%)")
    
    return accuracies


# ------------------------------
# Visualization
# ------------------------------
def plot_comparison(epsilons, normal_accs, defense_accs, save_path='./results/robustness_comparison.png'):
    """Plot robustness comparison between normal and defended models"""
    plt.figure(figsize=(10, 6))
    
    plt.plot(epsilons, [acc * 100 for acc in normal_accs], 
             marker='o', linewidth=2, markersize=8, label='Normal Training', color='red')
    plt.plot(epsilons, [acc * 100 for acc in defense_accs], 
             marker='s', linewidth=2, markersize=8, label='Adversarial Training (Defense)', color='blue')
    
    plt.xlabel('Epsilon (Perturbation Magnitude)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('FGSM Attack: Normal Training vs Adversarial Training Defense', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 100])
    
    # Add annotations for epsilon=0 (clean accuracy)
    plt.annotate(f'{normal_accs[0]*100:.1f}%', 
                xy=(epsilons[0], normal_accs[0]*100), 
                xytext=(10, -20), textcoords='offset points',
                fontsize=9, color='red')
    plt.annotate(f'{defense_accs[0]*100:.1f}%', 
                xy=(epsilons[0], defense_accs[0]*100), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, color='blue')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {save_path}")
    plt.show()


def visualize_adversarial_examples(model_normal, model_defense, loader, device, epsilon=0.03, num_samples=5):
    """Visualize adversarial examples and model predictions"""
    model_normal.eval()
    model_defense.eval()
    criterion = nn.CrossEntropyLoss()
    
    # CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Get one batch
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    
    # Take first num_samples images
    x = x[:num_samples]
    y = y[:num_samples]
    
    # Generate adversarial examples
    x_adv = fgsm_attack(model_normal, x, y, criterion, epsilon)
    
    # Get predictions
    with torch.no_grad():
        # Clean predictions
        pred_clean_normal = model_normal(x).argmax(dim=1)
        pred_clean_defense = model_defense(x).argmax(dim=1)
        
        # Adversarial predictions
        pred_adv_normal = model_normal(x_adv).argmax(dim=1)
        pred_adv_defense = model_defense(x_adv).argmax(dim=1)
    
    # Move to CPU for visualization
    x = x.cpu()
    x_adv = x_adv.cpu()
    y = y.cpu()
    pred_clean_normal = pred_clean_normal.cpu()
    pred_clean_defense = pred_clean_defense.cpu()
    pred_adv_normal = pred_adv_normal.cpu()
    pred_adv_defense = pred_adv_defense.cpu()
    
    # Denormalize for visualization
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    x_vis = x * std + mean
    x_adv_vis = x_adv * std + mean
    
    # Compute perturbation
    perturbation = (x_adv_vis - x_vis).abs()
    
    # Plot
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 4))
    
    for i in range(num_samples):
        true_label = classes[y[i]]
        
        # Clean image
        axes[i, 0].imshow(x_vis[i].permute(1, 2, 0).clamp(0, 1))
        axes[i, 0].set_title(f'Clean Image\nTrue: {true_label}', fontsize=10)
        axes[i, 0].axis('off')
        
        # Perturbation
        axes[i, 1].imshow(perturbation[i].permute(1, 2, 0) * 10)  # Amplified for visibility
        axes[i, 1].set_title(f'Perturbation (×10)\nε={epsilon}', fontsize=10)
        axes[i, 1].axis('off')
        
        # Adversarial image
        axes[i, 2].imshow(x_adv_vis[i].permute(1, 2, 0).clamp(0, 1))
        axes[i, 2].set_title(f'Adversarial Image', fontsize=10)
        axes[i, 2].axis('off')
        
        # Predictions comparison
        normal_clean_correct = '✓' if pred_clean_normal[i] == y[i] else '✗'
        normal_adv_correct = '✓' if pred_adv_normal[i] == y[i] else '✗'
        defense_clean_correct = '✓' if pred_clean_defense[i] == y[i] else '✗'
        defense_adv_correct = '✓' if pred_adv_defense[i] == y[i] else '✗'
        
        pred_text = (
            f"Normal Model:\n"
            f"  Clean: {classes[pred_clean_normal[i]]} {normal_clean_correct}\n"
            f"  Adv: {classes[pred_adv_normal[i]]} {normal_adv_correct}\n\n"
            f"Defense Model:\n"
            f"  Clean: {classes[pred_clean_defense[i]]} {defense_clean_correct}\n"
            f"  Adv: {classes[pred_adv_defense[i]]} {defense_adv_correct}"
        )
        
        axes[i, 3].text(0.1, 0.5, pred_text, fontsize=9, verticalalignment='center',
                       family='monospace')
        axes[i, 3].axis('off')
    
    plt.suptitle(f'FGSM Attack Visualization (ε={epsilon})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./results/adversarial_examples_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved to: ./results/adversarial_examples_comparison.png")
    plt.show()


# ------------------------------
# Main Evaluation
# ------------------------------
def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load test data
    _, _, test_loader = get_dataloaders(batch_size=128)
    
    # Load models
    print("Loading models...")
    
    # Normal trained model
    model_normal = SimpleCNN(num_classes=10).to(device)
    try:
        model_normal.load_state_dict(torch.load('./results/best_cnn_cifar10.pth', map_location=device))
        print("✓ Normal trained model loaded")
    except FileNotFoundError:
        print("✗ Normal trained model not found at './results/best_cnn_cifar10.pth'")
        return
    
    # Adversarially trained model (defense)
    model_defense = SimpleCNN(num_classes=10).to(device)
    try:
        model_defense.load_state_dict(torch.load('./results/best_cnn_cifar10_adv_trained.pth', map_location=device))
        print("✓ Adversarially trained model loaded")
    except FileNotFoundError:
        print("✗ Adversarially trained model not found at './results/best_cnn_cifar10_adv_trained.pth'")
        return
    
    print("\n" + "="*80)
    print("EVALUATION: Normal Training vs Adversarial Training Defense")
    print("="*80)
    
    # Define epsilon values to test
    epsilons = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    
    # Evaluate normal model
    print("\n[1/2] Evaluating Normal Trained Model:")
    print("-" * 80)
    normal_accuracies = evaluate_robustness_curve(model_normal, test_loader, device, epsilons)
    
    # Evaluate defense model
    print("\n[2/2] Evaluating Adversarially Trained Model (Defense):")
    print("-" * 80)
    defense_accuracies = evaluate_robustness_curve(model_defense, test_loader, device, epsilons)
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Epsilon':<12} {'Normal Model':<20} {'Defense Model':<20} {'Improvement':<15}")
    print("-" * 80)
    
    for eps, normal_acc, defense_acc in zip(epsilons, normal_accuracies, defense_accuracies):
        improvement = (defense_acc - normal_acc) * 100
        improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
        
        print(f"{eps:<12.3f} {normal_acc*100:>6.2f}%{'':<12} {defense_acc*100:>6.2f}%{'':<12} {improvement_str:<15}")
    
    print("="*80)
    
    # Plot comparison
    print("\nGenerating robustness comparison plot...")
    plot_comparison(epsilons, normal_accuracies, defense_accuracies)
    
    # Visualize adversarial examples
    print("\nGenerating adversarial examples visualization...")
    visualize_adversarial_examples(model_normal, model_defense, test_loader, device, epsilon=0.03, num_samples=5)
    
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()