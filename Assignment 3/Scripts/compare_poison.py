# compare_defense.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

# ---------------------------
# Model Definition
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

# ---------------------------
# Evaluation Functions
# ---------------------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

def add_patch_to_pil(img_pil, patch_size=6, color=(255,0,0), location='bottom_right'):
    img = img_pil.copy()
    w, h = img.size
    if location == 'bottom_right':
        x0, y0 = w - patch_size - 1, h - patch_size - 1
    elif location == 'top_left':
        x0, y0 = 1, 1
    else:
        x0, y0 = (w - patch_size)//2, (h - patch_size)//2
    draw = ImageDraw.Draw(img)
    draw.rectangle([x0, y0, x0+patch_size, y0+patch_size], fill=color)
    return img

def tensor_to_uint8(tensor):
    t = tensor.clone().cpu()
    for c in range(3):
        t[c] = t[c] * CIFAR_STD[c] + CIFAR_MEAN[c]
    t = torch.clamp(t, 0.0, 1.0)
    arr = (t.numpy().transpose(1,2,0) * 255).astype(np.uint8)
    return arr

def evaluate_model(model_path, device, test_loader):
    """Evaluate model on clean test set"""
    model = SimpleCNN(num_classes=10).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    
    correct = 0
    total = 0
    per_class = {i: {'correct': 0, 'total': 0} for i in range(10)}
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            
            for i in range(x.size(0)):
                gt = int(y[i].cpu())
                p = int(preds[i].cpu())
                per_class[gt]['total'] += 1
                if p == gt:
                    per_class[gt]['correct'] += 1
    
    acc = correct / total
    per_class_acc = {cls: (per_class[cls]['correct']/max(1, per_class[cls]['total'])) 
                     for cls in per_class}
    
    return acc, per_class_acc

def evaluate_backdoor(model_path, device, test_dataset, backdoor_target=0):
    """Evaluate backdoor attack success rate"""
    model = SimpleCNN(num_classes=10).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    
    # Create backdoored test set
    bd_images = []
    for i in range(len(test_dataset)):
        img, _ = test_dataset[i]
        # Convert tensor back to PIL
        pil = Image.fromarray(tensor_to_uint8(img))
        pil_bd = add_patch_to_pil(pil, patch_size=6, color=(255,0,0))
        bd_images.append(test_transform(pil_bd))
    
    total = len(bd_images)
    success = 0
    
    with torch.no_grad():
        for img in bd_images:
            img = img.unsqueeze(0).to(device)
            out = model(img)
            pred = out.argmax(1).item()
            if pred == backdoor_target:
                success += 1
    
    return success / total

def compute_confusion_matrix(model_path, device, test_loader):
    """Compute confusion matrix"""
    model = SimpleCNN(num_classes=10).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    
    conf = np.zeros((10, 10), dtype=int)
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)
            for gt, p in zip(y.cpu().numpy(), preds.cpu().numpy()):
                conf[gt, p] += 1
    
    return conf

# ---------------------------
# Comparison Functions
# ---------------------------

def load_results(results_dir):
    """Load results from JSON file"""
    json_path = os.path.join(results_dir, 'report.json')
    if not os.path.exists(json_path):
        json_path = os.path.join(results_dir, 'defense_results.json')
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return None

def print_per_class_accuracies(results):
    """Print per-class accuracies in a formatted table"""
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("\n" + "=" * 100)
    print("PER-CLASS ACCURACIES")
    print("=" * 100)
    
    # Header
    header = f"{'Class':<12}"
    for model_name in results.keys():
        header += f"{model_name.replace('_', ' ').title():<20}"
    print(header)
    print("-" * 100)
    
    # Per-class rows
    for i, class_name in enumerate(class_names):
        row = f"{class_name:<12}"
        for model_name in results.keys():
            acc = results[model_name]['per_class_acc'][i]
            row += f"{acc:>6.4f} ({acc*100:>5.1f}%)    "
        print(row)
    
    # Average row
    print("-" * 100)
    row = f"{'AVERAGE':<12}"
    for model_name in results.keys():
        avg_acc = results[model_name]['clean_acc']
        row += f"{avg_acc:>6.4f} ({avg_acc*100:>5.1f}%)    "
    print(row)
    print("=" * 100)

def print_confusion_matrices(results):
    """Print confusion matrices for all models"""
    class_names = ['air', 'auto', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'hrs', 'ship', 'trk']
    
    for model_name, res in results.items():
        print(f"\n{'=' * 80}")
        print(f"CONFUSION MATRIX: {model_name.replace('_', ' ').title()}")
        print("=" * 80)
        
        conf_matrix = np.array(res['confusion_matrix'])
        
        # Header
        header = "True\\Pred  "
        for name in class_names:
            header += f"{name:>6}"
        print(header)
        print("-" * 80)
        
        # Rows
        for i, name in enumerate(class_names):
            row = f"{name:<10}"
            for j in range(10):
                row += f"{conf_matrix[i, j]:>6}"
            print(row)
        print("=" * 80)

def print_detailed_metrics(results):
    """Print detailed metrics comparison"""
    print("\n" + "=" * 100)
    print("DETAILED METRICS COMPARISON")
    print("=" * 100)
    
    models = list(results.keys())
    
    # Clean Accuracy Section
    print("\n1. CLEAN TEST ACCURACY")
    print("-" * 100)
    for model_name in models:
        acc = results[model_name]['clean_acc']
        print(f"   {model_name.replace('_', ' ').title():<35} {acc:.6f} ({acc*100:.2f}%)")
    
    # Backdoor Success Section
    if 'backdoor_success' in results[models[0]]:
        print("\n2. BACKDOOR ATTACK SUCCESS RATE")
        print("-" * 100)
        for model_name in models:
            if 'backdoor_success' in results[model_name]:
                bd = results[model_name]['backdoor_success']
                print(f"   {model_name.replace('_', ' ').title():<35} {bd:.6f} ({bd*100:.2f}%)")
    
    # Improvement Metrics
    if len(models) > 1:
        print("\n3. IMPROVEMENT OVER NO DEFENSE")
        print("-" * 100)
        no_def_acc = results['no_defense']['clean_acc']
        
        for model_name in models[1:]:
            acc_imp = (results[model_name]['clean_acc'] - no_def_acc) * 100
            print(f"   {model_name.replace('_', ' ').title():<35} Accuracy: {acc_imp:+.2f}%")
            
            if 'backdoor_success' in results['no_defense'] and 'backdoor_success' in results[model_name]:
                no_def_bd = results['no_defense']['backdoor_success']
                bd_red = (no_def_bd - results[model_name]['backdoor_success']) * 100
                print(f"   {'':<35} Backdoor Reduction: {bd_red:+.2f}%")
    
    print("=" * 100)

def compare_models(no_defense_dir, defense_dirs, output_dir='./comparison_results'):
    """
    Compare models with and without defense
    
    Args:
        no_defense_dir: Directory with poisoned model (no defense)
        defense_dirs: List of tuples (name, directory) for defense models
        output_dir: Where to save comparison results
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"\nUsing device: {device}")
    
    # Load test data
    print("\nLoading CIFAR-10 test dataset...")
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print(f"Test dataset size: {len(test_dataset)} images")
    
    results = {}
    
    # Evaluate model without defense
    print("\n" + "=" * 70)
    print("EVALUATING MODEL WITHOUT DEFENSE")
    print("=" * 70)
    no_def_ckpt = os.path.join(no_defense_dir, 'chk_ep30.pth')
    if not os.path.exists(no_def_ckpt):
        # Try to find any checkpoint
        ckpts = [f for f in os.listdir(no_defense_dir) if f.endswith('.pth')]
        if ckpts:
            no_def_ckpt = os.path.join(no_defense_dir, ckpts[-1])
    
    print(f"Loading checkpoint: {no_def_ckpt}")
    no_def_acc, no_def_per_class = evaluate_model(no_def_ckpt, device, test_loader)
    print(f"Computing confusion matrix...")
    no_def_conf = compute_confusion_matrix(no_def_ckpt, device, test_loader)
    
    results['no_defense'] = {
        'clean_acc': no_def_acc,
        'per_class_acc': no_def_per_class,
        'confusion_matrix': no_def_conf.tolist()
    }
    
    print(f"\n✓ Clean Test Accuracy: {no_def_acc:.6f} ({no_def_acc*100:.2f}%)")
    
    # Try to evaluate backdoor if applicable
    no_def_results = load_results(no_defense_dir)
    if no_def_results and no_def_results.get('poison_type') == 'backdoor':
        print(f"Evaluating backdoor attack success rate...")
        bd_success = evaluate_backdoor(no_def_ckpt, device, test_dataset, backdoor_target=0)
        results['no_defense']['backdoor_success'] = bd_success
        print(f"✓ Backdoor Attack Success Rate: {bd_success:.6f} ({bd_success*100:.2f}%)")
    
    # Evaluate models with defense
    for def_name, def_dir in defense_dirs:
        print("\n" + "=" * 70)
        print(f"EVALUATING MODEL WITH {def_name.upper()}")
        print("=" * 70)
        def_ckpt = os.path.join(def_dir, 'defense_chk_ep30.pth')
        if not os.path.exists(def_ckpt):
            ckpts = [f for f in os.listdir(def_dir) if f.endswith('.pth')]
            if ckpts:
                def_ckpt = os.path.join(def_dir, ckpts[-1])
        
        print(f"Loading checkpoint: {def_ckpt}")
        def_acc, def_per_class = evaluate_model(def_ckpt, device, test_loader)
        print(f"Computing confusion matrix...")
        def_conf = compute_confusion_matrix(def_ckpt, device, test_loader)
        
        results[def_name] = {
            'clean_acc': def_acc,
            'per_class_acc': def_per_class,
            'confusion_matrix': def_conf.tolist()
        }
        
        print(f"\n✓ Clean Test Accuracy: {def_acc:.6f} ({def_acc*100:.2f}%)")
        
        # Try to evaluate backdoor if applicable
        def_results = load_results(def_dir)
        if def_results and no_def_results and no_def_results.get('poison_type') == 'backdoor':
            print(f"Evaluating backdoor attack success rate...")
            bd_success = evaluate_backdoor(def_ckpt, device, test_dataset, backdoor_target=0)
            results[def_name]['backdoor_success'] = bd_success
            print(f"✓ Backdoor Attack Success Rate: {bd_success:.6f} ({bd_success*100:.2f}%)")
    
    # Save comparison results
    print(f"\n{'=' * 70}")
    print("SAVING RESULTS")
    print("=" * 70)
    comparison_file = os.path.join(output_dir, 'comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Comparison results saved to: {comparison_file}")
    
    # Print detailed results to console
    print_detailed_metrics(results)
    print_per_class_accuracies(results)
    print_confusion_matrices(results)
    
    # Generate visualizations
    print(f"\n{'=' * 70}")
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    plot_comparisons(results, output_dir, no_def_results)
    
    print(f"\n{'=' * 70}")
    print("COMPARISON COMPLETE!")
    print("=" * 70)
    print(f"All results saved to: {output_dir}")
    print(f"  - comparison.json")
    print(f"  - accuracy_comparison.png")
    print(f"  - per_class_comparison.png")
    if 'backdoor_success' in results['no_defense']:
        print(f"  - backdoor_comparison.png")
    if len(results) > 1:
        print(f"  - improvement_metrics.png")
    
    return results

def plot_comparisons(results, output_dir, poison_info):
    """Generate comparison plots"""
    
    # Plot 1: Overall accuracy comparison
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    accuracies = [results[m]['clean_acc'] for m in models]
    colors = ['#e74c3c' if m == 'no_defense' else '#2ecc71' for m in models]
    
    bars = plt.bar(range(len(models)), accuracies, color=colors, alpha=0.7, edgecolor='black')
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    plt.title('Test Accuracy: Without vs With Defense', fontsize=14, fontweight='bold')
    plt.xticks(range(len(models)), [m.replace('_', ' ').title() for m in models], rotation=15, ha='right')
    plt.ylim([min(accuracies) - 0.05, 1.0])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'accuracy_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()
    
    # Plot 2: Per-class accuracy comparison
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))
    if len(results) == 1:
        axes = [axes]
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    for idx, (model_name, ax) in enumerate(zip(models, axes)):
        per_class = results[model_name]['per_class_acc']
        class_accs = [per_class[i] for i in range(10)]
        
        bars = ax.bar(range(10), class_accs, color='#3498db', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Class', fontsize=10, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=10, fontweight='bold')
        ax.set_title(f'{model_name.replace("_", " ").title()}', fontsize=11, fontweight='bold')
        ax.set_xticks(range(10))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=results[model_name]['clean_acc'], color='r', linestyle='--', 
                   label=f'Avg: {results[model_name]["clean_acc"]:.3f}')
        ax.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'per_class_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()
    
    # Plot 3: Backdoor success rate comparison (if applicable)
    if 'backdoor_success' in results['no_defense']:
        plt.figure(figsize=(10, 6))
        bd_models = [m for m in models if 'backdoor_success' in results[m]]
        bd_success = [results[m]['backdoor_success'] for m in bd_models]
        colors = ['#e74c3c' if m == 'no_defense' else '#2ecc71' for m in bd_models]
        
        bars = plt.bar(range(len(bd_models)), bd_success, color=colors, alpha=0.7, edgecolor='black')
        plt.xlabel('Model', fontsize=12, fontweight='bold')
        plt.ylabel('Backdoor Attack Success Rate', fontsize=12, fontweight='bold')
        plt.title('Backdoor Attack Success: Without vs With Defense', fontsize=14, fontweight='bold')
        plt.xticks(range(len(bd_models)), [m.replace('_', ' ').title() for m in bd_models], 
                   rotation=15, ha='right')
        plt.ylim([0, 1])
        plt.grid(axis='y', alpha=0.3)
        
        for bar, bd in zip(bars, bd_success):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{bd:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'backdoor_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_path}")
        plt.close()
    
    # Plot 4: Improvement metrics
    if len(models) > 1:
        plt.figure(figsize=(12, 5))
        
        # Accuracy improvement
        plt.subplot(1, 2, 1)
        no_def_acc = results['no_defense']['clean_acc']
        improvements = [(results[m]['clean_acc'] - no_def_acc) * 100 for m in models[1:]]
        colors_imp = ['#2ecc71' if x >= 0 else '#e74c3c' for x in improvements]
        
        bars = plt.barh(range(len(improvements)), improvements, color=colors_imp, alpha=0.7, edgecolor='black')
        plt.ylabel('Defense Method', fontsize=11, fontweight='bold')
        plt.xlabel('Accuracy Improvement (%)', fontsize=11, fontweight='bold')
        plt.title('Accuracy Improvement Over No Defense', fontsize=12, fontweight='bold')
        plt.yticks(range(len(improvements)), [m.replace('_', ' ').title() for m in models[1:]])
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.grid(axis='x', alpha=0.3)
        
        for bar, imp in zip(bars, improvements):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{imp:+.2f}%', ha='left' if width >= 0 else 'right', 
                    va='center', fontweight='bold', fontsize=9)
        
        # Backdoor reduction (if applicable)
        if 'backdoor_success' in results['no_defense']:
            plt.subplot(1, 2, 2)
            no_def_bd = results['no_defense']['backdoor_success']
            bd_models = [m for m in models[1:] if 'backdoor_success' in results[m]]
            reductions = [(no_def_bd - results[m]['backdoor_success']) * 100 for m in bd_models]
            colors_red = ['#2ecc71' if x >= 0 else '#e74c3c' for x in reductions]
            
            bars = plt.barh(range(len(reductions)), reductions, color=colors_red, alpha=0.7, edgecolor='black')
            plt.ylabel('Defense Method', fontsize=11, fontweight='bold')
            plt.xlabel('Backdoor Success Reduction (%)', fontsize=11, fontweight='bold')
            plt.title('Backdoor Attack Mitigation', fontsize=12, fontweight='bold')
            plt.yticks(range(len(reductions)), [m.replace('_', ' ').title() for m in bd_models])
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            plt.grid(axis='x', alpha=0.3)
            
            for bar, red in zip(bars, reductions):
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{red:+.2f}%', ha='left' if width >= 0 else 'right',
                        va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'improvement_metrics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_path}")
        plt.close()

# ---------------------------
# Main
# ---------------------------

if __name__ == '__main__':
    # Example usage:
    # Compare poisoned model (no defense) with defended models
    
    print("\n" + "=" * 70)
    print(" " * 20 + "DEFENSE COMPARISON TOOL")
    print("=" * 70)
    
    # Specify directories
    no_defense_dir = './poison_labelflip_10'  # Poisoned model without defense
    defense_dirs = [
        ('loss_filtering', './defense_loss_filtering'),
        # Add more defense methods here:
        # ('activation_clustering', './defense_activation_clustering'),
        # ('ensemble', './defense_ensemble'),
    ]
    
    print(f"\nConfiguration:")
    print(f"  No Defense Dir: {no_defense_dir}")
    for name, path in defense_dirs:
        print(f"  Defense ({name}): {path}")
    print(f"  Output Dir: ./comparison_results")
    
    # Run comparison
    results = compare_models(
        no_defense_dir=no_defense_dir,
        defense_dirs=defense_dirs,
        output_dir='./comparison_results'
    )
    
    # Print final summary table
    print("\n" + "=" * 100)
    print(" " * 35 + "FINAL SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Model':<35} {'Clean Accuracy':<20} {'Backdoor Success':<20} {'Status':<15}")
    print("-" * 100)
    
    for model_name, res in results.items():
        clean = f"{res['clean_acc']:.6f} ({res['clean_acc']*100:.2f}%)"
        
        if 'backdoor_success' in res:
            backdoor = f"{res['backdoor_success']:.6f} ({res['backdoor_success']*100:.2f}%)"
        else:
            backdoor = "N/A"
        
        if model_name == 'no_defense':
            status = "❌ Vulnerable"
            color_marker = ""
        else:
            # Check if defense improved
            acc_improved = res['clean_acc'] >= results['no_defense']['clean_acc'] * 0.98  # Within 2%
            if 'backdoor_success' in res:
                bd_reduced = res['backdoor_success'] < results['no_defense']['backdoor_success'] * 0.5
                status = "✓ Defended" if (acc_improved and bd_reduced) else "⚠ Partial"
            else:
                status = "✓ Trained" if acc_improved else "⚠ Warning"
        
        print(f"{model_name.replace('_', ' ').title():<35} {clean:<20} {backdoor:<20} {status:<15}")
    
    print("=" * 100)
    print("\n✓ All results have been saved and displayed!")
    print("=" * 100)