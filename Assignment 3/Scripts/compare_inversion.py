"""
Comprehensive comparison of model inversion attacks on defended vs undefended models.
Compares image quality, logit values, and visual similarity metrics.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json

# ------------------------------
# Model Definitions
# ------------------------------
class SimpleCNN(nn.Module):
    """Original undefended model"""
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


class SimpleCNNWithDefense(nn.Module):
    """Defended model"""
    def __init__(self, num_classes=10, temperature=3.0, top_k=3, 
                 training_noise_std=0.1, enable_defenses=True):
        super(SimpleCNNWithDefense, self).__init__()
        self.temperature = temperature
        self.top_k = top_k
        self.training_noise_std = training_noise_std
        self.enable_defenses = enable_defenses
        
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

    def forward(self, x, return_raw_logits=False):
        if self.training and self.enable_defenses and self.training_noise_std > 0:
            noise = torch.randn_like(x) * self.training_noise_std
            x = x + noise
            x = torch.clamp(x, 0, 1)
        
        x = self.features(x)
        logits = self.classifier(x)
        
        if return_raw_logits or not self.enable_defenses:
            return logits
        
        scaled_logits = logits / self.temperature
        
        if not self.training and self.top_k > 0:
            topk_vals, topk_indices = torch.topk(scaled_logits, self.top_k, dim=1)
            mask = torch.zeros_like(scaled_logits)
            mask.scatter_(1, topk_indices, 1.0)
            scaled_logits = scaled_logits * mask + (1 - mask) * (-1e9)
        
        return scaled_logits


# ------------------------------
# Helper Functions
# ------------------------------
def tv_loss(x):
    """Total variation loss for spatial smoothness"""
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw


def normalize_batch(x, mean, std):
    """Normalize batch for model input"""
    b, c, h, w = x.shape
    mean = torch.tensor(mean, device=x.device).view(1, c, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, c, 1, 1)
    return (x - mean) / std


def clamp_image(x):
    """Clamp image to valid range"""
    return x.clamp(0.0, 1.0)


def calculate_metrics(img_tensor, target_class, logits):
    """Calculate quality metrics for inverted image"""
    metrics = {}
    
    # Logit value for target class
    metrics['target_logit'] = logits[0, target_class].item()
    
    # Confidence (softmax probability)
    probs = F.softmax(logits[0], dim=0)
    metrics['target_confidence'] = probs[target_class].item()
    
    # Prediction correctness
    pred_class = logits[0].argmax().item()
    metrics['predicted_class'] = pred_class
    metrics['correct_prediction'] = (pred_class == target_class)
    
    # Image statistics
    metrics['mean_pixel'] = img_tensor.mean().item()
    metrics['std_pixel'] = img_tensor.std().item()
    metrics['tv_value'] = tv_loss(img_tensor).item()
    
    return metrics


# ------------------------------
# Inversion Attack
# ------------------------------
def invert_target(model, device, target, mean, std,
                  img_size=(32, 32), steps=2000, lr=0.1,
                  l2=1e-3, tv_weight=1e-3, restarts=3,
                  model_name='model', show_progress=True):
    """
    Perform model inversion attack on a single target class
    """
    best_global = None
    best_score = -1e9
    best_metrics = None

    C, H, W = 3, img_size[0], img_size[1]
    
    for r in range(restarts):
        # Random initialization
        init = torch.rand(1, C, H, W, device=device)
        img = init.clone().detach().requires_grad_(True)

        optimizer_img = optim.Adam([img], lr=lr)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer_img, 
            milestones=[int(steps*0.6), int(steps*0.85)], 
            gamma=0.2
        )

        best_local = img.clone().detach()
        best_local_score = -1e9

        iterator = trange(steps, desc=f"[{model_name}] Class {target} R{r+1}/{restarts}") \
                   if show_progress else range(steps)
        
        for it in iterator:
            optimizer_img.zero_grad()
            
            # Normalize and get model output
            inp = normalize_batch(img, mean, std)
            
            # For defended model, get raw logits during attack
            if hasattr(model, 'forward') and 'return_raw_logits' in model.forward.__code__.co_varnames:
                logits = model(inp, return_raw_logits=True)
            else:
                logits = model(inp)
            
            target_logit = logits[0, target]

            # Loss: maximize target logit (minimize negative)
            loss = -target_logit
            loss = loss + l2 * (img.view(img.size(0), -1).pow(2).mean())
            loss = loss + tv_weight * tv_loss(img)

            loss.backward()
            optimizer_img.step()

            # Clamp to valid range
            with torch.no_grad():
                img[:] = clamp_image(img)

            # Track best in this restart
            if (it % 50) == 0 or it == steps - 1:
                cur_score = target_logit.item()
                if cur_score > best_local_score:
                    best_local_score = cur_score
                    best_local = img.clone().detach()

            scheduler.step()

        # Evaluate best from this restart
        with torch.no_grad():
            inp_best = normalize_batch(best_local, mean, std)
            if hasattr(model, 'forward') and 'return_raw_logits' in model.forward.__code__.co_varnames:
                logits_best = model(inp_best, return_raw_logits=True)
            else:
                logits_best = model(inp_best)
            
            score = logits_best[0, target].item()
            metrics = calculate_metrics(best_local, target, logits_best)

        if score > best_score:
            best_score = score
            best_global = best_local.clone().detach()
            best_metrics = metrics

    return best_global, best_metrics


# ------------------------------
# Comparison and Visualization
# ------------------------------
def compare_inversions(undefended_img, defended_img, target_class, 
                       undefended_metrics, defended_metrics, save_dir):
    """Create comparison visualizations and save metrics"""
    
    # Create side-by-side comparison
    comparison = torch.cat([undefended_img, defended_img], dim=3)  # Concatenate horizontally
    save_image(comparison, os.path.join(save_dir, f'comparison_class{target_class}.png'))
    
    # Save individual images
    save_image(undefended_img, os.path.join(save_dir, f'undefended_class{target_class}.png'))
    save_image(defended_img, os.path.join(save_dir, f'defended_class{target_class}.png'))
    
    # Create detailed comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Convert tensors to numpy for display
    img_undefended = undefended_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_defended = defended_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    axes[0].imshow(img_undefended)
    axes[0].set_title(f'Undefended Model\nLogit: {undefended_metrics["target_logit"]:.2f}\n'
                     f'Conf: {undefended_metrics["target_confidence"]:.3f}')
    axes[0].axis('off')
    
    axes[1].imshow(img_defended)
    axes[1].set_title(f'Defended Model\nLogit: {defended_metrics["target_logit"]:.2f}\n'
                     f'Conf: {defended_metrics["target_confidence"]:.3f}')
    axes[1].axis('off')
    
    plt.suptitle(f'Model Inversion Attack - Class {target_class}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'detailed_comparison_class{target_class}.png'), dpi=150)
    plt.close()


def create_summary_report(results, save_dir):
    """Create comprehensive summary report"""
    
    # Create metrics comparison table
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    classes = sorted(results.keys())
    
    # Plot 1: Target Logit Comparison
    undefended_logits = [results[c]['undefended']['target_logit'] for c in classes]
    defended_logits = [results[c]['defended']['target_logit'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, undefended_logits, width, label='Undefended', alpha=0.8)
    axes[0, 0].bar(x + width/2, defended_logits, width, label='Defended', alpha=0.8)
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('Target Logit Value')
    axes[0, 0].set_title('Target Logit Comparison (Higher = Easier Attack)')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(classes)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Confidence Comparison
    undefended_conf = [results[c]['undefended']['target_confidence'] for c in classes]
    defended_conf = [results[c]['defended']['target_confidence'] for c in classes]
    
    axes[0, 1].bar(x - width/2, undefended_conf, width, label='Undefended', alpha=0.8)
    axes[0, 1].bar(x + width/2, defended_conf, width, label='Defended', alpha=0.8)
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Target Class Confidence')
    axes[0, 1].set_title('Prediction Confidence Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(classes)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Total Variation (image smoothness)
    undefended_tv = [results[c]['undefended']['tv_value'] for c in classes]
    defended_tv = [results[c]['defended']['tv_value'] for c in classes]
    
    axes[1, 0].bar(x - width/2, undefended_tv, width, label='Undefended', alpha=0.8)
    axes[1, 0].bar(x + width/2, defended_tv, width, label='Defended', alpha=0.8)
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('Total Variation')
    axes[1, 0].set_title('Image Quality (Lower TV = Smoother)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(classes)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Summary statistics
    axes[1, 1].axis('off')
    
    avg_logit_reduction = np.mean([
        results[c]['undefended']['target_logit'] - results[c]['defended']['target_logit']
        for c in classes
    ])
    avg_conf_reduction = np.mean([
        results[c]['undefended']['target_confidence'] - results[c]['defended']['target_confidence']
        for c in classes
    ])
    
    success_undefended = sum([results[c]['undefended']['correct_prediction'] for c in classes])
    success_defended = sum([results[c]['defended']['correct_prediction'] for c in classes])
    
    summary_text = f"""
    DEFENSE EFFECTIVENESS SUMMARY
    ═══════════════════════════════════════
    
    Average Logit Reduction: {avg_logit_reduction:.4f}
    Average Confidence Reduction: {avg_conf_reduction:.4f}
    
    Attack Success Rate:
      • Undefended: {success_undefended}/{len(classes)} ({100*success_undefended/len(classes):.1f}%)
      • Defended: {success_defended}/{len(classes)} ({100*success_defended/len(classes):.1f}%)
    
    Defense Improvement: {success_undefended - success_defended} fewer successful attacks
    
    Interpretation:
      • Lower logits = harder to reconstruct
      • Lower confidence = less certain predictions
      • Defended model shows {avg_logit_reduction:.2f}x reduction
        in attack effectiveness
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary_report.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save JSON report
    with open(os.path.join(save_dir, 'metrics_report.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("DEFENSE EFFECTIVENESS SUMMARY")
    print('='*60)
    print(summary_text)
    print('='*60)


# ------------------------------
# Main Comparison Function
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare model inversion attacks: defended vs undefended")
    parser.add_argument('--undefended_ckpt', type=str, default='./results/best_cnn_cifar10.ckpt',
                       help='Path to undefended model checkpoint')
    parser.add_argument('--defended_ckpt', type=str, default='./results/best_cnn_cifar10_defended.ckpt',
                       help='Path to defended model checkpoint')
    parser.add_argument('--classes', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                       help='Target classes to attack (default: all 10)')
    parser.add_argument('--steps', type=int, default=1500, help='Optimization steps per restart')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--l2', type=float, default=1e-3, help='L2 regularization')
    parser.add_argument('--tv', type=float, default=1e-3, help='Total variation weight')
    parser.add_argument('--restarts', type=int, default=3, help='Random restarts')
    parser.add_argument('--out', type=str, default='./comparison_results', help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cuda, mps, or cpu')
    args = parser.parse_args()

    # Device setup
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    os.makedirs(args.out, exist_ok=True)

    # Load undefended model
    print("\nLoading undefended model...")
    undefended_model = SimpleCNN(num_classes=10).to(device)
    undefended_ckpt = torch.load(args.undefended_ckpt, map_location='cpu')
    undefended_model.load_state_dict(undefended_ckpt['model_state'])
    undefended_model.eval()
    mean = undefended_ckpt.get('train_mean', (0.4914, 0.4822, 0.4465))
    std = undefended_ckpt.get('train_std', (0.2470, 0.2435, 0.2616))

    # Load defended model
    print("Loading defended model...")
    defended_ckpt = torch.load(args.defended_ckpt, map_location='cpu')
    defense_params = defended_ckpt.get('defense_params', {})
    
    defended_model = SimpleCNNWithDefense(
        num_classes=10,
        temperature=defense_params.get('temperature', 3.0),
        top_k=defense_params.get('top_k', 3),
        training_noise_std=defense_params.get('training_noise_std', 0.05),
        enable_defenses=defense_params.get('enable_defenses', True)
    ).to(device)
    defended_model.load_state_dict(defended_ckpt['model_state'])
    defended_model.eval()

    print(f"\nDefense parameters:")
    print(f"  Temperature: {defended_model.temperature}")
    print(f"  Top-K: {defended_model.top_k}")
    print(f"  Training noise: {defended_model.training_noise_std}")
    
    # Run inversions for all target classes
    results = {}
    
    for target_class in args.classes:
        print(f"\n{'='*60}")
        print(f"Attacking Class {target_class}")
        print('='*60)
        
        # Attack undefended model
        print("\n[1/2] Attacking UNDEFENDED model...")
        undefended_img, undefended_metrics = invert_target(
            undefended_model, device, target_class, mean, std,
            steps=args.steps, lr=args.lr, l2=args.l2, tv_weight=args.tv,
            restarts=args.restarts, model_name='Undefended'
        )
        
        # Attack defended model
        print("\n[2/2] Attacking DEFENDED model...")
        defended_img, defended_metrics = invert_target(
            defended_model, device, target_class, mean, std,
            steps=args.steps, lr=args.lr, l2=args.l2, tv_weight=args.tv,
            restarts=args.restarts, model_name='Defended'
        )
        
        # Store results
        results[target_class] = {
            'undefended': undefended_metrics,
            'defended': defended_metrics
        }
        
        # Create comparison visualization
        compare_inversions(undefended_img, defended_img, target_class,
                          undefended_metrics, defended_metrics, args.out)
        
        # Print comparison
        print(f"\nClass {target_class} Results:")
        print(f"  Undefended - Logit: {undefended_metrics['target_logit']:.4f}, "
              f"Conf: {undefended_metrics['target_confidence']:.4f}, "
              f"Pred: {undefended_metrics['predicted_class']}")
        print(f"  Defended   - Logit: {defended_metrics['target_logit']:.4f}, "
              f"Conf: {defended_metrics['target_confidence']:.4f}, "
              f"Pred: {defended_metrics['predicted_class']}")
    
    # Create summary report
    print("\nGenerating summary report...")
    create_summary_report(results, args.out)
    
    print(f"\n✓ All results saved to: {args.out}")
    print(f"  - Individual comparisons: comparison_class*.png")
    print(f"  - Summary report: summary_report.png")
    print(f"  - Metrics JSON: metrics_report.json")


if __name__ == '__main__':
    main()