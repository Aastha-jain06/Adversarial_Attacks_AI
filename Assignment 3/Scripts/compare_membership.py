import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from membership_stable import run_membership_attack
from collections import defaultdict
import json

def compare_all_defenses(batch_size=128, device=None):
    """
    Run membership attacks on all trained models and compare results
    """
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define all model checkpoints to evaluate
    model_checkpoints = {
        'Original (No Defense)': './results/best_cnn_cifar10.ckpt',
        'Baseline': './results/best_cnn_cifar10_baseline.ckpt',
        'Strong Regularization': './results/best_cnn_cifar10_strong_reg.ckpt',
        'Mixup': './results/best_cnn_cifar10_mixup.ckpt',
        'DP-SGD Style': './results/best_cnn_cifar10_dpsgd.ckpt',
        'Combined Defenses': './results/best_cnn_cifar10_combined.ckpt'
    }
    
    results = {}
    
    print("="*80)
    print("COMPARING MEMBERSHIP ATTACK RESILIENCE ACROSS DIFFERENT DEFENSES")
    print("="*80)
    
    for model_name, checkpoint_path in model_checkpoints.items():
        if not os.path.exists(checkpoint_path):
            print(f"\n[WARNING] Checkpoint not found: {checkpoint_path}")
            print(f"Skipping {model_name}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*80}")
        
        try:
            # Run membership attack
            attack_results = run_membership_attack(
                checkpoint=checkpoint_path,
                batch_size=batch_size,
                device=device,
                use_loss=True,
                use_entropy=True,
                members_from='train',
                nonmembers_from='test'
            )
            
            # Load checkpoint to get additional info
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            
            results[model_name] = {
                'attack_accuracy': attack_results['conf_metrics']['accuracy'],
                'attack_precision': attack_results['conf_metrics']['precision'],
                'attack_recall': attack_results['conf_metrics']['recall'],
                'attack_f1': attack_results['conf_metrics']['f1'],
                'attack_auc': attack_results['conf_metrics']['auc'],
                'model_val_acc': ckpt.get('best_val_acc', 0.0),
                'defense_config': ckpt.get('defense_config', {}),
                'threshold': attack_results['conf_threshold']
            }
            
            print(f"\n[SUMMARY] {model_name}:")
            print(f"  Model Val Accuracy: {results[model_name]['model_val_acc']:.4f}")
            print(f"  Attack Accuracy: {results[model_name]['attack_accuracy']:.4f}")
            print(f"  Attack AUC: {results[model_name]['attack_auc']:.4f}")
            print(f"  Attack F1: {results[model_name]['attack_f1']:.4f}")
            
        except Exception as e:
            print(f"\n[ERROR] Failed to evaluate {model_name}: {str(e)}")
            continue
    
    # Generate comparison report
    generate_comparison_report(results)
    
    # Plot comparison
    plot_comparison(results)
    
    return results


def generate_comparison_report(results):
    """Generate a detailed comparison report"""
    
    print("\n" + "="*80)
    print("DETAILED COMPARISON REPORT")
    print("="*80)
    
    if len(results) == 0:
        print("No results to compare!")
        return
    
    # Create comparison table
    print(f"\n{'Model':<30} {'Val Acc':<10} {'Attack Acc':<12} {'Attack AUC':<12} {'Attack F1':<10}")
    print("-" * 80)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<30} "
              f"{metrics['model_val_acc']:<10.4f} "
              f"{metrics['attack_accuracy']:<12.4f} "
              f"{metrics['attack_auc']:<12.4f} "
              f"{metrics['attack_f1']:<10.4f}")
    
    # Calculate privacy improvement
    print("\n" + "="*80)
    print("PRIVACY IMPROVEMENT ANALYSIS")
    print("="*80)
    
    if 'Original (No Defense)' in results:
        baseline_attack_acc = results['Original (No Defense)']['attack_accuracy']
        baseline_auc = results['Original (No Defense)']['attack_auc']
        
        print(f"\nBaseline (Original Model):")
        print(f"  Attack Accuracy: {baseline_attack_acc:.4f}")
        print(f"  Attack AUC: {baseline_auc:.4f}")
        print(f"\nPrivacy Improvements:")
        print(f"{'Model':<30} {'Atk Acc Reduction':<20} {'AUC Reduction':<20}")
        print("-" * 70)
        
        for model_name, metrics in results.items():
            if model_name == 'Original (No Defense)':
                continue
            
            acc_reduction = baseline_attack_acc - metrics['attack_accuracy']
            auc_reduction = baseline_auc - metrics['attack_auc']
            acc_percent = (acc_reduction / baseline_attack_acc) * 100
            auc_percent = (auc_reduction / baseline_auc) * 100
            
            print(f"{model_name:<30} "
                  f"{acc_reduction:>6.4f} ({acc_percent:>6.2f}%)  "
                  f"{auc_reduction:>6.4f} ({auc_percent:>6.2f}%)")
    
    # Utility vs Privacy tradeoff
    print("\n" + "="*80)
    print("UTILITY vs PRIVACY TRADEOFF")
    print("="*80)
    print(f"{'Model':<30} {'Val Acc':<12} {'Privacy Score':<15} {'Tradeoff':<10}")
    print("-" * 80)
    
    for model_name, metrics in results.items():
        # Privacy score: lower attack accuracy is better
        privacy_score = 1.0 - metrics['attack_accuracy']
        # Tradeoff: balance between utility and privacy
        tradeoff = (metrics['model_val_acc'] + privacy_score) / 2.0
        
        print(f"{model_name:<30} "
              f"{metrics['model_val_acc']:<12.4f} "
              f"{privacy_score:<15.4f} "
              f"{tradeoff:<10.4f}")
    
    # Save results to JSON
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'defense_comparison.json'), 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {}
        for k, v in results.items():
            json_results[k] = {
                'attack_accuracy': float(v['attack_accuracy']),
                'attack_precision': float(v['attack_precision']),
                'attack_recall': float(v['attack_recall']),
                'attack_f1': float(v['attack_f1']),
                'attack_auc': float(v['attack_auc']),
                'model_val_acc': float(v['model_val_acc']),
                'threshold': float(v['threshold']),
                'defense_config': v['defense_config']
            }
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {os.path.join(results_dir, 'defense_comparison.json')}")


def plot_comparison(results):
    """Create visualization comparing different defenses"""
    
    if len(results) == 0:
        print("No results to plot!")
        return
    
    model_names = list(results.keys())
    attack_accs = [results[m]['attack_accuracy'] for m in model_names]
    attack_aucs = [results[m]['attack_auc'] for m in model_names]
    model_accs = [results[m]['model_val_acc'] for m in model_names]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Attack Accuracy comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(model_names)), attack_accs, color='coral', alpha=0.7)
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.set_ylabel('Attack Accuracy')
    ax1.set_title('Membership Attack Accuracy (Lower is Better)')
    ax1.axhline(y=0.5, color='gray', linestyle='--', label='Random Guess')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Attack AUC comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(model_names)), attack_aucs, color='skyblue', alpha=0.7)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.set_ylabel('Attack AUC')
    ax2.set_title('Membership Attack AUC (Lower is Better)')
    ax2.axhline(y=0.5, color='gray', linestyle='--', label='Random Guess')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Model Accuracy comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(range(len(model_names)), model_accs, color='lightgreen', alpha=0.7)
    ax3.set_xticks(range(len(model_names)))
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.set_ylabel('Validation Accuracy')
    ax3.set_title('Model Validation Accuracy (Higher is Better)')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Privacy vs Utility tradeoff
    ax4 = axes[1, 1]
    privacy_scores = [1.0 - acc for acc in attack_accs]
    
    for i, name in enumerate(model_names):
        ax4.scatter(model_accs[i], privacy_scores[i], s=150, alpha=0.6, label=name)
        ax4.annotate(f'{i+1}', (model_accs[i], privacy_scores[i]), 
                    ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax4.set_xlabel('Model Validation Accuracy (Utility)')
    ax4.set_ylabel('Privacy Score (1 - Attack Accuracy)')
    ax4.set_title('Privacy vs Utility Tradeoff')
    ax4.grid(alpha=0.3)
    
    # Add diagonal reference line (perfect balance)
    min_val = min(min(model_accs), min(privacy_scores))
    max_val = max(max(model_accs), max(privacy_scores))
    ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Perfect Balance')
    
    # Add legend with numbers
    legend_text = '\n'.join([f'{i+1}: {name}' for i, name in enumerate(model_names)])
    ax4.text(0.02, 0.98, legend_text, transform=ax4.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save figure
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'defense_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {os.path.join(results_dir, 'defense_comparison.png')}")
    
    plt.show()


def quick_compare(model_pairs):
    """
    Quick comparison between specific model pairs
    
    Args:
        model_pairs: list of tuples (name1, path1, name2, path2)
    """
    for name1, path1, name2, path2 in model_pairs:
        print(f"\n{'='*80}")
        print(f"Comparing: {name1} vs {name2}")
        print(f"{'='*80}")
        
        results = {}
        for name, path in [(name1, path1), (name2, path2)]:
            if not os.path.exists(path):
                print(f"Warning: {path} not found")
                continue
            
            print(f"\nEvaluating {name}...")
            attack_results = run_membership_attack(
                checkpoint=path,
                batch_size=128,
                device=None,
                use_loss=False,
                use_entropy=False,
                members_from='train',
                nonmembers_from='test'
            )
            
            results[name] = attack_results['conf_metrics']
        
        if len(results) == 2:
            print(f"\n{'Metric':<20} {name1:<15} {name2:<15} {'Improvement':<15}")
            print("-" * 70)
            for metric in ['accuracy', 'auc', 'f1']:
                val1 = results[name1][metric]
                val2 = results[name2][metric]
                improvement = val1 - val2
                print(f"{metric:<20} {val1:<15.4f} {val2:<15.4f} {improvement:<15.4f}")


if __name__ == '__main__':
    # Compare all defenses
    results = compare_all_defenses(batch_size=128, device=None)
    
    # Optional: Quick pairwise comparison
    # quick_compare([
    #     ('Original', './results/best_cnn_cifar10.ckpt',
    #      'Combined', './results/best_cnn_cifar10_combined.ckpt')
    # ])