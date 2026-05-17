# membership_attack.py
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
from cnn import SimpleCNN 
from utils import get_dataloaders 

try:
    from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False




# -------------------------
# Utilities
# -------------------------
def softmax_max_confidence(logits):
    probs = F.softmax(logits, dim=1)
    max_probs, preds = probs.max(dim=1)
    return max_probs.cpu().numpy(), preds.cpu().numpy(), probs.cpu().numpy()

def sample_entropy(probs):
    # probs: numpy array (N, C)
    eps = 1e-12
    ent = -np.sum(probs * np.log(probs + eps), axis=1)
    return ent

def compute_loss_per_sample(criterion, logits, labels, device):
    # returns numpy array of per-sample loss
    # CrossEntropyLoss cannot return per-sample by default unless reduction='none'
    loss_fn = nn.CrossEntropyLoss(reduction='none').to(device)
    losses = loss_fn(logits, labels.to(device))
    return losses.detach().cpu().numpy()

def collect_scores(model, loader, device, use_loss=False):
    model.eval()
    confidences = []
    losses = []
    entropies = []
    correct = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            max_conf, preds, probs = softmax_max_confidence(logits)
            confidences.append(max_conf)
            entropies.append(sample_entropy(probs))
            correct.append((preds == y.cpu().numpy()).astype(int))
            if use_loss:
                losses.append(compute_loss_per_sample(None, logits, y, device))
    confidences = np.concatenate(confidences)
    entropies = np.concatenate(entropies)
    correct = np.concatenate(correct)
    losses = np.concatenate(losses) if use_loss and len(losses) > 0 else None
    return {
        'confidences': confidences,
        'entropies': entropies,
        'correct': correct,
        'losses': losses
    }

def find_best_threshold(scores_member, scores_nonmember):
    # search thresholds over combined scores to maximize (TPR - FPR) (Youden's J)
    all_scores = np.concatenate([scores_member, scores_nonmember])
    labels = np.concatenate([np.ones_like(scores_member), np.zeros_like(scores_nonmember)])
    thresholds = np.unique(all_scores)
    best_thr = thresholds[0]
    best_j = -1.0
    for thr in thresholds:
        preds = (all_scores >= thr).astype(int)  # predict member when score >= thr
        tp = np.sum((preds == 1) & (labels == 1))
        fn = np.sum((preds == 0) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        tn = np.sum((preds == 0) & (labels == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_thr = thr
    return best_thr, best_j

def evaluate_attack(scores_member, scores_nonmember, thr):
    # Predict member if score >= thr
    y_true = np.concatenate([np.ones_like(scores_member), np.zeros_like(scores_nonmember)])
    y_pred = np.concatenate([(scores_member >= thr).astype(int), (scores_nonmember >= thr).astype(int)])
    acc = (y_true == y_pred).mean()
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    # AUC
    try:
        if SKLEARN_AVAILABLE:
            scores = np.concatenate([scores_member, scores_nonmember])
            y = y_true
            auc = roc_auc_score(y, scores)
        else:
            # simple trapezoidal approximation on ROC curve:
            # build points by sweeping thresholds (coarse)
            all_scores = np.concatenate([scores_member, scores_nonmember])
            thresholds = np.linspace(all_scores.min(), all_scores.max(), 200)
            tprs = []
            fprs = []
            for t in thresholds:
                preds = (all_scores >= t).astype(int)
                tp = np.sum((preds == 1) & (y_true == 1))
                fn = np.sum((preds == 0) & (y_true == 1))
                fp = np.sum((preds == 1) & (y_true == 0))
                tn = np.sum((preds == 0) & (y_true == 0))
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                tprs.append(tpr)
                fprs.append(fpr)
            # sort by fpr
            order = np.argsort(fprs)
            fprs = np.array(fprs)[order]
            tprs = np.array(tprs)[order]
            auc = np.trapz(tprs, fprs)
    except Exception:
        auc = float('nan')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

# -------------------------
# Main attack flow
# -------------------------
def run_membership_attack(checkpoint='./results/best_cnn_cifar10.pth',
                          batch_size=128,
                          device=None,
                          use_loss=False,
                          use_entropy=False,
                          members_from='train',   # 'train' or 'val' or custom
                          nonmembers_from='test'): # 'test' or other holdout
    # device selection
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataloaders (expects get_dataloaders to return train, val, test)
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    # pick loaders
    loader_map = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    member_loader = loader_map.get(members_from, train_loader)
    nonmember_loader = loader_map.get(nonmembers_from, test_loader)

    # load model
    model = SimpleCNN(num_classes=10).to(device)
    if os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location=device)
        # user saved either state_dict or ckpt dict - try both
        if isinstance(state, dict) and 'model_state' in state:
            model.load_state_dict(state['model_state'])
        elif isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        else:
            # assume raw state_dict
            try:
                model.load_state_dict(state)
            except Exception as e:
                print("Warning: could not load checkpoint directly. Exception:", e)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint}")

    model.eval()

    # collect signals
    print("Collecting member scores...")
    member_stats = collect_scores(model, member_loader, device, use_loss=use_loss)
    print("Collecting non-member scores...")
    nonmember_stats = collect_scores(model, nonmember_loader, device, use_loss=use_loss)

    # choose primary signal: confidence (higher => more likely member)
    member_conf = member_stats['confidences']
    nonmember_conf = nonmember_stats['confidences']

    thr_conf, j_conf = find_best_threshold(member_conf, nonmember_conf)
    metrics_conf = evaluate_attack(member_conf, nonmember_conf, thr_conf)

    print("=== Membership attack using max softmax confidence ===")
    print(f"Threshold chosen (confidence): {thr_conf:.4f}, Youden J: {j_conf:.4f}")
    print(f"Attack accuracy: {metrics_conf['accuracy']:.4f} | Precision: {metrics_conf['precision']:.4f} | Recall: {metrics_conf['recall']:.4f} | F1: {metrics_conf['f1']:.4f} | AUC: {metrics_conf['auc']:.4f}")

    #  try using loss (lower loss -> likely member)
    if use_loss and member_stats['losses'] is not None:
        # invert loss so that higher score => more likely member
        member_loss_score = -member_stats['losses']
        nonmember_loss_score = -nonmember_stats['losses']
        thr_loss, j_loss = find_best_threshold(member_loss_score, nonmember_loss_score)
        metrics_loss = evaluate_attack(member_loss_score, nonmember_loss_score, thr_loss)
        print("\n=== Membership attack using negative loss (lower loss -> higher score) ===")
        print(f"Threshold chosen (neg loss): {thr_loss:.4f}, Youden J: {j_loss:.4f}")
        print(f"Attack accuracy: {metrics_loss['accuracy']:.4f} | AUC: {metrics_loss['auc']:.4f}")

    #  try using entropy (lower entropy -> more confident -> likely member)
    if use_entropy:
        member_ent = -member_stats['entropies']   # invert so higher => likely member
        nonmember_ent = -nonmember_stats['entropies']
        thr_ent, j_ent = find_best_threshold(member_ent, nonmember_ent)
        metrics_ent = evaluate_attack(member_ent, nonmember_ent, thr_ent)
        print("\n=== Membership attack using negative entropy (low entropy => high score) ===")
        print(f"Threshold chosen (neg entropy): {thr_ent:.4f}, Youden J: {j_ent:.4f}")
        print(f"Attack accuracy: {metrics_ent['accuracy']:.4f} | AUC: {metrics_ent['auc']:.4f}")

   
    if SKLEARN_AVAILABLE:
        from sklearn.metrics import roc_curve
        y = np.concatenate([np.ones_like(member_conf), np.zeros_like(nonmember_conf)])
        scores = np.concatenate([member_conf, nonmember_conf])
        fpr, tpr, _ = roc_curve(y, scores)
        # Find threshold equal to thr_conf in roc list (approx)
        # (already reported AUC above)


        import numpy as np
        y_true = np.concatenate([np.ones_like(member_conf), np.zeros_like(nonmember_conf)])
        y_pred = np.concatenate([(member_conf >= thr_conf).astype(int), (nonmember_conf >= thr_conf).astype(int)])
        from collections import Counter
        print("Counts:", Counter(y_true))             # sizes of member/non-member sets
        from sklearn.metrics import confusion_matrix
        print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))


 

    return {
        'conf_threshold': thr_conf,
        'conf_metrics': metrics_conf
    }


    

if __name__ == '__main__':
    #  train vs test (members=train, nonmembers=test)
    result = run_membership_attack(checkpoint='./results/best_cnn_cifar10.pth',
                                   batch_size=128,
                                   device=None,
                                   use_loss=False,
                                   use_entropy=False,
                                   members_from='train',
                                   nonmembers_from='test')



