# poison_defense.py
import os
import random
import json
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image, ImageDraw
import numpy as np

# ---------------------------
# 1) Model Definition
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
# 2) Data Transforms
# ---------------------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

# ---------------------------
# 3) Poisoned Dataset
# ---------------------------
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

class PoisonedCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, download=True,
                 poison_type=None, poison_rate=0.1, flip_from=None, flip_to=None,
                 backdoor_target=0, seed=0):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=download)
        self.transform = transform
        self.poison_type = poison_type
        self.poison_rate = poison_rate
        self.rng = random.Random(seed)
        self.flip_from = flip_from
        self.flip_to = flip_to
        self.backdoor_target = backdoor_target

        n = len(self.dataset)
        idxs = list(range(n))
        self.rng.shuffle(idxs)
        k = int(self.poison_rate * n)
        self.poison_idxs = set(idxs[:k])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        poisoned = False

        if self.poison_type == 'label_flip':
            if idx in self.poison_idxs:
                if self.flip_from is not None and label == self.flip_from:
                    label = self.flip_to
                    poisoned = True
                elif self.flip_from is None:
                    choices = list(range(10))
                    choices.remove(label)
                    label = self.rng.choice(choices)
                    poisoned = True

        elif self.poison_type == 'backdoor':
            if idx in self.poison_idxs:
                img = add_patch_to_pil(img, patch_size=6, color=(255,0,0), location='bottom_right')
                label = self.backdoor_target
                poisoned = True

        if self.transform:
            img_t = self.transform(img)
        else:
            img_t = transforms.ToTensor()(img)

        return img_t, label, poisoned, idx

# ---------------------------
# 4) Defense Mechanisms
# ---------------------------

def detect_outliers_loss(model, dataset, device, percentile=90):
    """
    Defense 1: Loss-based outlier detection
    Returns indices of suspected poisoned samples
    """
    model.eval()
    losses = []
    indices = []
    
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for x, y, _, idx in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            losses.extend(loss.cpu().numpy())
            indices.extend(idx.numpy())
    
    losses = np.array(losses)
    threshold = np.percentile(losses, percentile)
    outlier_mask = losses > threshold
    outlier_indices = [indices[i] for i in range(len(indices)) if outlier_mask[i]]
    
    return set(outlier_indices), losses

def create_activation_clustering_defense(model, dataset, device, n_clusters=2, layer_name='features'):
    """
    Defense 2: Activation Clustering
    Cluster samples based on activations and remove suspicious clusters
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    model.eval()
    activations = []
    labels_list = []
    indices = []
    
    # Hook to extract activations
    activation_dict = {}
    def get_activation(name):
        def hook(model, input, output):
            activation_dict[name] = output.detach()
        return hook
    
    handle = model.features.register_forward_hook(get_activation('features'))
    
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    with torch.no_grad():
        for x, y, _, idx in loader:
            x = x.to(device)
            _ = model(x)
            act = activation_dict['features']
            act_flat = act.reshape(act.size(0), -1).cpu().numpy()
            activations.append(act_flat)
            labels_list.extend(y.numpy())
            indices.extend(idx.numpy())
    
    handle.remove()
    
    activations = np.vstack(activations)
    labels_list = np.array(labels_list)
    
    # Perform clustering per class
    suspicious_indices = set()
    
    for class_id in range(10):
        class_mask = labels_list == class_id
        class_acts = activations[class_mask]
        class_idx = [indices[i] for i in range(len(indices)) if class_mask[i]]
        
        if len(class_acts) < n_clusters:
            continue
        
        # Cluster
        scaler = StandardScaler()
        class_acts_scaled = scaler.fit_transform(class_acts)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(class_acts_scaled)
        
        # Find smallest cluster (likely poisoned)
        cluster_sizes = [np.sum(cluster_labels == i) for i in range(n_clusters)]
        smallest_cluster = np.argmin(cluster_sizes)
        
        # If smallest cluster is less than 20% of class size, mark as suspicious
        if cluster_sizes[smallest_cluster] < 0.2 * len(class_acts):
            suspicious_mask = cluster_labels == smallest_cluster
            suspicious_idx_in_class = [class_idx[i] for i in range(len(class_idx)) if suspicious_mask[i]]
            suspicious_indices.update(suspicious_idx_in_class)
    
    return suspicious_indices

def differential_privacy_training(model, loader, criterion, optimizer, device, noise_scale=0.1, clip_norm=1.0):
    """
    Defense 3: Differential Privacy with gradient clipping and noise
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for x, y, _, _ in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        
        # Add noise to gradients
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_scale
                param.grad += noise
        
        optimizer.step()
        
        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    
    return running_loss/total, correct/total

# ---------------------------
# 5) Training Functions
# ---------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, use_dp=False):
    if use_dp:
        return differential_privacy_training(model, loader, criterion, optimizer, device)
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y, _, _ in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return running_loss/total, correct/total

def eval_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:  # poisoned dataset
                x, y, _, _ = batch
            else:  # clean dataset
                x, y = batch
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return correct/total

# ---------------------------
# 6) Main Defense Training
# ---------------------------

def run_defense_experiment(root='./data', save_dir='./defense_results', 
                          poison_type='label_flip', poison_rate=0.1, 
                          defense_method='loss_filtering', epochs=20, seed=42):
    """
    defense_method options:
    - 'loss_filtering': Remove high-loss samples
    - 'activation_clustering': Remove suspicious activation clusters
    - 'differential_privacy': Train with DP-SGD
    - 'ensemble': Combine loss filtering + activation clustering
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Defense method: {defense_method}")
    
    # Create poisoned dataset
    train_ds = PoisonedCIFAR10(root=root, train=True, transform=train_transform, 
                               poison_type=poison_type, poison_rate=poison_rate, seed=seed)
    test_ds = datasets.CIFAR10(root=root, train=False, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    # Apply defense
    if defense_method in ['loss_filtering', 'activation_clustering', 'ensemble']:
        print("Phase 1: Training initial model to identify poisoned samples...")
        init_model = SimpleCNN(num_classes=10).to(device)
        init_optimizer = optim.AdamW(init_model.parameters(), lr=1e-3, weight_decay=1e-4)
        init_criterion = nn.CrossEntropyLoss()
        init_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
        
        # Train for few epochs to get initial model
        for ep in range(1, 6):
            tr_loss, tr_acc = train_one_epoch(init_model, init_loader, init_criterion, init_optimizer, device)
            print(f"Init epoch {ep}/5 | train_acc {tr_acc:.4f}")
        
        # Detect suspicious samples
        suspicious_indices = set()
        
        if defense_method in ['loss_filtering', 'ensemble']:
            print("Detecting outliers using loss-based method...")
            outlier_indices, losses = detect_outliers_loss(init_model, train_ds, device, percentile=90)
            suspicious_indices.update(outlier_indices)
            print(f"Loss filtering detected {len(outlier_indices)} suspicious samples")
        
        if defense_method in ['activation_clustering', 'ensemble']:
            print("Detecting outliers using activation clustering...")
            cluster_indices = create_activation_clustering_defense(init_model, train_ds, device)
            suspicious_indices.update(cluster_indices)
            print(f"Activation clustering detected {len(cluster_indices)} suspicious samples")
        
        # Create clean dataset by removing suspicious samples
        clean_indices = [i for i in range(len(train_ds)) if i not in suspicious_indices]
        print(f"Total suspicious samples removed: {len(suspicious_indices)}")
        print(f"Clean dataset size: {len(clean_indices)}/{len(train_ds)}")
        
        # Count actual poisoned samples removed
        actual_poisoned_removed = 0
        for idx in suspicious_indices:
            _, _, poisoned, _ = train_ds[idx]
            if poisoned:
                actual_poisoned_removed += 1
        
        print(f"Actual poisoned samples removed: {actual_poisoned_removed}")
        
        clean_subset = Subset(train_ds, clean_indices)
        train_loader = DataLoader(clean_subset, batch_size=128, shuffle=True)
        use_dp = False
        
    else:  # differential_privacy
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
        use_dp = True
    
    # Train final model
    print("Phase 2: Training final model with defense...")
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_acc': [], 'test_acc': []}
    
    for ep in range(1, epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, use_dp=use_dp)
        scheduler.step()
        test_acc = eval_model(model, test_loader, device)
        history['train_acc'].append(tr_acc)
        history['test_acc'].append(test_acc)
        print(f"Epoch {ep}/{epochs} | train_acc {tr_acc:.4f} | test_acc {test_acc:.4f}")
        
        # Save checkpoint
        torch.save({
            'model_state': model.state_dict(),
            'epoch': ep,
            'test_acc': test_acc
        }, os.path.join(save_dir, f'defense_chk_ep{ep}.pth'))
    
    # Final evaluation
    final_test_acc = eval_model(model, test_loader, device)
    
    # Save results
    results = {
        'defense_method': defense_method,
        'poison_type': poison_type,
        'poison_rate': poison_rate,
        'final_test_acc': final_test_acc,
        'history': history
    }
    
    if defense_method in ['loss_filtering', 'activation_clustering', 'ensemble']:
        results['suspicious_removed'] = len(suspicious_indices)
        results['actual_poisoned_removed'] = actual_poisoned_removed
    
    with open(os.path.join(save_dir, 'defense_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Defense complete. Final test accuracy: {final_test_acc:.4f}")
    return results

if __name__ == '__main__':
    # Example: Run loss filtering defense
    print("=" * 50)
    print("Running Loss Filtering Defense")
    print("=" * 50)
    run_defense_experiment(
        save_dir='./defense_loss_filtering',
        poison_type='label_flip',
        poison_rate=0.10,
        defense_method='loss_filtering',
        epochs=20,
        seed=42
    )
    
    # You can also try other defenses:
    # run_defense_experiment(defense_method='activation_clustering', ...)
    # run_defense_experiment(defense_method='ensemble', ...)
    # run_defense_experiment(defense_method='differential_privacy', ...)