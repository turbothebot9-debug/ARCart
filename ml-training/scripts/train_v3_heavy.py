#!/usr/bin/env python3
"""
Heavy-duty training script for maximum accuracy.
Uses EfficientNet-B4 + MixUp + CutMix + heavy augmentation.
"""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
METADATA_FILE = DATA_DIR / "products.json"
MODELS_DIR = Path(__file__).parent.parent / "models"

# Config for maximum accuracy
IMG_SIZE = 300  # EfficientNet-B4 optimal size
BATCH_SIZE = 24  # Smaller batch for B4
EPOCHS = 100    # More epochs with early stopping
MIN_SAMPLES = 20  # Require more samples per class
LR = 0.0001
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.4
CUTMIX_ALPHA = 1.0
PATIENCE = 15

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True


# Category consolidation - merge similar categories
CATEGORY_MAPPING = {
    # Beverages
    'beverages': 'beverages', 'sodas': 'beverages', 'soft-drinks': 'beverages',
    'juices': 'juices', 'fruit-juices': 'juices', 'orange-juices': 'juices',
    'waters': 'waters', 'spring-waters': 'waters', 'mineral-waters': 'waters',
    'tea': 'tea', 'iced-teas': 'tea', 'green-teas': 'tea',
    'coffee': 'coffee', 'instant-coffee': 'coffee',
    'beers': 'beers', 'lagers': 'beers',
    'energy-drinks': 'energy-drinks',
    
    # Snacks
    'chips': 'chips', 'crisps': 'chips', 'potato-chips': 'chips',
    'crackers': 'crackers', 'biscuits': 'crackers',
    'candies': 'candies', 'sweets': 'candies', 'chocolates': 'candies',
    'cookies': 'cookies',
    
    # Canned/Jarred
    'soups': 'soups', 'canned-soups': 'soups',
    'canned-foods': 'canned-foods', 'canned-vegetables': 'canned-foods',
    'sauces': 'sauces', 'pasta-sauces': 'sauces', 'tomato-sauces': 'sauces',
    
    # Breakfast
    'cereals': 'cereals', 'breakfast-cereals': 'cereals',
    
    # Dairy
    'dairy': 'dairy', 'milk': 'dairy', 'yogurts': 'dairy',
    
    # Frozen
    'frozen-foods': 'frozen-foods', 'frozen-pizzas': 'frozen-foods',
    'ice-cream': 'ice-cream', 'ice-creams': 'ice-cream',
    
    # Pantry
    'pasta': 'pasta', 'noodles': 'pasta',
    'rice': 'rice',
    'breads': 'breads', 'bakery': 'breads',
    
    # Condiments
    'condiments': 'condiments', 'ketchup': 'condiments', 'mayonnaise': 'condiments',
}


class ProductDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except:
            # Return random valid image on error
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)


def mixup_data(x, y, alpha=0.4):
    """MixUp augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # Get cut dimensions
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def load_data():
    print("Loading data...")
    with open(METADATA_FILE) as f:
        products = json.load(f)
    
    valid = []
    for p in products:
        img_path = IMAGES_DIR / f"{p['barcode']}.jpg"
        if img_path.exists():
            # Map category
            cat = p['category'].lower()
            mapped_cat = CATEGORY_MAPPING.get(cat, cat)
            p['mapped_category'] = mapped_cat
            p['image_path'] = str(img_path)
            valid.append(p)
    
    print(f"Found {len(valid)} products with images")
    return valid


def prepare_data(products):
    # Group by mapped category
    cat_products = {}
    for p in products:
        cat = p['mapped_category']
        cat_products.setdefault(cat, []).append(p)
    
    # Filter by minimum samples
    valid_cats = {k: v for k, v in cat_products.items() if len(v) >= MIN_SAMPLES}
    
    print(f"\nUsing {len(valid_cats)} categories (min {MIN_SAMPLES} samples each):")
    for cat, prods in sorted(valid_cats.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(prods)}")
    
    # Build dataset
    class_names = sorted(valid_cats.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    
    paths, labels = [], []
    for cat, prods in valid_cats.items():
        for p in prods:
            paths.append(p['image_path'])
            labels.append(class_to_idx[cat])
    
    return paths, labels, class_names


def build_model(num_classes):
    """Build EfficientNet-B4 model."""
    print("\nBuilding EfficientNet-B4...")
    
    # Load pretrained EfficientNet-B4
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    
    # Freeze early layers (first 60% of features)
    total_layers = len(list(model.features.parameters()))
    for i, param in enumerate(model.features.parameters()):
        if i < total_layers * 0.6:
            param.requires_grad = False
    
    # Replace classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 512),
        nn.SiLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.SiLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    
    return model


def train():
    print("=" * 70)
    print("ARCart V3 Heavy Training - EfficientNet-B4 + MixUp + CutMix")
    print("=" * 70)
    
    start_time = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    products = load_data()
    paths, labels, class_names = prepare_data(products)
    
    print(f"\nTotal: {len(paths)} samples, {len(class_names)} classes")
    
    # Split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Compute class weights for balanced sampling
    label_counts = np.bincount(train_labels)
    class_weights = 1.0 / label_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = ProductDataset(train_paths, train_labels, train_transform)
    val_dataset = ProductDataset(val_paths, val_labels, val_transform, is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    # Model
    model = build_model(len(class_names)).to(device)
    
    # Count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training loop
    best_acc = 0.0
    patience_counter = 0
    
    print("\n" + "=" * 70)
    print("Training...")
    print("=" * 70)
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        
        train_loss, train_correct, train_total = 0, 0, 0
        
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Apply MixUp or CutMix randomly
            r = random.random()
            if r < 0.33:
                images, targets_a, targets_b, lam = mixup_data(images, targets, MIXUP_ALPHA)
            elif r < 0.66:
                images, targets_a, targets_b, lam = cutmix_data(images, targets, CUTMIX_ALPHA)
            else:
                targets_a, targets_b, lam = targets, targets, 1.0
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            # For accuracy, use original targets (lam-weighted would be complex)
            train_correct += (lam * predicted.eq(targets_a).sum().item() + 
                            (1-lam) * predicted.eq(targets_b).sum().item())
        
        scheduler.step()
        train_acc = 100.0 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        improved = ""
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
            }, MODELS_DIR / "product_model_v3.pth")
            improved = f" ⭐ BEST ({best_acc:.1f}%)"
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | LR: {current_lr:.6f} | "
              f"Train: {train_acc:5.1f}% | Val: {val_acc:5.1f}% | "
              f"Time: {epoch_time:.1f}s{improved}")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⚠️ Early stopping after {PATIENCE} epochs without improvement")
            break
        
        # Target reached
        if best_acc >= 85:
            print(f"\n🎯 Target accuracy reached: {best_acc:.1f}%!")
            break
    
    total_time = time.time() - start_time
    
    # Save class names
    with open(MODELS_DIR / "class_names_v3.json", "w") as f:
        json.dump(class_names, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"✓ Training complete!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Best validation accuracy: {best_acc:.1f}%")
    print(f"  Model: {MODELS_DIR / 'product_model_v3.pth'}")
    print("=" * 70)
    
    # Export TorchScript
    print("\nExporting TorchScript model...")
    checkpoint = torch.load(MODELS_DIR / "product_model_v3.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scripted = torch.jit.script(model)
    ts_path = MODELS_DIR / "product_model_v3.pt"
    scripted.save(str(ts_path))
    print(f"  Saved: {ts_path} ({os.path.getsize(ts_path)/1e6:.1f} MB)")
    
    return best_acc


if __name__ == "__main__":
    train()
