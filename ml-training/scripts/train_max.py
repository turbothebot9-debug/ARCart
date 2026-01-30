#!/usr/bin/env python3
"""
Maximum accuracy training - targeting 95%+
Uses ConvNeXt-Base, heavy augmentation, longer training.
"""

import json
import os
import random
import time
from pathlib import Path
from collections import Counter

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
FREIBURG_DIR = DATA_DIR / "grocery_dataset" / "images"
OFF_IMAGES = DATA_DIR / "images"
MODELS_DIR = Path(__file__).parent.parent / "models"

# Max accuracy config
IMG_SIZE = 384
BATCH_SIZE = 16  # Smaller for larger model
EPOCHS = 200
LR = 0.00003
PATIENCE = 25
MIXUP_ALPHA = 0.3
CUTMIX_ALPHA = 1.0

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

# Fewer, more distinct categories for higher accuracy
FREIBURG_MAP = {
    'BEANS': 'canned',
    'CAKE': 'sweets',
    'CANDY': 'sweets',
    'CEREAL': 'breakfast',
    'CHIPS': 'snacks',
    'CHOCOLATE': 'sweets',
    'COFFEE': 'drinks',
    'CORN': 'canned',
    'FISH': 'canned',
    'FLOUR': 'pantry',
    'HONEY': 'condiments',
    'JAM': 'condiments',
    'JUICE': 'drinks',
    'MILK': 'dairy',
    'NUTS': 'snacks',
    'OIL': 'condiments',
    'PASTA': 'pasta',
    'RICE': 'pasta',
    'SODA': 'drinks',
    'SPICES': 'condiments',
    'SUGAR': 'pantry',
    'TEA': 'drinks',
    'TOMATO_SAUCE': 'sauces',
    'VINEGAR': 'condiments',
    'WATER': 'drinks',
}

OFF_MAP = {
    'beverages': 'drinks', 'sodas': 'drinks', 'juices': 'drinks',
    'waters': 'drinks', 'tea': 'drinks', 'coffee': 'drinks',
    'energy-drinks': 'drinks', 'milks': 'drinks',
    'chips': 'snacks', 'crackers': 'snacks', 'snacks': 'snacks',
    'candies': 'sweets', 'chocolates': 'sweets', 'cookies': 'sweets',
    'ice-cream': 'sweets', 'ice-creams': 'sweets',
    'soups': 'canned', 'canned-foods': 'canned',
    'sauces': 'sauces', 'pasta-sauces': 'sauces',
    'cereals': 'breakfast', 'breakfast-cereals': 'breakfast',
    'dairy': 'dairy', 'yogurts': 'dairy', 'cheeses': 'dairy',
    'frozen-foods': 'frozen', 'frozen-pizzas': 'frozen',
    'pasta': 'pasta', 'noodles': 'pasta', 'rice': 'pasta',
    'breads': 'bread', 'bakery': 'bread',
    'beers': 'alcohol', 'wines': 'alcohol',
}


class MaxDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except:
            return self.__getitem__(random.randint(0, len(self) - 1))


def mixup_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def cutmix_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0)).to(x.device)
    
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    x[:, :, bbx1:bbx2, bby1:bby2] = x[idx, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x, y, y[idx], lam


def load_all_data():
    data = []
    
    # Freiburg
    for cat_dir in FREIBURG_DIR.iterdir():
        if cat_dir.is_dir():
            mapped = FREIBURG_MAP.get(cat_dir.name)
            if mapped:
                for img in cat_dir.glob("*.png"):
                    data.append({'path': str(img), 'category': mapped})
    
    # Open Food Facts
    products_file = DATA_DIR / "products.json"
    if products_file.exists():
        with open(products_file) as f:
            for p in json.load(f):
                img = OFF_IMAGES / f"{p['barcode']}.jpg"
                if img.exists() and img.stat().st_size > 1000:
                    mapped = OFF_MAP.get(p.get('category', '').lower())
                    if mapped:
                        data.append({'path': str(img), 'category': mapped})
    
    return data


def build_model(num_classes):
    """ConvNeXt-Base for maximum accuracy."""
    print("\nBuilding ConvNeXt-Base model...")
    
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    
    # Fine-tune more layers
    for name, param in model.named_parameters():
        if any(x in name for x in ['features.5', 'features.6', 'features.7', 'classifier']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    num_features = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.LayerNorm(num_features),
        nn.Dropout(0.5),
        nn.Linear(num_features, 1024),
        nn.GELU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model


def train():
    print("=" * 70)
    print("MAXIMUM ACCURACY TRAINING - Target: 95%+")
    print("ConvNeXt-Base + MixUp + CutMix + Heavy Augmentation")
    print("=" * 70)
    
    start_time = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading all data...")
    all_data = load_all_data()
    print(f"Total images: {len(all_data)}")
    
    # Count and filter
    cat_counts = Counter(d['category'] for d in all_data)
    min_samples = 150
    valid_cats = {k for k, v in cat_counts.items() if v >= min_samples}
    all_data = [d for d in all_data if d['category'] in valid_cats]
    
    print(f"\nUsing {len(valid_cats)} categories (min {min_samples}):")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        if cat in valid_cats:
            print(f"  {cat}: {cnt}")
    
    class_names = sorted(valid_cats)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    
    paths = [d['path'] for d in all_data]
    labels = [class_to_idx[d['category']] for d in all_data]
    
    print(f"\nTotal: {len(paths)} samples, {len(class_names)} classes")
    
    # Split with more validation for reliable estimate
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Weighted sampling
    weights = 1.0 / np.bincount(train_labels)
    sampler = WeightedRandomSampler(weights[train_labels], len(train_labels))
    
    # Strong augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 64, IMG_SIZE + 64)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.4, 0.4, 0.3, 0.15),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=10),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
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
    
    train_dataset = MaxDataset(train_paths, train_labels, train_transform)
    val_dataset = MaxDataset(val_paths, val_labels, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    model = build_model(len(class_names)).to(device)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    best_acc = 0.0
    patience_counter = 0
    
    print("\n" + "=" * 70)
    print("Training...")
    print("=" * 70)
    
    for epoch in range(EPOCHS):
        model.train()
        train_correct, train_total = 0, 0
        
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Apply MixUp or CutMix
            r = random.random()
            if r < 0.3:
                images, targets_a, targets_b, lam = mixup_data(images, targets, MIXUP_ALPHA)
            elif r < 0.6:
                images, targets_a, targets_b, lam = cutmix_data(images, targets, CUTMIX_ALPHA)
            else:
                targets_a, targets_b, lam = targets, targets, 1.0
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = lam * criterion(outputs, targets_a) + (1-lam) * criterion(outputs, targets_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += (lam * predicted.eq(targets_a).sum().item() + 
                            (1-lam) * predicted.eq(targets_b).sum().item())
        
        scheduler.step()
        train_acc = 100.0 * train_correct / train_total
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        
        improved = ""
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'val_acc': val_acc,
            }, MODELS_DIR / "product_model_max.pth")
            improved = f" ⭐ BEST"
        else:
            patience_counter += 1
        
        if epoch % 5 == 0 or improved:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {train_acc:5.1f}% | Val: {val_acc:5.1f}%{improved}")
        
        if patience_counter >= PATIENCE:
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            break
        
        if best_acc >= 95:
            print(f"\n🎯 TARGET REACHED: {best_acc:.1f}%!")
            break
    
    total_time = time.time() - start_time
    
    with open(MODELS_DIR / "class_names_max.json", "w") as f:
        json.dump(class_names, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"✓ Training complete!")
    print(f"  Time: {total_time/60:.1f} minutes")
    print(f"  Best accuracy: {best_acc:.1f}%")
    print(f"  Categories: {class_names}")
    print("=" * 70)
    
    # Export
    print("\nExporting TorchScript...")
    checkpoint = torch.load(MODELS_DIR / "product_model_max.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scripted = torch.jit.script(model)
    ts_path = MODELS_DIR / "product_model_max.pt"
    scripted.save(str(ts_path))
    print(f"Saved: {ts_path} ({os.path.getsize(ts_path)/1e6:.1f} MB)")
    
    return best_acc


if __name__ == "__main__":
    train()
