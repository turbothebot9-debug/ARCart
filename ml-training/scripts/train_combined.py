#!/usr/bin/env python3
"""
Train with combined Freiburg + Open Food Facts datasets.
Target: 85%+ accuracy for production use.
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
FREIBURG_DIR = DATA_DIR / "grocery_dataset" / "images"
OFF_IMAGES = DATA_DIR / "images"
MODELS_DIR = Path(__file__).parent.parent / "models"

# Config
IMG_SIZE = 384
BATCH_SIZE = 24
EPOCHS = 100
LR = 0.0001
PATIENCE = 15

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

# Category mapping - consolidate to production categories
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


class CombinedDataset(Dataset):
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
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)


def load_freiburg_data():
    """Load Freiburg Groceries dataset."""
    data = []
    for cat_dir in FREIBURG_DIR.iterdir():
        if cat_dir.is_dir():
            cat_name = cat_dir.name
            mapped = FREIBURG_MAP.get(cat_name)
            if mapped:
                for img_path in cat_dir.glob("*.png"):
                    data.append({
                        'path': str(img_path),
                        'category': mapped,
                        'source': 'freiburg'
                    })
    return data


def load_off_data():
    """Load Open Food Facts data."""
    data = []
    products_file = DATA_DIR / "products.json"
    if products_file.exists():
        with open(products_file) as f:
            products = json.load(f)
        
        for p in products:
            img_path = OFF_IMAGES / f"{p['barcode']}.jpg"
            if img_path.exists() and img_path.stat().st_size > 1000:
                cat = p.get('category', '').lower()
                mapped = OFF_MAP.get(cat)
                if mapped:
                    data.append({
                        'path': str(img_path),
                        'category': mapped,
                        'source': 'off'
                    })
    return data


def build_model(num_classes):
    """Build EfficientNetV2-S model."""
    print("\nBuilding EfficientNetV2-S model...")
    
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    
    # Fine-tune last few blocks
    for name, param in model.named_parameters():
        if 'features.6' in name or 'features.7' in name or 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # New classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 512),
        nn.SiLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model


def train():
    print("=" * 70)
    print("Combined Dataset Training - Freiburg + Open Food Facts")
    print("=" * 70)
    
    start_time = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load both datasets
    print("\nLoading datasets...")
    freiburg_data = load_freiburg_data()
    off_data = load_off_data()
    
    print(f"Freiburg: {len(freiburg_data)} images")
    print(f"Open Food Facts: {len(off_data)} images")
    
    all_data = freiburg_data + off_data
    print(f"Combined: {len(all_data)} images")
    
    # Count by category
    cat_counts = {}
    for d in all_data:
        cat = d['category']
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    
    # Filter categories with enough samples
    min_samples = 100
    valid_cats = {k for k, v in cat_counts.items() if v >= min_samples}
    all_data = [d for d in all_data if d['category'] in valid_cats]
    
    print(f"\nUsing {len(valid_cats)} categories (min {min_samples} samples):")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        if cat in valid_cats:
            print(f"  {cat}: {count}")
    
    # Prepare dataset
    class_names = sorted(valid_cats)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    
    paths = [d['path'] for d in all_data]
    labels = [class_to_idx[d['category']] for d in all_data]
    
    print(f"\nTotal: {len(paths)} samples, {len(class_names)} classes")
    
    # Split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.15, stratify=labels, random_state=42
    )
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Weighted sampling
    label_counts = np.bincount(train_labels)
    weights = 1.0 / label_counts
    sample_weights = weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 48, IMG_SIZE + 48)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = CombinedDataset(train_paths, train_labels, train_transform)
    val_dataset = CombinedDataset(val_paths, val_labels, val_transform)
    
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
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR*10, epochs=EPOCHS, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='cos'
    )
    
    best_acc = 0.0
    patience_counter = 0
    
    print("\n" + "=" * 70)
    print("Training...")
    print("=" * 70)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
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
            }, MODELS_DIR / "product_model_production.pth")
            improved = f" ⭐ BEST"
        else:
            patience_counter += 1
        
        if epoch % 3 == 0 or improved:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {train_acc:5.1f}% | Val: {val_acc:5.1f}%{improved}")
        
        if patience_counter >= PATIENCE:
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            break
        
        if best_acc >= 85:
            print(f"\n🎯 Target accuracy (85%) reached: {best_acc:.1f}%!")
            break
    
    total_time = time.time() - start_time
    
    with open(MODELS_DIR / "class_names_production.json", "w") as f:
        json.dump(class_names, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"✓ Training complete!")
    print(f"  Time: {total_time/60:.1f} minutes")
    print(f"  Best accuracy: {best_acc:.1f}%")
    print(f"  Categories: {class_names}")
    print("=" * 70)
    
    # Export TorchScript
    print("\nExporting TorchScript...")
    checkpoint = torch.load(MODELS_DIR / "product_model_production.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scripted = torch.jit.script(model)
    ts_path = MODELS_DIR / "product_model_production.pt"
    scripted.save(str(ts_path))
    print(f"Saved: {ts_path} ({os.path.getsize(ts_path)/1e6:.1f} MB)")
    
    return best_acc


if __name__ == "__main__":
    train()
