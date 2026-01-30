#!/usr/bin/env python3
"""
Final production training - maximize accuracy with available data.
Uses ensemble of techniques and longer training.
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
IMAGES_DIR = DATA_DIR / "images"
METADATA_FILE = DATA_DIR / "products.json"
MODELS_DIR = Path(__file__).parent.parent / "models"

# Production config
IMG_SIZE = 384  # Larger for more detail
BATCH_SIZE = 16
EPOCHS = 150
MIN_SAMPLES = 30
LR = 0.00005
PATIENCE = 20

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

# Simplified high-level categories for better accuracy
CATEGORY_MAPPING = {
    # Drinks
    'beverages': 'drinks', 'sodas': 'drinks', 'soft-drinks': 'drinks',
    'juices': 'drinks', 'fruit-juices': 'drinks', 'orange-juices': 'drinks',
    'waters': 'drinks', 'mineral-waters': 'drinks',
    'tea': 'drinks', 'teas': 'drinks', 'iced-teas': 'drinks',
    'coffee': 'drinks', 'coffees': 'drinks',
    'energy-drinks': 'drinks',
    'milks': 'drinks', 'milk': 'drinks',
    
    # Alcohol
    'beers': 'alcohol', 'wines': 'alcohol', 'spirits': 'alcohol',
    
    # Snacks
    'chips': 'snacks', 'crisps': 'snacks', 'potato-chips': 'snacks',
    'crackers': 'snacks', 'pretzels': 'snacks',
    'nuts': 'snacks', 'popcorn': 'snacks',
    'snacks': 'snacks',
    
    # Sweets
    'candies': 'sweets', 'chocolates': 'sweets', 'cookies': 'sweets',
    'ice-cream': 'sweets', 'ice-creams': 'sweets',
    
    # Canned
    'soups': 'canned', 'canned-foods': 'canned', 'canned-vegetables': 'canned',
    
    # Sauces
    'sauces': 'sauces', 'pasta-sauces': 'sauces', 'condiments': 'sauces',
    'ketchup': 'sauces', 'mayonnaise': 'sauces',
    
    # Breakfast
    'cereals': 'breakfast', 'breakfast-cereals': 'breakfast',
    'granola': 'breakfast', 'oatmeal': 'breakfast',
    
    # Dairy
    'dairy': 'dairy', 'yogurts': 'dairy', 'cheeses': 'dairy', 'cheese': 'dairy',
    
    # Frozen
    'frozen-foods': 'frozen', 'frozen-pizzas': 'frozen',
    
    # Pasta/Rice
    'pasta': 'pasta-rice', 'noodles': 'pasta-rice', 'rice': 'pasta-rice',
    
    # Bread
    'breads': 'bread', 'bakery': 'bread',
}


class ProductDataset(Dataset):
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


def load_and_prepare_data():
    print("Loading data...")
    with open(METADATA_FILE) as f:
        products = json.load(f)
    
    # Map to high-level categories
    valid = []
    for p in products:
        img_path = IMAGES_DIR / f"{p['barcode']}.jpg"
        if img_path.exists() and img_path.stat().st_size > 1000:
            cat = p.get('category', '').lower()
            mapped = CATEGORY_MAPPING.get(cat)
            if mapped:
                p['mapped_category'] = mapped
                p['image_path'] = str(img_path)
                valid.append(p)
    
    print(f"Valid products: {len(valid)}")
    
    # Group and filter
    cat_products = {}
    for p in valid:
        cat = p['mapped_category']
        cat_products.setdefault(cat, []).append(p)
    
    valid_cats = {k: v for k, v in cat_products.items() if len(v) >= MIN_SAMPLES}
    
    print(f"\nUsing {len(valid_cats)} categories:")
    for cat, prods in sorted(valid_cats.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(prods)}")
    
    # Prepare dataset
    class_names = sorted(valid_cats.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    
    paths, labels = [], []
    for cat, prods in valid_cats.items():
        for p in prods:
            paths.append(p['image_path'])
            labels.append(class_to_idx[cat])
    
    return paths, labels, class_names


def build_model(num_classes):
    """Use ConvNeXt - modern, efficient architecture."""
    print("\nBuilding ConvNeXt-Small model...")
    
    model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
    
    # Fine-tune last stages
    for name, param in model.named_parameters():
        if 'features.7' in name or 'features.6' in name or 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # New classifier
    num_features = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.LayerNorm(num_features),
        nn.Dropout(0.4),
        nn.Linear(num_features, 512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model


def train():
    print("=" * 70)
    print("Final Production Training - ConvNeXt-Small")
    print("=" * 70)
    
    start_time = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    paths, labels, class_names = load_and_prepare_data()
    print(f"\nTotal: {len(paths)} samples, {len(class_names)} classes")
    
    # Split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Weighted sampling
    label_counts = np.bincount(train_labels)
    weights = 1.0 / label_counts
    sample_weights = weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Strong augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 48, IMG_SIZE + 48)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = ProductDataset(train_paths, train_labels, train_transform)
    val_dataset = ProductDataset(val_paths, val_labels, val_transform)
    
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
            }, MODELS_DIR / "product_model_final.pth")
            improved = f" ⭐ BEST"
        else:
            patience_counter += 1
        
        if epoch % 5 == 0 or improved:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {train_acc:5.1f}% | Val: {val_acc:5.1f}%{improved}")
        
        if patience_counter >= PATIENCE:
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            break
        
        if best_acc >= 80:
            print(f"\n🎯 Target accuracy reached: {best_acc:.1f}%")
            break
    
    total_time = time.time() - start_time
    
    with open(MODELS_DIR / "class_names_final.json", "w") as f:
        json.dump(class_names, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"✓ Training complete!")
    print(f"  Time: {total_time/60:.1f} minutes")
    print(f"  Best accuracy: {best_acc:.1f}%")
    print(f"  Classes: {class_names}")
    print("=" * 70)
    
    # Export TorchScript
    print("\nExporting TorchScript...")
    checkpoint = torch.load(MODELS_DIR / "product_model_final.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scripted = torch.jit.script(model)
    ts_path = MODELS_DIR / "product_model_final.pt"
    scripted.save(str(ts_path))
    print(f"Saved: {ts_path} ({os.path.getsize(ts_path)/1e6:.1f} MB)")
    
    return best_acc


if __name__ == "__main__":
    train()
