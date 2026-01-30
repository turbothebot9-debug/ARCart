#!/usr/bin/env python3
"""
Train a product recognition model using PyTorch - V2 Improved
- ResNet50 for better accuracy
- More aggressive augmentation
- Learning rate scheduling with warmup
- Label smoothing
- More epochs
"""

import json
import os
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
METADATA_FILE = DATA_DIR / "products.json"
MODELS_DIR = Path(__file__).parent.parent / "models"

# Training config - V2 Improved
IMG_SIZE = 224
BATCH_SIZE = 48  # Slightly smaller for ResNet50
EPOCHS = 30      # More epochs
MIN_SAMPLES_PER_CLASS = 5
LEARNING_RATE = 0.0005  # Lower LR for fine-tuning
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


class ProductDataset(Dataset):
    """Custom dataset for product images."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            # Return a blank image on error
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='white')
            if self.transform:
                image = self.transform(image)
            return image, label


def load_data():
    """Load product metadata and filter valid entries."""
    print("Loading metadata...")
    
    with open(METADATA_FILE) as f:
        products = json.load(f)
    
    # Filter to products with downloaded images
    valid_products = []
    for p in products:
        image_path = IMAGES_DIR / f"{p['barcode']}.jpg"
        if image_path.exists():
            p["image_path"] = str(image_path)
            valid_products.append(p)
    
    print(f"Found {len(valid_products)} products with images")
    return valid_products


def prepare_dataset(products):
    """Prepare dataset with class balancing."""
    
    # Group by category
    category_products = {}
    for p in products:
        cat = p["category"]
        if cat not in category_products:
            category_products[cat] = []
        category_products[cat].append(p)
    
    # Filter categories with enough samples
    valid_categories = {
        cat: prods for cat, prods in category_products.items() 
        if len(prods) >= MIN_SAMPLES_PER_CLASS
    }
    
    print(f"Using {len(valid_categories)} categories")
    for cat, prods in sorted(valid_categories.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  {cat}: {len(prods)} samples")
    
    # Create class mapping
    class_names = sorted(valid_categories.keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Prepare data arrays
    image_paths = []
    labels = []
    
    for cat, prods in valid_categories.items():
        for p in prods:
            image_paths.append(p["image_path"])
            labels.append(class_to_idx[cat])
    
    return image_paths, labels, class_names


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay."""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + __import__('math').cos(__import__('math').pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def train():
    """Main training function."""
    print("=" * 60)
    print("ARCart Product Recognition Model Training V2")
    print("ResNet50 + Strong Augmentation + LR Warmup")
    print("=" * 60)
    
    start_time = time.time()
    
    # Setup
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    products = load_data()
    image_paths, labels, class_names = prepare_dataset(products)
    
    print(f"\nTotal samples: {len(image_paths)}")
    print(f"Classes: {len(class_names)}")
    
    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    # Data transforms - V2 with stronger augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),  # Resize larger for crop
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # Random cutout
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and loaders
    train_dataset = ProductDataset(train_paths, train_labels, train_transform)
    val_dataset = ProductDataset(val_paths, val_labels, val_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    # Build model (ResNet50 with transfer learning)
    print("\nBuilding model (ResNet50)...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # Unfreeze more layers for better fine-tuning
    # Freeze only the first conv and bn layers
    for name, param in model.named_parameters():
        if 'layer1' in name or 'layer2' in name:
            param.requires_grad = False
        # layer3 and layer4 will be fine-tuned
    
    # Replace final layer with larger head
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.2),
        nn.Linear(256, len(class_names))
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # LR scheduler with warmup
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=3, total_epochs=EPOCHS, 
        base_lr=LEARNING_RATE
    )
    
    # Training loop
    best_val_acc = 0.0
    patience = 8
    patience_counter = 0
    
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # Update learning rate
        current_lr = scheduler.step(epoch)
        
        # Train phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels_batch) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels_batch.size(0)
            train_correct += predicted.eq(labels_batch).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(device, non_blocking=True)
                labels_batch = labels_batch.to(device, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels_batch.size(0)
                val_correct += predicted.eq(labels_batch).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start
        
        # Print progress
        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
            }, MODELS_DIR / "product_model_v2.pth")
            improved = " ⭐ NEW BEST"
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | LR: {current_lr:.6f} | "
              f"Train: {train_acc:5.1f}% | Val: {val_acc:5.1f}% | "
              f"Time: {epoch_time:.1f}s{improved}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping - no improvement for {patience} epochs")
            break
    
    total_time = time.time() - start_time
    
    # Save class names separately
    class_names_path = MODELS_DIR / "class_names_v2.json"
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"✓ Training complete!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Best validation accuracy: {best_val_acc:.1f}%")
    print(f"  Model saved to: {MODELS_DIR / 'product_model_v2.pth'}")
    print("=" * 60)
    
    # Export to ONNX for web deployment
    print("\nExporting to ONNX...")
    
    # Load best model
    checkpoint = torch.load(MODELS_DIR / "product_model_v2.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    onnx_path = MODELS_DIR / "product_model_v2.onnx"
    
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=14
    )
    print(f"  ONNX model saved to: {onnx_path}")
    print(f"  ONNX size: {os.path.getsize(onnx_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    train()
