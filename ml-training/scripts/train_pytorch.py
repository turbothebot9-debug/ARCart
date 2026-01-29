#!/usr/bin/env python3
"""
Train a product recognition model using PyTorch with GPU acceleration.
Uses ResNet18 as base and fine-tunes on Open Food Facts data.
"""

import json
import os
from pathlib import Path

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

# Training config
IMG_SIZE = 224
BATCH_SIZE = 64  # RTX 4090 can handle big batches
EPOCHS = 15
MIN_SAMPLES_PER_CLASS = 5
LEARNING_RATE = 0.001

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


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


def train():
    """Main training function."""
    print("=" * 60)
    print("ARCart Product Recognition Model Training (PyTorch)")
    print("=" * 60)
    
    # Setup
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    products = load_data()
    image_paths, labels, class_names = prepare_dataset(products)
    
    print(f"Total samples: {len(image_paths)}")
    print(f"Classes: {len(class_names)}")
    
    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and loaders
    train_dataset = ProductDataset(train_paths, train_labels, train_transform)
    val_dataset = ProductDataset(val_paths, val_labels, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Build model (ResNet18 with transfer learning)
    print("\nBuilding model (ResNet18)...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze early layers
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False
    
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, len(class_names))
    )
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    best_val_acc = 0.0
    
    print("\n--- Training ---")
    for epoch in range(EPOCHS):
        # Train phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels_batch in train_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels_batch.size(0)
            train_correct += predicted.eq(labels_batch).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(device)
                labels_batch = labels_batch.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels_batch.size(0)
                val_correct += predicted.eq(labels_batch).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODELS_DIR / "product_model.pth")
        
        scheduler.step()
    
    # Save class names
    class_names_path = MODELS_DIR / "class_names.json"
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"  Best validation accuracy: {best_val_acc:.1f}%")
    print(f"  Model saved to: {MODELS_DIR / 'product_model.pth'}")
    print(f"  Classes saved to: {class_names_path}")
    
    # Export to ONNX for web deployment
    print("\nExporting to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    onnx_path = MODELS_DIR / "product_model.onnx"
    
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )
    print(f"  ONNX model saved to: {onnx_path}")


if __name__ == "__main__":
    train()
