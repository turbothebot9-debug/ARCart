#!/usr/bin/env python3
"""
Product Embedding System - Visual product recognition without retraining.

Instead of classifying into categories, this creates "fingerprints" (embeddings)
for products. New products can be added by just uploading photos - no retraining.

How it works:
1. Extract 1280-dim embedding from product image using EfficientNet
2. Store embeddings in database with product info
3. When user scans, find closest match using cosine similarity
"""

import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# Paths
DATA_DIR = Path(__file__).parent.parent / "ml-training/data"
EMBEDDINGS_DIR = Path(__file__).parent / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProductEmbedder:
    """Generates embeddings for product images using pretrained model."""
    
    def __init__(self, model_name: str = "efficientnet_v2_s"):
        self.model_name = model_name
        self.model = self._load_model()
        self.transform = self._get_transform()
        self.embedding_dim = 1280  # EfficientNetV2-S output dim
        
    def _load_model(self) -> nn.Module:
        """Load pretrained model and remove classification head."""
        print(f"Loading {self.model_name} for embeddings on {device}...")
        
        # Load EfficientNetV2-S (good balance of speed and accuracy)
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        
        # Remove classifier, keep feature extractor
        model.classifier = nn.Identity()
        
        model = model.to(device)
        model.eval()
        
        return model
    
    def _get_transform(self) -> transforms.Compose:
        """Image preprocessing for the model."""
        return transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract embedding from a single image."""
        img_tensor = self.transform(image).unsqueeze(0).to(device)
        embedding = self.model(img_tensor)
        return embedding.cpu().numpy().flatten()
    
    @torch.no_grad()
    def get_embeddings_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Extract embeddings from multiple images."""
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            tensors = torch.stack([self.transform(img) for img in batch]).to(device)
            embeddings = self.model(tensors)
            all_embeddings.append(embeddings.cpu().numpy())
            
            if (i + batch_size) % 500 == 0:
                print(f"  Processed {i + batch_size} images...")
        
        return np.vstack(all_embeddings)


class ProductDatabase:
    """
    Vector database for product embeddings using sklearn.
    Enables fast similarity search across products.
    """
    
    def __init__(self, embedding_dim: int = 1280):
        self.embedding_dim = embedding_dim
        self.embeddings: Optional[np.ndarray] = None
        self.products: List[Dict] = []
        self.nn_index: Optional[NearestNeighbors] = None
        self.db_path = EMBEDDINGS_DIR / "product_db.json"
        self.embeddings_path = EMBEDDINGS_DIR / "product_embeddings.npy"
        
    def build_index(self):
        """Build nearest neighbors index."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return
            
        # Normalize embeddings for cosine similarity
        normalized = normalize(self.embeddings)
        
        # Build index
        self.nn_index = NearestNeighbors(
            n_neighbors=min(10, len(self.products)),
            metric='cosine',
            algorithm='brute'  # Works well for <100k products
        )
        self.nn_index.fit(normalized)
        print(f"Built index with {len(self.products)} products")
    
    def add_products(self, embeddings: np.ndarray, products: List[Dict]):
        """Add products to database."""
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self.products.extend(products)
        self.build_index()
        print(f"Database now has {len(self.products)} products")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        """Find k most similar products."""
        if self.nn_index is None or len(self.products) == 0:
            return []
        
        # Normalize query
        query = normalize(query_embedding.reshape(1, -1))
        
        # Search
        k = min(k, len(self.products))
        distances, indices = self.nn_index.kneighbors(query, n_neighbors=k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # Convert cosine distance to similarity (1 - distance)
            similarity = 1 - dist
            results.append((self.products[idx], float(similarity)))
        
        return results
    
    def save(self):
        """Save database to disk."""
        # Save product metadata
        with open(self.db_path, 'w') as f:
            json.dump(self.products, f, indent=2)
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(self.embeddings_path, self.embeddings)
        
        print(f"Saved database: {len(self.products)} products")
    
    def load(self) -> bool:
        """Load database from disk."""
        if not self.db_path.exists() or not self.embeddings_path.exists():
            return False
        
        with open(self.db_path) as f:
            self.products = json.load(f)
        
        self.embeddings = np.load(self.embeddings_path)
        self.build_index()
        
        print(f"Loaded database: {len(self.products)} products")
        return True


class ProductMatcher:
    """
    Main interface for product matching.
    Combines embedder + database for end-to-end product recognition.
    """
    
    def __init__(self):
        self.embedder = ProductEmbedder()
        self.db = ProductDatabase(self.embedder.embedding_dim)
        
        # Try to load existing database
        self.db.load()
    
    def add_product(self, image: Image.Image, product_info: Dict) -> bool:
        """Add a single product to the database."""
        try:
            embedding = self.embedder.get_embedding(image)
            self.db.add_products(
                np.array([embedding]), 
                [product_info]
            )
            return True
        except Exception as e:
            print(f"Error adding product: {e}")
            return False
    
    def add_products_from_folder(self, folder: Path, category: str = None):
        """Add all images from a folder as products."""
        images = []
        products = []
        
        for img_path in folder.glob("*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                    products.append({
                        'id': img_path.stem,
                        'name': img_path.stem.replace('_', ' ').replace('-', ' ').title(),
                        'category': category or folder.name,
                        'image_path': str(img_path)
                    })
                except Exception as e:
                    pass
        
        if images:
            print(f"Generating embeddings for {len(images)} images...")
            embeddings = self.embedder.get_embeddings_batch(images)
            self.db.add_products(embeddings, products)
    
    def identify(self, image: Image.Image, threshold: float = 0.7) -> Optional[Dict]:
        """
        Identify a product from an image.
        Returns best match if confidence > threshold, else None.
        """
        embedding = self.embedder.get_embedding(image)
        results = self.db.search(embedding, k=1)
        
        if results and results[0][1] >= threshold:
            product, confidence = results[0]
            return {
                **product,
                'confidence': round(confidence, 3)
            }
        return None
    
    def search(self, image: Image.Image, k: int = 5) -> List[Dict]:
        """Find k most similar products."""
        embedding = self.embedder.get_embedding(image)
        results = self.db.search(embedding, k=k)
        
        return [
            {**product, 'confidence': round(score, 3)}
            for product, score in results
        ]
    
    def save(self):
        """Save the database."""
        self.db.save()
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        return {
            'total_products': len(self.db.products),
            'embedding_dim': self.embedder.embedding_dim,
            'categories': list(set(p.get('category', 'unknown') for p in self.db.products))
        }


def build_database_from_training_data():
    """Build product database from our training images."""
    print("=" * 60)
    print("Building Product Embedding Database")
    print("=" * 60)
    
    matcher = ProductMatcher()
    
    # Add Freiburg dataset
    freiburg_dir = DATA_DIR / "grocery_dataset" / "images"
    if freiburg_dir.exists():
        print(f"\nProcessing Freiburg dataset...")
        for category_dir in sorted(freiburg_dir.iterdir()):
            if category_dir.is_dir():
                print(f"  {category_dir.name}...", end=" ", flush=True)
                matcher.add_products_from_folder(category_dir, category_dir.name)
                print("✓")
    
    # Add Open Food Facts images  
    off_images = DATA_DIR / "images"
    off_products = DATA_DIR / "products.json"
    
    if off_images.exists() and off_products.exists():
        print(f"\nProcessing Open Food Facts...")
        
        with open(off_products) as f:
            products_meta = json.load(f)
        
        # Create lookup by barcode
        meta_lookup = {p['barcode']: p for p in products_meta}
        
        images = []
        products = []
        
        for img_path in list(off_images.glob("*.jpg"))[:]:
            barcode = img_path.stem
            if barcode in meta_lookup:
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                    meta = meta_lookup[barcode]
                    products.append({
                        'id': barcode,
                        'barcode': barcode,
                        'name': meta.get('name', barcode)[:100],
                        'category': meta.get('category', 'unknown'),
                        'image_path': str(img_path)
                    })
                except:
                    pass
        
        if images:
            print(f"  Generating embeddings for {len(images)} products...")
            embeddings = matcher.embedder.get_embeddings_batch(images)
            matcher.db.add_products(embeddings, products)
    
    # Save database
    matcher.save()
    
    print("\n" + "=" * 60)
    stats = matcher.get_stats()
    print(f"✓ Database built!")
    print(f"  Total products: {stats['total_products']}")
    print(f"  Categories: {len(stats['categories'])}")
    print(f"  Embedding dimension: {stats['embedding_dim']}")
    print("=" * 60)
    
    return matcher


if __name__ == "__main__":
    build_database_from_training_data()
