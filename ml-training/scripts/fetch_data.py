#!/usr/bin/env python3
"""
Fetch product data and images from Open Food Facts API
for training the ARCart product recognition model.
"""

import os
import json
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

# Config
DATA_DIR = Path(__file__).parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
METADATA_FILE = DATA_DIR / "products.json"

# Categories to fetch (common grocery items)
CATEGORIES = [
    "beverages",
    "sodas",
    "waters",
    "juices",
    "cereals",
    "snacks",
    "chips",
    "chocolates",
    "candies",
    "dairy",
    "milks",
    "yogurts",
    "cheeses",
    "breads",
    "canned-foods",
    "soups",
    "pasta",
    "sauces",
    "condiments",
    "coffee",
    "tea",
    "energy-drinks",
    "beers",
    "frozen-foods",
    "ice-creams",
    "cookies",
    "crackers"
]

# API settings
API_BASE = "https://world.openfoodfacts.org"
PRODUCTS_PER_CATEGORY = 100  # Adjust based on needs
REQUEST_DELAY = 0.5  # Be nice to the API


def setup_dirs():
    """Create necessary directories."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {DATA_DIR}")
    print(f"Images directory: {IMAGES_DIR}")


def fetch_category_products(category: str, limit: int = PRODUCTS_PER_CATEGORY) -> list:
    """Fetch products from a specific category."""
    url = f"{API_BASE}/category/{category}.json"
    params = {
        "page_size": min(limit, 100),
        "page": 1,
        "fields": "code,product_name,brands,categories_tags,image_url,image_small_url,image_front_url"
    }
    
    products = []
    
    try:
        while len(products) < limit:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "products" not in data or not data["products"]:
                break
                
            for p in data["products"]:
                # Only include products with images and names
                if p.get("product_name") and (p.get("image_url") or p.get("image_front_url")):
                    products.append({
                        "barcode": p.get("code", ""),
                        "name": p.get("product_name", "Unknown"),
                        "brand": p.get("brands", ""),
                        "category": category,
                        "categories": p.get("categories_tags", []),
                        "image_url": p.get("image_front_url") or p.get("image_url"),
                        "image_small_url": p.get("image_small_url", "")
                    })
                    
                    if len(products) >= limit:
                        break
            
            params["page"] += 1
            time.sleep(REQUEST_DELAY)
            
            # Check if more pages exist
            if params["page"] > data.get("page_count", 1):
                break
                
    except Exception as e:
        print(f"  Error fetching {category}: {e}")
    
    return products


def download_image(product: dict) -> bool:
    """Download product image."""
    if not product.get("image_url"):
        return False
        
    barcode = product.get("barcode", "unknown")
    image_path = IMAGES_DIR / f"{barcode}.jpg"
    
    # Skip if already downloaded
    if image_path.exists():
        return True
    
    try:
        response = requests.get(product["image_url"], timeout=30)
        response.raise_for_status()
        
        # Verify it's an image
        content_type = response.headers.get("content-type", "")
        if "image" not in content_type:
            return False
        
        image_path.write_bytes(response.content)
        return True
        
    except Exception as e:
        return False


def main():
    print("=" * 60)
    print("Open Food Facts Data Fetcher for ARCart")
    print("=" * 60)
    
    setup_dirs()
    
    all_products = []
    
    # Fetch products from each category
    print(f"\nFetching products from {len(CATEGORIES)} categories...")
    for i, category in enumerate(CATEGORIES, 1):
        print(f"[{i}/{len(CATEGORIES)}] Fetching {category}...", end=" ")
        products = fetch_category_products(category)
        print(f"got {len(products)} products")
        all_products.extend(products)
        time.sleep(REQUEST_DELAY)
    
    # Remove duplicates (same barcode)
    seen_barcodes = set()
    unique_products = []
    for p in all_products:
        if p["barcode"] and p["barcode"] not in seen_barcodes:
            seen_barcodes.add(p["barcode"])
            unique_products.append(p)
    
    print(f"\nTotal unique products: {len(unique_products)}")
    
    # Save metadata
    print(f"Saving metadata to {METADATA_FILE}...")
    with open(METADATA_FILE, "w") as f:
        json.dump(unique_products, f, indent=2)
    
    # Download images
    print(f"\nDownloading images...")
    downloaded = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_image, p): p for p in unique_products}
        
        for i, future in enumerate(as_completed(futures), 1):
            if future.result():
                downloaded += 1
            else:
                failed += 1
            
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(unique_products)} (downloaded: {downloaded}, failed: {failed})")
    
    print(f"\nComplete!")
    print(f"  Products: {len(unique_products)}")
    print(f"  Images downloaded: {downloaded}")
    print(f"  Failed: {failed}")
    print(f"\nData saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
