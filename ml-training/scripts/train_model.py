#!/usr/bin/env python3
"""
Train a product recognition model using transfer learning.
Uses MobileNetV2 as base and fine-tunes on Open Food Facts data.
Exports to TensorFlow.js format for browser use.
"""

import os
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
METADATA_FILE = DATA_DIR / "products.json"
MODELS_DIR = Path(__file__).parent.parent / "models"
TFJS_DIR = MODELS_DIR / "tfjs"

# Training config
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
MIN_SAMPLES_PER_CLASS = 5  # Minimum images needed per category
MAX_CLASSES = 100  # Limit number of classes for manageability


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
    
    # Limit number of classes
    if len(valid_categories) > MAX_CLASSES:
        sorted_cats = sorted(valid_categories.items(), key=lambda x: -len(x[1]))
        valid_categories = dict(sorted_cats[:MAX_CLASSES])
    
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


def load_and_preprocess_image(path):
    """Load and preprocess a single image."""
    try:
        img = load_img(path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def create_dataset(image_paths, labels, batch_size, shuffle=True):
    """Create a tf.data.Dataset."""
    
    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, label
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))
    
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def build_model(num_classes):
    """Build transfer learning model based on MobileNetV2."""
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base_model


def train():
    """Main training function."""
    print("=" * 60)
    print("ARCart Product Recognition Model Training")
    print("=" * 60)
    
    # Setup
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TFJS_DIR.mkdir(parents=True, exist_ok=True)
    
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
    
    # Create datasets
    train_dataset = create_dataset(train_paths, train_labels, BATCH_SIZE, shuffle=True)
    val_dataset = create_dataset(val_paths, val_labels, BATCH_SIZE, shuffle=False)
    
    # Build model
    print("\nBuilding model...")
    model, base_model = build_model(len(class_names))
    
    # Phase 1: Train only top layers
    print("\n--- Phase 1: Training top layers ---")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS // 2,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]
    )
    
    # Phase 2: Fine-tune some base layers
    print("\n--- Phase 2: Fine-tuning ---")
    base_model.trainable = True
    
    # Freeze all layers except the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS // 2,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]
    )
    
    # Save Keras model
    keras_path = MODELS_DIR / "product_model.keras"
    print(f"\nSaving Keras model to {keras_path}...")
    model.save(keras_path)
    
    # Save class names
    class_names_path = MODELS_DIR / "class_names.json"
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved class names to {class_names_path}")
    
    # Convert to TensorFlow.js
    print("\nConverting to TensorFlow.js format...")
    try:
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(model, str(TFJS_DIR))
        print(f"TensorFlow.js model saved to {TFJS_DIR}")
    except ImportError:
        print("tensorflowjs not installed. Run: pip install tensorflowjs")
        print("Then manually convert with:")
        print(f"  tensorflowjs_converter --input_format=keras {keras_path} {TFJS_DIR}")
    
    # Final evaluation
    print("\n--- Final Evaluation ---")
    loss, accuracy = model.evaluate(val_dataset)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    print("\n✓ Training complete!")
    print(f"  Model: {keras_path}")
    print(f"  Classes: {class_names_path}")
    print(f"  TensorFlow.js: {TFJS_DIR}")


if __name__ == "__main__":
    train()
