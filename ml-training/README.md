# ARCart ML Training

Train a custom product recognition model using Open Food Facts data.

## Setup

```bash
cd ml-training
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

## Step 1: Fetch Training Data

Downloads product images and metadata from Open Food Facts:

```bash
python scripts/fetch_data.py
```

This will:
- Fetch ~2,500 products across 25+ categories
- Download product images
- Save metadata to `data/products.json`

Takes about 10-15 minutes depending on connection.

## Step 2: Train the Model

```bash
python scripts/train_model.py
```

This will:
- Load downloaded images
- Fine-tune MobileNetV2 on product categories
- Export to TensorFlow.js format

Training takes ~30-60 minutes on CPU, ~10 minutes on GPU.

## Step 3: Use in ARCart

Copy the trained model to the web app:

```bash
cp -r models/tfjs/* ../public/models/
cp models/class_names.json ../public/models/
```

Then update `src/ml/recognizer.js` to load your custom model.

## Output

```
models/
├── product_model.keras    # Full Keras model
├── class_names.json       # Category labels
└── tfjs/                  # TensorFlow.js model
    ├── model.json
    └── group1-shard1of1.bin
```

## Customization

Edit `scripts/fetch_data.py` to change:
- `CATEGORIES` - which product types to fetch
- `PRODUCTS_PER_CATEGORY` - how many products per category

Edit `scripts/train_model.py` to change:
- `EPOCHS` - training iterations
- `IMG_SIZE` - input image dimensions
- `MIN_SAMPLES_PER_CLASS` - minimum images needed per category

## Adding Your Own Products

1. Take photos of products
2. Add to `data/images/` as `{barcode}.jpg`
3. Add entry to `data/products.json`
4. Re-run training
