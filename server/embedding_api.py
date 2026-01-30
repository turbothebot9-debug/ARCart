#!/usr/bin/env python3
"""
Product Recognition API - Embedding-based visual search.

Endpoints:
- POST /identify - Identify product from image (returns best match)
- POST /search - Search for similar products (returns top k)
- POST /add - Add new product to database
- GET /stats - Database statistics
- GET /health - Health check
"""

import io
import base64
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image

from product_embeddings import ProductMatcher

# Initialize matcher (loads existing database)
print("Loading product matcher...")
matcher = ProductMatcher()
print(f"Ready! {matcher.get_stats()['total_products']} products in database")

# FastAPI app
app = FastAPI(
    title="ARCart Product Recognition API",
    description="Visual product identification using embedding similarity",
    version="2.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ImageRequest(BaseModel):
    image: str  # Base64 encoded image


class ProductResult(BaseModel):
    id: str
    name: str
    category: str
    confidence: float
    barcode: Optional[str] = None


class IdentifyResponse(BaseModel):
    success: bool
    product: Optional[ProductResult] = None
    message: str = ""


class SearchResponse(BaseModel):
    success: bool
    results: List[ProductResult]
    count: int


class AddProductRequest(BaseModel):
    image: str  # Base64
    name: str
    category: str
    barcode: Optional[str] = None
    price: Optional[float] = None


class StatsResponse(BaseModel):
    total_products: int
    categories: List[str]
    embedding_dim: int


def decode_image(base64_str: str) -> Image.Image:
    """Decode base64 image to PIL Image."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


@app.get("/")
async def root():
    stats = matcher.get_stats()
    return {
        "service": "ARCart Product Recognition API v2.0",
        "method": "embedding-based similarity search",
        "products": stats['total_products'],
        "status": "ready"
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "products": len(matcher.db.products)}


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    stats = matcher.get_stats()
    return StatsResponse(
        total_products=stats['total_products'],
        categories=stats['categories'],
        embedding_dim=stats['embedding_dim']
    )


@app.post("/identify", response_model=IdentifyResponse)
async def identify_product(request: ImageRequest):
    """
    Identify a product from an image.
    Returns the best matching product if confidence > threshold.
    """
    try:
        image = decode_image(request.image)
        result = matcher.identify(image, threshold=0.6)
        
        if result:
            return IdentifyResponse(
                success=True,
                product=ProductResult(
                    id=result.get('id', result.get('barcode', 'unknown')),
                    name=result['name'],
                    category=result['category'],
                    confidence=result['confidence'],
                    barcode=result.get('barcode')
                ),
                message=f"Product identified with {result['confidence']:.1%} confidence"
            )
        else:
            return IdentifyResponse(
                success=False,
                message="No matching product found. Try adding it to the database."
            )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_products(request: ImageRequest, k: int = 5):
    """
    Search for similar products.
    Returns top k matches with confidence scores.
    """
    try:
        image = decode_image(request.image)
        results = matcher.search(image, k=k)
        
        return SearchResponse(
            success=True,
            results=[
                ProductResult(
                    id=r.get('id', r.get('barcode', 'unknown')),
                    name=r['name'],
                    category=r['category'],
                    confidence=r['confidence'],
                    barcode=r.get('barcode')
                )
                for r in results
            ],
            count=len(results)
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/add")
async def add_product(request: AddProductRequest):
    """
    Add a new product to the database.
    No retraining needed - just adds the embedding.
    """
    try:
        image = decode_image(request.image)
        
        product_info = {
            'id': request.barcode or request.name.lower().replace(' ', '_'),
            'name': request.name,
            'category': request.category,
            'barcode': request.barcode,
            'price': request.price
        }
        
        success = matcher.add_product(image, product_info)
        
        if success:
            matcher.save()  # Persist to disk
            return {
                "success": True,
                "message": f"Added '{request.name}' to database",
                "total_products": len(matcher.db.products)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add product")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# File upload endpoint removed - use base64 /add endpoint instead


# ============================================================
# BACKWARDS COMPATIBILITY - /predict endpoint for existing app
# ============================================================

@app.post("/predict")
async def predict_legacy(request: ImageRequest):
    """
    Legacy endpoint for backwards compatibility with existing app.
    Returns format expected by src/ml/recognizer.js
    """
    try:
        image = decode_image(request.image)
        results = matcher.search(image, k=5)
        
        if results:
            best = results[0]
            return {
                "success": True,
                "predicted_class": best['name'],
                "confidence": best['confidence'],
                "top_predictions": [
                    {
                        "label": r['name'],
                        "score": r['confidence'],
                        "category": r['category']
                    }
                    for r in results
                ]
            }
        else:
            return {
                "success": False,
                "predicted_class": "Unknown",
                "confidence": 0.0,
                "top_predictions": []
            }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
