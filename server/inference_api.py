#!/usr/bin/env python3
"""
FastAPI inference server for ARCart product recognition.
Run with: uvicorn inference_api:app --host 0.0.0.0 --port 8000
"""

import json
import io
import base64
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Paths (Production model with 85.3% accuracy - EfficientNetV2-S)
MODEL_PATH = Path(__file__).parent.parent / "ml-training/models/product_model_production.pt"
CLASSES_PATH = Path(__file__).parent.parent / "ml-training/models/class_names_production.json"

# Load model and classes
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(MODEL_PATH, map_location=device)
model.eval()
print(f"Model loaded on {device}")

with open(CLASSES_PATH) as f:
    class_names = json.load(f)
print(f"Loaded {len(class_names)} classes")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# FastAPI app
app = FastAPI(title="ARCart Product Recognition API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    image: str  # Base64 encoded image


class PredictionResponse(BaseModel):
    category: str
    confidence: float
    top_predictions: list


@app.get("/")
async def root():
    return {"status": "ok", "model": "ARCart Product Recognition", "classes": len(class_names)}


@app.get("/classes")
async def get_classes():
    return {"classes": class_names}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Decode base64 image
        if "," in request.image:
            # Handle data URL format
            image_data = base64.b64decode(request.image.split(",")[1])
        else:
            image_data = base64.b64decode(request.image)
        
        # Open and preprocess image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, k=5)
        
        top_predictions = [
            {"category": class_names[idx.item()], "confidence": prob.item()}
            for prob, idx in zip(top_probs, top_indices)
        ]
        
        # Best prediction
        best_idx = top_indices[0].item()
        best_conf = top_probs[0].item()
        
        return PredictionResponse(
            category=class_names[best_idx],
            confidence=best_conf,
            top_predictions=top_predictions
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy", "gpu": torch.cuda.is_available()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
