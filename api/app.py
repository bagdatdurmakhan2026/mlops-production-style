import logging
import pickle
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, FiniteFloat

app = FastAPI(title="MLOps Production Pipeline", version="1.0.0")
logger = logging.getLogger(__name__)

# Определяем абсолютный путь к модели
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "models" / "model.pkl"

model = None
model_features = ["votes", "approx_cost"]
binary_defaults = {}

# Загружаем модель при старте приложения
try:
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        model_features = artifact.get("features", model_features)
        binary_defaults = artifact.get("binary_defaults", {})
    else:
        model = artifact
    logger.info("Model loaded from %s", MODEL_PATH)
except Exception as e:
    logger.exception("Could not load model artifact from %s: %s", MODEL_PATH, e)

class PredictionInput(BaseModel):
    features: List[FiniteFloat] = Field(
        ...,
        min_length=2,
        max_length=4,
        description=(
            "Признаки: [votes, approx_cost] или эти признаки и "
            "binary-признаки [online_order, book_table]"
        ),
    )


@app.get("/healthz", status_code=status.HTTP_200_OK)
def health_check():
    """Эндпоинт для проверки здоровья сервиса (liveness/readiness probe)"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model is not loaded"
        )
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(data: PredictionInput):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model is not loaded. Service unavailable."
        )
    
    if len(data.features) not in {2, len(model_features)}:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Ожидалось 2 или {len(model_features)} признака "
                f"в порядке {model_features}"
            ),
        )

    values = data.features
    if len(values) == 2 and len(model_features) > 2:
        values = values + [
            binary_defaults.get(feature, 0.0)
            for feature in model_features[2:]
        ]
    features_frame = pd.DataFrame([values], columns=model_features)
    
    prediction = model.predict(features_frame)
    return {"prediction": float(prediction[0])}