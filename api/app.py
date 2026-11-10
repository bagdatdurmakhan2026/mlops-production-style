import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, FiniteFloat

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "models" / "model.pkl"
PREDICTION_REQUESTS = Counter(
    "restaurant_prediction_requests_total",
    "Total prediction requests accepted by the API",
)
INPUT_DRIFT = Counter(
    "restaurant_input_drift_total",
    "Prediction feature values outside the training reference range",
    ["feature"],
)
PREDICTION_LATENCY = Histogram(
    "restaurant_prediction_duration_seconds",
    "Time spent generating a restaurant rating prediction",
)


def load_model_artifact(path=MODEL_PATH):
    with path.open("rb") as model_file:
        artifact = pickle.load(model_file)

    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        features = artifact.get("features", ["votes", "approx_cost"])
        defaults = artifact.get("binary_defaults", {})
        input_reference = artifact.get("input_reference", {})
        training_environment = artifact.get("training_environment", {})
    else:
        model = artifact
        features = ["votes", "approx_cost"]
        defaults = {}
        input_reference = {}
        training_environment = {}

    if not callable(getattr(model, "predict", None)):
        raise TypeError(f"Model artifact at {path} does not expose predict().")
    if not isinstance(training_environment, dict):
        raise ValueError(f"Model artifact at {path} has invalid environment metadata.")
    for package, runtime_version in (
        ("scikit-learn", sklearn.__version__),
        ("numpy", np.__version__),
    ):
        training_version = training_environment.get(package)
        if training_version and training_version != runtime_version:
            raise RuntimeError(
                f"Model artifact at {path} was trained with {package} "
                f"{training_version}, but runtime has {runtime_version}."
            )
    if not features or not all(isinstance(feature, str) for feature in features):
        raise ValueError(f"Model artifact at {path} has an invalid feature schema.")
    if len(set(features)) != len(features):
        raise ValueError(f"Model artifact at {path} contains duplicate features.")
    return model, features, defaults, input_reference


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, model_features, binary_defaults, input_reference = load_model_artifact(
        MODEL_PATH
    )
    app.state.model = model
    app.state.model_features = model_features
    app.state.binary_defaults = binary_defaults
    app.state.input_reference = input_reference
    logger.info("Model loaded from %s", MODEL_PATH)
    yield


app = FastAPI(
    title="Restaurant Rating API",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionInput(BaseModel):
    features: Annotated[
        list[FiniteFloat],
        Field(
            min_length=2,
            max_length=4,
            description=(
                "Ordered model features. The first two are votes and approximate "
                "cost; omitted binary features use saved training defaults."
            ),
        ),
    ]


@app.get("/healthz", status_code=status.HTTP_200_OK)
def health_check(request: Request):
    return {
        "status": "healthy",
        "model_loaded": hasattr(request.app.state, "model"),
    }


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(data: PredictionInput, request: Request):
    model = request.app.state.model
    model_features = request.app.state.model_features
    binary_defaults = request.app.state.binary_defaults
    if len(data.features) not in {2, len(model_features)}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Expected 2 or {len(model_features)} features "
                f"in this order: {model_features}"
            ),
        )

    if data.features[0] < 0 or data.features[1] < 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="votes and approx_cost must be non-negative",
        )

    values = data.features
    if len(values) == 2 and len(model_features) > 2:
        values = values + [
            binary_defaults.get(feature, 0.0)
            for feature in model_features[2:]
        ]
    features_frame = pd.DataFrame([values], columns=model_features)
    reference = request.app.state.input_reference
    for feature, value in zip(model_features, values):
        bounds = reference.get(feature)
        if bounds and not bounds["lower"] <= value <= bounds["upper"]:
            INPUT_DRIFT.labels(feature=feature).inc()
            logger.warning(
                "Prediction input outside training reference range",
                extra={
                    "feature": feature,
                    "value": value,
                    "lower": bounds["lower"],
                    "upper": bounds["upper"],
                },
            )

    PREDICTION_REQUESTS.inc()
    with PREDICTION_LATENCY.time():
        prediction = model.predict(features_frame)
    return {"prediction": float(prediction[0])}
