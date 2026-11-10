import pickle

import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor

from api import app as api_module


@pytest.fixture
def client():
    with TestClient(api_module.app) as test_client:
        yield test_client


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data.get("model_loaded") is True

def test_predict(client):
    # Передаем корректные реалистичные признаки: [votes, approx_cost]
    payload = {"features": [1200.0, 1500.0]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Жестко проверяем, что API возвращает поле "prediction" со значением-числом
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))

def test_predict_rejects_wrong_feature_count(client):
    response = client.post("/predict", json={"features": [100.0, 500.0, 1.0]})
    assert response.status_code == 422


def test_predict_rejects_negative_counts(client):
    response = client.post("/predict", json={"features": [-1, 500]})
    assert response.status_code == 422


def test_metrics_endpoint_exposes_prediction_metrics(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "restaurant_prediction_requests_total" in response.text


def test_out_of_reference_input_is_counted_as_drift(client):
    response = client.post(
        "/predict",
        json={"features": [1_000_000_000, 1_000_000_000]},
    )
    metrics_response = client.get("/metrics")

    assert response.status_code == 200
    assert 'restaurant_input_drift_total{feature="votes"}' in metrics_response.text


def test_predict_rejects_non_finite_features():
    with pytest.raises(ValidationError):
        api_module.PredictionInput(features=[float("nan"), 500.0])


def test_startup_fails_when_model_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(api_module, "MODEL_PATH", tmp_path / "missing.pkl")
    with pytest.raises(FileNotFoundError):
        with TestClient(api_module.app):
            pass


def test_model_loading_rejects_incompatible_sklearn_version(tmp_path):
    artifact_path = tmp_path / "model.pkl"
    with artifact_path.open("wb") as model_file:
        pickle.dump(
            {
                "model": DummyRegressor(),
                "features": ["votes", "approx_cost"],
                "training_environment": {"scikit-learn": "0.0"},
            },
            model_file,
        )

    with pytest.raises(RuntimeError, match="was trained with scikit-learn"):
        api_module.load_model_artifact(artifact_path)