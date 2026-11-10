import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient
from api.app import PredictionInput, app

client = TestClient(app) # явно передаем через app=app

def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data.get("model_loaded") is True

def test_predict():
    # Передаем корректные реалистичные признаки: [votes, approx_cost]
    payload = {"features": [1200.0, 1500.0]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Жестко проверяем, что API возвращает поле "prediction" со значением-числом
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))

def test_predict_rejects_wrong_feature_count():
    response = client.post("/predict", json={"features": [100.0, 500.0, 1.0]})
    assert response.status_code == 422

def test_predict_rejects_non_finite_features():
    with pytest.raises(ValidationError):
        PredictionInput(features=[float("nan"), 500.0])