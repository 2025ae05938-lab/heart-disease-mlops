"""Tests for the API. Skipped automatically if no trained model exists yet."""
import os

import pytest
from fastapi.testclient import TestClient

MODEL_FILE = os.path.join("models", "model.joblib")

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_FILE), reason="train the model first (python src/train.py)"
)

SAMPLE = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


def get_client():
    from api.app import app  # imported late so the skip check runs first
    return TestClient(app)


def test_health():
    assert get_client().get("/health").json() == {"status": "ok"}


def test_predict_returns_prediction_and_confidence():
    resp = get_client().post("/predict", json=SAMPLE)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["confidence"] <= 1


def test_predict_rejects_bad_input():
    bad = {**SAMPLE, "age": -5}  # invalid age
    assert get_client().post("/predict", json=bad).status_code == 422
