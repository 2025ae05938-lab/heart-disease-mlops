"""Step 6 and 8: FastAPI service that serves the model, with logging + metrics.

Run locally from the project root:
    uvicorn api.app:app --reload --port 8000
Then open http://localhost:8000/docs to try it in the browser.
"""
import logging
import os
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

# ---------- Logging setup (assignment: "log API requests") ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",  # timestamp | level | text
)
logger = logging.getLogger("heart-api")

# ---------- Load the trained pipeline once, when the app starts ----------
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "model.joblib"))
model = joblib.load(MODEL_PATH)  # includes preprocessing + classifier

# Column order the pipeline was trained with
FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak",
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
]

app = FastAPI(title="Heart Disease Risk API", version="1.0.0")

# Adds a /metrics endpoint (request counts, latency...) that Prometheus can scrape
Instrumentator().instrument(app).expose(app)


class Patient(BaseModel):
    """Input JSON schema. FastAPI validates every request against this."""

    age: float = Field(..., ge=1, le=120, examples=[63])
    sex: int = Field(..., ge=0, le=1, examples=[1])
    cp: int = Field(..., ge=1, le=4, examples=[1])
    trestbps: float = Field(..., ge=50, le=300, examples=[145])
    chol: float = Field(..., ge=50, le=700, examples=[233])
    fbs: int = Field(..., ge=0, le=1, examples=[1])
    restecg: int = Field(..., ge=0, le=2, examples=[2])
    thalach: float = Field(..., ge=30, le=250, examples=[150])
    exang: int = Field(..., ge=0, le=1, examples=[0])
    oldpeak: float = Field(..., ge=0, le=10, examples=[2.3])
    slope: int = Field(..., ge=1, le=3, examples=[3])
    ca: int = Field(..., ge=0, le=4, examples=[0])
    thal: int = Field(..., ge=3, le=7, examples=[6])


@app.get("/health")
def health():
    """Simple check used by Docker/Kubernetes to see if the app is alive."""
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: Patient):
    """Return the prediction (0/1) and the model's confidence."""
    start = time.time()  # to measure how long the request takes
    try:
        # Build a one-row table in the exact column order the model expects
        row = pd.DataFrame([patient.model_dump()])[FEATURES]
        proba = float(model.predict_proba(row)[0][1])  # P(heart disease)
        prediction = int(proba >= 0.5)  # 1 if probability is at least 50%
    except Exception as exc:  # any unexpected error -> HTTP 500 with a message
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))

    # Confidence = probability of the predicted class
    confidence = proba if prediction == 1 else 1 - proba
    elapsed_ms = (time.time() - start) * 1000
    # One log line per request: input, output, and latency
    logger.info(
        "predict input=%s prediction=%d confidence=%.3f time_ms=%.1f",
        patient.model_dump(), prediction, confidence, elapsed_ms,
    )
    return {
        "prediction": prediction,
        "label": "heart disease" if prediction == 1 else "no heart disease",
        "confidence": round(confidence, 4),
    }
