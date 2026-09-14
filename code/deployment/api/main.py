from enum import Enum
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model" / "model.pkl"

app = FastAPI(title="Penguin Classifier API", version="1.0")

# Load once at startup, not per request. If the file is missing, still start
# and report via /health — easier to diagnose than a container that
# silently crashes on boot.
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None

# Enums restrict input to valid categories: bad values get a clear 422
# validation error instead of reaching the model.
class Island(str, Enum):
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"


class Sex(str, Enum):
    male = "male"
    female = "female"


class PenguinFeatures(BaseModel):
    # Physically plausible ranges: reject garbage before it hits the model.
    bill_length_mm: float = Field(..., ge=20, le=70)
    bill_depth_mm: float = Field(..., ge=10, le=30)
    flipper_length_mm: float = Field(..., ge=150, le=250)
    body_mass_g: float = Field(..., ge=2000, le=8000)
    island: Island
    sex: Sex


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

# The sklearn Pipeline selects columns BY NAME, so the request is wrapped
# in a one-row DataFrame with the training-time column names; field order
# in the request does not matter.
@app.post("/predict")
def predict(features: PenguinFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    df = pd.DataFrame([features.model_dump()])
    # Return the full class distribution, not just the label — the UI
    # visualizes per-class probabilities.
    probs = model.predict_proba(df)[0]
    # classes_ gives the class order matching predict_proba output columns.
    probabilities = {cls: round(float(p), 4) for cls, p in zip(model.classes_, probs)}

    return {
        "prediction": max(probabilities, key=probabilities.get),
        "probabilities": probabilities,
    }