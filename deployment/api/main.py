"""
main.py — FastAPI REST API for stock price prediction.

Endpoints:
    POST /predict          — Predict next-day price from raw OHLCV window
    POST /predict/raw      — Predict from pre-normalized tensor (advanced)
    GET  /models           — List available models
    GET  /health           — Health check

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os, json, glob
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DL4AI Stock Prediction API",
    description="LSTM-based stock price prediction for Vietnam market (CS313 Final Project)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Adjusted Close', 'Volume',
    'SMA_10', 'SMA_20', 'MACD', 'MACD_Signal',
    'RSI', 'BB_Upper', 'BB_Lower', 'Returns', 'Volatility',
]
WINDOW_SIZE = 30


# ── Model registry ────────────────────────────────────────────────────────────
models = {}  # ticker → tf.saved_model
metadata = {}  # ticker → dict


def load_models():
    """Load all exported SavedModels from MODEL_DIR."""
    if not os.path.exists(MODEL_DIR):
        print(f"WARNING: MODEL_DIR {MODEL_DIR} does not exist. No models loaded.")
        return
    for meta_path in glob.glob(os.path.join(MODEL_DIR, "*/metadata.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        ticker = meta["ticker"]
        model_path = os.path.dirname(meta_path)
        try:
            models[ticker] = tf.saved_model.load(model_path)
            metadata[ticker] = meta
            print(f"  Loaded model: {ticker} (MAE={meta['test_mae']:.4f})")
        except Exception as e:
            print(f"  Failed to load {ticker}: {e}")
    print(f"Loaded {len(models)} models from {MODEL_DIR}")


@app.on_event("startup")
def startup():
    load_models()


# ── Request/Response schemas ──────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """Send the last 30 days of OHLCV data for a stock."""
    ticker: str = Field(..., description="Stock ticker (e.g., FPT, VCB, HPG)")
    window: List[List[float]] = Field(
        ...,
        description=(
            f"2D array of shape [{WINDOW_SIZE}, {len(FEATURE_COLS)}]. "
            f"Each row = 1 trading day with columns: {FEATURE_COLS}"
        ),
    )

class RawTensorRequest(BaseModel):
    """Send a pre-normalized tensor (advanced users)."""
    ticker: str
    tensor: List[List[float]] = Field(
        ..., description=f"Normalized 2D array [{WINDOW_SIZE}, {len(FEATURE_COLS)}]"
    )

class PredictResponse(BaseModel):
    ticker: str
    predicted_price: float = Field(..., description="Predicted next-day closing price (VND)")
    normalized_prediction: float = Field(..., description="Raw model output [0-1]")
    window_min: float
    window_range: float
    model_test_mae: Optional[float] = None

class ModelInfo(BaseModel):
    ticker: str
    test_mae: float
    features: int
    window_size: int
    forecast_k: int


# ── Helper: per-window MinMax normalization ───────────────────────────────────
def normalize_window(window: np.ndarray):
    """Apply per-window MinMax normalization (same as training pipeline)."""
    w_min = window.min(axis=0)
    w_max = window.max(axis=0)
    w_range = w_max - w_min
    w_range[w_range == 0] = 1
    normalized = (window - w_min) / w_range
    target_idx = FEATURE_COLS.index('Close')
    return normalized, float(w_min[target_idx]), float(w_range[target_idx])


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(models.keys())}


@app.get("/models", response_model=List[ModelInfo])
def list_models():
    """List all available prediction models."""
    return [
        ModelInfo(
            ticker=t,
            test_mae=metadata[t]["test_mae"],
            features=len(metadata[t]["feature_cols"]),
            window_size=metadata[t]["window_size"],
            forecast_k=metadata[t]["forecast_k"],
        )
        for t in sorted(models.keys())
    ]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict next-day closing price from raw OHLCV+indicators window.

    The API applies per-window MinMax normalization internally (same as training)
    and inverse-transforms the output to the original price scale.
    """
    ticker = req.ticker.upper()
    if ticker not in models:
        raise HTTPException(404, f"Model for '{ticker}' not found. Available: {list(models.keys())}")

    window = np.array(req.window, dtype=np.float32)
    if window.shape != (WINDOW_SIZE, len(FEATURE_COLS)):
        raise HTTPException(
            400,
            f"Expected window shape ({WINDOW_SIZE}, {len(FEATURE_COLS)}), "
            f"got {window.shape}",
        )

    # Normalize
    norm_window, w_min, w_range = normalize_window(window)

    # Predict
    input_tensor = tf.constant(norm_window[np.newaxis], dtype=tf.float32)
    model = models[ticker]
    prediction = model(input_tensor)
    norm_pred = float(prediction.numpy().flatten()[0])

    # Inverse transform to original scale
    predicted_price = norm_pred * w_range + w_min

    return PredictResponse(
        ticker=ticker,
        predicted_price=round(predicted_price, 2),
        normalized_prediction=round(norm_pred, 6),
        window_min=round(w_min, 2),
        window_range=round(w_range, 2),
        model_test_mae=metadata[ticker].get("test_mae"),
    )


@app.post("/predict/raw", response_model=PredictResponse)
def predict_raw(req: RawTensorRequest):
    """Predict from a pre-normalized tensor (for advanced integrations)."""
    ticker = req.ticker.upper()
    if ticker not in models:
        raise HTTPException(404, f"Model for '{ticker}' not found.")

    tensor = np.array(req.tensor, dtype=np.float32)
    if tensor.shape != (WINDOW_SIZE, len(FEATURE_COLS)):
        raise HTTPException(400, f"Expected shape ({WINDOW_SIZE}, {len(FEATURE_COLS)})")

    input_tensor = tf.constant(tensor[np.newaxis], dtype=tf.float32)
    prediction = models[ticker](input_tensor)
    norm_pred = float(prediction.numpy().flatten()[0])

    return PredictResponse(
        ticker=ticker,
        predicted_price=0.0,  # Cannot inverse-transform without norm params
        normalized_prediction=round(norm_pred, 6),
        window_min=0.0,
        window_range=0.0,
        model_test_mae=metadata[ticker].get("test_mae"),
    )


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
