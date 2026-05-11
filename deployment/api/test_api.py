"""
test_api.py — Test the prediction API with sample data.

Usage:
    # Start the API first:
    #   uvicorn main:app --port 8000
    # Then run:
    python test_api.py
"""

import requests
import json
import numpy as np

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    r = requests.get(f"{BASE_URL}/health")
    print("=== Health Check ===")
    print(json.dumps(r.json(), indent=2))
    assert r.status_code == 200
    print("PASS\n")


def test_list_models():
    """Test model listing."""
    r = requests.get(f"{BASE_URL}/models")
    print("=== Available Models ===")
    for m in r.json():
        print(f"  {m['ticker']}: MAE={m['test_mae']:.4f}, "
              f"{m['features']} features, window={m['window_size']}")
    assert r.status_code == 200
    print("PASS\n")
    return [m["ticker"] for m in r.json()]


def test_predict(ticker="FPT"):
    """Test prediction with synthetic data."""
    print(f"=== Predict {ticker} ===")

    # Generate synthetic window (30 days × 15 features)
    # In production, use real OHLCV + indicator data
    np.random.seed(42)
    base_price = 100.0
    window = []
    for day in range(30):
        price = base_price + np.random.randn() * 2
        row = [
            price - 1,     # Open
            price + 1.5,   # High
            price - 1.5,   # Low
            price,         # Close
            price,         # Adjusted Close
            1e6 + np.random.randn() * 1e5,  # Volume
            price,         # SMA_10
            price,         # SMA_20
            0.5,           # MACD
            0.3,           # MACD_Signal
            55.0,          # RSI
            price + 4,     # BB_Upper
            price - 4,     # BB_Lower
            0.01,          # Returns
            0.02,          # Volatility
        ]
        window.append(row)
        base_price = price

    payload = {"ticker": ticker, "window": window}
    r = requests.post(f"{BASE_URL}/predict", json=payload)

    if r.status_code == 200:
        result = r.json()
        print(f"  Predicted price: {result['predicted_price']:.2f}")
        print(f"  Normalized pred: {result['normalized_prediction']:.6f}")
        print(f"  Window min: {result['window_min']}, range: {result['window_range']}")
        print("PASS\n")
    elif r.status_code == 404:
        print(f"  Model not found (expected if model not exported yet)")
        print(f"  Response: {r.json()}")
        print("SKIP\n")
    else:
        print(f"  ERROR {r.status_code}: {r.json()}")
        print("FAIL\n")

    return r


def test_predict_bad_shape():
    """Test error handling for wrong input shape."""
    print("=== Bad Shape Test ===")
    payload = {"ticker": "FPT", "window": [[1, 2, 3]]}  # Wrong shape
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    assert r.status_code == 400
    print(f"  Correctly rejected: {r.json()['detail'][:80]}")
    print("PASS\n")


def test_predict_unknown_ticker():
    """Test error handling for unknown ticker."""
    print("=== Unknown Ticker Test ===")
    np.random.seed(0)
    window = np.random.randn(30, 15).tolist()
    payload = {"ticker": "UNKNOWN", "window": window}
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    assert r.status_code == 404
    print(f"  Correctly rejected: {r.json()['detail'][:80]}")
    print("PASS\n")


if __name__ == "__main__":
    print("=" * 60)
    print("DL4AI Stock Prediction API — Test Suite")
    print("=" * 60 + "\n")

    test_health()
    tickers = test_list_models()

    if tickers:
        test_predict(tickers[0])
    else:
        test_predict("FPT")

    test_predict_bad_shape()
    test_predict_unknown_ticker()

    print("All tests completed!")
