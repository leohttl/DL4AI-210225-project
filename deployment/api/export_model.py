"""
export_model.py — Export trained LSTM models to SavedModel format for serving.

Run this script AFTER training in the notebook to produce deployable model artifacts.
Usage:
    python export_model.py --data-dir ../data_vn --output-dir ../models
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

# ── Feature engineering (mirrors notebook) ────────────────────────────────────
FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Adjusted Close', 'Volume',
    'SMA_10', 'SMA_20', 'MACD', 'MACD_Signal',
    'RSI', 'BB_Upper', 'BB_Lower', 'Returns', 'Volatility',
]
WINDOW_SIZE = 30
FORECAST_K = 1


def add_technical_indicators(df):
    """Add technical indicators matching the notebook pipeline."""
    df = df.copy()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    # Bollinger Bands
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_Upper'] = sma20 + 2 * std20
    df['BB_Lower'] = sma20 - 2 * std20
    # Returns & Volatility
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(20).std()
    df.dropna(inplace=True)
    return df


def build_lstm(input_shape, output_size=1):
    """Build LSTM matching notebook architecture."""
    return Sequential([
        LSTM(32, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(16),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(output_size),
    ])


def create_windows(df, feature_cols, target_col='Close', window_size=30,
                   forecast_k=1, mode='kth_day'):
    """Sliding windows with per-window MinMax normalization."""
    data = df[feature_cols].values
    target = df[target_col].values
    t_idx = feature_cols.index(target_col)
    X, y, norms = [], [], []
    for i in range(len(data) - window_size - forecast_k):
        window = data[i:i + window_size]
        w_min = window.min(axis=0)
        w_max = window.max(axis=0)
        w_range = w_max - w_min
        w_range[w_range == 0] = 1
        window_norm = (window - w_min) / w_range
        norms.append({'min': float(w_min[t_idx]), 'range': float(w_range[t_idx])})
        if mode == 'kth_day':
            label = (target[i + window_size + forecast_k - 1] - w_min[t_idx]) / w_range[t_idx]
            y.append(label)
        else:
            labels = [(target[i + window_size + j] - w_min[t_idx]) / w_range[t_idx]
                      for j in range(forecast_k)]
            y.append(labels)
        X.append(window_norm)
    return np.array(X, np.float32), np.array(y, np.float32), norms


def train_and_export(ticker, df, output_dir):
    """Train LSTM on a single stock and export SavedModel."""
    df = add_technical_indicators(df)
    X, y, norms = create_windows(df, FEATURE_COLS, 'Close', WINDOW_SIZE, FORECAST_K)

    n = len(X)
    t = int(n * 0.7)
    v = int(n * 0.85)

    model = build_lstm((WINDOW_SIZE, len(FEATURE_COLS)))
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    model.fit(
        X[:t], y[:t],
        validation_data=(X[t:v], y[t:v]),
        epochs=50, batch_size=32, verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6),
        ]
    )

    # Evaluate
    y_pred = model.predict(X[v:], verbose=0).flatten()
    mae = np.mean(np.abs(y[v:] - y_pred))
    print(f'  {ticker}: test MAE = {mae:.4f}')

    # Export
    model_path = os.path.join(output_dir, ticker)
    model.export(model_path)

    # Save metadata for inference
    meta = {
        'ticker': ticker,
        'feature_cols': FEATURE_COLS,
        'window_size': WINDOW_SIZE,
        'forecast_k': FORECAST_K,
        'target_col': 'Close',
        'test_mae': float(mae),
    }
    with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    return mae


def main():
    parser = argparse.ArgumentParser(description='Export LSTM models for deployment')
    parser.add_argument('--data-dir', default='../data_vn', help='Directory with stock CSVs')
    parser.add_argument('--output-dir', default='../models', help='Output directory for SavedModels')
    parser.add_argument('--tickers', nargs='+', default=['FPT', 'VCB', 'HPG', 'VNM'],
                        help='Tickers to train and export')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for ticker in args.tickers:
        csv_path = os.path.join(args.data_dir, f'{ticker}.csv')
        if not os.path.exists(csv_path):
            print(f'  {ticker}: CSV not found at {csv_path}, skipping')
            continue
        df = pd.read_csv(csv_path, parse_dates=['Date'])
        df = df[df['Date'] >= '2012-01-01'].reset_index(drop=True)
        print(f'Training {ticker} ({len(df)} rows)...')
        train_and_export(ticker, df, args.output_dir)

    print(f'\nModels exported to {args.output_dir}/')


if __name__ == '__main__':
    main()
