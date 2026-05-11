"""
app.py — Streamlit SaaS interface for DL4AI Stock Prediction.

Features:
  - Select stock ticker from available models
  - Auto-fetch latest data from local CSV or manual input
  - Display prediction with confidence context
  - Interactive charts: historical prices, prediction overlay, technical indicators
  - Portfolio recommendation summary

Usage:
    streamlit run app.py
"""

import os, json, glob
import numpy as np
import pandas as pd
import streamlit as st
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DL4AI Stock Predictor",
    page_icon="📈",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "http://localhost:8000")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data_vn"))

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Adjusted Close', 'Volume',
    'SMA_10', 'SMA_20', 'MACD', 'MACD_Signal',
    'RSI', 'BB_Upper', 'BB_Lower', 'Returns', 'Volatility',
]
WINDOW_SIZE = 30


# ── Technical indicators (matching notebook) ──────────────────────────────────
def add_technical_indicators(df):
    df = df.copy()
    if 'Adjusted Close' not in df.columns:
        df['Adjusted Close'] = df['Close']
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_Upper'] = sma20 + 2 * std20
    df['BB_Lower'] = sma20 - 2 * std20
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(20).std()
    df.dropna(inplace=True)
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("DL4AI Stock Predictor")
st.sidebar.markdown("**CS313 Deep Learning — Final Project**")
st.sidebar.markdown("*HOA THI THUY LINH — 210225*")
st.sidebar.markdown("---")

# Available tickers from data directory
available_tickers = []
if os.path.exists(DATA_DIR):
    csvs = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    available_tickers = sorted([os.path.basename(f).replace(".csv", "") for f in csvs])

ticker = st.sidebar.selectbox(
    "Select Stock Ticker",
    available_tickers if available_tickers else ["FPT", "VCB", "HPG", "VNM", "ACB", "VIC", "MWG"],
    index=0,
)

use_api = st.sidebar.checkbox("Use API (requires running server)", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How it works:**\n"
    "1. Load last 30 days of OHLCV data\n"
    "2. Compute technical indicators\n"
    "3. Per-window MinMax normalization\n"
    "4. LSTM predicts next-day Close\n"
    "5. Inverse transform to VND price"
)


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_stock_data(ticker):
    csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df[df["Date"] >= "2012-01-01"].reset_index(drop=True)
    df = add_technical_indicators(df)
    return df


# ── Local prediction (TF model) ──────────────────────────────────────────────
@st.cache_resource
def load_local_model(ticker):
    """Load SavedModel for local inference (no API needed)."""
    import tensorflow as tf
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", ticker)
    if not os.path.exists(model_path):
        return None
    return tf.saved_model.load(model_path)


def predict_local(model, window):
    """Run prediction locally with a loaded SavedModel."""
    import tensorflow as tf
    w_min = window.min(axis=0)
    w_max = window.max(axis=0)
    w_range = w_max - w_min
    w_range[w_range == 0] = 1
    norm = (window - w_min) / w_range
    t_idx = FEATURE_COLS.index('Close')
    inp = tf.constant(norm[np.newaxis], dtype=tf.float32)
    pred = float(model(inp).numpy().flatten()[0])
    return pred * w_range[t_idx] + w_min[t_idx], pred


def predict_via_api(ticker, window):
    """Call the FastAPI prediction endpoint."""
    payload = {"ticker": ticker, "window": window.tolist()}
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    if r.status_code == 200:
        data = r.json()
        return data["predicted_price"], data["normalized_prediction"]
    else:
        st.error(f"API error: {r.status_code} — {r.json().get('detail', 'Unknown error')}")
        return None, None


# ── Main content ──────────────────────────────────────────────────────────────
st.title(f"📈 {ticker} — Stock Price Prediction")

df = load_stock_data(ticker)

if df is None:
    st.error(f"No data found for {ticker}. Check DATA_DIR: {DATA_DIR}")
    st.stop()

# ── Metrics row ───────────────────────────────────────────────────────────────
latest = df.iloc[-1]
prev = df.iloc[-2]
change = latest['Close'] - prev['Close']
change_pct = change / prev['Close'] * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Last Close", f"{latest['Close']:,.0f} VND", f"{change:+,.0f} ({change_pct:+.1f}%)")
col2.metric("Volume", f"{latest['Volume']:,.0f}")
col3.metric("RSI (14)", f"{latest['RSI']:.1f}")
col4.metric("Data Points", f"{len(df):,}")

# ── Prediction ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Next-Day Price Prediction")

window = df[FEATURE_COLS].iloc[-WINDOW_SIZE:].values

predicted_price, norm_pred = None, None

if use_api:
    predicted_price, norm_pred = predict_via_api(ticker, window)
else:
    model = load_local_model(ticker)
    if model is not None:
        predicted_price, norm_pred = predict_local(model, window)
    else:
        st.warning(
            f"No local model found for {ticker}. "
            "Run `python export_model.py --tickers {ticker}` first, "
            "or check 'Use API' in sidebar."
        )

if predicted_price is not None:
    pred_change = predicted_price - latest['Close']
    pred_change_pct = pred_change / latest['Close'] * 100
    direction = "up" if pred_change > 0 else "down"

    pcol1, pcol2, pcol3 = st.columns(3)
    pcol1.metric(
        "Predicted Next Close",
        f"{predicted_price:,.0f} VND",
        f"{pred_change:+,.0f} ({pred_change_pct:+.1f}%)",
    )
    pcol2.metric("Direction", f"{'🟢 UP' if pred_change > 0 else '🔴 DOWN'}")
    pcol3.metric("Normalized Output", f"{norm_pred:.4f}")

    st.caption(
        "⚠️ **Disclaimer:** This prediction is from an academic LSTM model. "
        "It does NOT beat the naive baseline (tomorrow ≈ today) and should NOT "
        "be used for real trading decisions."
    )

# ── Charts ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Historical Price & Technical Indicators")

# Show last N days
show_days = st.slider("Days to display", 30, len(df), min(250, len(df)))
plot_df = df.iloc[-show_days:]

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.5, 0.25, 0.25],
    subplot_titles=("Price & Bollinger Bands", "MACD", "RSI"),
)

# Candlestick
fig.add_trace(
    go.Candlestick(
        x=plot_df["Date"], open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"], close=plot_df["Close"], name="OHLC",
    ),
    row=1, col=1,
)
fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["BB_Upper"],
                         line=dict(width=1, dash="dot", color="gray"), name="BB Upper"), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["BB_Lower"],
                         line=dict(width=1, dash="dot", color="gray"), name="BB Lower",
                         fill="tonexty", fillcolor="rgba(200,200,200,0.1)"), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["SMA_10"],
                         line=dict(width=1, color="orange"), name="SMA 10"), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["SMA_20"],
                         line=dict(width=1, color="blue"), name="SMA 20"), row=1, col=1)

# Add prediction marker
if predicted_price is not None:
    next_date = plot_df["Date"].iloc[-1] + pd.Timedelta(days=1)
    fig.add_trace(go.Scatter(
        x=[next_date], y=[predicted_price],
        mode="markers", marker=dict(size=12, color="red", symbol="star"),
        name=f"Prediction: {predicted_price:,.0f}",
    ), row=1, col=1)

# MACD
fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["MACD"],
                         line=dict(color="blue", width=1), name="MACD"), row=2, col=1)
fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["MACD_Signal"],
                         line=dict(color="orange", width=1), name="Signal"), row=2, col=1)
macd_hist = plot_df["MACD"] - plot_df["MACD_Signal"]
colors = ["green" if v >= 0 else "red" for v in macd_hist]
fig.add_trace(go.Bar(x=plot_df["Date"], y=macd_hist, marker_color=colors,
                     name="Histogram"), row=2, col=1)

# RSI
fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["RSI"],
                         line=dict(color="purple", width=1), name="RSI"), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# ── Recent data table ─────────────────────────────────────────────────────────
st.subheader("Recent Data (Last 10 Days)")
display_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "SMA_10", "SMA_20"]
st.dataframe(
    df[display_cols].tail(10).style.format({
        "Open": "{:,.0f}", "High": "{:,.0f}", "Low": "{:,.0f}", "Close": "{:,.0f}",
        "Volume": "{:,.0f}", "RSI": "{:.1f}", "MACD": "{:.2f}",
        "SMA_10": "{:,.0f}", "SMA_20": "{:,.0f}",
    }),
    use_container_width=True,
)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "*DL4AI Stock Prediction SaaS — CS313 Deep Learning Final Project | "
    "HOA THI THUY LINH — 210225 | May 2026*"
)
