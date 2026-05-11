# DL4AI-210225-project

**CS313 Deep Learning for Artificial Intelligence — Final Project (Spring 2026)**

Time-Series Stock Market Prediction using LSTM, GRU, and Conv1D

**Author:** HOA THI THUY LINH | **Student ID:** 210225

---

## Project Overview

This project applies deep learning to financial time-series prediction across 6 tasks:

| Task | Description | Weight |
|------|-------------|--------|
| Task 1 | Nasdaq Stock Price Prediction (multi-feature, k-th day, k consecutive) | 15% |
| Task 2 | Vietnam Stock Price Prediction (HOSE market, 8 stocks) | 15% |
| Task 3 | Buy/Sell Trading Signal Identification | 20% |
| Task 4 | Portfolio Optimization & Risk Management | 30% |
| Task 5 | Model Deployment (API + Web App + Workflow) | 30% (extra credit) |
| Task 6 | Report & Documentation | 20% |

## Key Results

- **Task 1-2:** LSTM/GRU/Conv1D models trained with 15 features and per-window MinMax normalization. Honest finding: models do not beat the naive baseline at k=1 horizon due to strong price autocorrelation.
- **Task 3:** Buy/sell signal classification with AUC ~0.55 (realistic for stock prediction). PR-AUC, validation threshold tuning, rolling CV, and trading backtest included.
- **Task 4:** DL-Enhanced Markowitz portfolio optimization improves ex-ante Sharpe by +0.95 over classical approach. 7-factor profitability scoring, 8-factor risk heuristic.
- **Task 5:** Full deployment stack: FastAPI REST API, Streamlit dashboard, Airflow/dbt/Airbyte workflow.

## Repository Structure

```
DL4AI-210225-project/
├── DL4AI_Final_Project.ipynb     # Main Jupyter notebook (Tasks 1-4)
├── 210225-project-report.docx    # Project report (Task 6.1)
├── README.md                     # This file (Task 6.3)
├── data_nasdaq/                  # Nasdaq OHLCV data
│   └── *.csv
├── data_vn/                      # Vietnam stock data
│   ├── *.csv                     # Daily OHLCV per stock
│   ├── financial-ratio/          # Quarterly financial ratios
│   ├── dividend-history/         # Dividend data
│   ├── industry-analysis/        # Industry classifications
│   └── ticker-overview.csv       # Stock metadata
├── deployment/                   # Task 5: Deployment
│   ├── api/                      # Task 5.1: FastAPI REST API
│   │   ├── main.py               # API server
│   │   ├── export_model.py       # Model export script
│   │   ├── test_api.py           # API test suite
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── webapp/                   # Task 5.2: Streamlit Web App
│   │   ├── app.py                # Dashboard application
│   │   └── requirements.txt
│   ├── workflow/                  # Task 5.3: AI Engineering Workflow
│   │   ├── dags/
│   │   │   └── stock_prediction_dag.py  # Airflow DAG
│   │   ├── dbt/
│   │   │   ├── dbt_project.yml
│   │   │   └── models/
│   │   │       ├── staging/stg_stock_prices.sql
│   │   │       └── marts/technical_indicators.sql
│   │   ├── docker-compose.yml    # Full stack deployment
│   │   ├── init.sql              # Database schema
│   │   └── architecture.mermaid  # System diagram
│   └── models/                   # Exported SavedModels (generated)
└── *.png                         # EDA and result plots
```

## Setup & Reproduction

### Prerequisites

```bash
python >= 3.10
pip install tensorflow pandas numpy scikit-learn matplotlib plotly
```

### Running the Notebook (Tasks 1-4)

```bash
# Navigate to project directory
cd DL4AI-210225-project

# Launch Jupyter
jupyter notebook DL4AI_Final_Project.ipynb

# Or with increased output rate limit:
jupyter notebook --ServerApp.iopub_msg_rate_limit=10000 --ServerApp.rate_limit_window=5
```

Run all cells sequentially. Expected runtime: ~30-45 minutes on a modern laptop.

### Task 5.1: API Deployment

```bash
cd deployment/api

# Export models (run after notebook training)
python export_model.py --tickers FPT VCB HPG VNM

# Start API server
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# Test
python test_api.py
```

**Sample API call:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "FPT", "window": [[...]]}'
```

**Response:**
```json
{
  "ticker": "FPT",
  "predicted_price": 125400.50,
  "normalized_prediction": 0.6523,
  "model_test_mae": 0.1168
}
```

### Task 5.2: Web Dashboard

```bash
cd deployment/webapp
pip install -r requirements.txt
streamlit run app.py
# Open http://localhost:8501
```

### Task 5.3: Full Stack (Docker)

```bash
cd deployment/workflow
docker compose up -d

# Access:
#   API:       http://localhost:8000
#   Dashboard: http://localhost:8501
#   Airflow:   http://localhost:8080 (admin/admin)
```

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | TensorFlow/Keras (LSTM, GRU, Conv1D, BiLSTM) |
| Data Processing | pandas, numpy, scikit-learn |
| Visualization | matplotlib, plotly |
| API | FastAPI, uvicorn |
| Web App | Streamlit |
| Orchestration | Apache Airflow |
| Data Transform | dbt |
| Data Ingestion | Airbyte |
| Database | PostgreSQL |
| Containerization | Docker, Docker Compose |

## Model Architecture

```
Input: (batch, 30, 15) — 30-day window × 15 features
  ↓
LSTM(32, return_sequences=True) → Dropout(0.3)
  ↓
LSTM(16) → Dropout(0.3)
  ↓
Dense(16, relu) → Dense(1)
  ↓
Output: normalized predicted price
```

**Features (15):** Open, High, Low, Close, Adjusted Close, Volume, SMA-10, SMA-20, MACD, MACD Signal, RSI, BB Upper, BB Lower, Returns, Volatility

**Normalization:** Per-window MinMax (each 30-day window independently scaled to [0,1])

## Important Disclaimers

- Models do **not** beat the naive baseline for next-day (k=1) prediction. This is expected and honestly reported.
- Trading signals (AUC ~0.55) represent weak but non-random statistical edges that do not translate to profitable trading after transaction costs.
- Portfolio optimization results are based on in-sample optimization, not rolling out-of-sample backtest.
- This is an **academic project** and should not be used for real investment decisions.

## License

This project is submitted as coursework for CS313 Deep Learning for Artificial Intelligence, Spring 2026.
