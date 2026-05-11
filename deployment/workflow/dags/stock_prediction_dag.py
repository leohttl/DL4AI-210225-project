"""
stock_prediction_dag.py — Airflow DAG for automated stock prediction pipeline.

Pipeline:
    1. ingest_data      — Fetch latest OHLCV from market data API (simulated via Airbyte)
    2. transform_data   — dbt: clean, compute technical indicators, validate
    3. run_prediction   — Load LSTM model, predict next-day prices
    4. store_results    — Write predictions to PostgreSQL/MongoDB
    5. notify           — Send summary notification

Schedule: Daily at 18:00 ICT (after market close at 15:00)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# ── Default args ──────────────────────────────────────────────────────────────
default_args = {
    'owner': 'dl4ai-210225',
    'depends_on_past': False,
    'email': ['leohoa@rezona.ai'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


# ── Task functions ────────────────────────────────────────────────────────────
def ingest_stock_data(**context):
    """
    Step 1: Ingest latest OHLCV data.

    In production, Airbyte would sync from data sources:
      - Source: Vietnam stock market API (SSI, VNDirect, or cafef.vn)
      - Destination: PostgreSQL raw schema

    For this demo, we read from local CSV and simulate the ingestion.
    """
    import pandas as pd
    import json

    tickers = ['FPT', 'VCB', 'HPG', 'VNM', 'ACB', 'VIC', 'MWG', 'HPG']
    ingested = {}

    for ticker in tickers:
        csv_path = f'/opt/airflow/data/data_vn/{ticker}.csv'
        try:
            df = pd.read_csv(csv_path, parse_dates=['Date'])
            latest_date = df['Date'].max().strftime('%Y-%m-%d')
            ingested[ticker] = {
                'rows': len(df),
                'latest_date': latest_date,
                'status': 'ok',
            }
        except Exception as e:
            ingested[ticker] = {'status': 'error', 'error': str(e)}

    # Push to XCom for downstream tasks
    context['ti'].xcom_push(key='ingested_tickers', value=json.dumps(ingested))
    print(f"Ingested {len([t for t in ingested.values() if t['status'] == 'ok'])} stocks")
    return ingested


def transform_features(**context):
    """
    Step 2: Feature engineering (dbt equivalent).

    In production, dbt would run:
      dbt run --models staging.stg_stock_prices marts.technical_indicators

    Models:
      - stg_stock_prices: clean raw data, handle missing values, filter date range
      - technical_indicators: compute SMA, MACD, RSI, Bollinger Bands, Returns, Volatility
      - feature_windows: create 30-day sliding windows with per-window normalization
    """
    import pandas as pd
    import numpy as np
    import json

    ingested = json.loads(context['ti'].xcom_pull(key='ingested_tickers'))

    FEATURE_COLS = [
        'Open', 'High', 'Low', 'Close', 'Adjusted Close', 'Volume',
        'SMA_10', 'SMA_20', 'MACD', 'MACD_Signal',
        'RSI', 'BB_Upper', 'BB_Lower', 'Returns', 'Volatility',
    ]

    transformed = {}
    for ticker, info in ingested.items():
        if info['status'] != 'ok':
            continue

        df = pd.read_csv(f'/opt/airflow/data/data_vn/{ticker}.csv', parse_dates=['Date'])
        df = df[df['Date'] >= '2012-01-01'].reset_index(drop=True)

        # Compute indicators (same as notebook pipeline)
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

        # Save transformed data
        output_path = f'/opt/airflow/data/transformed/{ticker}_features.parquet'
        df.to_parquet(output_path, index=False)

        transformed[ticker] = {
            'rows': len(df),
            'features': len(FEATURE_COLS),
            'latest_date': df['Date'].max().strftime('%Y-%m-%d'),
        }

    context['ti'].xcom_push(key='transformed', value=json.dumps(transformed))
    print(f"Transformed {len(transformed)} stocks with {len(FEATURE_COLS)} features")


def run_predictions(**context):
    """
    Step 3: Run LSTM predictions for all stocks.

    Loads exported SavedModels and predicts next-day close price
    using the latest 30-day window.
    """
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    import json

    FEATURE_COLS = [
        'Open', 'High', 'Low', 'Close', 'Adjusted Close', 'Volume',
        'SMA_10', 'SMA_20', 'MACD', 'MACD_Signal',
        'RSI', 'BB_Upper', 'BB_Lower', 'Returns', 'Volatility',
    ]
    WINDOW_SIZE = 30

    transformed = json.loads(context['ti'].xcom_pull(key='transformed'))
    predictions = {}

    for ticker in transformed:
        model_path = f'/opt/airflow/models/{ticker}'
        data_path = f'/opt/airflow/data/transformed/{ticker}_features.parquet'

        try:
            model = tf.saved_model.load(model_path)
            df = pd.read_parquet(data_path)

            # Get latest window
            window = df[FEATURE_COLS].iloc[-WINDOW_SIZE:].values

            # Per-window MinMax normalization
            w_min = window.min(axis=0)
            w_max = window.max(axis=0)
            w_range = w_max - w_min
            w_range[w_range == 0] = 1
            norm = (window - w_min) / w_range

            t_idx = FEATURE_COLS.index('Close')
            inp = tf.constant(norm[np.newaxis], dtype=tf.float32)
            pred_norm = float(model(inp).numpy().flatten()[0])
            pred_price = pred_norm * w_range[t_idx] + w_min[t_idx]

            last_close = float(df['Close'].iloc[-1])
            change_pct = (pred_price - last_close) / last_close * 100

            predictions[ticker] = {
                'last_close': last_close,
                'predicted_price': round(pred_price, 2),
                'change_pct': round(change_pct, 2),
                'prediction_date': context['ds'],
            }
            print(f"  {ticker}: {last_close:,.0f} → {pred_price:,.0f} ({change_pct:+.1f}%)")

        except Exception as e:
            predictions[ticker] = {'error': str(e)}
            print(f"  {ticker}: ERROR — {e}")

    context['ti'].xcom_push(key='predictions', value=json.dumps(predictions))


def store_predictions(**context):
    """
    Step 4: Store predictions in database.

    In production, this writes to PostgreSQL or MongoDB:

    PostgreSQL schema:
        CREATE TABLE predictions (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10),
            prediction_date DATE,
            last_close FLOAT,
            predicted_price FLOAT,
            change_pct FLOAT,
            created_at TIMESTAMP DEFAULT NOW()
        );

    MongoDB collection: predictions
    """
    import json

    predictions = json.loads(context['ti'].xcom_pull(key='predictions'))

    # Demo: write to JSON file (replace with DB insert in production)
    output_path = f"/opt/airflow/data/predictions/{context['ds']}.json"
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)

    successful = sum(1 for p in predictions.values() if 'error' not in p)
    print(f"Stored {successful} predictions for {context['ds']}")

    # In production:
    # from sqlalchemy import create_engine
    # engine = create_engine(os.environ['DATABASE_URL'])
    # pd.DataFrame(predictions).T.to_sql('predictions', engine, if_exists='append')


def send_notification(**context):
    """Step 5: Send daily prediction summary."""
    import json

    predictions = json.loads(context['ti'].xcom_pull(key='predictions'))

    summary_lines = [f"📊 Stock Predictions for {context['ds']}:\n"]
    for ticker, pred in sorted(predictions.items()):
        if 'error' in pred:
            summary_lines.append(f"  ❌ {ticker}: Error")
        else:
            emoji = "🟢" if pred['change_pct'] > 0 else "🔴"
            summary_lines.append(
                f"  {emoji} {ticker}: {pred['last_close']:,.0f} → "
                f"{pred['predicted_price']:,.0f} ({pred['change_pct']:+.1f}%)"
            )

    summary = "\n".join(summary_lines)
    print(summary)

    # In production: send via Slack/Email
    # requests.post(SLACK_WEBHOOK, json={'text': summary})


# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id='stock_prediction_pipeline',
    default_args=default_args,
    description='Daily LSTM stock prediction pipeline for Vietnam market',
    schedule_interval='0 18 * * 1-5',  # 18:00 ICT, weekdays only
    start_date=days_ago(1),
    catchup=False,
    tags=['dl4ai', 'stock-prediction', 'lstm'],
) as dag:

    t_ingest = PythonOperator(
        task_id='ingest_stock_data',
        python_callable=ingest_stock_data,
    )

    # In production, this would be: BashOperator(bash_command='dbt run --models ...')
    t_transform = PythonOperator(
        task_id='transform_features',
        python_callable=transform_features,
    )

    t_predict = PythonOperator(
        task_id='run_predictions',
        python_callable=run_predictions,
    )

    t_store = PythonOperator(
        task_id='store_predictions',
        python_callable=store_predictions,
    )

    t_notify = PythonOperator(
        task_id='send_notification',
        python_callable=send_notification,
    )

    # Pipeline: ingest → transform → predict → store → notify
    t_ingest >> t_transform >> t_predict >> t_store >> t_notify
