-- init.sql — Database schema for stock prediction pipeline

-- Raw data (populated by Airbyte)
CREATE SCHEMA IF NOT EXISTS raw;
CREATE TABLE IF NOT EXISTS raw.stock_prices (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    adjusted_close FLOAT,
    volume BIGINT,
    synced_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ticker, date)
);

-- Predictions (written by Airflow DAG)
CREATE TABLE IF NOT EXISTS public.predictions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    prediction_date DATE NOT NULL,
    last_close FLOAT,
    predicted_price FLOAT,
    change_pct FLOAT,
    model_version VARCHAR(50) DEFAULT 'lstm-v1',
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ticker, prediction_date, model_version)
);

-- Portfolio recommendations
CREATE TABLE IF NOT EXISTS public.portfolio_recommendations (
    id SERIAL PRIMARY KEY,
    recommendation_date DATE NOT NULL,
    profile VARCHAR(20) NOT NULL, -- aggressive, balanced, conservative
    ticker VARCHAR(10) NOT NULL,
    weight FLOAT NOT NULL,
    expected_return FLOAT,
    risk_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_predictions_date ON public.predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON public.predictions(ticker);
CREATE INDEX IF NOT EXISTS idx_raw_prices_ticker_date ON raw.stock_prices(ticker, date);
