-- technical_indicators.sql
-- Mart model: compute technical indicators for LSTM input features
-- Mirrors the notebook's add_technical_indicators() function

WITH prices AS (
    SELECT * FROM {{ ref('stg_stock_prices') }}
),

with_sma AS (
    SELECT *,
        AVG(close_price) OVER (
            PARTITION BY ticker ORDER BY trading_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) AS sma_10,
        AVG(close_price) OVER (
            PARTITION BY ticker ORDER BY trading_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS sma_20,
        STDDEV(close_price) OVER (
            PARTITION BY ticker ORDER BY trading_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS std_20,
        -- Daily returns
        (close_price - LAG(close_price) OVER (PARTITION BY ticker ORDER BY trading_date))
            / NULLIF(LAG(close_price) OVER (PARTITION BY ticker ORDER BY trading_date), 0)
            AS daily_return
    FROM prices
),

with_bands AS (
    SELECT *,
        sma_20 + 2 * std_20 AS bb_upper,
        sma_20 - 2 * std_20 AS bb_lower,
        -- Rolling volatility (20-day std of returns)
        STDDEV(daily_return) OVER (
            PARTITION BY ticker ORDER BY trading_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS volatility
    FROM with_sma
),

-- EMA for MACD requires recursive CTE or window function approximation
-- Using simplified EMA via exponential weighting
with_ema AS (
    SELECT *,
        -- Approximate EMA12 and EMA26 using window functions
        -- Note: In production, use a UDF for exact EMA calculation
        AVG(close_price) OVER (
            PARTITION BY ticker ORDER BY trading_date ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
        ) AS ema_12_approx,
        AVG(close_price) OVER (
            PARTITION BY ticker ORDER BY trading_date ROWS BETWEEN 25 PRECEDING AND CURRENT ROW
        ) AS ema_26_approx
    FROM with_bands
),

with_macd AS (
    SELECT *,
        ema_12_approx - ema_26_approx AS macd,
        AVG(ema_12_approx - ema_26_approx) OVER (
            PARTITION BY ticker ORDER BY trading_date ROWS BETWEEN 8 PRECEDING AND CURRENT ROW
        ) AS macd_signal
    FROM with_ema
),

-- RSI calculation
with_gains AS (
    SELECT *,
        CASE WHEN daily_return > 0 THEN daily_return ELSE 0 END AS gain,
        CASE WHEN daily_return < 0 THEN ABS(daily_return) ELSE 0 END AS loss
    FROM with_macd
),

with_rsi AS (
    SELECT *,
        AVG(gain) OVER (
            PARTITION BY ticker ORDER BY trading_date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) AS avg_gain,
        AVG(loss) OVER (
            PARTITION BY ticker ORDER BY trading_date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) AS avg_loss
    FROM with_gains
)

SELECT
    ticker,
    trading_date,
    open_price,
    high_price,
    low_price,
    close_price,
    adjusted_close,
    volume,
    sma_10,
    sma_20,
    macd,
    macd_signal,
    CASE
        WHEN avg_loss = 0 THEN 100
        ELSE 100 - (100 / (1 + avg_gain / NULLIF(avg_loss, 0)))
    END AS rsi,
    bb_upper,
    bb_lower,
    daily_return AS returns,
    volatility
FROM with_rsi
WHERE sma_20 IS NOT NULL  -- Filter out rows without enough history
ORDER BY ticker, trading_date
