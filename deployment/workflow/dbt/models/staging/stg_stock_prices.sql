-- stg_stock_prices.sql
-- Staging model: clean raw stock price data
-- Source: Airbyte sync from Vietnam market data API → PostgreSQL raw schema

WITH raw AS (
    SELECT
        ticker,
        date::DATE AS trading_date,
        open::FLOAT AS open_price,
        high::FLOAT AS high_price,
        low::FLOAT AS low_price,
        close::FLOAT AS close_price,
        COALESCE(adjusted_close::FLOAT, close::FLOAT) AS adjusted_close,
        volume::BIGINT AS volume
    FROM {{ source('raw', 'stock_prices') }}
    WHERE date >= '2012-01-01'
      AND close IS NOT NULL
      AND volume > 0
),

-- Remove duplicates (keep latest sync)
deduplicated AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY ticker, trading_date ORDER BY trading_date) AS rn
    FROM raw
)

SELECT
    ticker,
    trading_date,
    open_price,
    high_price,
    low_price,
    close_price,
    adjusted_close,
    volume
FROM deduplicated
WHERE rn = 1
ORDER BY ticker, trading_date
