-- Trades table (insiders, politicians, funds)
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,               -- openinsider, quiver, etc.
    ticker TEXT NOT NULL,
    transaction_date DATE NOT NULL,
    transaction_type TEXT,              -- Purchase, Sale, etc.
    insider_name TEXT,
    politician_name TEXT,
    fund_name TEXT,
    shares INTEGER,
    price REAL,
    filing_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Stock prices table (daily OHLCV)
CREATE TABLE IF NOT EXISTS stock_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date DATE NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)                -- avoid duplicate rows
);

-- Persisted backtest snapshots — populated by `python backtest.py --persist`
-- (also run automatically at the end of pipeline.py). Each pipeline run
-- writes one batch of segment rows under a single run_id, so the API can
-- serve the most recent snapshot without recomputing.
CREATE TABLE IF NOT EXISTS backtest_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,                  -- groups rows from one run
    run_at          TIMESTAMP NOT NULL,
    since_filter    DATE,                           -- --since arg, may be NULL
    horizon_days    INTEGER NOT NULL,
    segment_field   TEXT NOT NULL,                  -- e.g. 'side', 'pattern_set', 'ALL'
    segment_value   TEXT NOT NULL,
    adjusted        TEXT NOT NULL,                  -- 'raw' | 'spy_adj' | 'sector_adj'
    n               INTEGER NOT NULL,
    mean_return     REAL,
    median_return   REAL,
    std_return      REAL,
    hit_rate        REAL,
    sharpe          REAL
);

CREATE INDEX IF NOT EXISTS idx_backtest_runs_run_at ON backtest_runs(run_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_run_id ON backtest_runs(run_id);

-- Performance indexes.
-- The analyzer repeatedly filters trades by ticker, by trader name, and by
-- transaction_date; without these every report scan is a full table scan.
CREATE INDEX IF NOT EXISTS idx_trades_ticker           ON trades(ticker);
CREATE INDEX IF NOT EXISTS idx_trades_transaction_date ON trades(transaction_date);
CREATE INDEX IF NOT EXISTS idx_trades_insider_name     ON trades(insider_name);
CREATE INDEX IF NOT EXISTS idx_trades_politician_name  ON trades(politician_name);
CREATE INDEX IF NOT EXISTS idx_trades_fund_name        ON trades(fund_name);
CREATE INDEX IF NOT EXISTS idx_trades_source           ON trades(source);

-- Price lookups are almost always (ticker, date); the UNIQUE constraint
-- above already covers that, so no extra index is needed here.
