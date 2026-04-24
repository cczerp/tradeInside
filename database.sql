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
