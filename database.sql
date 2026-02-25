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
