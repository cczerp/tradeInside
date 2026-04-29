#!/usr/bin/env python3
"""
Add performance indexes to existing database
"""
import sqlite3
import os
import sys
import io

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DB_FILE = "tradeinsider.db"

def add_indexes():
    print("[*] Adding performance indexes to database...")

    if not os.path.exists(DB_FILE):
        print(f"[-] Database not found: {DB_FILE}")
        return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker)",
        "CREATE INDEX IF NOT EXISTS idx_trades_transaction_date ON trades(transaction_date)",
        "CREATE INDEX IF NOT EXISTS idx_trades_insider_name ON trades(insider_name)",
        "CREATE INDEX IF NOT EXISTS idx_trades_ticker_date ON trades(ticker, transaction_date)",
        "CREATE INDEX IF NOT EXISTS idx_trades_transaction_type ON trades(transaction_type)",
        "CREATE INDEX IF NOT EXISTS idx_prices_ticker ON stock_prices(ticker)",
        "CREATE INDEX IF NOT EXISTS idx_prices_date ON stock_prices(date)",
        "CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON stock_prices(ticker, date)",
    ]

    for idx_sql in indexes:
        try:
            cursor.execute(idx_sql)
            idx_name = idx_sql.split("idx_")[1].split(" ")[0]
            print(f"  [+] Created index: idx_{idx_name}")
        except Exception as e:
            print(f"  [-] Failed to create index: {e}")

    conn.commit()
    conn.close()

    print("[+] Indexes added successfully!")

if __name__ == "__main__":
    add_indexes()
