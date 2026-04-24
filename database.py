# database.py
import sqlite3
import os
import sys
import io

from logging_config import get_logger

log = get_logger(__name__)


if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DB_FILE = os.path.join(os.path.dirname(__file__), "tradeinsider.db")
SCHEMA_FILE = os.path.join(os.path.dirname(__file__), "database.sql")

class Database:
    def __init__(self, db_file=DB_FILE):
        self.db_file = db_file
        self.conn = None
        self.cursor = None

    def connect(self):
        try:
            self.conn = sqlite3.connect(self.db_file)
            self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            log.error("DB connection error: %s", e)
            print(f"[-] DB connection error: {e}")
            return False

    def close(self):
        if self.conn:
            self.conn.close()

    def migrate(self, schema_file=SCHEMA_FILE):
        with open(schema_file, "r") as f:
            schema = f.read()
        self.cursor.executescript(schema)
        self.conn.commit()
        print("[✓] Database schema applied")

    # migrate_data.py expects this name; keep it as an alias so the migration
    # script works end-to-end.
    def create_tables(self, schema_file=SCHEMA_FILE):
        self.migrate(schema_file)

    def insert_trade(self, trade):
        # Check for duplicate first
        check_sql = """
        SELECT id FROM trades 
        WHERE source = ? AND ticker = ? AND transaction_date = ? 
        AND (insider_name = ? OR politician_name = ?)
        """
        self.cursor.execute(check_sql, (
            trade.get("source"),
            trade.get("ticker"),
            trade.get("transaction_date"),
            trade.get("insider_name"),
            trade.get("politician_name")
        ))
        
        if self.cursor.fetchone():
            return  # Skip duplicate
        
        # Original insert code
        sql = """
        INSERT INTO trades 
        (source, ticker, transaction_date, transaction_type, insider_name,
        politician_name, fund_name, shares, price, filing_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.cursor.execute(sql, (
            trade.get("source"),
            trade.get("ticker"),
            trade.get("transaction_date"),
            trade.get("transaction_type"),
            trade.get("insider_name"),
            trade.get("politician_name"),
            trade.get("fund_name"),
            trade.get("shares"),
            trade.get("price"),
            trade.get("filing_date"),
        ))
        self.conn.commit()

    def insert_price(self, price_row):
        sql = """
        INSERT OR IGNORE INTO stock_prices
        (ticker, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        self.cursor.execute(sql, (
            price_row["ticker"],
            price_row["date"],
            price_row.get("open"),
            price_row.get("high"),
            price_row.get("low"),
            price_row.get("close"),
            price_row.get("volume"),
        ))
        self.conn.commit()

    def bulk_insert_trades(self, trades):
        """Insert a batch of trade dicts in a single transaction.

        Each dict may contain any subset of the trades columns. Rows that
        duplicate an existing (source, ticker, transaction_date, insider_name
        OR politician_name) are skipped. Returns True on success.
        """
        if not trades:
            return True

        check_sql = (
            "SELECT 1 FROM trades "
            "WHERE source = ? AND ticker = ? AND transaction_date = ? "
            "AND (insider_name = ? OR politician_name = ?) LIMIT 1"
        )
        insert_sql = (
            "INSERT INTO trades "
            "(source, ticker, transaction_date, transaction_type, insider_name, "
            " politician_name, fund_name, shares, price, filing_date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )

        inserted = 0
        try:
            for t in trades:
                ticker = t.get("ticker")
                txn_date = t.get("transaction_date")
                if not ticker or not txn_date:
                    continue

                self.cursor.execute(check_sql, (
                    t.get("source"),
                    ticker,
                    txn_date,
                    t.get("insider_name"),
                    t.get("politician_name"),
                ))
                if self.cursor.fetchone():
                    continue

                self.cursor.execute(insert_sql, (
                    t.get("source"),
                    ticker,
                    txn_date,
                    t.get("transaction_type"),
                    t.get("insider_name"),
                    t.get("politician_name"),
                    t.get("fund_name"),
                    t.get("shares"),
                    t.get("price"),
                    t.get("filed_date") or t.get("filing_date"),
                ))
                inserted += 1

            self.conn.commit()
            log.info("bulk_insert_trades: %d inserted / %d submitted", inserted, len(trades))
            return True
        except Exception as e:
            self.conn.rollback()
            log.exception("bulk_insert_trades failed: %s", e)
            return False

    def get_database_stats(self):
        stats = {}
        self.cursor.execute("SELECT COUNT(*) FROM trades")
        stats["total_trades"] = self.cursor.fetchone()[0]
        self.cursor.execute("SELECT COUNT(*) FROM stock_prices")
        stats["total_prices"] = self.cursor.fetchone()[0]

        # Extended stats used by migrate_data.py summary output.
        self.cursor.execute(
            "SELECT COUNT(DISTINCT COALESCE(insider_name, politician_name, fund_name)) "
            "FROM trades "
            "WHERE COALESCE(insider_name, politician_name, fund_name) IS NOT NULL"
        )
        stats["unique_traders"] = self.cursor.fetchone()[0]

        self.cursor.execute("SELECT COUNT(DISTINCT ticker) FROM trades")
        stats["unique_tickers"] = self.cursor.fetchone()[0]

        self.cursor.execute("SELECT source, COUNT(*) FROM trades GROUP BY source")
        stats["by_source"] = dict(self.cursor.fetchall())

        return stats
