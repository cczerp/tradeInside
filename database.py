# database.py
import sqlite3
import os
import sys
import io


if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
DB_FILE = os.path.join(os.path.dirname(__file__), "tradeinsider.db")

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
            print(f"[-] DB connection error: {e}")
            return False

    def close(self):
        if self.conn:
            self.conn.close()

    def migrate(self, schema_file="database.sql"):
        with open(schema_file, "r") as f:
            schema = f.read()
        self.cursor.executescript(schema)
        self.conn.commit()
        print("[✓] Database schema applied")

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

    def get_database_stats(self):
        stats = {}
        self.cursor.execute("SELECT COUNT(*) FROM trades")
        stats["total_trades"] = self.cursor.fetchone()[0]
        self.cursor.execute("SELECT COUNT(*) FROM stock_prices")
        stats["total_prices"] = self.cursor.fetchone()[0]
        return stats
