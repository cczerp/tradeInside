# fetch_data.py
import yfinance as yf
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from database import Database

BATCH_SIZE = 200  # yfinance batch download limit

def fetch_prices_for_all_tickers(days_back=730):
    """Fetch stock prices for all tickers in database using batch downloads"""
    print("\n" + "="*60)
    print("FETCHING STOCK PRICES")
    print("="*60)

    db = Database()
    db.connect()

    db.cursor.execute("SELECT DISTINCT ticker FROM trades WHERE ticker IS NOT NULL")
    rows = db.cursor.fetchall()

    tickers = []
    for row in rows:
        if row[0]:
            ticker = str(row[0]).strip().split()[0].upper()
            if ticker and len(ticker) <= 5:
                tickers.append(ticker)
    tickers = list(set(tickers))

    # Skip tickers that already have fresh data
    tickers_to_fetch = []
    for ticker in tickers:
        db.cursor.execute("SELECT COUNT(*) FROM stock_prices WHERE ticker = ?", (ticker,))
        if db.cursor.fetchone()[0] == 0:
            tickers_to_fetch.append(ticker)

    db.close()

    if not tickers_to_fetch:
        print(f"All {len(tickers)} tickers already have price data")
        return

    print(f"Found {len(tickers)} tickers, fetching {len(tickers_to_fetch)} new ones\n")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    total = 0
    batches = [tickers_to_fetch[i:i+BATCH_SIZE] for i in range(0, len(tickers_to_fetch), BATCH_SIZE)]

    for b_idx, batch in enumerate(batches, 1):
        print(f"  Batch {b_idx}/{len(batches)}: downloading {len(batch)} tickers...")
        try:
            data = yf.download(batch, start=start_date, end=end_date,
                               auto_adjust=True, progress=False, threads=True)
            if data.empty:
                print(f"  Batch {b_idx}: no data returned")
                continue

            # Multi-ticker download has MultiIndex columns; single ticker is flat
            is_multi = isinstance(data.columns, pd.MultiIndex)

            db = Database()
            db.connect()

            for ticker in batch:
                try:
                    if is_multi:
                        if ticker not in data.columns.get_level_values(1):
                            continue
                        hist = data.xs(ticker, axis=1, level=1).dropna(how='all')
                    else:
                        hist = data.dropna(how='all')

                    if hist.empty:
                        continue

                    rows_inserted = 0
                    for date, row in hist.iterrows():
                        db.cursor.execute("""
                            INSERT OR IGNORE INTO stock_prices (ticker, date, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (ticker, date.strftime('%Y-%m-%d'),
                              float(row['Open']), float(row['High']),
                              float(row['Low']), float(row['Close']),
                              int(row.get('Volume', 0))))
                        rows_inserted += db.cursor.rowcount

                    db.conn.commit()
                    total += rows_inserted
                except Exception:
                    pass

            db.close()
            print(f"  Batch {b_idx}: done")

        except Exception as e:
            print(f"  Batch {b_idx}: ERROR — {e}")

    print(f"\n{'='*60}")
    print(f"COMPLETE: {total} price records inserted")
    print(f"{'='*60}")


def fetch_events_for_all_tickers():
    """Fetch corporate events for all tickers in database"""
    print("\n" + "="*60)
    print("FETCHING CORPORATE EVENTS")
    print("="*60)
    
    db = Database()
    db.connect()
    
    # Create table
    db.cursor.execute("""
        CREATE TABLE IF NOT EXISTS corporate_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            event_date DATE NOT NULL,
            event_type TEXT NOT NULL,
            description TEXT,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, event_date, event_type, source)
        )
    """)
    db.conn.commit()
    
    # Get unique tickers
    db.cursor.execute("SELECT DISTINCT ticker FROM trades WHERE ticker IS NOT NULL")
    rows = db.cursor.fetchall()
    
    # Clean tickers - CRITICAL FIX
    tickers = []
    for row in rows:
        if row[0]:
            ticker = str(row[0]).strip().split()[0].upper()
            if ticker and len(ticker) <= 5:
                tickers.append(ticker)
    tickers = list(set(tickers))
    
    db.close()
    
    if not tickers:
        print("No tickers found")
        return
    
    print(f"Found {len(tickers)} unique tickers\n")
    
    def fetch_single_ticker(ticker):
        thread_db = Database()
        thread_db.connect()
        events = 0

        try:
            # Skip if already has events
            thread_db.cursor.execute("SELECT COUNT(*) FROM corporate_events WHERE ticker = ?", (ticker,))
            if thread_db.cursor.fetchone()[0] > 0:
                thread_db.close()
                return f"{ticker}: already exists"

            # Yfinance earnings
            try:
                stock = yf.Ticker(ticker)
                earnings = stock.earnings_dates
                if earnings is not None and not earnings.empty:
                    for date, row in earnings.head(8).iterrows():
                        thread_db.cursor.execute(
                            "INSERT OR IGNORE INTO corporate_events (ticker, event_date, event_type, description, source) VALUES (?, ?, ?, ?, ?)",
                            (ticker, date.strftime('%Y-%m-%d'), 'earnings', 'Earnings', 'yfinance')
                        )
                        if thread_db.cursor.rowcount > 0:
                            events += 1
                    thread_db.conn.commit()
            except:
                pass
            
            # SEC 8-Ks
            try:
                url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=8-K&dateb=&owner=exclude&count=100&output=atom"
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                
                if response.status_code == 200:
                    root = ET.fromstring(response.content)
                    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                        date_elem = entry.find('{http://www.w3.org/2005/Atom}updated')
                        if date_elem:
                            thread_db.cursor.execute(
                                "INSERT OR IGNORE INTO corporate_events (ticker, event_date, event_type, description, source) VALUES (?, ?, ?, ?, ?)",
                                (ticker, date_elem.text.split('T')[0], '8k_filing', '8-K', 'sec_edgar')
                            )
                            if thread_db.cursor.rowcount > 0:
                                events += 1
                    thread_db.conn.commit()
                
                time.sleep(0.12)
            except:
                pass
            
            thread_db.close()
            return f"{ticker}: {events} events"
        except:
            thread_db.close()
            return f"{ticker}: 0 events"
    
    # Parallel fetch
    total = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_single_ticker, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f"  [{i}/{len(tickers)}] {result}")
            if " events" in result:
                try:
                    total += int(result.split(": ")[1].split(" ")[0])
                except:
                    pass
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {total} events")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fetch_data.py [prices|events|all]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "prices":
        fetch_prices_for_all_tickers()
    elif command == "events":
        fetch_events_for_all_tickers()
    elif command == "all":
        fetch_prices_for_all_tickers()
        fetch_events_for_all_tickers()
    else:
        print("Unknown command. Use: prices, events, or all")