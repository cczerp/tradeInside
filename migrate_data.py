import os
import pandas as pd
import glob
from datetime import datetime
from database import Database

def parse_date(date_str):
    """Convert date string to datetime object, handling various formats"""
    if pd.isna(date_str) or date_str == '':
        return None
    
    # Try mm/dd/yyyy format first
    try:
        return datetime.strptime(str(date_str), '%m/%d/%Y').date()
    except:
        pass
    
    # Try other common formats
    for fmt in ['%Y-%m-%d', '%m-%d-%Y', '%d/%m/%Y']:
        try:
            return datetime.strptime(str(date_str), fmt).date()
        except:
            continue
    
    return None

def clean_numeric(value):
    """Clean numeric values, handling currency symbols and commas"""
    if pd.isna(value) or value == '':
        return None
    
    # Remove $, commas, and whitespace
    cleaned = str(value).replace('$', '').replace(',', '').strip()
    
    try:
        return float(cleaned)
    except:
        return None

def prepare_trade_data(row, source, section):
    """Convert a DataFrame row to trade data dict"""
    return {
        'source': source,
        'section': section,
        'insider_name': row.get('InsiderName'),
        'position': row.get('Position'),
        'politician_name': row.get('PoliticianName'),
        'chamber': row.get('Chamber'),
        'party': row.get('Party'),
        'fund_name': row.get('FundName') or row.get('Fund'),
        'ticker': row.get('Ticker'),
        'stock_name': row.get('StockName'),
        'transaction_type': row.get('TransactionType'),
        'shares': clean_numeric(row.get('Shares')),
        'price': clean_numeric(row.get('Price')),
        'value': clean_numeric(row.get('Value')),
        'amount_range': row.get('Amount'),
        'transaction_date': parse_date(row.get('TransactionDate')),
        'disclosed_date': parse_date(row.get('Disclosed')),
        'filed_date': parse_date(row.get('Filed')),
        'traded_date': parse_date(row.get('Traded')),
        'description': row.get('Description'),
        'return_pct': clean_numeric(row.get('Return')),
        'scraped_at': parse_date(row.get('ScrapedAt')) or datetime.now()
    }

def load_csv_files(data_folder='./data'):
    """Load all CSV files from data folder"""
    print(f"\n[+] Scanning for CSV files in {data_folder}...")
    
    all_files = glob.glob(os.path.join(data_folder, '**', '*.csv'), recursive=True)
    
    if not all_files:
        print(f"[-] No CSV files found in {data_folder}")
        return []
    
    print(f"[*] Found {len(all_files)} CSV files")
    
    all_trades = []
    
    for file_path in all_files:
        try:
            # Extract source and section from file path
            # Example: ./data/quiver/insider_trading_20250929.csv
            path_parts = file_path.split(os.sep)
            
            if len(path_parts) >= 3:
                source = path_parts[-2]  # 'quiver'
                filename = path_parts[-1]  # 'insider_trading_20250929.csv'
                section = filename.split('_')[0] + '_' + filename.split('_')[1] if '_' in filename else 'unknown'
            else:
                source = 'unknown'
                section = 'unknown'
            
            print(f"\n[*] Loading: {file_path}")
            print(f"    Source: {source}, Section: {section}")
            
            # Read CSV
            df = pd.read_csv(file_path)
            print(f"    Rows: {len(df)}, Columns: {list(df.columns)}")
            
            # Convert each row to trade data
            for _, row in df.iterrows():
                trade_data = prepare_trade_data(row, source, section)
                
                # Only add if we have at least a ticker
                if trade_data.get('ticker'):
                    all_trades.append(trade_data)
            
            print(f"[✓] Loaded {len(df)} trades from {file_path}")
            
        except Exception as e:
            print(f"[-] Error loading {file_path}: {e}")
            continue
    
    return all_trades

def load_json_files(data_folder='./data'):
    """Load all JSON files from data folder"""
    print(f"\n[+] Scanning for JSON files in {data_folder}...")
    
    all_files = glob.glob(os.path.join(data_folder, '**', '*.json'), recursive=True)
    
    if not all_files:
        print(f"[-] No JSON files found in {data_folder}")
        return []
    
    print(f"[*] Found {len(all_files)} JSON files")
    
    all_trades = []
    
    for file_path in all_files:
        try:
            # Extract source and section from file path
            path_parts = file_path.split(os.sep)
            
            if len(path_parts) >= 3:
                source = path_parts[-2]
                filename = path_parts[-1]
                section = filename.split('_')[0] + '_' + filename.split('_')[1] if '_' in filename else 'unknown'
            else:
                source = 'unknown'
                section = 'unknown'
            
            print(f"\n[*] Loading: {file_path}")
            print(f"    Source: {source}, Section: {section}")
            
            # Read JSON
            df = pd.read_json(file_path)
            print(f"    Rows: {len(df)}, Columns: {list(df.columns)}")
            
            # Convert each row to trade data
            for _, row in df.iterrows():
                trade_data = prepare_trade_data(row, source, section)
                
                if trade_data.get('ticker'):
                    all_trades.append(trade_data)
            
            print(f"[✓] Loaded {len(df)} trades from {file_path}")
            
        except Exception as e:
            print(f"[-] Error loading {file_path}: {e}")
            continue
    
    return all_trades

def migrate_to_database(data_folder='./data'):
    """Main migration function"""
    print("\n" + "="*60)
    print("DATA MIGRATION TO POSTGRESQL")
    print("="*60)
    
    # Initialize database
    db = Database()
    
    if not db.connect():
        print("[-] Failed to connect to database. Exiting.")
        return
    
    # Create tables if they don't exist
    print("\n[+] Creating database tables...")
    db.create_tables()
    
    # Load all trade data from CSV and JSON files
    csv_trades = load_csv_files(data_folder)
    json_trades = load_json_files(data_folder)
    
    all_trades = csv_trades + json_trades
    
    if not all_trades:
        print("\n[-] No trades found to migrate")
        db.close()
        return
    
    print(f"\n[+] Total trades to migrate: {len(all_trades)}")
    
    # Insert trades in batches of 1000
    batch_size = 1000
    total_inserted = 0
    
    for i in range(0, len(all_trades), batch_size):
        batch = all_trades[i:i+batch_size]
        if db.bulk_insert_trades(batch):
            total_inserted += len(batch)
            print(f"    Progress: {total_inserted}/{len(all_trades)}")
    
    print(f"\n[✓] Migration complete! Inserted {total_inserted} trades")
    
    # Show database stats
    stats = db.get_database_stats()
    print(f"\n[*] Database Statistics:")
    print(f"    Total Trades: {stats['total_trades']}")
    print(f"    Unique Traders: {stats['unique_traders']}")
    print(f"    Unique Tickers: {stats['unique_tickers']}")
    print(f"\n    Trades by Source:")
    for source, count in stats['by_source'].items():
        print(f"      {source}: {count}")
    
    db.close()

if __name__ == "__main__":
    # Run migration
    migrate_to_database('./data')