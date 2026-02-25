# analyzer.py
import pandas as pd
import sqlite3
import os
import glob
import csv
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict
import sys
import io

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    

DB_FILE = "tradeinsider.db"
DATA_DIR = "./data/csv"
REPORT_DIR = "./reports"
REPEAT_OFFENDERS_FILE = "repeat_offenders.csv"
os.makedirs(REPORT_DIR, exist_ok=True)

# ============================================================================
# SECTOR MAPPING
# ============================================================================

SECTOR_MAP = {
    'GOOGL': 'Tech', 'NVDA': 'Tech', 'TSLA': 'Tech', 'TEAM': 'Tech', 'PATH': 'Tech',
    'MRVL': 'Tech', 'CRM': 'Tech', 'MPWR': 'Tech', 'GWRE': 'Software', 'NTRA': 'Software',
    'ASGI': 'Software', 'FFAI': 'Software', 'NEXT': 'Software', 'MNTR': 'Software',
    'SMRT': 'Software', 'GCTS': 'Tech', 'GPUS': 'Tech', 'ASIC': 'Tech',
    'MKZR': 'Biotech', 'IONS': 'Biotech', 'SAVA': 'Biotech', 'NVCR': 'Biotech',
    'AGIO': 'Biotech', 'CRNX': 'Biotech', 'NAUT': 'Biotech', 'UTHR': 'Pharma',
    'SION': 'Biotech', 'BLNE': 'Biotech', 'AQST': 'Pharma', 'MBX': 'Biotech',
    'NUVL': 'Biotech', 'CCEL': 'Biotech', 'ABEO': 'Biotech',
    'PSEC': 'Finance', 'GSBD': 'Finance', 'IBKR': 'Finance', 'AFCG': 'Finance',
    'MBIN': 'Finance', 'MIAX': 'Finance', 'ACP': 'Finance', 'AWP': 'Finance',
    'IAF': 'Finance', 'AOD': 'Finance', 'FAX': 'Finance', 'OBLG': 'Finance',
    'BCDA': 'Finance', 'MSIF': 'Finance', 'NZF': 'Finance', 'MUNEX': 'Finance',
    'RIG': 'Energy', 'EONR': 'Energy', 'MTDR': 'Energy', 'EP': 'Energy',
    'GTE': 'Energy', 'RNGE': 'Energy', 'STEP': 'Energy', 'NRGV': 'Energy',
    'SPG': 'RealEstate', 'UMH': 'RealEstate', 'TPL': 'RealEstate', 'OPEN': 'RealEstate',
    'YORK': 'RealEstate', 'CTO': 'RealEstate', 'PROP': 'RealEstate', 'VWFB': 'RealEstate',
    'CVNA': 'Retail', 'DLTH': 'Retail', 'DLTR': 'Retail', 'COCO': 'Retail',
    'HAIN': 'Consumer', 'KLG': 'Consumer', 'CAMP': 'Consumer', 'CHE': 'Consumer',
    'HTH': 'Healthcare', 'TRUP': 'Healthcare', 'DRTTF': 'Healthcare', 'NTST': 'Healthcare',
    'HALO': 'Healthcare', 'GTHP': 'Healthcare',
    'VST': 'Industrial', 'SI': 'Industrial', 'FUL': 'Industrial', 'AGX': 'Industrial',
    'MUX': 'Mining', 'UAMY': 'Mining', 'STRR': 'Mining',
    'RYM': 'Insurance', 'AVBC': 'Insurance',
    'NCMI': 'Media', 'EA': 'Media', 'EMIS': 'Media',
    'DX': 'Telecom',
    'RIOT': 'Crypto', 'MRAI': 'Crypto',
    'CWAN': 'Cannabis',
    'RSI': 'Gaming', 'NORD': 'Gaming',
    'OCTO': 'Aerospace',
    'AGRI': 'Agriculture',
    'PGEN': 'Shipping', 'NXXT': 'Shipping',
    'DMII': 'SpecialtyFinance', 'UWMC': 'SpecialtyFinance', 'PEPG': 'SpecialtyFinance',
    'ADMQ': 'Services', 'TEM': 'Services', 'RYAN': 'Services', 'NIQ': 'Services',
    'STEX': 'Construction', 'DOMH': 'Construction',
    'CPK': 'Chemical', 'MG': 'Chemical',
    'SMR': 'Semiconductor', 'JSPR': 'Semiconductor',
    'SKYT': 'Diagnostics', 'DTIL': 'Diagnostics',
    'QNBC': 'Marketing',
    'SFD': 'Utilities',
    'MPB': 'Marine',
    'RMTI': 'Research', 'GLIBK': 'Research',
    'AREB': 'Environmental',
    'ONEW': 'Holdings', 'UUU': 'Holdings', 'RCG': 'Holdings', 'CRSF': 'Holdings',
    'AENT': 'Holdings', 'AMBC': 'Holdings',
    'MDRR': 'Other', 'PHXE.': 'Other', 'GGZ.E': 'Other',
}

def get_sector(ticker):
    return SECTOR_MAP.get(ticker, 'Other')

# ============================================================================
# ROLE WEIGHTING
# ============================================================================

def get_role_multiplier(role):
    if pd.isna(role):
        return 1.0
    
    role_lower = str(role).lower()
    
    if any(x in role_lower for x in ['ceo', 'chief executive', 'president']):
        return 2.0
    if any(x in role_lower for x in ['cfo', 'chief financial']):
        return 2.0
    if any(x in role_lower for x in ['coo', 'chief operating']):
        return 1.8
    if any(x in role_lower for x in ['director', 'board']):
        return 1.5
    if any(x in role_lower for x in ['officer', 'vp', 'vice president', 'evp', 'svp']):
        return 1.25
    
    return 1.0

# ============================================================================
# TRANSACTION TYPE FILTERING
# ============================================================================

def is_suspicious_transaction(transaction_type):
    if pd.isna(transaction_type):
        return True
    
    trans_lower = str(transaction_type).lower()
    
    noise_patterns = [
        'award', 'grant', 'restricted stock award',
        'option exercise', 'automatic', '10b5-1',
        'conversion', 'gift', 'inheritance'
    ]
    
    if any(pattern in trans_lower for pattern in noise_patterns):
        return False
    
    return True

# ============================================================================
# REPEAT OFFENDER TRACKING
# ============================================================================

def load_repeat_offenders():
    """Load the repeat offenders database"""
    if not os.path.exists(REPEAT_OFFENDERS_FILE):
        return {}
    
    offenders = {}
    with open(REPEAT_OFFENDERS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trader = row['trader_name']
            if trader not in offenders:
                offenders[trader] = {
                    'appearances': int(row['appearances']),
                    'history': []
                }
            
            offenders[trader]['history'].append({
                'date': row['trade_date'],
                'ticker': row['ticker'],
                'patterns': row['patterns'],
                'risk_score': int(row['risk_score']),
                'trade_price': float(row['trade_price']) if row['trade_price'] else None,
                'price_movement': row['price_movement']
            })
    
    return offenders

def save_repeat_offenders(offenders_data):
    """Save repeat offenders to CSV"""
    with open(REPEAT_OFFENDERS_FILE, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['trader_name', 'appearances', 'trade_date', 'ticker', 'patterns', 
                      'risk_score', 'trade_price', 'price_movement']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for trader, data in offenders_data.items():
            for entry in data['history']:
                writer.writerow({
                    'trader_name': trader,
                    'appearances': data['appearances'],
                    'trade_date': entry['date'],
                    'ticker': entry['ticker'],
                    'patterns': entry['patterns'],
                    'risk_score': entry['risk_score'],
                    'trade_price': entry.get('trade_price', ''),
                    'price_movement': entry.get('price_movement', '')
                })

def get_repeat_offender_multiplier(appearances):
    """Calculate multiplier based on number of appearances in top 10"""
    if appearances == 1:
        return 1.0
    elif appearances == 2:
        return 1.5
    elif appearances == 3:
        return 2.0
    else:
        return 2.5

# ============================================================================
# PRICE MANAGEMENT
# ============================================================================

def check_price_freshness(conn):
    """Check if prices are older than 24 hours"""
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(date) FROM stock_prices")
    result = cursor.fetchone()
    
    if not result or not result[0]:
        return False
    
    latest_date = datetime.strptime(result[0], '%Y-%m-%d')
    age_hours = (datetime.now() - latest_date).total_seconds() / 3600
    
    return age_hours < 24

def fetch_current_prices(tickers):
    """Fetch current prices for given tickers"""
    print(f"\n[*] Fetching current prices for {len(tickers)} tickers...")
    
    current_prices = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1d')
            if not hist.empty:
                current_prices[ticker] = float(hist['Close'].iloc[-1])
                print(f"  ✓ {ticker}: ${current_prices[ticker]:.2f}")
        except Exception as e:
            print(f"  ✗ {ticker}: Failed")
            current_prices[ticker] = None
    
    return current_prices

def update_prices_if_stale(conn):
    """Update prices if they're older than 24 hours"""
    if check_price_freshness(conn):
        print("\n[✓] Price data is fresh (< 24 hours old)")
        return
    
    print("\n[!] Price data is stale (> 24 hours old)")
    print("[*] Fetching updated prices...")
    
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM trades WHERE ticker IS NOT NULL")
    rows = cursor.fetchall()
    
    tickers = []
    for row in rows:
        if row[0]:
            ticker = str(row[0]).strip().split()[0].upper()
            if ticker and len(ticker) <= 5:
                tickers.append(ticker)
    tickers = list(set(tickers))[:20]  # Limit to 20 for speed
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='5d')
            
            if not hist.empty:
                for date, row in hist.iterrows():
                    cursor.execute("""
                        INSERT OR REPLACE INTO stock_prices (ticker, date, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (ticker, date.strftime('%Y-%m-%d'), float(row['Open']), 
                          float(row['High']), float(row['Low']), float(row['Close']), int(row['Volume'])))
                
                conn.commit()
                print(f"  ✓ Updated {ticker}")
        except:
            print(f"  ✗ Failed {ticker}")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_scraped_data():
    print("\n[*] Loading all scraped data...")
    
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not all_files:
        print("[-] No CSV files found")
        return pd.DataFrame(), pd.DataFrame()
    
    insider_dfs = []
    institutional_dfs = []
    
    for f in all_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            
            if '13f' in f.lower() or 'institutional' in f.lower():
                institutional_dfs.append(df)
            else:
                insider_dfs.append(df)
        except Exception as e:
            print(f"   Error loading {f}: {e}")
    
    insider_df = pd.concat(insider_dfs, ignore_index=True) if insider_dfs else pd.DataFrame()
    institutional_df = pd.concat(institutional_dfs, ignore_index=True) if institutional_dfs else pd.DataFrame()
    
    print(f"[✓] Loaded {len(insider_df)} insider trades from {len(insider_dfs)} files")
    print(f"[✓] Loaded {len(institutional_df)} institutional records from {len(institutional_dfs)} files")
    
    return insider_df, institutional_df


def load_corporate_events():
    conn = sqlite3.connect(DB_FILE)
    
    try:
        events_df = pd.read_sql("SELECT * FROM corporate_events", conn)
        conn.close()
        
        if events_df.empty:
            print("  [!] No corporate events in database")
            return {}
        
        events_df['event_date'] = pd.to_datetime(events_df['event_date'])
        
        events_lookup = {}
        for ticker in events_df['ticker'].unique():
            ticker_events = events_df[events_df['ticker'] == ticker]['event_date'].tolist()
            events_lookup[ticker] = ticker_events
        
        print(f"  [✓] Loaded {len(events_df)} corporate events for {len(events_lookup)} tickers")
        return events_lookup
        
    except Exception as e:
        conn.close()
        print(f"  [!] Could not load corporate events: {e}")
        return {}


def check_event_relationship(ticker, trade_date, events_lookup):
    """
    Returns:
    - ('exclude', reason) if trade should be excluded (3 days POST event)
    - ('bonus', reason) if trade gets bonus points (7 days PRE event)
    - (None, None) if no special treatment
    """
    if ticker not in events_lookup:
        return (None, None)
    
    event_dates = events_lookup[ticker]
    
    for event_date in event_dates:
        days_diff = (event_date - trade_date).days
        
        # 3 days POST announcement - EXCLUDE
        if 0 <= days_diff <= 3:
            return ('exclude', f"Trade within 3 days after event on {event_date.date()}")
        
        # 7 days PRE announcement - BONUS
        if -7 <= days_diff < 0:
            return ('bonus', f"Trade {abs(days_diff)} days before event on {event_date.date()}")
    
    return (None, None)


def load_price_data():
    conn = sqlite3.connect(DB_FILE)
    
    # Update if stale
    update_prices_if_stale(conn)
    
    df = pd.read_sql("SELECT * FROM stock_prices", conn)
    conn.close()
    
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    
    return df

# ============================================================================
# PATTERN CONSOLIDATION
# ============================================================================

def consolidate_patterns(patterns_list):
    """Consolidate duplicate patterns into ranges and counts"""
    if not patterns_list:
        return []
    
    consolidated = []
    pattern_groups = defaultdict(list)
    
    for pattern in patterns_list:
        if 'Perfect timing:' in pattern:
            # Extract ticker and percentage
            parts = pattern.split('(')
            if len(parts) >= 2:
                ticker_part = parts[0].replace('Perfect timing:', '').strip()
                pct_part = parts[1].split('%')[0].replace('+', '')
                
                try:
                    pct = float(pct_part)
                    pattern_groups[('timing', ticker_part)].append(pct)
                except:
                    consolidated.append(pattern)
            else:
                consolidated.append(pattern)
        elif 'Coordinated:' in pattern:
            pattern_groups[('coordinated', pattern)].append(pattern)
        else:
            consolidated.append(pattern)
    
    # Consolidate timing patterns
    for (ptype, ticker), values in pattern_groups.items():
        if ptype == 'timing':
            if len(values) == 1:
                consolidated.append(f"Perfect timing: {ticker} (+{values[0]:.1f}%)")
            else:
                min_pct = min(values)
                max_pct = max(values)
                if min_pct == max_pct:
                    consolidated.append(f"Perfect timing: {ticker} (+{min_pct:.1f}%) x{len(values)}")
                else:
                    consolidated.append(f"Perfect timing: {ticker} (+{min_pct:.1f}-{max_pct:.1f}%) x{len(values)}")
        elif ptype == 'coordinated':
            # Only show once for coordinated
            consolidated.append(values[0])
    
    return consolidated

# ============================================================================
# REFINED PATTERN DETECTION
# ============================================================================

def detect_patterns(insider_df, institutional_df, prices_df):
    
    # Filter out non-suspicious transactions
    print("\n[*] Filtering transaction types...")
    original_count = len(insider_df)
    insider_df = insider_df[insider_df['transaction_type'].apply(is_suspicious_transaction)].copy()
    filtered_count = original_count - len(insider_df)
    print(f"  Filtered out {filtered_count} non-suspicious transactions (awards, grants, etc.)")
    print(f"  Analyzing {len(insider_df)} suspicious transactions")
    
    # Normalize dates
    if 'transaction_date' in insider_df.columns:
        insider_df['transaction_date'] = pd.to_datetime(insider_df['transaction_date'], errors='coerce')
    if 'filing_date' in insider_df.columns:
        insider_df['filing_date'] = pd.to_datetime(insider_df['filing_date'], errors='coerce')
    
    # Add sector
    if 'ticker' in insider_df.columns:
        insider_df['sector'] = insider_df['ticker'].apply(get_sector)
    
    # Load events and repeat offenders
    events_lookup = load_corporate_events()
    repeat_offenders = load_repeat_offenders()
    
    # Create trader lookup for role/company info
    trader_info = {}
    trade_prices = {}  # Store trade prices
    
    if 'trader_name' in insider_df.columns:
        for trader in insider_df['trader_name'].unique():
            if pd.isna(trader):
                continue
            trader_rows = insider_df[insider_df['trader_name'] == trader]
            if len(trader_rows) == 0:
                continue
            role = trader_rows['role'].iloc[0] if 'role' in trader_rows.columns and len(trader_rows) > 0 else 'Unknown'
            company = trader_rows['ticker'].iloc[0] if 'ticker' in trader_rows.columns and len(trader_rows) > 0 else 'Unknown'
            trader_info[trader] = {'role': role, 'company': company}
    
    # Storage
    patterns = defaultdict(lambda: {
        'patterns': [],
        'base_score': 0,
        'pattern_count': 0,
        'role_multiplier': 1.0,
        'role': 'Unknown',
        'company': 'Unknown',
        'trade_prices': {},
        'price_movements': {}
    })
    
    ticker_patterns = defaultdict(lambda: {'patterns': [], 'score': 0})
    late_filings = {}
    detected_clusters = set()
    
    print("\n[*] Refined pattern detection...")
    
    # PATTERN 1: SECTOR CLUSTERING
    print("  [1/6] Sector-based clustering...")
    clusters_filtered = 0
    
    if 'ticker' in insider_df.columns and 'trader_name' in insider_df.columns and 'transaction_date' in insider_df.columns:
        df_sorted = insider_df.dropna(subset=['ticker', 'trader_name', 'transaction_date']).sort_values('transaction_date')

        # Pre-group by ticker to avoid O(n²) full-DataFrame scans per row
        ticker_groups_p1 = {t: grp for t, grp in df_sorted.groupby('ticker')}

        for ticker, ticker_df in ticker_groups_p1.items():
            sector = ticker_df['sector'].iloc[0]

            for idx, row in ticker_df.iterrows():
                window_start = row['transaction_date']
                window_end = window_start + timedelta(days=3)

                sector_trades = ticker_df[
                    (ticker_df['transaction_date'] >= window_start) &
                    (ticker_df['transaction_date'] <= window_end)
                ]

                unique_traders = sector_trades['trader_name'].nunique()

                if unique_traders >= 3:
                    cluster_id = f"{ticker}_{window_start.date()}"

                    if cluster_id not in detected_clusters:
                        event_status, event_desc = check_event_relationship(ticker, window_start, events_lookup)

                        if event_status == 'exclude':
                            clusters_filtered += 1
                            continue

                        detected_clusters.add(cluster_id)

                        coordinated_list = []
                        for trader in sector_trades['trader_name'].unique():
                            if pd.isna(trader):
                                continue
                            info = trader_info.get(trader, {'role': 'Unknown', 'company': 'Unknown'})
                            coordinated_list.append(f"{trader} ({info['role']}, {info['company']})")

                        for trader in sector_trades['trader_name'].unique():
                            if pd.isna(trader):
                                continue
                            info = trader_info.get(trader, {'role': 'Unknown', 'company': 'Unknown'})

                            other_traders = [t for t in coordinated_list if str(trader) not in t]

                            patterns[trader]['patterns'].append(
                                f"Sector cluster: {ticker} ({sector}) - {unique_traders} traders on {window_start.date()} | Coordinated with: {', '.join(other_traders)}"
                            )
                            patterns[trader]['base_score'] += 5
                            patterns[trader]['pattern_count'] += 1
                            patterns[trader]['role'] = info['role']
                            patterns[trader]['company'] = info['company']

                            if 'role' in sector_trades.columns:
                                trader_rows = sector_trades[sector_trades['trader_name'] == trader]
                                if len(trader_rows) > 0:
                                    trader_role = trader_rows['role'].iloc[0]
                                    role_mult = get_role_multiplier(trader_role)
                                    patterns[trader]['role_multiplier'] = max(patterns[trader]['role_multiplier'], role_mult)

                        ticker_patterns[ticker]['patterns'].append(
                            f"Cluster: {unique_traders} traders ({sector} sector, {window_start.date()})"
                        )
                        ticker_patterns[ticker]['score'] += 5

    print(f"    Found {len(detected_clusters)} clusters ({clusters_filtered} filtered by public events)")
    
    # PATTERN 1.5: SAME-COMPANY CLUSTERING (Multiple insiders from SAME company)
    print("  [1.5/7] Same-company insider clustering...")
    company_clusters_detected = set()
    company_clusters_filtered = 0
    
    if 'ticker' in insider_df.columns and 'trader_name' in insider_df.columns and 'transaction_date' in insider_df.columns:
        df_sorted = insider_df.dropna(subset=['ticker', 'trader_name', 'transaction_date']).sort_values('transaction_date')

        # Pre-group by ticker to avoid O(n²) full-DataFrame scans per row
        ticker_groups_p15 = {t: grp for t, grp in df_sorted.groupby('ticker')}

        for ticker, ticker_df in ticker_groups_p15.items():
            for idx, row in ticker_df.iterrows():
                window_start = row['transaction_date']
                window_end = window_start + timedelta(days=7)

                # Only search within the pre-filtered ticker group
                company_trades = ticker_df[
                    (ticker_df['transaction_date'] >= window_start) &
                    (ticker_df['transaction_date'] <= window_end)
                ]

                unique_traders = company_trades['trader_name'].nunique()

                if unique_traders >= 3:
                    cluster_id = f"{ticker}_company_{window_start.date()}"

                    if cluster_id not in company_clusters_detected:
                        event_status, event_desc = check_event_relationship(ticker, window_start, events_lookup)

                        if event_status == 'exclude':
                            company_clusters_filtered += 1
                            continue

                        company_clusters_detected.add(cluster_id)

                        # Build list of insiders with roles
                        insider_list = []
                        role_diversity = set()
                        for trader in company_trades['trader_name'].unique():
                            if pd.isna(trader):
                                continue
                            info = trader_info.get(trader, {'role': 'Unknown', 'company': 'Unknown'})
                            insider_list.append(f"{trader} ({info['role']})")

                            # Track role diversity
                            role_lower = str(info['role']).lower()
                            if any(x in role_lower for x in ['ceo', 'chief executive']):
                                role_diversity.add('CEO')
                            elif any(x in role_lower for x in ['cfo', 'chief financial']):
                                role_diversity.add('CFO')
                            elif any(x in role_lower for x in ['director', 'board']):
                                role_diversity.add('Director')
                            elif any(x in role_lower for x in ['officer', 'vp', 'president']):
                                role_diversity.add('Officer')

                        # Calculate bonus for role diversity
                        diversity_bonus = 0
                        if len(role_diversity) >= 3:
                            diversity_bonus = 5  # CEO + CFO + Director/Officer = strong signal
                        elif len(role_diversity) == 2:
                            diversity_bonus = 3

                        base_score = 10 + diversity_bonus

                        for trader in company_trades['trader_name'].unique():
                            if pd.isna(trader):
                                continue
                            info = trader_info.get(trader, {'role': 'Unknown', 'company': 'Unknown'})

                            # Filter out current trader from list
                            other_insiders = [t for t in insider_list if str(trader) not in t]

                            diversity_note = f" (Role diversity: {', '.join(role_diversity)})" if len(role_diversity) >= 2 else ""

                            patterns[trader]['patterns'].append(
                                f"Same-company cluster: {ticker} - {unique_traders} insiders within 7 days{diversity_note} | Coordinated with: {', '.join(other_insiders)}"
                            )
                            patterns[trader]['base_score'] += base_score
                            patterns[trader]['pattern_count'] += 1
                            patterns[trader]['role'] = info['role']
                            patterns[trader]['company'] = info['company']

                            if 'role' in company_trades.columns:
                                trader_rows = company_trades[company_trades['trader_name'] == trader]
                                if len(trader_rows) > 0:
                                    trader_role = trader_rows['role'].iloc[0]
                                    role_mult = get_role_multiplier(trader_role)
                                    patterns[trader]['role_multiplier'] = max(patterns[trader]['role_multiplier'], role_mult)

                        ticker_patterns[ticker]['patterns'].append(
                            f"Same-company cluster: {unique_traders} insiders ({window_start.date()}){diversity_note}"
                        )
                        ticker_patterns[ticker]['score'] += base_score

    print(f"    Found {len(company_clusters_detected)} same-company clusters ({company_clusters_filtered} filtered by public events)")
    
    # PATTERN 1.75: LARGE TRADES (3x+ historical average)
    print("  [1.75/7] Large trade detection (3x+ historical avg)...")
    
    # Calculate historical averages for each trader
    trader_historical_avg = {}
    large_trades_detected = 0
    
    if 'trader_name' in insider_df.columns and 'shares' in insider_df.columns and 'transaction_date' in insider_df.columns:
        # Filter trades from last 12 months
        twelve_months_ago = datetime.now() - timedelta(days=365)
        recent_trades = insider_df[insider_df['transaction_date'] >= twelve_months_ago].copy()
        
        # Build price lookup if available
        trade_price_lookup = {}
        if not prices_df.empty:
            for ticker in prices_df['ticker'].unique():
                ticker_prices = prices_df[prices_df['ticker'] == ticker].copy()
                ticker_prices = ticker_prices.sort_values('date')
                trade_price_lookup[ticker] = ticker_prices.set_index('date')['close'].to_dict()
        
        # Calculate trade values using vectorized apply (avoids slow row-by-row .at[] mutation)
        def _calc_trade_value(row):
            shares = row.get('shares')
            if pd.isna(shares) or shares == 0:
                return 0.0
            shares = float(shares)
            price = None
            if 'price' in row.index and pd.notna(row.get('price')):
                try:
                    price = float(row['price'])
                    if price <= 0:
                        price = None
                except (ValueError, TypeError):
                    price = None
            if price is None:
                tkr = row.get('ticker')
                td = row.get('transaction_date')
                if tkr in trade_price_lookup and pd.notna(td):
                    price = trade_price_lookup[tkr].get(pd.Timestamp(td))
            return abs(shares * price) if price else 0.0

        recent_trades = recent_trades.assign(trade_value=recent_trades.apply(_calc_trade_value, axis=1))
        
        # Calculate historical average per trader
        for trader in recent_trades['trader_name'].unique():
            if pd.isna(trader):
                continue
            
            trader_trades = recent_trades[
                (recent_trades['trader_name'] == trader) & 
                (recent_trades['trade_value'] > 0)
            ]
            
            if len(trader_trades) > 0:
                avg_value = trader_trades['trade_value'].mean()
                trader_historical_avg[trader] = avg_value
        
        # Detect large trades (aggregate within 7 days for same ticker)
        for trader in insider_df['trader_name'].unique():
            if pd.isna(trader) or trader not in trader_historical_avg:
                continue
            
            trader_trades = insider_df[insider_df['trader_name'] == trader].copy()
            
            # Group by ticker and 7-day windows
            for ticker in trader_trades['ticker'].unique():
                if pd.isna(ticker):
                    continue
                
                ticker_trades = trader_trades[trader_trades['ticker'] == ticker].sort_values('transaction_date')
                
                # Sliding window for aggregation
                processed_dates = set()
                
                for idx, row in ticker_trades.iterrows():
                    trade_date = row['transaction_date']
                    
                    if pd.isna(trade_date) or trade_date in processed_dates:
                        continue
                    
                    # Find all trades within 7 days
                    window_start = trade_date
                    window_end = trade_date + timedelta(days=7)
                    
                    window_trades = ticker_trades[
                        (ticker_trades['transaction_date'] >= window_start) &
                        (ticker_trades['transaction_date'] <= window_end)
                    ]
                    
                    # Calculate total value
                    total_value = 0.0
                    total_shares = 0.0
                    
                    for _, wtrade in window_trades.iterrows():
                        if pd.isna(wtrade.get('shares')) or wtrade.get('shares') == 0:
                            continue
                        
                        shares = float(wtrade['shares'])
                        price = None
                        
                        if 'price' in wtrade and pd.notna(wtrade.get('price')) and wtrade.get('price') > 0:
                            price = float(wtrade['price'])
                        elif ticker in trade_price_lookup:
                            price = trade_price_lookup[ticker].get(pd.Timestamp(wtrade['transaction_date']))
                        
                        if price:
                            total_value += abs(shares * price)
                            total_shares += abs(shares)
                        
                        processed_dates.add(wtrade['transaction_date'])
                    
                    # Check if 3x+ historical average
                    if total_value > 0:
                        historical_avg = trader_historical_avg[trader]
                        multiplier = total_value / historical_avg if historical_avg > 0 else 0
                        
                        if multiplier >= 3.0:
                            info = trader_info.get(trader, {'role': 'Unknown', 'company': 'Unknown'})
                            
                            patterns[trader]['patterns'].append(
                                f"Large trade: {ticker} (${total_value:,.0f} = {multiplier:.1f}x avg, {int(total_shares):,} shares)"
                            )
                            patterns[trader]['base_score'] += 8
                            patterns[trader]['pattern_count'] += 1
                            patterns[trader]['role'] = info['role']
                            patterns[trader]['company'] = info['company']
                            
                            large_trades_detected += 1
    
    print(f"    Found {large_trades_detected} large trades (3x+ historical average)")
    
    # PATTERN 2: TIMING
    print("  [2/6] Sliding scale profitable timing...")
    timing_filtered = 0
    timing_bonused = 0
    
    if not prices_df.empty and 'ticker' in insider_df.columns and 'transaction_date' in insider_df.columns:
        price_lookup = {}
        for ticker in prices_df['ticker'].unique():
            ticker_prices = prices_df[prices_df['ticker'] == ticker].copy()
            ticker_prices = ticker_prices.sort_values('date')
            price_lookup[ticker] = ticker_prices.set_index('date')['close'].to_dict()
        
        timing_df = insider_df.dropna(subset=['ticker', 'transaction_date', 'trader_name']).copy()
        
        windows = [
            (7, 14, 15, 8),
            (15, 30, 15, 5),
            (31, 60, 18, 4),
            (61, 90, 20, 3),
        ]
        
        timing_count = 0
        for _, row in timing_df.iterrows():
            ticker = row['ticker']
            trade_date = row['transaction_date']
            trader = row['trader_name']

            if ticker not in price_lookup:
                continue

            ticker_prices = price_lookup[ticker]
            trade_price = ticker_prices.get(pd.Timestamp(trade_date))
            if not trade_price:
                continue

            # Check event relationship once per trade (applies equally to all windows)
            event_status, event_desc = check_event_relationship(ticker, trade_date, events_lookup)
            if event_status == 'exclude':
                timing_filtered += 1
                continue

            # Score ALL qualifying windows, not just the single best-gain window
            is_bonused = False
            for start_day, end_day, min_gain, score in windows:
                check_date = trade_date + timedelta(days=end_day)

                future_price = None
                for offset in range(6):
                    future_price = ticker_prices.get(pd.Timestamp(check_date + timedelta(days=offset)))
                    if future_price:
                        break

                if not future_price:
                    continue

                gain = ((future_price - trade_price) / trade_price) * 100

                if gain >= min_gain:
                    final_score = score
                    if event_status == 'bonus':
                        final_score += 15
                        if not is_bonused:
                            timing_bonused += 1
                            is_bonused = True

                    patterns[trader]['patterns'].append(
                        f"Perfect timing: {ticker} (+{gain:.1f}% in {start_day}-{end_day} days)"
                    )
                    patterns[trader]['base_score'] += final_score
                    patterns[trader]['pattern_count'] += 1

                    # Store trade price; keep the highest gain seen for this ticker
                    patterns[trader]['trade_prices'][ticker] = trade_price
                    existing = patterns[trader]['price_movements'].get(ticker, '')
                    try:
                        existing_pct = float(existing.replace('+', '').replace('%', '')) if existing else 0
                    except ValueError:
                        existing_pct = 0
                    if gain > existing_pct:
                        patterns[trader]['price_movements'][ticker] = f"+{gain:.1f}%"

                    info = trader_info.get(trader, {'role': 'Unknown', 'company': 'Unknown'})
                    patterns[trader]['role'] = info['role']
                    patterns[trader]['company'] = info['company']

                    if 'role' in timing_df.columns and pd.notna(row.get('role')):
                        trader_role = row.get('role')
                        role_mult = get_role_multiplier(trader_role)
                        patterns[trader]['role_multiplier'] = max(patterns[trader]['role_multiplier'], role_mult)

                    ticker_patterns[ticker]['patterns'].append(
                        f"Timed trade by {trader} (+{gain:.1f}% in {start_day}-{end_day}d)"
                    )
                    ticker_patterns[ticker]['score'] += 3
                    timing_count += 1

        print(f"    Found {timing_count} timing patterns ({timing_filtered} filtered, {timing_bonused} bonused for pre-event)")
    
    # PATTERN 3: LATE FILING
    print("  [3/6] Late filing detection...")
    if 'filing_date' in insider_df.columns and 'transaction_date' in insider_df.columns:
        delays = insider_df.dropna(subset=['transaction_date', 'filing_date', 'trader_name']).copy()
        delays['delay_days'] = (delays['filing_date'] - delays['transaction_date']).dt.days
        
        late_filings_df = delays[delays['delay_days'] >= 7]
        
        for _, row in late_filings_df.iterrows():
            trader = row['trader_name']
            ticker = row.get('ticker', 'Unknown')
            delay = row['delay_days']
            
            if trader not in late_filings:
                late_filings[trader] = []
            
            late_filings[trader].append({
                'ticker': ticker,
                'days': delay,
                'pattern': f"Late filing: {ticker} ({int(delay)} days)"
            })
    
    print(f"    Found {len(late_filings)} traders with late filings")
    
    # PATTERN 4: COORDINATED
    print("  [4/6] Coordinated sector activity...")
    if 'transaction_date' in insider_df.columns and 'trader_name' in insider_df.columns:
        df_dated = insider_df.dropna(subset=['transaction_date', 'trader_name', 'sector']).copy()
        df_dated['date_only'] = df_dated['transaction_date'].dt.date
        
        date_sector_activity = defaultdict(lambda: defaultdict(set))
        
        for _, row in df_dated.iterrows():
            date = row['date_only']
            sector = row['sector']
            trader = row['trader_name']
            date_sector_activity[date][sector].add(trader)
        
        coordinated_count = 0
        for date, sectors in date_sector_activity.items():
            for sector, traders in sectors.items():
                if len(traders) >= 5:
                    coordinated_list = []
                    for t in traders:
                        info = trader_info.get(t, {'role': 'Unknown', 'company': 'Unknown'})
                        coordinated_list.append(f"{t} ({info['role']}, {info['company']})")
                    
                    for trader in traders:
                        info = trader_info.get(trader, {'role': 'Unknown', 'company': 'Unknown'})
                        patterns[trader]['patterns'].append(
                            f"Coordinated: {date} ({sector} sector, {len(traders)} traders) | Coordinated with: {', '.join([t for t in coordinated_list if trader not in t])}"
                        )
                        patterns[trader]['base_score'] += 3
                        patterns[trader]['pattern_count'] += 1
                        patterns[trader]['role'] = info['role']
                        patterns[trader]['company'] = info['company']
                    coordinated_count += 1
        
        print(f"    Found {coordinated_count} coordinated events")
    
    # PATTERN 5: REPEAT OFFENDERS
    print("  [5/6] Repeat offender detection...")
    if 'trader_name' in insider_df.columns and 'ticker' in insider_df.columns:
        ticker_counts = insider_df.dropna(subset=['trader_name', 'ticker']).groupby(
            ['trader_name', 'ticker']
        ).size().reset_index(name='count')
        
        repeat_traders = ticker_counts[ticker_counts['count'] >= 5]
        
        for _, row in repeat_traders.iterrows():
            trader = row['trader_name']
            ticker = row['ticker']
            count = row['count']
            
            patterns[trader]['patterns'].append(f"Repeat: {ticker} ({count}x)")
            patterns[trader]['base_score'] += 2
            patterns[trader]['pattern_count'] += 1
            
            ticker_patterns[ticker]['patterns'].append(f"Repeatedly by {trader} ({count}x)")
            ticker_patterns[ticker]['score'] += 1
        
        print(f"    Found {len(repeat_traders)} repeat patterns")
    
    # PATTERN 6: INSTITUTIONAL (placeholder)
    print("  [6/6] Insider + Institution coordination...")
    if not institutional_df.empty:
        print(f"    Found {len(institutional_df)} institutional records (full analysis needs ticker mapping)")
    
    # CONDITIONAL LATE FILING
    print("\n  [SCORING] Applying conditional late filing...")
    late_filing_added = 0
    for trader, filings in late_filings.items():
        if patterns[trader]['pattern_count'] > 0:
            for filing in filings:
                patterns[trader]['patterns'].append(filing['pattern'])
                patterns[trader]['base_score'] += 2
                patterns[trader]['pattern_count'] += 1
                late_filing_added += 1
                
                if filing['ticker'] != 'Unknown':
                    ticker_patterns[filing['ticker']]['patterns'].append(f"Late filing by {trader}")
                    ticker_patterns[filing['ticker']]['score'] += 1
    
    print(f"    Added {late_filing_added} late filing scores")
    
    # MULTIPLICATIVE SCORING + REPEAT OFFENDER BONUS
    print("\n  [SCORING] Applying multiplicative + role + repeat offender bonuses...")
    trader_scores = defaultdict(lambda: {'score': 0, 'patterns': [], 'is_repeat': False})
    
    for trader, data in patterns.items():
        base = data['base_score']
        count = data['pattern_count']
        role_mult = data['role_multiplier']
        
        if count == 0:
            pattern_mult = 0
        elif count == 1:
            pattern_mult = 1.0
        elif count == 2:
            pattern_mult = 1.5
        else:
            pattern_mult = 2.5
        
        # Check repeat offender status
        repeat_mult = 1.0
        is_repeat = False
        if trader in repeat_offenders:
            appearances = repeat_offenders[trader]['appearances']
            repeat_mult = get_repeat_offender_multiplier(appearances + 1)  # +1 for current appearance
            is_repeat = True
        
        final_score = int(base * pattern_mult * role_mult * repeat_mult)

        # Consolidate patterns
        consolidated_patterns = consolidate_patterns(data['patterns'])

        trader_scores[trader] = {
            'score': final_score,
            'base_score': base,
            'pattern_mult': pattern_mult,
            'patterns': consolidated_patterns,
            'role_multiplier': role_mult,
            'role': data['role'],
            'company': data['company'],
            'is_repeat': is_repeat,
            'repeat_mult': repeat_mult if is_repeat else None,
            'trade_prices': data['trade_prices'],
            'price_movements': data['price_movements']
        }
    
    print(f"\n  [✓] Detection complete:")
    print(f"      • {len(trader_scores)} traders with patterns")
    print(f"      • {len(ticker_patterns)} tickers flagged")
    print(f"      • {sum(1 for t in trader_scores.values() if t['is_repeat'])} repeat offenders")
    
    return dict(trader_scores), dict(ticker_patterns), {}

# ============================================================================
# REPORTING
# ============================================================================

def generate_report(trader_scores, ticker_scores, company_scores, insider_df):
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_top10_path = os.path.join(REPORT_DIR, f"insider_TOP10_{timestamp}.txt")
    
    top_traders = sorted(trader_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Get current prices for top 10 traders' tickers
    top_10_tickers = set()
    for trader, data in top_traders[:10]:
        top_10_tickers.update(data['trade_prices'].keys())
    
    current_prices = fetch_current_prices(list(top_10_tickers))
    
    # Update repeat offenders database
    repeat_offenders = load_repeat_offenders()
    new_offenders = {}
    
    for trader, data in top_traders[:10]:
        if trader in repeat_offenders:
            # Update existing
            appearances = repeat_offenders[trader]['appearances'] + 1
        else:
            # New entry
            appearances = 1
        
        # Build pattern summary
        pattern_types = []
        if any('Perfect timing' in p for p in data['patterns']):
            pattern_types.append('Perfect Timing')
        if any('Coordinated' in p for p in data['patterns']):
            pattern_types.append('Coordinated')
        
        new_offenders[trader] = {
            'appearances': appearances,
            'history': repeat_offenders.get(trader, {}).get('history', []) + [{
                'date': datetime.now().strftime('%Y-%m-%d'),
                'ticker': data['company'],
                'patterns': ', '.join(pattern_types),
                'risk_score': data['score'],
                'trade_price': list(data['trade_prices'].values())[0] if data['trade_prices'] else None,
                'price_movement': list(data['price_movements'].values())[0] if data['price_movements'] else ''
            }]
        }
    
    save_repeat_offenders(new_offenders)
    
    # ============================================================================
    # TOP 10 REPORT
    # ============================================================================
    with open(report_top10_path, 'w', encoding='utf-8') as f:
        f.write("="*120 + "\n")
        f.write("TOP 10 MOST SUSPICIOUS TRADERS - INSIDER TRADING PATTERN DETECTION\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Records Analyzed: {len(insider_df)}\n")
        f.write("="*120 + "\n\n")
        
        f.write("DETECTION RULES (7 pattern types):\n")
        f.write("  [1] Transaction Filtering : Excluded awards, grants, option exercises, 10b5-1 plans\n")
        f.write("  [2] Sector Clustering     : 3+ traders in same sector/ticker within 3 days (+5pts)\n")
        f.write("  [3] Same-Company Cluster  : 3+ insiders from SAME company within 7 days (+10-15pts)\n")
        f.write("  [4] Large Trade           : Volume 3x+ trader's 12-month average (+8pts)\n")
        f.write("  [5] Perfect Timing        : ALL qualifying windows scored independently\n")
        f.write("                              7-14d(+8pts, ≥15%)  15-30d(+5pts, ≥15%)\n")
        f.write("                              31-60d(+4pts, ≥18%) 61-90d(+3pts, ≥20%)\n")
        f.write("  [6] Late Filing           : 7+ days late, only counted alongside other patterns (+2pts)\n")
        f.write("  [7] Coordinated Activity  : 5+ traders in same sector on the same day (+3pts)\n")
        f.write("\n")
        f.write("  Score Multipliers:\n")
        f.write("    Pre-Event Bonus    : +15pts for trades 1-7 days BEFORE earnings/8-K\n")
        f.write("    Role Weighting     : CEO/CFO(2.0x)  COO(1.8x)  Director(1.5x)  Officer(1.25x)\n")
        f.write("    Pattern Multiplier : 1 pattern(1.0x)  2 patterns(1.5x)  3+ patterns(2.5x)\n")
        f.write("    Repeat Offender    : 2nd appearance(1.5x)  3rd(2.0x)  4th+(2.5x)\n")
        f.write("    Event Filtering    : Excludes trades within 3 days AFTER an earnings/8-K filing\n")
        f.write("\n")
        f.write("  Formula: Score = base_pts × pattern_mult × role_mult × repeat_mult\n\n")
        
        f.write("="*120 + "\n")
        f.write("TOP 10 MOST SUSPICIOUS TRADERS\n")
        f.write("="*120 + "\n\n")
        
        for rank, (trader, data) in enumerate(top_traders[:10], 1):
            role_mult = data.get('role_multiplier', 1.0)
            role = data.get('role', 'Unknown')
            company = data.get('company', 'Unknown')
            
            f.write(f"#{rank} - {trader} ({role}, {company})\n")
            
            base = data.get('base_score', '?')
            p_mult = data.get('pattern_mult', '?')
            r_mult = data.get('repeat_mult') or 1.0
            score_formula = f"{base} (base) × {p_mult}x (patterns) × {role_mult}x (role)"
            if data.get('is_repeat'):
                score_formula += f" × {r_mult}x (repeat ⚠️)"
            f.write(f"Risk Score: {data['score']}  [{score_formula}]\n")
            
            # Price information
            if data['trade_prices']:
                f.write("Trade Prices & Current Values:\n")
                for ticker, trade_price in data['trade_prices'].items():
                    current = current_prices.get(ticker)
                    movement = data['price_movements'].get(ticker, 'N/A')
                    if current:
                        f.write(f"  • {ticker}: Traded at ${trade_price:.2f} → Now ${current:.2f} ({movement})\n")
                    else:
                        f.write(f"  • {ticker}: Traded at ${trade_price:.2f} → Current price unavailable\n")
            
            f.write(f"Patterns Detected ({len(data['patterns'])}):\n")
            for pattern in data['patterns']:
                if ' | ' in pattern:
                    main, detail = pattern.split(' | ', 1)
                    f.write(f"  • {main}\n    └─ {detail}\n")
                else:
                    f.write(f"  • {pattern}\n")
            f.write("\n" + "-"*120 + "\n\n")
        
        f.write("\n" + "="*120 + "\n")
        f.write("END OF TOP 10 REPORT\n")
        f.write("="*120 + "\n")
    
    print(f"\n[✓] Top 10 Report saved: {report_top10_path}")
    print(f"[✓] Repeat offenders database updated: {REPEAT_OFFENDERS_FILE}")

# ============================================================================
# MAIN
# ============================================================================

def run_full_analysis():
    print("\n" + "="*100)
    print("REFINED INSIDER TRADING PATTERN ANALYSIS")
    print("="*100)
    
    insider_df, institutional_df = load_all_scraped_data()
    if insider_df.empty:
        print("[-] No insider data to analyze")
        return
    
    prices_df = load_price_data()
    if prices_df.empty:
        print("[!] No price data found. Run: python fetch_data.py prices")
    
    trader_scores, ticker_scores, company_scores = detect_patterns(insider_df, institutional_df, prices_df)
    
    generate_report(trader_scores, ticker_scores, company_scores, insider_df)
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)

if __name__ == "__main__":
    run_full_analysis()