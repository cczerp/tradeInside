# analyzer.py
import pandas as pd
import sqlite3
import os
import glob
from datetime import datetime, timedelta
from collections import defaultdict

DB_FILE = "tradeinsider.db"
DATA_DIR = "./data/csv"
REPORT_DIR = "./reports"
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


def has_public_event(ticker, date, events_lookup, window_days=7):
    if ticker not in events_lookup:
        return (False, None)
    
    event_dates = events_lookup[ticker]
    
    for event_date in event_dates:
        days_diff = abs((event_date - date).days)
        if days_diff <= window_days:
            return (True, f"Public event on {event_date.date()}")
    
    return (False, None)


def load_price_data():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM stock_prices", conn)
    conn.close()
    
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    
    return df

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
    
    # Load events
    events_lookup = load_corporate_events()
    
    # Storage
    patterns = defaultdict(lambda: {
        'patterns': [],
        'base_score': 0,
        'pattern_count': 0,
        'role_multiplier': 1.0
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
        
        for idx, row in df_sorted.iterrows():
            window_start = row['transaction_date']
            window_end = window_start + timedelta(days=3)
            
            sector_trades = insider_df[
                (insider_df['sector'] == row['sector']) &
                (insider_df['ticker'] == row['ticker']) &
                (insider_df['transaction_date'] >= window_start) &
                (insider_df['transaction_date'] <= window_end)
            ]
            
            unique_traders = sector_trades['trader_name'].nunique()
            
            if unique_traders >= 3:
                cluster_id = f"{row['ticker']}_{window_start.date()}"
                
                if cluster_id not in detected_clusters:
                    has_event, event_desc = has_public_event(row['ticker'], window_start, events_lookup, 7)
                    
                    if has_event:
                        clusters_filtered += 1
                        continue
                    
                    detected_clusters.add(cluster_id)
                    
                    for trader in sector_trades['trader_name'].unique():
                        patterns[trader]['patterns'].append(
                            f"Sector cluster: {row['ticker']} ({row['sector']}) - {unique_traders} traders on {window_start.date()}"
                        )
                        patterns[trader]['base_score'] += 5
                        patterns[trader]['pattern_count'] += 1
                        
                        # SAFE role lookup
                        if 'role' in sector_trades.columns:
                            trader_rows = sector_trades[sector_trades['trader_name'] == trader]
                            if len(trader_rows) > 0:
                                trader_role = trader_rows['role'].iloc[0]
                                role_mult = get_role_multiplier(trader_role)
                                patterns[trader]['role_multiplier'] = max(patterns[trader]['role_multiplier'], role_mult)
                    
                    ticker_patterns[row['ticker']]['patterns'].append(
                        f"Cluster: {unique_traders} traders ({row['sector']} sector, {window_start.date()})"
                    )
                    ticker_patterns[row['ticker']]['score'] += 5
    
    print(f"    Found {len(detected_clusters)} clusters ({clusters_filtered} filtered by public events)")
    
    # PATTERN 2: TIMING
    print("  [2/6] Sliding scale profitable timing...")
    timing_filtered = 0
    
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
            
            best_window = None
            best_gain = 0
            best_score = 0
            
            for start_day, end_day, min_gain, score in windows:
                check_date = trade_date + timedelta(days=end_day)
                
                future_price = None
                for offset in range(6):
                    future_price = ticker_prices.get(pd.Timestamp(check_date + timedelta(days=offset)))
                    if future_price:
                        break
                
                if future_price:
                    gain = ((future_price - trade_price) / trade_price) * 100
                    
                    if gain >= min_gain and gain > best_gain:
                        best_gain = gain
                        best_score = score
                        best_window = f"{start_day}-{end_day} days"
            
            if best_window:
                has_event, event_desc = has_public_event(ticker, trade_date, events_lookup, 7)
                
                if has_event:
                    timing_filtered += 1
                    continue
                
                patterns[trader]['patterns'].append(
                    f"Perfect timing: {ticker} (+{best_gain:.1f}% in {best_window})"
                )
                patterns[trader]['base_score'] += best_score
                patterns[trader]['pattern_count'] += 1
                
                # SAFE role lookup
                if 'role' in timing_df.columns and pd.notna(row.get('role')):
                    trader_role = row.get('role')
                    role_mult = get_role_multiplier(trader_role)
                    patterns[trader]['role_multiplier'] = max(patterns[trader]['role_multiplier'], role_mult)
                
                ticker_patterns[ticker]['patterns'].append(
                    f"Timed trade by {trader} (+{best_gain:.1f}%)"
                )
                ticker_patterns[ticker]['score'] += 3
                timing_count += 1
        
        print(f"    Found {timing_count} timing patterns ({timing_filtered} filtered by events)")
    
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
                    for trader in traders:
                        patterns[trader]['patterns'].append(
                            f"Coordinated: {date} ({sector} sector, {len(traders)} traders)"
                        )
                        patterns[trader]['base_score'] += 3
                        patterns[trader]['pattern_count'] += 1
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
    
    # MULTIPLICATIVE SCORING
    print("\n  [SCORING] Applying multiplicative + role bonuses...")
    trader_scores = defaultdict(lambda: {'score': 0, 'patterns': []})
    
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
        
        final_score = int(base * pattern_mult * role_mult)
        
        trader_scores[trader] = {
            'score': final_score,
            'patterns': data['patterns'],
            'role_multiplier': role_mult
        }
    
    print(f"\n  [✓] Detection complete:")
    print(f"      • {len(trader_scores)} traders with patterns")
    print(f"      • {len(ticker_patterns)} tickers flagged")
    
    return dict(trader_scores), dict(ticker_patterns), {}

# ============================================================================
# REPORTING
# ============================================================================

def generate_report(trader_scores, ticker_scores, company_scores, insider_df):
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"insider_analysis_{timestamp}.txt")
    
    top_traders = sorted(trader_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    top_tickers = sorted(ticker_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*120 + "\n")
        f.write("REFINED INSIDER TRADING PATTERN DETECTION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Records Analyzed: {len(insider_df)}\n")
        f.write(f"Total Patterns Detected: {sum(len(t[1]['patterns']) for t in top_traders)}\n")
        f.write("="*120 + "\n\n")
        
        f.write("DETECTION RULES:\n")
        f.write("  • Transaction Filtering: Excluded awards, grants, option exercises, 10b5-1 plans\n")
        f.write("  • Sector Clustering: 3+ traders in same sector/ticker within 3 days\n")
        f.write("  • Perfect Timing: 7-14d(8pts), 15-30d(5pts), 31-60d(4pts), 61-90d(3pts)\n")
        f.write("  • Late Filing: 7+ days late (only scores if combined with other patterns)\n")
        f.write("  • Role Weighting: CEO/CFO(2.0x), Director(1.5x), Officer(1.25x)\n")
        f.write("  • Pattern Multiplier: 1 pattern(1.0x), 2 patterns(1.5x), 3+ patterns(2.5x)\n")
        f.write("  • Event Filtering: Excludes patterns within ±7 days of earnings/8-K filings\n\n")
        
        f.write("="*120 + "\n")
        f.write("TOP 20 MOST SUSPICIOUS TRADERS\n")
        f.write("="*120 + "\n\n")
        
        for rank, (trader, data) in enumerate(top_traders[:20], 1):
            role_mult = data.get('role_multiplier', 1.0)
            f.write(f"#{rank} - {trader}\n")
            f.write(f"Risk Score: {data['score']} (Role Multiplier: {role_mult}x)\n")
            f.write(f"Patterns Detected ({len(data['patterns'])}):\n")
            for pattern in data['patterns']:
                f.write(f"  • {pattern}\n")
            f.write("\n" + "-"*120 + "\n\n")
        
        f.write("\n" + "="*120 + "\n")
        f.write("TOP 10 MOST SUSPICIOUS TICKERS\n")
        f.write("="*120 + "\n\n")
        
        for rank, (ticker, data) in enumerate(top_tickers[:10], 1):
            f.write(f"#{rank} - {ticker}\n")
            f.write(f"Risk Score: {data['score']}\n")
            f.write(f"Patterns Detected ({len(data['patterns'])}):\n")
            for pattern in data['patterns'][:20]:
                f.write(f"  • {pattern}\n")
            f.write("\n" + "-"*120 + "\n\n")
        
        f.write("\n" + "="*120 + "\n")
        f.write(f"COMPLETE TRADER LIST ({len(top_traders)} total)\n")
        f.write("="*120 + "\n\n")
        
        for trader, data in top_traders:
            f.write(f"{trader}: Score {data['score']} ({len(data['patterns'])} patterns)\n")
        
        f.write("\n" + "="*120 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*120 + "\n")
    
    print(f"\n[✓] Report saved: {report_path}")

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