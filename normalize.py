# normalize.py
"""
Maps all possible column name variations to standardized output columns
"""

COLUMN_MAP = {
    # TICKER/STOCK variations → "ticker"
    "ticker": "ticker",
    "symbol": "ticker",
    "stock": "ticker",
    
    # WHO (trader name) variations → "trader_name"
    "insider name": "trader_name",
    "insider_name": "trader_name",
    "insider\xa0name": "trader_name",
    "reporting owner": "trader_name",
    "officer": "trader_name",
    "name / title": "trader_name",
    "politician": "trader_name",
    "fund": "trader_name",
    "manager": "trader_name",
    "filer": "trader_name",
    
    # TRANSACTION TYPE variations → "transaction_type"
    "transaction": "transaction_type",
    "type": "transaction_type",
    "trade_type": "transaction_type",
    "trade\xa0type": "transaction_type",
    "purchase / sale": "transaction_type",
    
    # WHEN (date) variations → "transaction_date"
    "trade date": "transaction_date",
    "trade\xa0date": "transaction_date",
    "transaction_date": "transaction_date",
    "date": "transaction_date",
    "traded": "transaction_date",
    
    # AMOUNT variations
    "shares": "shares",
    "qty": "shares",
    "quantity": "shares",
    "value": "value",
    
    # PRICE variations
    "price": "price",
    "share price": "price",
    
    # COMPANY variations
    "company": "company",
    "company\xa0name": "company",
    "company name": "company",
    "company_name": "company",
    
    # ROLE variations
    "position": "role",
    "title": "role",
    "role": "role",
    
    # FILING DATE variations
    "filing date": "filing_date",
    "filing\xa0date": "filing_date",
    "filed": "filing_date",
    "filing_date": "filing_date",
    "disclosed": "filing_date",
    "disclosed (est)": "filing_date",
}

def normalize_columns(df):
    """
    Renames DataFrame columns to standard names using COLUMN_MAP
    Returns DataFrame with standardized column names
    """
    import pandas as pd
    
    # Make all column names lowercase for matching
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # Create rename mapping
    rename_map = {}
    for old_col in df.columns:
        if old_col in COLUMN_MAP:
            rename_map[old_col] = COLUMN_MAP[old_col]
    
    # Rename columns
    df = df.rename(columns=rename_map)
    
    return df

def get_standard_columns():
    """Returns the list of standardized output columns"""
    return [
        "trader_name",
        "ticker",
        "transaction_type",
        "transaction_date",
        "shares",
        "value",
        "price",
        "company",
        "role",
        "filing_date",
        "source",
        "section",
    ]