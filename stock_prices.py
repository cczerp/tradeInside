# stock_prices.py
import yfinance as yf
from datetime import datetime, timedelta

def fetch_stock_prices(ticker, days_back=730):
    """Fetch historical stock prices for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            return []
        
        prices = []
        for date, row in hist.iterrows():
            prices.append({
                'ticker': ticker,
                'date': date.date(),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': int(row['Volume'])
            })
        
        return prices
    except Exception as e:
        return []