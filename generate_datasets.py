import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_base_dates(start_date='2023-01-01', days=90):
    """Generate date range for datasets"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    dates = [start + timedelta(days=i) for i in range(days)]
    return dates

def generate_volatile_stock(dates):
    """Generate high volatility stock data with large random swings"""
    np.random.seed(42)  # For reproducibility

    # Start with base price around 150
    base_price = 150.0
    prices = [base_price]

    # Generate daily returns with high volatility (±5-10%)
    for i in range(1, len(dates)):
        # Random return between -10% and +10%
        daily_return = np.random.normal(0, 0.08)  # Mean 0, std 8%
        daily_return = np.clip(daily_return, -0.10, 0.10)  # Cap at ±10%
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)

    # Generate OHLC data
    data = []
    for i, price in enumerate(prices):
        # Add some intraday volatility
        volatility = abs(np.random.normal(0, 0.02))  # 2% intraday volatility
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = prices[i-1] if i > 0 else price
        close = price

        # Ensure OHLC logic: O ≤ H, O ≥ L, C ≤ H, C ≥ L
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Volume: 1M to 4M with some randomness
        volume = int(np.random.uniform(1000000, 4000000))

        data.append({
            'Date': dates[i].strftime('%Y-%m-%d'),
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume
        })

    return pd.DataFrame(data)

def generate_stable_stock(dates):
    """Generate low volatility stock data with small steady movements"""
    np.random.seed(123)  # Different seed for variety

    # Start with base price around 200
    base_price = 200.0
    prices = [base_price]

    # Generate daily returns with low volatility (±0.5-2%)
    for i in range(1, len(dates)):
        # Random return between -2% and +2%
        daily_return = np.random.normal(0, 0.01)  # Mean 0, std 1%
        daily_return = np.clip(daily_return, -0.02, 0.02)  # Cap at ±2%
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)

    # Generate OHLC data with minimal intraday movement
    data = []
    for i, price in enumerate(prices):
        # Low intraday volatility
        volatility = abs(np.random.normal(0, 0.005))  # 0.5% intraday volatility
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = prices[i-1] if i > 0 else price
        close = price

        # Ensure OHLC logic
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Volume: Lower and more consistent (800K to 2M)
        volume = int(np.random.uniform(800000, 2000000))

        data.append({
            'Date': dates[i].strftime('%Y-%m-%d'),
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume
        })

    return pd.DataFrame(data)

def generate_trending_stock(dates):
    """Generate trending stock data with upward bias and moderate volatility"""
    np.random.seed(456)  # Different seed for variety

    # Start with base price around 100
    base_price = 100.0
    prices = [base_price]

    # Generate daily returns with upward bias (+0.5-1% trend + noise)
    for i in range(1, len(dates)):
        # Upward trend + random noise
        trend = 0.007  # +0.7% daily trend
        noise = np.random.normal(0, 0.015)  # ±1.5% noise
        daily_return = trend + noise
        daily_return = np.clip(daily_return, -0.05, 0.05)  # Cap at ±5%
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)

    # Generate OHLC data
    data = []
    for i, price in enumerate(prices):
        # Moderate intraday volatility
        volatility = abs(np.random.normal(0, 0.015))  # 1.5% intraday volatility
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = prices[i-1] if i > 0 else price
        close = price

        # Ensure OHLC logic
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Volume: Moderate and increasing slightly (1.2M to 3M)
        volume = int(np.random.uniform(1200000, 3000000))

        data.append({
            'Date': dates[i].strftime('%Y-%m-%d'),
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume
        })

    return pd.DataFrame(data)

def main():
    """Generate all three datasets"""
    dates = generate_base_dates()

    print("Generating volatile stock data...")
    volatile_df = generate_volatile_stock(dates)
    volatile_df.to_csv('data/volatile_stock.csv', index=False)
    print(f"Created volatile_stock.csv with {len(volatile_df)} rows")

    print("Generating stable stock data...")
    stable_df = generate_stable_stock(dates)
    stable_df.to_csv('data/stable_stock.csv', index=False)
    print(f"Created stable_stock.csv with {len(stable_df)} rows")

    print("Generating trending stock data...")
    trending_df = generate_trending_stock(dates)
    trending_df.to_csv('data/trending_stock.csv', index=False)
    print(f"Created trending_stock.csv with {len(trending_df)} rows")

    print("\nDataset generation complete!")
    print("Files created in data/ directory:")
    print("- volatile_stock.csv: High volatility stock")
    print("- stable_stock.csv: Low volatility stock")
    print("- trending_stock.csv: Upward trending stock")

if __name__ == '__main__':
    main()
