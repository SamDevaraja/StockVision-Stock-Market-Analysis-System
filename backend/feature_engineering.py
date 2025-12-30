import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from typing import Optional

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for stock data
    Expected columns: Open, High, Low, Close, Volume
    """
    df = df.copy()
    
    # Ensure Date is datetime if present
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    
    # Moving Averages
    df['SMA_5'] = SMAIndicator(close=df['Close'], window=5).sma_indicator()
    df['SMA_10'] = SMAIndicator(close=df['Close'], window=10).sma_indicator()
    df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['EMA_12'] = EMAIndicator(close=df['Close'], window=12).ema_indicator()
    df['EMA_26'] = EMAIndicator(close=df['Close'], window=26).ema_indicator()
    
    # MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    
    # RSI
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    # Stochastic Oscillator
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_middle'] = bb.bollinger_mavg()
    df['BB_lower'] = bb.bollinger_lband()
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    df['BB_percent'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Average True Range (ATR)
    df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
    
    # Volume indicators
    if 'Volume' in df.columns:
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        df['VWAP'] = VolumeWeightedAveragePrice(
            high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']
        ).volume_weighted_average_price()
    
    # Price-based features
    df['Price_change'] = df['Close'].pct_change()
    df['Price_change_abs'] = df['Close'].diff()
    df['High_Low_ratio'] = df['High'] / df['Low']
    df['Close_Open_ratio'] = df['Close'] / df['Open']
    
    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag) if 'Volume' in df.columns else None
    
    # Rolling statistics
    for window in [5, 10, 20]:
        df[f'Close_std_{window}'] = df['Close'].rolling(window=window).std()
        df[f'Close_min_{window}'] = df['Close'].rolling(window=window).min()
        df[f'Close_max_{window}'] = df['Close'].rolling(window=window).max()
        df[f'Close_mean_{window}'] = df['Close'].rolling(window=window).mean()
    
    # Percentage changes
    for period in [1, 2, 5, 10]:
        df[f'Return_{period}d'] = df['Close'].pct_change(periods=period)
    
    return df

def prepare_features(df: pd.DataFrame, target_col: str = 'Close', 
                     lookahead: int = 1) -> tuple:
    """
    Prepare features and target for ML model
    Returns: (X, y) where X is features and y is target
    """
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Create target variable (future price)
    df['target'] = df[target_col].shift(-lookahead)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in [
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'target',
        'Volume_lag_1', 'Volume_lag_2', 'Volume_lag_3', 'Volume_lag_5', 'Volume_lag_10'
    ]]
    
    # Remove any remaining NaN columns
    feature_cols = [col for col in feature_cols if df[col].notna().any()]
    
    X = df[feature_cols].values
    y = df['target'].values
    
    return X, y, feature_cols

def get_feature_names(df: pd.DataFrame) -> list:
    """Get list of feature names after engineering"""
    df_temp = calculate_technical_indicators(df.head(100))
    feature_cols = [col for col in df_temp.columns if col not in [
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Volume_lag_1', 'Volume_lag_2', 'Volume_lag_3', 'Volume_lag_5', 'Volume_lag_10'
    ]]
    return [col for col in feature_cols if df_temp[col].notna().any()]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess stock data"""
    df = df.copy()
    
    # Convert Date column if present
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    
    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data")
    
    # Handle missing values
    df = df.ffill().bfill()
    
    # Remove any remaining NaN rows
    df = df.dropna(subset=required_cols)
    
    # Ensure numeric columns are numeric
    for col in required_cols + (['Volume'] if 'Volume' in df.columns else []):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove invalid data (negative prices, etc.)
    df = df[(df['Close'] > 0) & (df['High'] >= df['Low'])]
    
    return df
