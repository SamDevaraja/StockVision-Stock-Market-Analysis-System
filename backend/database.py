import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional

DB_PATH = 'stockvision.db'

def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Uploads table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_date TEXT NOT NULL,
            filepath TEXT NOT NULL,
            ticker TEXT,
            row_count INTEGER
        )
    ''')
    
    # Models table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            filepath TEXT NOT NULL,
            created_date TEXT NOT NULL,
            training_data_id INTEGER,
            accuracy REAL,
            rmse REAL,
            feature_count INTEGER,
            FOREIGN KEY (training_data_id) REFERENCES uploads(id)
        )
    ''')
    
    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            predicted_value REAL,
            confidence REAL,
            created_date TEXT NOT NULL,
            FOREIGN KEY (model_id) REFERENCES models(id)
        )
    ''')
    
    # Backtests table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS backtests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            initial_capital REAL NOT NULL,
            final_capital REAL,
            return_percentage REAL,
            total_trades INTEGER,
            win_rate REAL,
            created_date TEXT NOT NULL,
            FOREIGN KEY (model_id) REFERENCES models(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def add_upload(filename: str, filepath: str, ticker: Optional[str] = None, row_count: int = 0) -> int:
    """Add an upload record to the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO uploads (filename, upload_date, filepath, ticker, row_count)
        VALUES (?, ?, ?, ?, ?)
    ''', (filename, datetime.now().isoformat(), filepath, ticker, row_count))
    upload_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return upload_id

def get_upload(upload_id: int) -> Optional[Dict]:
    """Get upload details by ID"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM uploads WHERE id = ?', (upload_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_uploads() -> List[Dict]:
    """Get all uploads"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM uploads ORDER BY upload_date DESC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def add_model(model_name: str, model_type: str, filepath: str, 
              training_data_id: Optional[int] = None, accuracy: Optional[float] = None,
              rmse: Optional[float] = None, feature_count: int = 0) -> int:
    """Add a model record to the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO models (model_name, model_type, filepath, created_date, 
                          training_data_id, accuracy, rmse, feature_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (model_name, model_type, filepath, datetime.now().isoformat(),
          training_data_id, accuracy, rmse, feature_count))
    model_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return model_id

def get_model(model_id: int) -> Optional[Dict]:
    """Get model details by ID"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM models WHERE id = ?', (model_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_models() -> List[Dict]:
    """Get all models"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM models ORDER BY created_date DESC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def add_prediction(model_id: int, ticker: str, prediction_date: str,
                   predicted_value: float, confidence: Optional[float] = None) -> int:
    """Add a prediction record"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (model_id, ticker, prediction_date, predicted_value,
                               confidence, created_date)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (model_id, ticker, prediction_date, predicted_value, confidence,
          datetime.now().isoformat()))
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return prediction_id

def get_predictions(model_id: Optional[int] = None, limit: int = 100) -> List[Dict]:
    """Get predictions, optionally filtered by model_id"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    if model_id:
        cursor.execute('''
            SELECT * FROM predictions WHERE model_id = ? 
            ORDER BY created_date DESC LIMIT ?
        ''', (model_id, limit))
    else:
        cursor.execute('SELECT * FROM predictions ORDER BY created_date DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def add_backtest(model_id: int, start_date: str, end_date: str, initial_capital: float,
                 final_capital: float, return_percentage: float, total_trades: int,
                 win_rate: float) -> int:
    """Add a backtest record"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO backtests (model_id, start_date, end_date, initial_capital,
                              final_capital, return_percentage, total_trades, win_rate, created_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (model_id, start_date, end_date, initial_capital, final_capital,
          return_percentage, total_trades, win_rate, datetime.now().isoformat()))
    backtest_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return backtest_id

def get_backtests(model_id: Optional[int] = None, limit: int = 50) -> List[Dict]:
    """Get backtests, optionally filtered by model_id"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    if model_id:
        cursor.execute('''
            SELECT * FROM backtests WHERE model_id = ? 
            ORDER BY created_date DESC LIMIT ?
        ''', (model_id, limit))
    else:
        cursor.execute('SELECT * FROM backtests ORDER BY created_date DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

if __name__ == '__main__':
    init_db()
    print("Database initialized successfully!")
