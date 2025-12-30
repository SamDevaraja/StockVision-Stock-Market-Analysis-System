import sys
import os
# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from werkzeug.utils import secure_filename
import traceback

from backend.database import (
    init_db, add_upload, get_all_uploads, get_upload,
    add_model, get_all_models, get_model,
    add_prediction, get_predictions,
    add_backtest, get_backtests
)
from backend.feature_engineering import prepare_features, clean_data, get_feature_names
from backend.models import StockPredictor
from backend.backtest import BacktestEngine

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'csv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'StockVision API is running'})

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload CSV file for training"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only CSV files are allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read and validate CSV
        try:
            df = pd.read_csv(filepath)
            required_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_cols):
                os.remove(filepath)
                return jsonify({
                    'error': f'CSV must contain columns: {", ".join(required_cols)}'
                }), 400
            
            # Clean data
            df = clean_data(df)
            row_count = len(df)
            
            # Extract ticker from filename if possible
            ticker = request.form.get('ticker', filename.split('.')[0].upper())
            
            # Save to database
            upload_id = add_upload(filename, filepath, ticker, row_count)
            
            return jsonify({
                'message': 'File uploaded successfully',
                'upload_id': upload_id,
                'filename': filename,
                'row_count': row_count,
                'ticker': ticker
            }), 201
        
        except Exception as e:
            os.remove(filepath)
            return jsonify({'error': f'Error processing CSV: {str(e)}'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Train ML model on uploaded data"""
    try:
        data = request.get_json()
        upload_id = data.get('upload_id')
        model_type = data.get('model_type', 'random_forest')
        model_name = data.get('model_name', f'{model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        test_size = data.get('test_size', 0.2)
        lookahead = data.get('lookahead', 1)
        
        if not upload_id:
            return jsonify({'error': 'upload_id is required'}), 400
        
        # Get upload info
        upload = get_upload(upload_id)
        if not upload:
            return jsonify({'error': 'Upload not found'}), 404
        
        # Load data
        df = pd.read_csv(upload['filepath'])
        df = clean_data(df)
        
        # Prepare features
        X, y, feature_names = prepare_features(df, lookahead=lookahead)
        
        if len(X) < 10:
            return jsonify({'error': 'Insufficient data for training'}), 400
        
        # Train/test split (time series split)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        predictor = StockPredictor(model_type=model_type)
        
        # Get model parameters from request
        model_params = data.get('model_params', {})
        
        metrics = predictor.train(
            X_train, y_train,
            X_test, y_test,
            feature_names=feature_names,
            **model_params
        )
        
        # Save model
        model_filename = f"{model_name}.joblib"
        model_filepath = os.path.join(MODELS_FOLDER, model_filename)
        predictor.save(model_filepath)
        
        # Save to database
        model_id = add_model(
            model_name=model_name,
            model_type=model_type,
            filepath=model_filepath,
            training_data_id=upload_id,
            accuracy=metrics.get('val_r2', metrics.get('train_r2')),
            rmse=metrics.get('val_rmse', metrics.get('train_rmse')),
            feature_count=len(feature_names)
        )
        
        return jsonify({
            'message': 'Model trained successfully',
            'model_id': model_id,
            'model_name': model_name,
            'metrics': metrics,
            'feature_count': len(feature_names)
        }), 201
    
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction for a ticker and date"""
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        ticker = data.get('ticker', '').strip()
        date = data.get('date')
        use_yfinance = data.get('use_yfinance', False)

        print(f"Received prediction request: model_id={model_id}, ticker='{ticker}', date={date}, use_yfinance={use_yfinance}")

        if not model_id:
            return jsonify({'error': 'model_id is required'}), 400

        if use_yfinance and not ticker:
            return jsonify({'error': 'ticker is required when use_yfinance is True'}), 400
        
        # Get model
        model_record = get_model(model_id)
        if not model_record:
            return jsonify({'error': 'Model not found'}), 404
        
        # Load model
        predictor = StockPredictor.load(model_record['filepath'])
        
        # Get data
        if use_yfinance and ticker:
            # Fetch from yfinance
            end_date = pd.to_datetime(date) if date else datetime.now()
            start_date = end_date - pd.Timedelta(days=365)
            print(f"Fetching data for {ticker} from {start_date.date()} to {end_date.date()}")
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            print(f"Raw data shape: {df.shape}")
            print(f"Raw data columns: {list(df.columns)}")
            if df.empty:
                print(f"No data returned from yfinance for {ticker}")
                return jsonify({'error': f'No data found for ticker {ticker}. Please check the ticker symbol.'}), 400
            df = df.reset_index()
            df.columns = [col.capitalize() if col != 'Date' else 'Date' for col in df.columns]
            print(f"After column rename: {list(df.columns)}")
            df = clean_data(df)
            print(f"After cleaning: {df.shape}")
            if df.empty:
                print(f"Data became empty after cleaning")
                return jsonify({'error': f'No valid data found for ticker {ticker} after cleaning. Please check the data.'}), 400
        else:
            # Use uploaded data
            upload_id = model_record.get('training_data_id')
            if not upload_id:
                return jsonify({'error': 'No training data associated with model'}), 400
            
            upload = get_upload(upload_id)
            df = pd.read_csv(upload['filepath'])
            df = clean_data(df)
        
        if len(df) < 50:
            return jsonify({'error': 'Insufficient historical data'}), 400
        
        # Prepare features
        X, y, feature_names = prepare_features(df, lookahead=1)

        # Filter to only use features that the model was trained on
        if predictor.feature_names:
            # Get indices of model's features in current feature_names
            indices = []
            for feat in predictor.feature_names:
                if feat in feature_names:
                    indices.append(feature_names.index(feat))
                else:
                    # If a training feature is missing, we can't make a prediction
                    return jsonify({'error': f'Required feature {feat} not available in current data'}), 400
            X = X[:, indices]

        # Use the most recent data point for prediction
        X_pred = X[-1:].reshape(1, -1)
        
        # Make prediction
        prediction, confidence = predictor.predict_single(X[-1])
        
        # Save prediction
        pred_id = add_prediction(
            model_id=model_id,
            ticker=ticker or upload.get('ticker', 'UNKNOWN'),
            prediction_date=date or datetime.now().isoformat(),
            predicted_value=prediction,
            confidence=confidence
        )
        
        current_price = df['Close'].iloc[-1]
        change_pct = ((prediction - current_price) / current_price) * 100
        
        return jsonify({
            'prediction_id': pred_id,
            'ticker': ticker or upload.get('ticker', 'UNKNOWN'),
            'date': date or datetime.now().isoformat(),
            'current_price': float(current_price),
            'predicted_price': prediction,
            'predicted_change_pct': change_pct,
            'confidence': confidence,
            'model_id': model_id,
            'model_name': model_record['model_name']
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List all trained models"""
    try:
        models = get_all_models()
        return jsonify({
            'models': models,
            'count': len(models)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models/<int:model_id>', methods=['GET'])
def get_model_details(model_id):
    """Get details of a specific model"""
    try:
        model = get_model(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        # Get predictions for this model
        predictions = get_predictions(model_id=model_id, limit=10)
        
        return jsonify({
            'model': model,
            'recent_predictions': predictions
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/backtest', methods=['POST'])
def run_backtest():
    """Run backtesting simulation"""
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        initial_capital = data.get('initial_capital', 10000.0)
        stop_loss = data.get('stop_loss', 0.02)
        take_profit = data.get('take_profit', 0.05)
        transaction_cost = data.get('transaction_cost', 0.001)
        
        if not model_id:
            return jsonify({'error': 'model_id is required'}), 400
        
        # Get model
        model_record = get_model(model_id)
        if not model_record:
            return jsonify({'error': 'Model not found'}), 404
        
        # Get training data
        upload_id = model_record.get('training_data_id')
        if not upload_id:
            return jsonify({'error': 'No training data associated with model'}), 400

        upload = get_upload(upload_id)
        df = pd.read_csv(upload['filepath'])
        df = clean_data(df)

        # Filter data by date range if provided
        if start_date or end_date:
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                mask = pd.Series(True, index=df.index)
                if start_date:
                    mask &= (df['Date'] >= pd.to_datetime(start_date))
                if end_date:
                    mask &= (df['Date'] <= pd.to_datetime(end_date))
                df = df[mask].reset_index(drop=True)

        if len(df) < 2:
            return jsonify({'error': 'Insufficient data for backtesting after date filtering'}), 400

        # Load model first to get feature names
        predictor = StockPredictor.load(model_record['filepath'])

        # Prepare features on the filtered data
        X, y, feature_names = prepare_features(df, lookahead=1)

        # Filter to only use features that the model was trained on
        if predictor.feature_names:
            # Get indices of model's features in current feature_names
            indices = []
            for feat in predictor.feature_names:
                if feat in feature_names:
                    indices.append(feature_names.index(feat))
                else:
                    # If a training feature is missing, we can't make a prediction
                    return jsonify({'error': f'Required feature {feat} not available in current data'}), 400
            X = X[:, indices]
            feature_names = predictor.feature_names  # Use model's feature names

        if len(X) < 2:
            return jsonify({'error': 'Insufficient feature data for backtesting'}), 400

        # Run backtest
        engine = BacktestEngine(predictor, df, X, feature_names)
        results = engine.run_backtest(
            initial_capital=initial_capital,
            stop_loss=stop_loss,
            take_profit=take_profit,
            transaction_cost=transaction_cost,
            start_date=start_date,
            end_date=end_date
        )
        
        # Save backtest results
        backtest_id = add_backtest(
            model_id=model_id,
            start_date=start_date or df['Date'].iloc[0].isoformat() if 'Date' in df.columns else '',
            end_date=end_date or df['Date'].iloc[-1].isoformat() if 'Date' in df.columns else '',
            initial_capital=initial_capital,
            final_capital=results['final_capital'],
            return_percentage=results['total_return'] * 100,
            total_trades=results['total_trades'],
            win_rate=results['win_rate']
        )
        
        # Prepare response (exclude large trade data for initial response)
        response_results = {
            'backtest_id': backtest_id,
            'initial_capital': results['initial_capital'],
            'final_capital': results['final_capital'],
            'total_return': results['total_return'],
            'total_return_pct': results['total_return_pct'],
            'total_trades': results['total_trades'],
            'winning_trades': results['winning_trades'],
            'losing_trades': results['losing_trades'],
            'win_rate': results['win_rate'],
            'avg_return': results['avg_return'],
            'max_return': results['max_return'],
            'min_return': results['min_return'],
            'trade_count': len(results['trades']),
            'equity_curve_points': len(results['equity_curve'])
        }
        
        return jsonify({
            'message': 'Backtest completed',
            'results': response_results
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/data-profile', methods=['GET'])
def data_profile():
    """Get statistical profile of uploaded data"""
    try:
        upload_id = request.args.get('upload_id', type=int)
        
        if not upload_id:
            return jsonify({'error': 'upload_id is required'}), 400
        
        upload = get_upload(upload_id)
        if not upload:
            return jsonify({'error': 'Upload not found'}), 404
        
        # Load data
        df = pd.read_csv(upload['filepath'])
        df = clean_data(df)
        
        # Calculate statistics
        profile = {
            'upload_id': upload_id,
            'filename': upload['filename'],
            'row_count': len(df),
            'date_range': {
                'start': df['Date'].min().isoformat() if 'Date' in df.columns else None,
                'end': df['Date'].max().isoformat() if 'Date' in df.columns else None
            },
            'columns': list(df.columns),
            'statistics': {}
        }
        
        # Calculate statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            profile['statistics'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median()),
                'q25': float(df[col].quantile(0.25)),
                'q75': float(df[col].quantile(0.75))
            }
        
        # Calculate correlations if Close exists
        if 'Close' in df.columns:
            correlations = {}
            for col in numeric_cols:
                if col != 'Close':
                    corr = df['Close'].corr(df[col])
                    if not pd.isna(corr):
                        correlations[col] = float(corr)
            profile['correlations_with_close'] = correlations
        
        return jsonify(profile), 200
    
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/uploads', methods=['GET'])
def list_uploads():
    """List all uploaded files"""
    try:
        uploads = get_all_uploads()
        return jsonify({
            'uploads': uploads,
            'count': len(uploads)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize database
    init_db()
    print("Database initialized")
    print("Routes:", [str(rule) for rule in app.url_map.iter_rules()])
    print("Starting Flask API server on http://localhost:5000")

    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
