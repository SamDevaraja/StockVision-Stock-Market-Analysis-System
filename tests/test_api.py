"""
Test script for StockVision API
Run this script to verify that upload, train, and predict functionality works correctly.
Make sure the Flask API is running before executing this script.
"""

import requests
import time
import os
import sys

API_BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Please ensure the backend is running.")
        print("   Start the API with: cd backend && python app.py")
        return False

def test_upload():
    """Test file upload"""
    print("\nTesting file upload...")
    sample_file = "data/sample_data.csv"
    
    if not os.path.exists(sample_file):
        print(f"❌ Sample file not found: {sample_file}")
        return None
    
    try:
        with open(sample_file, 'rb') as f:
            files = {'file': ('sample_data.csv', f, 'text/csv')}
            data = {'ticker': 'TEST'}
            response = requests.post(f"{API_BASE_URL}/upload", files=files, data=data)
        
        if response.status_code == 201:
            result = response.json()
            print(f"✅ Upload successful: Upload ID {result['upload_id']}")
            return result['upload_id']
        else:
            print(f"❌ Upload failed: {response.json().get('error', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        return None

def test_data_profile(upload_id):
    """Test data profile endpoint"""
    print(f"\nTesting data profile for upload {upload_id}...")
    try:
        response = requests.get(f"{API_BASE_URL}/data-profile?upload_id={upload_id}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Data profile generated: {result['row_count']} rows")
            return True
        else:
            print(f"❌ Data profile failed: {response.json().get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Data profile error: {str(e)}")
        return False

def test_train(upload_id):
    """Test model training"""
    print(f"\nTesting model training for upload {upload_id}...")
    try:
        train_data = {
            'upload_id': upload_id,
            'model_type': 'random_forest',
            'model_name': 'test_model_rf',
            'test_size': 0.2,
            'lookahead': 1,
            'model_params': {
                'n_estimators': 50,
                'max_depth': 5
            }
        }
        
        print("Training model (this may take a minute)...")
        response = requests.post(f"{API_BASE_URL}/train", json=train_data)
        
        if response.status_code == 201:
            result = response.json()
            print(f"✅ Model trained successfully: Model ID {result['model_id']}")
            print(f"   Training RMSE: {result['metrics'].get('train_rmse', 'N/A'):.4f}")
            if 'val_rmse' in result['metrics']:
                print(f"   Validation RMSE: {result['metrics']['val_rmse']:.4f}")
            return result['model_id']
        else:
            print(f"❌ Training failed: {response.json().get('error', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"❌ Training error: {str(e)}")
        return None

def test_predict(model_id):
    """Test prediction"""
    print(f"\nTesting prediction with model {model_id}...")
    try:
        predict_data = {
            'model_id': model_id,
            'ticker': 'TEST',
            'date': '2020-04-01',
            'use_yfinance': False
        }
        
        response = requests.post(f"{API_BASE_URL}/predict", json=predict_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful:")
            print(f"   Current Price: ${result['current_price']:.2f}")
            print(f"   Predicted Price: ${result['predicted_price']:.2f}")
            print(f"   Predicted Change: {result['predicted_change_pct']:.2f}%")
            return True
        else:
            print(f"❌ Prediction failed: {response.json().get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return False

def test_models():
    """Test listing models"""
    print("\nTesting models list...")
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Found {result['count']} models")
            return True
        else:
            print(f"❌ Models list failed: {response.json().get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Models list error: {str(e)}")
        return False

def test_backtest(model_id):
    """Test backtesting"""
    print(f"\nTesting backtest with model {model_id}...")
    try:
        backtest_data = {
            'model_id': model_id,
            'start_date': '2020-02-01',
            'end_date': '2020-03-31',
            'initial_capital': 10000.0,
            'stop_loss': 0.02,
            'take_profit': 0.05,
            'transaction_cost': 0.001
        }
        
        print("Running backtest (this may take a minute)...")
        response = requests.post(f"{API_BASE_URL}/backtest", json=backtest_data)
        
        if response.status_code == 200:
            result = response.json()
            results = result.get('results', {})
            print(f"✅ Backtest completed:")
            print(f"   Initial Capital: ${results['initial_capital']:,.2f}")
            print(f"   Final Capital: ${results['final_capital']:,.2f}")
            print(f"   Total Return: {results['total_return_pct']:.2f}%")
            print(f"   Total Trades: {results['total_trades']}")
            print(f"   Win Rate: {results['win_rate']:.2%}")
            return True
        else:
            print(f"❌ Backtest failed: {response.json().get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Backtest error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("StockVision API Test Suite")
    print("=" * 60)
    
    # Test health
    if not test_health():
        sys.exit(1)
    
    # Test upload
    upload_id = test_upload()
    if not upload_id:
        print("\n❌ Upload test failed. Cannot continue.")
        sys.exit(1)
    
    # Test data profile
    test_data_profile(upload_id)
    
    # Test train
    model_id = test_train(upload_id)
    if not model_id:
        print("\n❌ Training test failed. Cannot continue.")
        sys.exit(1)
    
    # Wait a bit for model to be saved
    time.sleep(1)
    
    # Test models list
    test_models()
    
    # Test predict
    test_predict(model_id)
    
    # Test backtest
    test_backtest(model_id)
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
