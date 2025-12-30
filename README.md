# StockVision - Stock Prediction System

A comprehensive web-based stock prediction system built with Flask (backend) and Streamlit (frontend), featuring multiple machine learning models and backtesting capabilities.

## Features

- **Multiple ML Models**: Random Forest, XGBoost, LightGBM
- **Technical Indicators**: MA, RSI, MACD, Bollinger Bands, and more
- **Backtesting**: Customizable parameters with stop-loss, take-profit, transaction costs
- **Data Management**: CSV upload, data profiling, and visualization
- **Model Management**: Version control, model listing, and SHAP explanations
- **Real-time Data**: Integration with yfinance for live stock data

## Project Structure

```
StockVision/
├── backend/
│   ├── app.py              # Flask API application
│   ├── models.py           # ML model implementations
│   ├── database.py         # Database utilities
│   ├── feature_engineering.py  # Technical indicators and features
│   └── backtest.py         # Backtesting logic
├── frontend/
│   └── app.py              # Streamlit web application
├── data/                  # Generated or sample datasets (not tracked)
├── uploads/               # Runtime CSV uploads (auto-created)
├── models/                # Trained ML models (auto-created)
├── tests/
│   └── test_api.py         # API test script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Repository Notes (Important)

The following directories/files are not included in the GitHub repository:

- `venv/` – Virtual environment (system-specific)
- `data/` – Sample or generated datasets
- `uploads/` – User-uploaded CSV files
- `models/` – Trained ML model artifacts
- `*.db` – Local database files

These will be automatically created when running the project or setup script.

## Git Ignore

This project uses a `.gitignore` file to exclude virtual environments,
datasets, trained models, uploads, cache files, and local databases.

## Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd StockVision
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run setup script** (initializes database and creates directories)
   ```bash
   python setup.py
   ```
   
   Or manually initialize the database:
   ```bash
   python -c "from backend.database import init_db; init_db()"
   ```

## Usage

### Starting the Backend API

```bash
cd backend
python app.py
```

The API will be available at `http://localhost:5000`

### Starting the Frontend

```bash
streamlit run frontend/app.py
```

The web interface will open in your browser at `http://localhost:8501`

## API Endpoints

- `POST /upload`: Upload CSV files for training
- `POST /train`: Train ML models on uploaded data
- `POST /predict`: Make predictions for a ticker and date
- `GET /models`: List available trained models
- `POST /backtest`: Run backtesting simulations
- `GET /data-profile`: Get statistical profile of uploaded data

## Testing

Run the test script to verify functionality:

```bash
python tests/test_api.py
```

## Sample Data

A sample CSV file named **`sample_data.csv`** is provided for reference.  
The dataset must follow the format below:

**Required Columns:**
- Date
- Open
- High
- Low
- Close
- Volume

**Notes:**
- The CSV file must contain a header row.
- Date should be in a valid date format (YYYY-MM-DD recommended).


## Model Training

1. Upload a CSV file through the web interface or API
2. Select model type (Random Forest, XGBoost, or LightGBM)
3. Configure training parameters
4. Train the model
5. Use the trained model for predictions

## Backtesting

Configure backtesting parameters:
- Start and end dates
- Initial capital
- Stop-loss percentage
- Take-profit percentage
- Transaction costs

## Requirements

- Python 3.8+
- All dependencies listed in `requirements.txt`

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Commands to be executed (shortcut)

PS C:\Users\Sam Devaraja\Desktop\StockVision> powershell -ExecutionPolicy Bypass -File start_apps.ps1
Starting StockVision...
[INFO] Starting Backend API...
[INFO] Starting Streamlit Frontend...
Waiting for services to start...
[INFO] Opening browser...

========================================
StockVision is starting!
========================================
Backend API:  http://localhost:5000
Frontend UI:  http://localhost:8501

The browser should open automatically.

If not, manually open: http://localhost:8501
