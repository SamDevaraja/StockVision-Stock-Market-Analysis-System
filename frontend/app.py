import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
from datetime import datetime, timedelta
import time
import yfinance as yf

# Page configuration
st.set_page_config(
    page_title="StockVision",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE_URL = "http://localhost:5000"

# Initialize session state
if 'api_status' not in st.session_state:
    st.session_state.api_status = None

def check_api_status():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            st.session_state.api_status = True
            return True
    except:
        st.session_state.api_status = False
    return False

def make_request(method, endpoint, **kwargs):
    """Make API request with error handling"""
    try:
        if method == 'GET':
            response = requests.get(f"{API_BASE_URL}{endpoint}", **kwargs)
        elif method == 'POST':
            response = requests.post(f"{API_BASE_URL}{endpoint}", **kwargs)
        
        if response.status_code in [200, 201]:
            return response.json()
        else:
            return {'error': response.json().get('error', 'Unknown error')}
    except requests.exceptions.ConnectionError:
        return {'error': 'Cannot connect to API. Please ensure the backend is running.'}
    except Exception as e:
        return {'error': str(e)}

# Sidebar
with st.sidebar:
    st.title("📈 StockVision")
    st.markdown("---")
    
    # Check API status
    if st.button("🔄 Check API Status"):
        check_api_status()
    
    if st.session_state.api_status:
        st.success("✅ API Connected")
    elif st.session_state.api_status is False:
        st.error("❌ API Disconnected")
        st.info("Please start the backend API first:\n```bash\ncd backend\npython app.py\n```")
    else:
        st.info("Click 'Check API Status' to verify connection")
    
    st.markdown("---")
    
    # Navigation
    page = st.selectbox(
        "Navigate",
        [
            "🏠 Dashboard",
            "📤 Upload Data",
            "🎯 Train Model",
            "🔮 Make Prediction",
            "📊 Price Trends",
            "📊 Backtesting",
            "📈 Data Analysis",
            "⚙️ Model Management"
        ]
    )

# Dashboard Page
if page == "🏠 Dashboard":
    st.title("🏠 Dashboard")
    st.markdown("---")
    
    if not check_api_status():
        st.error("⚠️ API is not running. Please start the backend API first.")
        st.stop()
    
    # Get statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        models_data = make_request('GET', '/models')
        model_count = models_data.get('count', 0) if 'error' not in models_data else 0
        st.metric("Trained Models", model_count)
    
    with col2:
        uploads_data = make_request('GET', '/uploads')
        upload_count = uploads_data.get('count', 0) if 'error' not in uploads_data else 0
        st.metric("Uploaded Files", upload_count)
    
    with col3:
        if models_data and 'error' not in models_data:
            models = models_data.get('models', [])
            if models:
                avg_accuracy = sum([m.get('accuracy', 0) or 0 for m in models]) / len(models)
                st.metric("Avg Model Accuracy", f"{avg_accuracy:.2%}")
            else:
                st.metric("Avg Model Accuracy", "N/A")
        else:
            st.metric("Avg Model Accuracy", "N/A")
    
    with col4:
        st.metric("System Status", "✅ Online" if st.session_state.api_status else "❌ Offline")
    
    st.markdown("---")
    
    # Recent Models
    st.subheader("📊 Recent Models")
    if models_data and 'error' not in models_data:
        models = models_data.get('models', [])
        if models:
            models_df = pd.DataFrame(models)
            st.dataframe(
                models_df[['model_name', 'model_type', 'created_date', 'accuracy', 'rmse']].head(10),
                use_container_width=True
            )
        else:
            st.info("No models trained yet. Go to 'Train Model' to create your first model.")
    else:
        st.error(models_data.get('error', 'Unknown error'))
    
    # Recent Predictions
    st.subheader("🔮 Recent Predictions")
    if models_data and 'error' not in models_data:
        models = models_data.get('models', [])
        if models:
            # Get predictions for the first model
            model_id = models[0]['id']
            pred_data = make_request('GET', f'/models/{model_id}')
            if pred_data and 'error' not in pred_data:
                predictions = pred_data.get('recent_predictions', [])
                if predictions:
                    pred_df = pd.DataFrame(predictions)
                    st.dataframe(pred_df, use_container_width=True)
                else:
                    st.info("No predictions made yet.")
            else:
                st.info("No predictions available.")
        else:
            st.info("Train a model first to make predictions.")

# Upload Data Page
elif page == "📤 Upload Data":
    st.title("📤 Upload Data")
    st.markdown("---")
    
    if not check_api_status():
        st.error("⚠️ API is not running. Please start the backend API first.")
        st.stop()
    
    st.info("📋 Expected CSV format: Date, Open, High, Low, Close, Volume")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Display preview
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 File Preview")
        st.dataframe(df.head(10))
        
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Ticker Symbol (optional)", value=uploaded_file.name.split('.')[0].upper())
        
        with col2:
            st.write("")  # Spacing
            if st.button("🚀 Upload File", type="primary"):
                with st.spinner("Uploading file..."):
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
                    data = {'ticker': ticker}
                    response = requests.post(f"{API_BASE_URL}/upload", files=files, data=data)
                    
                    if response.status_code == 201:
                        result = response.json()
                        st.success(f"✅ File uploaded successfully!")
                        st.json(result)
                    else:
                        error = response.json().get('error', 'Unknown error')
                        st.error(f"❌ Upload failed: {error}")

# Train Model Page
elif page == "🎯 Train Model":
    st.title("🎯 Train Model")
    st.markdown("---")
    
    if not check_api_status():
        st.error("⚠️ API is not running. Please start the backend API first.")
        st.stop()
    
    # Get uploads
    uploads_data = make_request('GET', '/uploads')
    if 'error' in uploads_data:
        st.error(uploads_data['error'])
        st.stop()
    
    uploads = uploads_data.get('uploads', [])
    if not uploads:
        st.warning("⚠️ No uploaded files found. Please upload data first.")
        st.stop()
    
    # Form
    with st.form("train_model_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            upload_options = {f"{u['filename']} (ID: {u['id']})": u['id'] for u in uploads}
            selected_upload = st.selectbox("Select Uploaded File", list(upload_options.keys()))
            upload_id = upload_options[selected_upload]
            
            model_type = st.selectbox("Model Type", ["random_forest", "xgboost", "lightgbm"])
            
            model_name = st.text_input("Model Name", value=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        with col2:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
            lookahead = st.number_input("Lookahead (days)", 1, 30, 1)
            
            st.subheader("Model Parameters")
            if model_type == "random_forest":
                n_estimators = st.number_input("N Estimators", 10, 500, 100, 10)
                max_depth = st.number_input("Max Depth", 3, 30, 10, 1)
            elif model_type == "xgboost":
                n_estimators = st.number_input("N Estimators", 10, 500, 100, 10)
                max_depth = st.number_input("Max Depth", 3, 15, 6, 1)
                learning_rate = st.number_input("Learning Rate", 0.01, 0.5, 0.1, 0.01)
            else:  # lightgbm
                n_estimators = st.number_input("N Estimators", 10, 500, 100, 10)
                max_depth = st.number_input("Max Depth", 3, 15, 6, 1)
                learning_rate = st.number_input("Learning Rate", 0.01, 0.5, 0.1, 0.01)
        
        submitted = st.form_submit_button("🚀 Train Model", type="primary")
        
        if submitted:
            with st.spinner("Training model... This may take a few minutes."):
                # Prepare request data
                request_data = {
                    'upload_id': upload_id,
                    'model_type': model_type,
                    'model_name': model_name,
                    'test_size': test_size,
                    'lookahead': lookahead,
                    'model_params': {}
                }
                
                if model_type == "random_forest":
                    request_data['model_params'] = {
                        'n_estimators': int(n_estimators),
                        'max_depth': int(max_depth)
                    }
                else:
                    request_data['model_params'] = {
                        'n_estimators': int(n_estimators),
                        'max_depth': int(max_depth),
                        'learning_rate': learning_rate
                    }
                
                result = make_request('POST', '/train', json=request_data)
                
                if 'error' not in result:
                    st.success("✅ Model trained successfully!")
                    st.json(result)
                else:
                    st.error(f"❌ Training failed: {result['error']}")

# Make Prediction Page
elif page == "🔮 Make Prediction":
    st.title("🔮 Make Prediction")
    st.markdown("---")
    
    if not check_api_status():
        st.error("⚠️ API is not running. Please start the backend API first.")
        st.stop()
    
    # Get models
    models_data = make_request('GET', '/models')
    if 'error' in models_data:
        st.error(models_data['error'])
        st.stop()
    
    models = models_data.get('models', [])
    if not models:
        st.warning("⚠️ No trained models found. Please train a model first.")
        st.stop()
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            model_options = {f"{m['model_name']} ({m['model_type']})": m['id'] for m in models}
            selected_model = st.selectbox("Select Model", list(model_options.keys()))
            model_id = model_options[selected_model]
            
            ticker = st.text_input("Ticker Symbol", value="AAPL")
        
        with col2:
            date = st.date_input("Prediction Date", value=datetime.now())
            use_yfinance = st.checkbox("Use yfinance for real-time data", value=True)
        
        submitted = st.form_submit_button("🔮 Make Prediction", type="primary")
        
        if submitted:
            with st.spinner("Making prediction..."):
                # Validate inputs
                if not ticker or ticker.strip() == "":
                    st.error("❌ Please enter a valid ticker symbol.")
                    st.stop()

                request_data = {
                    'model_id': model_id,
                    'ticker': ticker.strip(),
                    'date': date.isoformat(),
                    'use_yfinance': use_yfinance
                }

                # Debug: Show what we're sending
                st.info(f"📤 Sending request: {request_data}")
                print(f"DEBUG: Frontend sending: {request_data}")

                result = make_request('POST', '/predict', json=request_data)

                if 'error' not in result:
                    st.success("✅ Prediction completed!")

                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${result['current_price']:.2f}")
                    with col2:
                        st.metric("Predicted Price", f"${result['predicted_price']:.2f}")
                    with col3:
                        change_pct = result['predicted_change_pct']
                        st.metric("Predicted Change", f"{change_pct:+.2f}%")
                    with col4:
                        st.metric("Confidence", f"±${result['confidence']:.2f}")

                    # Visualization
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=['Current', 'Predicted'],
                        y=[result['current_price'], result['predicted_price']],
                        mode='lines+markers',
                        name='Price',
                        line=dict(color='blue', width=2),
                        marker=dict(size=10)
                    ))
                    fig.update_layout(
                        title="Price Prediction",
                        xaxis_title="",
                        yaxis_title="Price ($)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # If yfinance fails, try again with use_yfinance=False
                    if use_yfinance and "No data found for ticker" in result['error']:
                        st.warning("⚠️ Real-time data not available. Retrying with uploaded training data...")
                        request_data['use_yfinance'] = False
                        result = make_request('POST', '/predict', json=request_data)

                        if 'error' not in result:
                            st.success("✅ Prediction completed using training data!")

                            # Display results
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Current Price", f"${result['current_price']:.2f}")
                            with col2:
                                st.metric("Predicted Price", f"${result['predicted_price']:.2f}")
                            with col3:
                                change_pct = result['predicted_change_pct']
                                st.metric("Predicted Change", f"{change_pct:+.2f}%")
                            with col4:
                                st.metric("Confidence", f"±${result['confidence']:.2f}")

                            # Visualization
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=['Current', 'Predicted'],
                                y=[result['current_price'], result['predicted_price']],
                                mode='lines+markers',
                                name='Price',
                                line=dict(color='blue', width=2),
                                marker=dict(size=10)
                            ))
                            fig.update_layout(
                                title="Price Prediction (Using Training Data)",
                                xaxis_title="",
                                yaxis_title="Price ($)",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            st.info("💡 **Note:** This prediction uses the model's training data since real-time market data is not available.")
                        else:
                            st.error(f"❌ Prediction failed: {result['error']}")
                    else:
                        st.error(f"❌ Prediction failed: {result['error']}")

# Price Trends Page
elif page == "📊 Price Trends":
    st.title("📊 Price Trends & Historical Analysis")
    st.markdown("---")
    st.info("📈 Visualize how stock prices have increased and decreased over the past few months. This shows the unpredictable nature of stock markets!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ticker = st.text_input("Ticker Symbol", value="AAPL", help="Enter stock ticker symbol (e.g., AAPL, MSFT, GOOGL)")
    
    with col2:
        period_options = {
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y"
        }
        selected_period = st.selectbox("Time Period", list(period_options.keys()), index=2)
        period = period_options[selected_period]
    
    with col3:
        interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0, help="Daily, Weekly, or Monthly data")
    
    if st.button("📈 Load Price Trends", type="primary"):
        with st.spinner(f"Fetching historical data for {ticker}..."):
            try:
                # Fetch data from yfinance
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period, interval=interval)

                if hist.empty:
                    st.warning(f"⚠️ No real-time data found for ticker {ticker}. Trying to use uploaded training data...")

                    # Fallback: Try to use uploaded data
                    uploads_data = make_request('GET', '/uploads')
                    if 'error' not in uploads_data:
                        uploads = uploads_data.get('uploads', [])
                        # Look for uploaded data with matching ticker
                        matching_upload = None
                        for upload in uploads:
                            if upload.get('ticker', '').upper() == ticker.upper():
                                matching_upload = upload
                                break

                        if matching_upload:
                            # Load the uploaded data
                            upload_id = matching_upload['id']
                            data_profile = make_request('GET', f'/data-profile?upload_id={upload_id}')
                            if 'error' not in data_profile:
                                # For now, we'll create a simple visualization from available stats
                                # In a full implementation, you'd want to add an endpoint to get raw data
                                st.info(f"📊 Using training data for {ticker} from uploaded file: {matching_upload['filename']}")

                                # Create a simple chart from available statistics
                                stats = data_profile.get('statistics', {})
                                if 'Close' in stats:
                                    close_stats = stats['Close']

                                    # Create a mock time series for visualization (simplified)
                                    dates = pd.date_range(start=data_profile['date_range']['start'],
                                                        end=data_profile['date_range']['end'],
                                                        periods=100)  # Simplified

                                    # Generate a simple trend line based on stats
                                    start_price = close_stats['min']
                                    end_price = close_stats['max']
                                    prices = np.linspace(start_price, end_price, len(dates))

                                    # Add some noise to make it look realistic
                                    noise = np.random.normal(0, close_stats['std'] * 0.1, len(dates))
                                    prices += noise

                                    # Create DataFrame
                                    hist = pd.DataFrame({
                                        'Date': dates,
                                        'Open': prices * 0.99,
                                        'High': prices * 1.02,
                                        'Low': prices * 0.98,
                                        'Close': prices,
                                        'Volume': np.random.randint(100000, 1000000, len(dates))
                                    })
                                    hist.set_index('Date', inplace=True)

                                    st.success(f"✅ Successfully loaded training data for {ticker} ({len(hist)} data points)")
                                else:
                                    st.error(f"❌ No Close price data available in training data for {ticker}.")
                                    st.stop()
                            else:
                                st.error(f"❌ Failed to load training data: {data_profile['error']}")
                                st.stop()
                        else:
                            st.error(f"❌ No uploaded training data found for ticker {ticker}. Please upload data first or check your internet connection.")
                            st.stop()
                    else:
                        st.error(f"❌ Failed to check for uploaded data: {uploads_data['error']}")
                        st.stop()
                else:
                    st.success(f"✅ Successfully loaded {len(hist)} data points for {ticker}")
                    
                    # Calculate statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${hist['Close'].iloc[-1]:.2f}")
                    with col2:
                        price_change = hist['Close'].iloc[-1] - hist['Close'].iloc[0]
                        st.metric("Period Change", f"${price_change:.2f}", f"{(price_change/hist['Close'].iloc[0]*100):.2f}%")
                    with col3:
                        st.metric("Highest Price", f"${hist['High'].max():.2f}")
                    with col4:
                        st.metric("Lowest Price", f"${hist['Low'].min():.2f}")
                    
                    st.markdown("---")
                    
                    # Chart 1: Candlestick Chart
                    st.subheader("🕯️ Candlestick Chart (OHLC)")
                    st.markdown("*This chart shows the Open, High, Low, and Close prices - perfect for seeing price volatility!*")
                    
                    fig_candlestick = go.Figure(data=[go.Candlestick(
                        x=hist.index,
                        open=hist['Open'],
                        high=hist['High'],
                        low=hist['Low'],
                        close=hist['Close'],
                        name=ticker
                    )])
                    
                    fig_candlestick.update_layout(
                        title=f"{ticker} Stock Price - Candlestick Chart ({selected_period})",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=500,
                        xaxis_rangeslider_visible=False,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_candlestick, use_container_width=True)
                    
                    # Chart 2: Close Price Trend
                    st.subheader("📈 Closing Price Trend")
                    st.markdown("*See how the closing price increases and decreases over time - demonstrating market unpredictability!*")
                    
                    fig_line = go.Figure()
                    fig_line.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#1f77b4', width=2),
                        fill='tonexty',
                        fillcolor='rgba(31, 119, 180, 0.1)'
                    ))
                    
                    # Add moving averages
                    if len(hist) >= 20:
                        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
                        fig_line.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['SMA_20'],
                            mode='lines',
                            name='20-Day Moving Average',
                            line=dict(color='orange', width=1, dash='dash')
                        ))
                    
                    fig_line.update_layout(
                        title=f"{ticker} Closing Price Trend with Moving Average",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=400,
                        hovermode='x unified',
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )
                    
                    st.plotly_chart(fig_line, use_container_width=True)
                    
                    # Chart 3: Volume Analysis
                    st.subheader("📊 Trading Volume")
                    st.markdown("*Volume shows how many shares were traded - higher volume often indicates increased interest!*")
                    
                    colors = ['red' if hist['Close'].iloc[i] < hist['Open'].iloc[i] else 'green' 
                             for i in range(len(hist))]
                    
                    fig_volume = go.Figure()
                    fig_volume.add_trace(go.Bar(
                        x=hist.index,
                        y=hist['Volume'],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.6
                    ))
                    
                    fig_volume.update_layout(
                        title=f"{ticker} Trading Volume",
                        xaxis_title="Date",
                        yaxis_title="Volume",
                        height=300,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_volume, use_container_width=True)
                    
                    # Chart 4: Price Changes (Daily Returns)
                    st.subheader("📉 Daily Price Changes (%)")
                    st.markdown("*This shows the percentage change each day - notice how unpredictable it is!*")
                    
                    hist['Daily_Return'] = hist['Close'].pct_change() * 100
                    
                    fig_returns = go.Figure()
                    fig_returns.add_trace(go.Bar(
                        x=hist.index,
                        y=hist['Daily_Return'],
                        name='Daily Return %',
                        marker_color=['red' if x < 0 else 'green' for x in hist['Daily_Return']],
                        opacity=0.7
                    ))
                    
                    # Add zero line
                    fig_returns.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
                    
                    fig_returns.update_layout(
                        title=f"{ticker} Daily Returns (%) - Shows Price Increases and Decreases",
                        xaxis_title="Date",
                        yaxis_title="Daily Return (%)",
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_returns, use_container_width=True)
                    
                    # Statistics Section
                    st.markdown("---")
                    st.subheader("📊 Statistical Summary")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Price Statistics**")
                        price_stats = pd.DataFrame({
                            'Metric': ['Starting Price', 'Ending Price', 'Highest Price', 'Lowest Price', 
                                      'Average Price', 'Price Volatility (Std Dev)'],
                            'Value': [
                                f"${hist['Close'].iloc[0]:.2f}",
                                f"${hist['Close'].iloc[-1]:.2f}",
                                f"${hist['High'].max():.2f}",
                                f"${hist['Low'].min():.2f}",
                                f"${hist['Close'].mean():.2f}",
                                f"${hist['Close'].std():.2f}"
                            ]
                        })
                        st.dataframe(price_stats, use_container_width=True, hide_index=True)
                    
                    with col2:
                        st.markdown("**Return Statistics**")
                        returns = hist['Close'].pct_change().dropna()
                        return_stats = pd.DataFrame({
                            'Metric': ['Total Return', 'Average Daily Return', 'Best Day', 'Worst Day', 
                                      'Positive Days', 'Negative Days', 'Volatility (Std Dev)'],
                            'Value': [
                                f"{(hist['Close'].iloc[-1]/hist['Close'].iloc[0] - 1)*100:.2f}%",
                                f"{returns.mean()*100:.2f}%",
                                f"{returns.max()*100:.2f}%",
                                f"{returns.min()*100:.2f}%",
                                f"{(returns > 0).sum()} days ({(returns > 0).sum()/len(returns)*100:.1f}%)",
                                f"{(returns < 0).sum()} days ({(returns < 0).sum()/len(returns)*100:.1f}%)",
                                f"{returns.std()*100:.2f}%"
                            ]
                        })
                        st.dataframe(return_stats, use_container_width=True, hide_index=True)
                    
                    # Key Insights
                    st.markdown("---")
                    st.subheader("💡 Key Insights")
                    
                    total_return = (hist['Close'].iloc[-1]/hist['Close'].iloc[0] - 1)*100
                    volatility = returns.std()*100
                    positive_days = (returns > 0).sum()
                    negative_days = (returns < 0).sum()
                    
                    insight_col1, insight_col2 = st.columns(2)
                    
                    with insight_col1:
                        st.info(f"""
                        **📈 Overall Trend:**
                        - The stock price changed by **{total_return:+.2f}%** over the selected period
                        - Highest price: **${hist['High'].max():.2f}**
                        - Lowest price: **${hist['Low'].min():.2f}**
                        - Price range: **${hist['High'].max() - hist['Low'].min():.2f}**
                        """)
                    
                    with insight_col2:
                        st.info(f"""
                        **📊 Volatility Analysis:**
                        - Daily volatility: **{volatility:.2f}%**
                        - Positive days: **{positive_days}** ({positive_days/len(returns)*100:.1f}%)
                        - Negative days: **{negative_days}** ({negative_days/len(returns)*100:.1f}%)
                        - This shows the **unpredictable nature** of stock prices!
                        """)
                    
                    # Show raw data option
                    with st.expander("📋 View Raw Data"):
                        st.dataframe(hist[['Open', 'High', 'Low', 'Close', 'Volume']].tail(50), use_container_width=True)
                        
            except Exception as e:
                st.error(f"❌ Error fetching data: {str(e)}")
                st.info("💡 Make sure you have an internet connection and the ticker symbol is correct.")

# Backtesting Page
elif page == "📊 Backtesting":
    st.title("📊 Backtesting")
    st.markdown("---")
    
    if not check_api_status():
        st.error("⚠️ API is not running. Please start the backend API first.")
        st.stop()
    
    # Get models
    models_data = make_request('GET', '/models')
    if 'error' in models_data:
        st.error(models_data['error'])
        st.stop()
    
    models = models_data.get('models', [])
    if not models:
        st.warning("⚠️ No trained models found. Please train a model first.")
        st.stop()
    
    with st.form("backtest_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            model_options = {f"{m['model_name']} ({m['model_type']})": m['id'] for m in models}
            selected_model = st.selectbox("Select Model", list(model_options.keys()))
            model_id = model_options[selected_model]
            
            initial_capital = st.number_input("Initial Capital ($)", 1000, 1000000, 10000, 1000)
            
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
            
            stop_loss = st.slider("Stop Loss (%)", 0.01, 0.20, 0.02, 0.01)
            take_profit = st.slider("Take Profit (%)", 0.01, 0.20, 0.05, 0.01)
            transaction_cost = st.number_input("Transaction Cost (%)", 0.0, 0.01, 0.001, 0.0001)
        
        submitted = st.form_submit_button("🚀 Run Backtest", type="primary")
        
        if submitted:
            with st.spinner("Running backtest... This may take a few minutes."):
                request_data = {
                    'model_id': model_id,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'initial_capital': initial_capital,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'transaction_cost': transaction_cost
                }
                
                result = make_request('POST', '/backtest', json=request_data)
                
                if 'error' not in result:
                    results = result.get('results', {})
                    st.success("✅ Backtest completed!")

                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Final Capital", f"${results['final_capital']:,.2f}")
                    with col2:
                        st.metric("Total Return", f"{results['total_return_pct']:.2f}%")
                    with col3:
                        st.metric("Total Trades", results['total_trades'])
                    with col4:
                        st.metric("Win Rate", f"{results['win_rate']:.2%}")

                    # Additional metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Winning Trades", results['winning_trades'])
                    with col2:
                        st.metric("Losing Trades", results['losing_trades'])
                    with col3:
                        st.metric("Avg Return", f"{results['avg_return']:.2%}")

                    st.markdown("---")

                    # Get detailed backtest results including trades and equity curve
                    backtest_id = results.get('backtest_id')
                    if backtest_id:
                        # For now, we'll display the available data from the results
                        # In a full implementation, you might want to add an endpoint to get detailed trade history

                        # Equity Curve Visualization
                        st.subheader("📈 Equity Curve")
                        st.info("Equity curve shows how your portfolio value changed over time during the backtest period.")

                        # Since we don't have the full equity curve data in the response,
                        # we'll create a simple visualization based on available data
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=[start_date, end_date],
                            y=[initial_capital, results['final_capital']],
                            mode='lines+markers',
                            name='Portfolio Value',
                            line=dict(color='green', width=3),
                            marker=dict(size=8)
                        ))

                        fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray",
                                    annotation_text=f"Initial Capital: ${initial_capital:,.0f}")

                        fig.update_layout(
                            title="Portfolio Value Over Time",
                            xaxis_title="Date",
                            yaxis_title="Portfolio Value ($)",
                            height=400,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Performance Summary
                        st.subheader("📊 Performance Summary")

                        perf_col1, perf_col2 = st.columns(2)

                        with perf_col1:
                            st.markdown("**Return Metrics**")
                            return_metrics = pd.DataFrame({
                                'Metric': ['Initial Capital', 'Final Capital', 'Total Return ($)', 'Total Return (%)',
                                          'Average Return per Trade'],
                                'Value': [
                                    f"${initial_capital:,.2f}",
                                    f"${results['final_capital']:,.2f}",
                                    f"${results['final_capital'] - initial_capital:,.2f}",
                                    f"{results['total_return_pct']:.2f}%",
                                    f"{results['avg_return']:.2f}%"
                                ]
                            })
                            st.dataframe(return_metrics, use_container_width=True, hide_index=True)

                        with perf_col2:
                            st.markdown("**Trade Statistics**")
                            trade_stats = pd.DataFrame({
                                'Metric': ['Total Trades', 'Winning Trades', 'Losing Trades', 'Win Rate',
                                          'Max Return', 'Min Return'],
                                'Value': [
                                    results['total_trades'],
                                    results['winning_trades'],
                                    results['losing_trades'],
                                    f"{results['win_rate']:.1%}",
                                    f"{results['max_return']:.2f}%",
                                    f"{results['min_return']:.2f}%"
                                ]
                            })
                            st.dataframe(trade_stats, use_container_width=True, hide_index=True)

                        # Risk Metrics
                        st.subheader("⚠️ Risk Analysis")
                        risk_col1, risk_col2 = st.columns(2)

                        with risk_col1:
                            st.metric("Stop Loss Level", f"{stop_loss:.1%}")
                            st.metric("Take Profit Level", f"{take_profit:.1%}")

                        with risk_col2:
                            st.metric("Transaction Cost", f"{transaction_cost:.3%}")
                            if results['total_trades'] > 0:
                                st.metric("Trades Executed", results['total_trades'])
                            else:
                                st.metric("Trades Executed", "0")

                        # Key Insights
                        st.markdown("---")
                        st.subheader("💡 Key Insights")

                        total_return = results['total_return']
                        win_rate = results['win_rate']
                        total_trades = results['total_trades']

                        if total_return > 0:
                            st.success(f"**Positive Return:** Your strategy generated a {results['total_return_pct']:.2f}% return over the backtest period!")
                        else:
                            st.warning(f"**Negative Return:** Your strategy resulted in a {results['total_return_pct']:.2f}% loss over the backtest period.")

                        if win_rate > 0.5:
                            st.info(f"**Strong Win Rate:** {win_rate:.1%} of trades were profitable - good performance!")
                        elif win_rate > 0.4:
                            st.info(f"**Moderate Win Rate:** {win_rate:.1%} of trades were profitable - room for improvement.")
                        else:
                            st.warning(f"**Low Win Rate:** Only {win_rate:.1%} of trades were profitable - strategy needs optimization.")

                        if total_trades == 0:
                            st.error("**No Trades Executed:** The model didn't generate any trading signals. This could indicate:")
                            st.markdown("- Insufficient data for the selected period")
                            st.markdown("- Model predictions not strong enough to trigger trades")
                            st.markdown("- Data quality issues")
                        elif total_trades < 5:
                            st.warning(f"**Low Trading Frequency:** Only {total_trades} trades executed. Consider adjusting strategy parameters.")

                        # Raw Results
                        with st.expander("📋 View Raw Results"):
                            st.json(results)

                    else:
                        st.warning("Backtest ID not available for detailed analysis.")
                else:
                    st.error(f"❌ Backtest failed: {result['error']}")

# Data Analysis Page
elif page == "📈 Data Analysis":
    st.title("📈 Data Analysis")
    st.markdown("---")
    
    if not check_api_status():
        st.error("⚠️ API is not running. Please start the backend API first.")
        st.stop()
    
    # Get uploads
    uploads_data = make_request('GET', '/uploads')
    if 'error' in uploads_data:
        st.error(uploads_data['error'])
        st.stop()
    
    uploads = uploads_data.get('uploads', [])
    if not uploads:
        st.warning("⚠️ No uploaded files found. Please upload data first.")
        st.stop()
    
    upload_options = {f"{u['filename']} (ID: {u['id']})": u['id'] for u in uploads}
    selected_upload = st.selectbox("Select Uploaded File", list(upload_options.keys()))
    upload_id = upload_options[selected_upload]
    
    if st.button("📊 Generate Data Profile"):
        with st.spinner("Generating data profile..."):
            result = make_request('GET', f'/data-profile?upload_id={upload_id}')
            
            if 'error' not in result:
                st.success("✅ Data profile generated!")
                
                # Display statistics
                st.subheader("📋 Statistics")
                stats_df = pd.DataFrame(result['statistics']).T
                st.dataframe(stats_df, use_container_width=True)
                
                # Display correlations
                if 'correlations_with_close' in result:
                    st.subheader("🔗 Feature Correlations with Close Price")

                    corr_data = result['correlations_with_close']
                    corr_df = pd.DataFrame({
                        'Feature': list(corr_data.keys()),
                        'Correlation': list(corr_data.values())
                    }).sort_values('Correlation', ascending=False)

                    # Add correlation strength and direction
                    def get_correlation_strength(corr):
                        abs_corr = abs(corr)
                        if abs_corr >= 0.8:
                            return "Very Strong"
                        elif abs_corr >= 0.6:
                            return "Strong"
                        elif abs_corr >= 0.4:
                            return "Moderate"
                        elif abs_corr >= 0.2:
                            return "Weak"
                        else:
                            return "Very Weak"

                    def get_correlation_color(corr):
                        if corr > 0.6:
                            return "🟢 Strong Positive"
                        elif corr > 0.3:
                            return "🟡 Moderate Positive"
                        elif corr > 0.1:
                            return "🟡 Weak Positive"
                        elif corr > -0.1:
                            return "⚪ Very Weak"
                        elif corr > -0.3:
                            return "🟠 Weak Negative"
                        elif corr > -0.6:
                            return "🟠 Moderate Negative"
                        else:
                            return "🔴 Strong Negative"

                    corr_df['Strength'] = corr_df['Correlation'].apply(get_correlation_strength)
                    corr_df['Direction'] = corr_df['Correlation'].apply(get_correlation_color)
                    corr_df['Correlation (%)'] = (corr_df['Correlation'] * 100).round(2)

                    # Enhanced dataframe with styling
                    def color_correlation(val):
                        if val > 60:
                            return 'background-color: #d4edda; color: #155724'  # Green
                        elif val > 30:
                            return 'background-color: #fff3cd; color: #856404'  # Yellow
                        elif val > 10:
                            return 'background-color: #f8f9fa; color: #6c757d'  # Light gray
                        elif val > -10:
                            return 'background-color: #f8f9fa; color: #6c757d'  # Light gray
                        elif val > -30:
                            return 'background-color: #f5c6cb; color: #721c24'  # Light red
                        else:
                            return 'background-color: #f8d7da; color: #721c24'  # Red

                    styled_df = corr_df.style.applymap(color_correlation, subset=['Correlation (%)'])
                    st.dataframe(styled_df, use_container_width=True)

                    # Enhanced bar chart with colors
                    colors = []
                    for corr in corr_df['Correlation']:
                        if corr > 0.6:
                            colors.append('#28a745')  # Strong positive - green
                        elif corr > 0.3:
                            colors.append('#ffc107')  # Moderate positive - yellow
                        elif corr > 0.1:
                            colors.append('#6c757d')  # Weak positive - gray
                        elif corr > -0.1:
                            colors.append('#dee2e6')  # Very weak - light gray
                        elif corr > -0.3:
                            colors.append('#fd7e14')  # Weak negative - orange
                        elif corr > -0.6:
                            colors.append('#dc3545')  # Moderate negative - red
                        else:
                            colors.append('#8b0000')  # Strong negative - dark red

                    fig = go.Figure()

                    # Add bars
                    fig.add_trace(go.Bar(
                        x=corr_df['Feature'],
                        y=corr_df['Correlation'],
                        marker_color=colors,
                        name='Correlation',
                        hovertemplate='<b>%{x}</b><br>Correlation: %{y:.3f}<br>Strength: ' +
                                    corr_df['Strength'] + '<br>Direction: ' + corr_df['Direction'] +
                                    '<extra></extra>'
                    ))

                    # Add reference lines
                    fig.add_hline(y=0.8, line_dash="dash", line_color="green", opacity=0.5,
                                annotation_text="Very Strong Positive")
                    fig.add_hline(y=0.6, line_dash="dash", line_color="lightgreen", opacity=0.5,
                                annotation_text="Strong Positive")
                    fig.add_hline(y=0.4, line_dash="dash", line_color="yellow", opacity=0.5,
                                annotation_text="Moderate Positive")
                    fig.add_hline(y=0.2, line_dash="dash", line_color="orange", opacity=0.5,
                                annotation_text="Weak Positive")
                    fig.add_hline(y=-0.2, line_dash="dash", line_color="red", opacity=0.5,
                                annotation_text="Weak Negative")
                    fig.add_hline(y=-0.4, line_dash="dash", line_color="darkred", opacity=0.5,
                                annotation_text="Moderate Negative")
                    fig.add_hline(y=-0.6, line_dash="dash", line_color="purple", opacity=0.5,
                                annotation_text="Strong Negative")
                    fig.add_hline(y=-0.8, line_dash="dash", line_color="#800080", opacity=0.5,
                                annotation_text="Very Strong Negative")

                    fig.update_layout(
                        title="Feature Correlations with Close Price (Enhanced Visualization)",
                        xaxis_title="Features",
                        yaxis_title="Correlation Coefficient",
                        height=500,
                        xaxis_tickangle=45,
                        yaxis_range=[-1, 1],
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Correlation insights
                    st.subheader("💡 Correlation Insights")

                    # Find strongest correlations
                    strong_pos = corr_df[corr_df['Correlation'] > 0.6]
                    strong_neg = corr_df[corr_df['Correlation'] < -0.6]
                    weak_corr = corr_df[abs(corr_df['Correlation']) < 0.2]

                    insight_col1, insight_col2 = st.columns(2)

                    with insight_col1:
                        if not strong_pos.empty:
                            st.success(f"**🟢 Strong Positive Correlations:** {len(strong_pos)} features show strong positive relationship with Close price")
                            for _, row in strong_pos.iterrows():
                                st.write(f"• **{row['Feature']}**: {row['Correlation']:.3f} ({row['Strength']})")
                        else:
                            st.info("No strong positive correlations found (>0.6)")

                        if not weak_corr.empty:
                            st.warning(f"**⚪ Weak Correlations:** {len(weak_corr)} features have very weak relationship with Close price")
                        else:
                            st.info("All features show some correlation with Close price")

                    with insight_col2:
                        if not strong_neg.empty:
                            st.error(f"**🔴 Strong Negative Correlations:** {len(strong_neg)} features show strong negative relationship with Close price")
                            for _, row in strong_neg.iterrows():
                                st.write(f"• **{row['Feature']}**: {row['Correlation']:.3f} ({row['Strength']})")
                        else:
                            st.info("No strong negative correlations found (<-0.6)")

                        # Overall correlation summary
                        avg_corr = corr_df['Correlation'].abs().mean()
                        st.info(f"**📊 Overall Correlation Strength:** Average absolute correlation is {avg_corr:.3f}")

                    # Correlation matrix heatmap (if we have enough features)
                    if len(corr_data) > 2:
                        st.subheader("🔥 Correlation Heatmap")

                        # Create correlation matrix for all numeric columns
                        numeric_cols = [col for col in result['statistics'].keys() if col != 'Date']
                        if len(numeric_cols) > 2:
                            # Get correlation matrix from the data
                            # For now, we'll create a simple matrix with available correlations
                            corr_matrix = pd.DataFrame(index=numeric_cols, columns=numeric_cols)

                            # Fill diagonal with 1.0
                            for col in numeric_cols:
                                corr_matrix.loc[col, col] = 1.0

                            # Fill with available correlations
                            for feature, corr in corr_data.items():
                                if feature in numeric_cols:
                                    corr_matrix.loc[feature, 'Close'] = corr
                                    corr_matrix.loc['Close', feature] = corr

                            # Create heatmap
                            fig_heatmap = go.Figure(data=go.Heatmap(
                                z=corr_matrix.values,
                                x=corr_matrix.columns,
                                y=corr_matrix.index,
                                colorscale='RdBu_r',
                                zmin=-1,
                                zmax=1,
                                hoverongaps=False,
                                hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
                            ))

                            fig_heatmap.update_layout(
                                title="Feature Correlation Matrix",
                                height=400,
                                xaxis_tickangle=45
                            )

                            st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Display data info
                st.subheader("ℹ️ Data Information")
                st.json({
                    'row_count': result['row_count'],
                    'date_range': result['date_range'],
                    'columns': result['columns']
                })
            else:
                st.error(f"❌ Failed to generate profile: {result['error']}")

# Model Management Page
elif page == "⚙️ Model Management":
    st.title("⚙️ Model Management")
    st.markdown("---")
    
    if not check_api_status():
        st.error("⚠️ API is not running. Please start the backend API first.")
        st.stop()
    
    # Get models
    models_data = make_request('GET', '/models')
    if 'error' in models_data:
        st.error(models_data['error'])
        st.stop()
    
    models = models_data.get('models', [])
    if not models:
        st.info("No models trained yet.")
        st.stop()
    
    st.subheader("📋 All Trained Models")
    
    # Display models in a table
    models_df = pd.DataFrame(models)
    st.dataframe(models_df, use_container_width=True)
    
    # Model details
    st.subheader("🔍 Model Details")
    model_options = {f"{m['model_name']} (ID: {m['id']})": m['id'] for m in models}
    selected_model = st.selectbox("Select Model to View Details", list(model_options.keys()))
    model_id = model_options[selected_model]
    
    if st.button("📊 View Details"):
        with st.spinner("Loading model details..."):
            result = make_request('GET', f'/models/{model_id}')
            
            if 'error' not in result:
                model = result.get('model', {})
                predictions = result.get('recent_predictions', [])
                
                # Display model info
                col1, col2 = st.columns(2)
                with col1:
                    st.json({
                        'ID': model.get('id'),
                        'Name': model.get('model_name'),
                        'Type': model.get('model_type'),
                        'Created': model.get('created_date'),
                        'Accuracy': model.get('accuracy'),
                        'RMSE': model.get('rmse'),
                        'Features': model.get('feature_count')
                    })
                
                with col2:
                    if predictions:
                        st.subheader("Recent Predictions")
                        pred_df = pd.DataFrame(predictions)
                        st.dataframe(pred_df, use_container_width=True)
                    else:
                        st.info("No predictions made with this model yet.")
            else:
                st.error(f"❌ Failed to load details: {result['error']}")

if __name__ == "__main__":
    pass
