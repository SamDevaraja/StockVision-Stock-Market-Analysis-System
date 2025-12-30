import pandas as pd
import numpy as np
from typing import Dict, Tuple
from backend.models import StockPredictor

class BacktestEngine:
    def __init__(self, model: StockPredictor, data: pd.DataFrame, 
                 features: np.ndarray, feature_names: list):
        self.model = model
        self.data = data.copy()
        self.features = features
        self.feature_names = feature_names
        
    def run_backtest(self, initial_capital: float = 10000.0,
                    stop_loss: float = 0.02,
                    take_profit: float = 0.05,
                    transaction_cost: float = 0.001,
                    start_date: str = None,
                    end_date: str = None) -> Dict:
        """
        Run backtesting simulation
        Returns: Dictionary with backtest results
        """
        # Data and features are already filtered in the API layer
        # Use the pre-calculated features passed to the constructor
        features_to_use = self.features
        data_to_use = self.data

        if len(data_to_use) < 2:
            raise ValueError("Insufficient data for backtesting")

        if len(features_to_use) < 2:
            raise ValueError("Insufficient feature data for backtesting")
        
        capital = initial_capital
        shares = 0
        position = None  # 'long' or None
        entry_price = 0
        trades = []
        
        # Make predictions
        predictions = self.model.predict(features_to_use)
        
        # Reset index for data_to_use
        data_to_use = data_to_use.reset_index(drop=True)
        
        for i in range(1, len(data_to_use)):
            current_price = data_to_use.iloc[i]['Close']
            previous_price = data_to_use.iloc[i-1]['Close']
            predicted_price = predictions[i] if i < len(predictions) else current_price
            
            # Simple trading strategy: buy if predicted price > current price, sell otherwise
            # More sophisticated strategies can be implemented
            
            if position is None:  # No position
                # Buy signal: predicted price is higher than current price
                if predicted_price > current_price:
                    # Calculate number of shares we can buy
                    cost_per_share = current_price * (1 + transaction_cost)
                    shares_to_buy = capital / cost_per_share
                    
                    if shares_to_buy > 0:
                        shares = shares_to_buy
                        entry_price = current_price
                        position = 'long'
                        capital = 0
                        
                        trades.append({
                            'date': data_to_use.iloc[i]['Date'] if 'Date' in data_to_use.columns else i,
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares,
                            'capital': capital
                        })
            else:  # In position
                # Calculate current P&L
                current_value = shares * current_price
                pnl_pct = (current_price - entry_price) / entry_price
                
                # Check stop loss
                if pnl_pct <= -stop_loss:
                    # Sell at stop loss
                    capital = current_value * (1 - transaction_cost)
                    trades.append({
                        'date': data_to_use.iloc[i]['Date'] if 'Date' in data_to_use.columns else i,
                        'action': 'SELL',
                        'price': current_price,
                        'shares': shares,
                        'capital': capital,
                        'pnl_pct': pnl_pct,
                        'reason': 'STOP_LOSS'
                    })
                    shares = 0
                    position = None
                    entry_price = 0
                
                # Check take profit
                elif pnl_pct >= take_profit:
                    # Sell at take profit
                    capital = current_value * (1 - transaction_cost)
                    trades.append({
                        'date': data_to_use.iloc[i]['Date'] if 'Date' in data_to_use.columns else i,
                        'action': 'SELL',
                        'price': current_price,
                        'shares': shares,
                        'capital': capital,
                        'pnl_pct': pnl_pct,
                        'reason': 'TAKE_PROFIT'
                    })
                    shares = 0
                    position = None
                    entry_price = 0
                
                # Sell signal: predicted price is lower than current price
                elif predicted_price < current_price:
                    # Sell
                    capital = current_value * (1 - transaction_cost)
                    trades.append({
                        'date': data_to_use.iloc[i]['Date'] if 'Date' in data_to_use.columns else i,
                        'action': 'SELL',
                        'price': current_price,
                        'shares': shares,
                        'capital': capital,
                        'pnl_pct': pnl_pct,
                        'reason': 'SIGNAL'
                    })
                    shares = 0
                    position = None
                    entry_price = 0
        
        # Close any remaining position
        if position is not None:
            final_price = data_to_use.iloc[-1]['Close']
            current_value = shares * final_price
            capital = current_value * (1 - transaction_cost)
            pnl_pct = (final_price - entry_price) / entry_price
            trades.append({
                'date': data_to_use.iloc[-1]['Date'] if 'Date' in data_to_use.columns else len(data_to_use)-1,
                'action': 'SELL',
                'price': final_price,
                'shares': shares,
                'capital': capital,
                'pnl_pct': pnl_pct,
                'reason': 'END_OF_PERIOD'
            })
        
        # Calculate statistics
        final_capital = capital
        total_return = (final_capital - initial_capital) / initial_capital
        
        # Separate buy and sell trades
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL' and 'pnl_pct' in t]
        
        total_trades = len(buy_trades)
        winning_trades = len([t for t in sell_trades if t.get('pnl_pct', 0) > 0])
        losing_trades = len([t for t in sell_trades if t.get('pnl_pct', 0) <= 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_return = np.mean([t.get('pnl_pct', 0) for t in sell_trades]) if sell_trades else 0
        max_return = np.max([t.get('pnl_pct', 0) for t in sell_trades]) if sell_trades else 0
        min_return = np.min([t.get('pnl_pct', 0) for t in sell_trades]) if sell_trades else 0
        
        # Calculate equity curve
        equity_curve = []
        current_capital = initial_capital
        current_shares = 0
        current_entry = 0
        
        for i in range(len(data_to_use)):
            price = data_to_use.iloc[i]['Close']
            if current_shares > 0:
                current_capital = current_shares * price
            equity_curve.append({
                'date': data_to_use.iloc[i]['Date'] if 'Date' in data_to_use.columns else i,
                'equity': current_capital
            })
            
            # Update based on trades
            for trade in trades:
                trade_date = trade['date']
                data_date = data_to_use.iloc[i]['Date'] if 'Date' in data_to_use.columns else i
                if trade_date == data_date:
                    if trade['action'] == 'BUY':
                        current_shares = trade['shares']
                        current_entry = trade['price']
                        current_capital = 0
                    elif trade['action'] == 'SELL':
                        current_capital = trade['capital']
                        current_shares = 0
                        current_entry = 0
        
        results = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'max_return': max_return,
            'min_return': min_return,
            'trades': trades,
            'equity_curve': equity_curve
        }
        
        return results
