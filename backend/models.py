import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from typing import Tuple, Optional
import pandas as pd

MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

class StockPredictor:
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize stock predictor
        model_type: 'random_forest', 'xgboost', or 'lightgbm'
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
    def create_model(self, **kwargs):
        """Create the model based on type"""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 2),
                min_samples_leaf=kwargs.get('min_samples_leaf', 1),
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              feature_names: Optional[list] = None, **kwargs) -> dict:
        """
        Train the model
        Returns: Dictionary with training metrics
        """
        if self.model is None:
            self.create_model(**kwargs)
        
        self.feature_names = feature_names
        
        # Train the model
        if self.model_type == 'lightgbm' and X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
            )
        else:
            self.model.fit(X_train, y_train)
        
        self.is_trained = True
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        metrics = {
            'train_rmse': float(train_rmse),
            'train_mae': float(train_mae),
            'train_r2': float(train_r2)
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            metrics.update({
                'val_rmse': float(val_rmse),
                'val_mae': float(val_mae),
                'val_r2': float(val_r2)
            })
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_single(self, X: np.ndarray) -> Tuple[float, float]:
        """
        Make a single prediction with confidence interval
        Returns: (prediction, confidence_std)
        """
        if self.model_type == 'random_forest':
            # For Random Forest, use tree predictions for confidence
            predictions = np.array([tree.predict(X.reshape(1, -1))[0] 
                                   for tree in self.model.estimators_])
            pred = np.mean(predictions)
            confidence = np.std(predictions)
        else:
            pred = self.model.predict(X.reshape(1, -1))[0]
            # For tree-based models, use a simple confidence estimate
            # In production, you might want to use prediction intervals
            confidence = pred * 0.05  # 5% of prediction as confidence estimate
        
        return float(pred), float(abs(confidence))
    
    def save(self, filepath: str):
        """Save the model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    @staticmethod
    def load(filepath: str) -> 'StockPredictor':
        """Load a model from disk"""
        model_data = joblib.load(filepath)
        predictor = StockPredictor(model_type=model_data['model_type'])
        predictor.model = model_data['model']
        predictor.feature_names = model_data['feature_names']
        predictor.is_trained = model_data['is_trained']
        return predictor
    
    def get_feature_importance(self) -> dict:
        """Get feature importance scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            if self.feature_names:
                return dict(zip(self.feature_names, importances))
            else:
                return {f'feature_{i}': float(imp) for i, imp in enumerate(importances)}
        else:
            return {}
