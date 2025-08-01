"""
Simplified LTV Model focusing purely on regression performance
Removes multi-task complexity to focus on getting positive R² score
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from typing import Dict, Any

logger = logging.getLogger(__name__)


class SimpleLTVModel(nn.Module):
    """Simplified neural network focusing only on LTV regression"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.2):
        super(SimpleLTVModel, self).__init__()
        
        # Much simpler architecture for better regression performance
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU()  # Ensure positive output
        )
    
    def forward(self, x):
        return self.network(x)


class SimpleLTVRegressor:
    """Simplified LTV regressor with multiple model options"""
    
    def __init__(self, model_type: str = 'neural', use_robust_scaling: bool = True):
        self.model_type = model_type  # 'neural', 'lgb', 'rf'
        self.use_robust_scaling = use_robust_scaling
        
        # Use RobustScaler for better handling of outliers and small values
        self.scaler = RobustScaler() if use_robust_scaling else StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Track training stats
        self.training_stats = {}
        
    def _preprocess_targets(self, y):
        """Preprocess LTV targets with better handling of small values"""
        
        # Option 1: No transformation (direct regression)
        # This often works better for small values
        return y
        
        # Alternative preprocessing approaches:
        # Option 2: Square root transformation (less aggressive than log)
        # return np.sqrt(y)
        
        # Option 3: Simple scaling
        # return y / y.std() if y.std() > 0 else y
    
    def _inverse_preprocess_targets(self, y_transformed):
        """Inverse of target preprocessing"""
        return y_transformed  # No transformation used
        
        # For square root: return y_transformed ** 2
        # For scaling: return y_transformed * self.target_std
    
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fit the simplified LTV model"""
        
        logger.info(f"Training Simplified LTV Model ({self.model_type})...")
        
        # Store training statistics
        self.training_stats = {
            'total_samples': len(y),
            'payers': np.sum(y > 0),  
            'non_payers': np.sum(y == 0),
            'payer_rate': np.mean(y > 0),
            'mean_ltv': np.mean(y),
            'std_ltv': np.std(y),
            'median_ltv': np.median(y),
            'max_ltv': np.max(y),
            'q75_ltv': np.percentile(y, 75),
            'q95_ltv': np.percentile(y, 95)
        }
        
        logger.info(f"Training data: {len(y):,} samples")
        logger.info(f"Payer rate: {self.training_stats['payer_rate']*100:.1f}%")
        logger.info(f"LTV stats: mean=${self.training_stats['mean_ltv']:.3f}, std=${self.training_stats['std_ltv']:.3f}")
        logger.info(f"LTV range: $0 - ${self.training_stats['max_ltv']:.2f}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Preprocess targets
        y_processed = self._preprocess_targets(y)
        
        if self.model_type == 'neural':
            self._fit_neural_network(X_scaled, y_processed)
        elif self.model_type == 'lgb':
            self._fit_lightgbm(X_scaled, y_processed)
        elif self.model_type == 'rf':
            self._fit_random_forest(X_scaled, y_processed)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        return self
    
    def _fit_neural_network(self, X: np.ndarray, y: np.ndarray):
        """Fit neural network with focus on regression performance"""
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Initialize model with smaller architecture
        self.model = SimpleLTVModel(
            input_dim=X.shape[1], 
            hidden_dim=32,  # Much smaller
            dropout_rate=0.1  # Less dropout
        ).to(self.device)
        
        # Use simpler optimizer settings
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-4
        )
        
        # Simpler loss function - just MSE for regression
        criterion = nn.MSELoss()
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        self.model.train()
        for epoch in range(100):  # Fewer epochs
            # Training
            optimizer.zero_grad()
            train_pred = self.model(X_train_tensor).squeeze()
            train_loss = criterion(train_pred, y_train_tensor)
            train_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_tensor).squeeze()
                val_loss = criterion(val_pred, y_val_tensor)
            self.model.train()
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_simple_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('best_simple_model.pth'))
        self.model.eval()
        
        logger.info(f"Neural network training completed. Best val loss: {best_val_loss:.6f}")
    
    def _fit_lightgbm(self, X: np.ndarray, y: np.ndarray):
        """Fit LightGBM regressor optimized for small LTV values"""
        
        # LightGBM parameters optimized for regression
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 0.1,  # L2 regularization
        }
        
        self.model = lgb.LGBMRegressor(**params, n_estimators=200)
        self.model.fit(X, y)
        
        logger.info("LightGBM training completed")
    
    def _fit_random_forest(self, X: np.ndarray, y: np.ndarray):
        """Fit Random Forest regressor"""
        
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        logger.info("Random Forest training completed")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'neural':
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                predictions = self.model(X_tensor).squeeze().cpu().numpy()
        else:
            predictions = self.model.predict(X_scaled)
        
        # Inverse transform targets
        predictions = self._inverse_preprocess_targets(predictions)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        
        predictions = self.predict(X)
        
        # Regression metrics
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        
        # Additional metrics for small values
        mape = np.mean(np.abs((y - predictions) / np.maximum(y, 0.001))) * 100
        
        # Directional accuracy (for values > 0)
        payer_mask = y > 0
        if np.sum(payer_mask) > 0:
            payer_mae = mean_absolute_error(y[payer_mask], predictions[payer_mask])
            payer_r2 = r2_score(y[payer_mask], predictions[payer_mask])
        else:
            payer_mae = mae
            payer_r2 = r2
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'payer_mae': payer_mae,
            'payer_r2': payer_r2,
            'mean_prediction': np.mean(predictions),
            'std_prediction': np.std(predictions),
            'total_actual': np.sum(y),
            'total_predicted': np.sum(predictions),
            'prediction_error_pct': abs(np.sum(predictions) - np.sum(y)) / np.sum(y) * 100 if np.sum(y) > 0 else 0
        }


def compare_simple_models(X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
    """Compare different simple model approaches"""
    
    logger.info("Comparing simplified LTV models...")
    
    # Split data for fair comparison
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    results = {}
    
    # Test different model types
    for model_type in ['lgb', 'rf', 'neural']:
        logger.info(f"\\nTraining {model_type.upper()} model...")
        
        try:
            model = SimpleLTVRegressor(model_type=model_type)
            model.fit(X_train, y_train)
            
            # Evaluate
            metrics = model.evaluate(X_test, y_test)
            results[model_type] = {
                'model': model,
                'metrics': metrics
            }
            
            logger.info(f"{model_type.upper()} Results:")
            logger.info(f"  R² Score: {metrics['r2']:.4f}")
            logger.info(f"  RMSE: {metrics['rmse']:.4f}")
            logger.info(f"  MAE: {metrics['mae']:.4f}")
            
        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")
            results[model_type] = {'error': str(e)}
    
    # Find best model
    best_model_type = None
    best_r2 = -float('inf')
    
    for model_type, result in results.items():
        if 'metrics' in result and result['metrics']['r2'] > best_r2:
            best_r2 = result['metrics']['r2']
            best_model_type = model_type
    
    logger.info(f"\\nBest model: {best_model_type.upper()} with R² = {best_r2:.4f}")
    
    results['best_model'] = best_model_type
    results['test_data'] = {'X': X_test, 'y': y_test}
    
    return results


if __name__ == "__main__":
    # This can be run standalone for testing
    logging.basicConfig(level=logging.INFO)
    logger.info("Simple LTV Model module loaded")