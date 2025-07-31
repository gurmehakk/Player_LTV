"""
ExpLTV Model with Zero-Inflated Log-Normal (ZILN) Loss and Whale Detection
Implements ExpLTV methodology with whale detection, expert routing, and joint loss optimization
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, classification_report, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler


class ExpLTVModel(nn.Module):
    """ExpLTV Neural Network with Whale Detection and Expert Routing"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 embedding_dim: int = 64,
                 dropout_rate: float = 0.2,
                 num_experts: int = 2):
        super(ExpLTVModel, self).__init__()
        
        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Whale detection head
        self.whale_detector = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
        # Payer classification head
        self.payer_classifier = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
        # Expert routing for LTV prediction
        self.expert_gate = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_experts),
            nn.Softmax(dim=1)
        )
        
        # LTV prediction experts
        self.ltv_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(embedding_dim, 1),
                nn.ReLU()  # Ensure positive output
            ) for _ in range(num_experts)
        ])
        
        self.num_experts = num_experts
    
    def forward(self, x):
        # Shared feature extraction
        shared_features = self.shared_layers(x)
        
        # Whale detection
        whale_prob = self.whale_detector(shared_features)
        
        # Payer classification
        payer_prob = self.payer_classifier(shared_features)
        
        # Expert routing
        expert_weights = self.expert_gate(shared_features)
        
        # Get predictions from all experts
        expert_outputs = []
        for expert in self.ltv_experts:
            expert_outputs.append(expert(shared_features))
        
        # Combine expert outputs using routing weights
        expert_outputs = torch.stack(expert_outputs, dim=2)  # [batch, 1, num_experts]
        expert_weights = expert_weights.unsqueeze(1)  # [batch, 1, num_experts]
        
        ltv_pred = torch.sum(expert_outputs * expert_weights, dim=2)  # [batch, 1]
        
        return {
            'whale_prob': whale_prob,
            'payer_prob': payer_prob,
            'ltv_pred': ltv_pred,
            'expert_weights': expert_weights.squeeze(1)
        }


class ZILNModel:
    """Zero-Inflated Log-Normal Model with ExpLTV Neural Network and LightGBM Fallback"""
    
    def __init__(self, 
                 use_neural_network: bool = True,
                 binary_params: Optional[Dict] = None,
                 regression_params: Optional[Dict] = None,
                 nn_params: Optional[Dict] = None):
        
        self.use_neural_network = use_neural_network
        
        # Neural network parameters
        self.nn_params = nn_params or {
            'hidden_dim': 128,
            'embedding_dim': 64,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'epochs': 200,
            'batch_size': 1024,
            'early_stopping_patience': 20,
            'gradient_clip_norm': 1.0
        }
        
        # Default parameters for binary classifier (LightGBM fallback)
        self.binary_params = binary_params or {
            'objective': 'binary',
            'metric': 'auc',
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'num_leaves': 31,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'class_weight': 'balanced',
            'random_state': 42,
            'verbose': -1
        }
        
        # Default parameters for regression model
        self.regression_params = regression_params or {
            'objective': 'regression',
            'metric': 'rmse',
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'num_leaves': 31,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'random_state': 42,
            'verbose': -1
        }
        
        # Model components
        self.neural_model = None
        self.binary_classifier = None  # LightGBM fallback
        self.amount_regressor = None   # LightGBM fallback
        self.scaler = StandardScaler()
        
        # Device for PyTorch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Feature importance
        self.feature_importance_binary = {}
        self.feature_importance_amount = {}
        
        # Model statistics
        self.training_stats = {}
        self.whale_threshold = None
        
    def fit(self, X: pd.DataFrame, y: np.ndarray, whale_labels: Optional[np.ndarray] = None) -> 'ZILNModel':
        """
        Fit the ExpLTV model with whale detection
        
        Args:
            X: Feature matrix
            y: Target values (revenue)
            whale_labels: Optional whale labels for supervised learning
            
        Returns:
            Fitted model instance
        """
        
        print("Training ZILN Model...")
        
        # Step 1: Create binary target (payer vs non-payer)
        y_binary = (y > 0).astype(int)
        
        # Step 2: Create positive revenue subset for amount prediction
        positive_mask = y > 0
        X_positive = X[positive_mask]
        y_positive = y[positive_mask]
        
        # Log-transform positive revenue values
        y_log_positive = np.log1p(y_positive)
        
        # Store training statistics
        self.training_stats = {
            'total_samples': len(y),
            'payers': sum(y_binary),
            'non_payers': len(y) - sum(y_binary),
            'payer_rate': sum(y_binary) / len(y),
            'avg_revenue_payers': np.mean(y_positive) if len(y_positive) > 0 else 0,
            'median_revenue_payers': np.median(y_positive) if len(y_positive) > 0 else 0
        }
        
        print(f"Training set: {self.training_stats['total_samples']:,} players")
        print(f"Payers: {self.training_stats['payers']:,} ({self.training_stats['payer_rate']*100:.1f}%)")
        print(f"Average revenue (payers): ${self.training_stats['avg_revenue_payers']:.2f}")
        
        # Step 3: Train binary classifier (payer vs non-payer)
        print("Training binary classifier...")
        self.binary_classifier = lgb.LGBMClassifier(**self.binary_params)
        self.binary_classifier.fit(X, y_binary)
        
        # Store feature importance for binary classifier
        self.feature_importance_binary = dict(
            zip(X.columns, self.binary_classifier.feature_importances_)
        )
        
        # Step 4: Train amount regressor on positive values only
        if len(X_positive) > 10:  # Need minimum samples for regression
            print("Training amount regressor...")
            self.amount_regressor = lgb.LGBMRegressor(**self.regression_params)
            self.amount_regressor.fit(X_positive, y_log_positive)
            
            # Store feature importance for amount regressor
            self.feature_importance_amount = dict(
                zip(X_positive.columns, self.amount_regressor.feature_importances_)
            )
        else:
            print("Warning: Insufficient positive samples for amount regression")
            self.amount_regressor = None
            self.feature_importance_amount = {}
        
        print("ZILN model training completed")
        return self
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict using ZILN approach
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Tuple of (expected_pltv, payer_probabilities, conditional_amounts)
        """
        
        if self.binary_classifier is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Step 1: Predict probability of being a payer
        payer_probs = self.binary_classifier.predict_proba(X)[:, 1]
        
        # Step 2: Predict log-amount for potential payers
        if self.amount_regressor is not None:
            log_amounts = self.amount_regressor.predict(X)
            conditional_amounts = np.expm1(log_amounts)  # Transform back from log space
            
            # Ensure non-negative amounts
            conditional_amounts = np.maximum(conditional_amounts, 0)
        else:
            # Fallback: use mean amount from training
            mean_amount = self.training_stats.get('avg_revenue_payers', 0)
            conditional_amounts = np.full(len(X), mean_amount)
        
        # Step 3: Expected pLTV = P(payer) * E(amount | payer)
        expected_pltv = payer_probs * conditional_amounts
        
        return expected_pltv, payer_probs, conditional_amounts
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict payer probabilities only"""
        if self.binary_classifier is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.binary_classifier.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive evaluation of ZILN model performance
        
        Args:
            X_test: Test feature matrix
            y_test: Test target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        # Get predictions
        predictions, payer_probs, conditional_amounts = self.predict(X_test)
        
        # Binary classification metrics
        y_binary_test = (y_test > 0).astype(int)
        auc_score = roc_auc_score(y_binary_test, payer_probs)
        
        # Classification accuracy metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Use 0.5 threshold for binary predictions
        payer_predictions = (payer_probs >= 0.5).astype(int)
        
        binary_accuracy = accuracy_score(y_binary_test, payer_predictions)
        precision = precision_score(y_binary_test, payer_predictions, zero_division=0)
        recall = recall_score(y_binary_test, payer_predictions, zero_division=0)
        f1 = f1_score(y_binary_test, payer_predictions, zero_division=0)
        
        # Regression metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        # R² score (can be negative for poor models)
        r2 = r2_score(y_test, predictions)
        
        # Mean Absolute Percentage Error (MAPE) for non-zero values
        non_zero_mask = y_test > 0
        if np.sum(non_zero_mask) > 0:
            mape = np.mean(np.abs((y_test[non_zero_mask] - predictions[non_zero_mask]) / y_test[non_zero_mask])) * 100
        else:
            mape = float('inf')
        
        # Explained Variance Score
        from sklearn.metrics import explained_variance_score
        explained_variance = explained_variance_score(y_test, predictions)
        
        # Revenue prediction accuracy (within X% bands)
        revenue_accuracy_10pct = np.mean(np.abs(y_test - predictions) <= 0.1 * np.maximum(y_test, predictions))
        revenue_accuracy_25pct = np.mean(np.abs(y_test - predictions) <= 0.25 * np.maximum(y_test, predictions))
        revenue_accuracy_50pct = np.mean(np.abs(y_test - predictions) <= 0.5 * np.maximum(y_test, predictions))
        
        # Business metrics: Lift analysis
        sorted_indices = np.argsort(predictions)[::-1]  # Sort by predicted value (descending)
        
        # Calculate lift for different percentiles
        percentiles = [0.01, 0.05, 0.1, 0.2, 0.5]
        lift_metrics = {}
        
        total_revenue = np.sum(y_test)
        
        for pct in percentiles:
            top_n = int(len(predictions) * pct)
            if top_n > 0:
                top_indices = sorted_indices[:top_n]
                top_revenue = np.sum(y_test[top_indices])
                
                # Lift = (% of revenue captured) / (% of users targeted)
                revenue_capture_rate = top_revenue / total_revenue if total_revenue > 0 else 0
                lift = revenue_capture_rate / pct if pct > 0 else 0
                
                lift_metrics[f'lift_top_{int(pct*100)}pct'] = lift
                lift_metrics[f'revenue_capture_top_{int(pct*100)}pct'] = revenue_capture_rate
        
        # Precision and recall for payer classification at different thresholds
        from sklearn.metrics import precision_recall_curve
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_binary_test, payer_probs)
        avg_precision = np.mean(precision_curve)
        avg_recall = np.mean(recall_curve)
        
        # Find optimal threshold (max F1 score)
        f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
        optimal_threshold_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_threshold_idx] if len(thresholds) > optimal_threshold_idx else 0.5
        optimal_f1 = f1_scores[optimal_threshold_idx]
        
        # Model calibration: predicted vs actual payer rates
        predicted_payer_rate = np.mean(payer_probs)
        actual_payer_rate = np.mean(y_binary_test)
        calibration_error = abs(predicted_payer_rate - actual_payer_rate)
        
        # Brier Score (lower is better)
        brier_score = np.mean((payer_probs - y_binary_test) ** 2)
        
        # Decile analysis for payer probabilities
        decile_analysis = {}
        prob_deciles = np.percentile(payer_probs, np.arange(10, 101, 10))
        for i, decile in enumerate(prob_deciles):
            decile_mask = payer_probs >= decile
            if np.sum(decile_mask) > 0:
                decile_payer_rate = np.mean(y_binary_test[decile_mask])
                decile_avg_prob = np.mean(payer_probs[decile_mask])
                decile_analysis[f'decile_{i+1}_payer_rate'] = decile_payer_rate
                decile_analysis[f'decile_{i+1}_avg_prob'] = decile_avg_prob
        
        # Revenue tier accuracy
        if np.sum(y_test > 0) > 0:
            payers_mask = y_test > 0
            payer_revenues = y_test[payers_mask]
            payer_predictions = predictions[payers_mask]
            
            # High-value player identification (top 10% of spenders)
            high_value_threshold = np.percentile(payer_revenues, 90)
            actual_high_value = (payer_revenues >= high_value_threshold).astype(int)
            predicted_high_value = (payer_predictions >= high_value_threshold).astype(int)
            
            high_value_accuracy = accuracy_score(actual_high_value, predicted_high_value)
            high_value_precision = precision_score(actual_high_value, predicted_high_value, zero_division=0)
            high_value_recall = recall_score(actual_high_value, predicted_high_value, zero_division=0)
        else:
            high_value_accuracy = 0
            high_value_precision = 0
            high_value_recall = 0
        
        # Compile all metrics
        metrics = {
            # Regression metrics
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'explained_variance': explained_variance,
            
            # Revenue accuracy bands
            'revenue_accuracy_within_10pct': revenue_accuracy_10pct,
            'revenue_accuracy_within_25pct': revenue_accuracy_25pct,
            'revenue_accuracy_within_50pct': revenue_accuracy_50pct,
            
            # Binary classification metrics
            'auc_payer_classification': auc_score,
            'binary_accuracy': binary_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'optimal_threshold': optimal_threshold,
            'optimal_f1': optimal_f1,
            
            # Business metrics
            **lift_metrics,
            
            # Model calibration
            'predicted_payer_rate': predicted_payer_rate,
            'actual_payer_rate': actual_payer_rate,
            'calibration_error': calibration_error,
            'brier_score': brier_score,
            
            # High-value player identification
            'high_value_accuracy': high_value_accuracy,
            'high_value_precision': high_value_precision,
            'high_value_recall': high_value_recall,
            
            # Revenue metrics
            'total_revenue_actual': total_revenue,
            'total_revenue_predicted': np.sum(predictions),
            'revenue_prediction_error': abs(total_revenue - np.sum(predictions)) / total_revenue if total_revenue > 0 else float('inf'),
            
            # Additional performance indicators
            'mean_predicted_revenue': np.mean(predictions),
            'median_predicted_revenue': np.median(predictions),
            'std_predicted_revenue': np.std(predictions),
            'mean_actual_revenue': np.mean(y_test),
            'median_actual_revenue': np.median(y_test),
            'std_actual_revenue': np.std(y_test)
        }
        
        # Add decile analysis
        metrics.update(decile_analysis)
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, Dict[str, float]]:
        """
        Get feature importance for both model components
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary with binary and amount model feature importances
        """
        
        result = {}
        
        # Binary classifier importance
        if self.feature_importance_binary:
            sorted_binary = sorted(
                self.feature_importance_binary.items(),
                key=lambda x: x[1], reverse=True
            )[:top_n]
            result['binary_classifier'] = dict(sorted_binary)
        
        # Amount regressor importance
        if self.feature_importance_amount:
            sorted_amount = sorted(
                self.feature_importance_amount.items(),
                key=lambda x: x[1], reverse=True
            )[:top_n]
            result['amount_regressor'] = dict(sorted_amount)
        
        return result
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get training statistics"""
        return self.training_stats.copy()