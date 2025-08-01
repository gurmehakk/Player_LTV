"""
ExpLTV Model with Zero-Inflated Log-Normal (ZILN) Loss and Whale Detection
Implements ExpLTV methodology with whale detection, expert routing, and joint loss optimization
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)
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
                 hidden_dim: int = 256,
                 embedding_dim: int = 128,
                 dropout_rate: float = 0.3,
                 num_experts: int = 3):
        super(ExpLTVModel, self).__init__()
        
        # Enhanced shared feature extraction layers with batch normalization
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        shared_output_dim = hidden_dim // 2
        
        # Enhanced whale detection head
        self.whale_detector = nn.Sequential(
            nn.Linear(shared_output_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Enhanced payer classification head
        self.payer_classifier = nn.Sequential(
            nn.Linear(shared_output_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Enhanced expert routing for LTV prediction
        self.expert_gate = nn.Sequential(
            nn.Linear(shared_output_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(embedding_dim, num_experts),
            nn.Softmax(dim=1)
        )
        
        # Enhanced LTV prediction experts with residual connections
        self.ltv_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_output_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(embedding_dim // 2, 1),
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
        
        # Enhanced neural network parameters
        self.nn_params = nn_params or {
            'hidden_dim': 256,
            'embedding_dim': 128,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'epochs': 500,
            'batch_size': 512,
            'early_stopping_patience': 30,
            'gradient_clip_norm': 1.0,
            'weight_decay': 1e-4,
            'lr_scheduler_factor': 0.8,
            'lr_scheduler_patience': 15,
            'min_lr': 1e-6
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
        self.target_scaler = StandardScaler()  # For LTV target scaling
        
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
        Fit the ExpLTV model with neural network and whale detection
        
        Args:
            X: Feature matrix
            y: Target values (revenue)
            whale_labels: Optional whale labels for supervised learning
            
        Returns:
            Fitted model instance
        """
        
        logger = logging.getLogger(__name__)
        logger.info("Training Enhanced ZILN Model with Neural Network...")
        
        # Step 1: Create binary target (payer vs non-payer)  
        y_binary = (y > 0).astype(int)
        
        # Store training statistics
        positive_mask = y > 0
        y_positive = y[positive_mask]
        
        self.training_stats = {
            'total_samples': len(y),
            'payers': sum(y_binary),
            'non_payers': len(y) - sum(y_binary),
            'payer_rate': sum(y_binary) / len(y),
            'avg_revenue_payers': np.mean(y_positive) if len(y_positive) > 0 else 0,
            'median_revenue_payers': np.median(y_positive) if len(y_positive) > 0 else 0
        }
        
        logger.info(f"Training set: {self.training_stats['total_samples']:,} players")
        logger.info(f"Payers: {self.training_stats['payers']:,} ({self.training_stats['payer_rate']*100:.1f}%)")
        logger.info(f"Average revenue (payers): ${self.training_stats['avg_revenue_payers']:.2f}")
        
        if self.use_neural_network and len(y_positive) > 100:
            # Train neural network
            logger.info("Training Neural Network Model...")
            self._train_neural_network(X, y, y_binary, whale_labels)
        else:
            # Fallback to LightGBM
            print("Using LightGBM fallback...")
            self._train_lightgbm_fallback(X, y, y_binary)
        
        logger.info("Enhanced ZILN model training completed")
        return self
    
    def _train_neural_network(self, X: pd.DataFrame, y: np.ndarray, y_binary: np.ndarray, 
                             whale_labels: Optional[np.ndarray] = None):
        """Train the neural network model with under/oversampling for imbalanced data"""
        
        from sklearn.model_selection import train_test_split
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.combine import SMOTEENN
        import matplotlib.pyplot as plt
        
        # SMOTE is now handled in the feature engineering stage
        # No need for duplicate SMOTE application here
        
        # Prepare data
        X_scaled = self.scaler.fit_transform(X)
        
        # Scale LTV targets to improve regression learning
        # Use log1p transformation for better handling of zero values
        y_log = np.log1p(y)  # log(1 + y) to handle zeros
        y_scaled = self.target_scaler.fit_transform(y_log.reshape(-1, 1)).flatten()
        
        # Create whale labels if not provided
        if whale_labels is None:
            whale_threshold = np.percentile(y[y > 0], 95) if len(y[y > 0]) > 0 else 0
            whale_labels = (y >= whale_threshold).astype(int)
        
        # Use original data (SMOTE already applied in feature engineering if requested)
        X_for_split = X_scaled
        y_for_split = y_scaled  # Use scaled targets for training
        y_binary_for_split = y_binary
        whale_for_split = whale_labels
        
        # Train/validation split on balanced data
        X_train, X_val, y_train, y_val = train_test_split(
            X_for_split, y_for_split, test_size=0.2, random_state=42, stratify=whale_for_split
        )
        y_binary_train = (y_train > 0).astype(int)
        y_binary_val = (y_val > 0).astype(int)
        
        # Create whale labels if not provided
        if whale_labels is None:
            whale_threshold = np.percentile(y[y > 0], 95) if len(y[y > 0]) > 0 else 0
            whale_labels_train = (y_train >= whale_threshold).astype(int)
            whale_labels_val = (y_val >= whale_threshold).astype(int)
        else:
            whale_train, whale_val = train_test_split(whale_labels, test_size=0.2, random_state=42)
            whale_labels_train, whale_labels_val = whale_train, whale_val
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        y_binary_train_tensor = torch.FloatTensor(y_binary_train).to(self.device)
        y_binary_val_tensor = torch.FloatTensor(y_binary_val).to(self.device)
        whale_train_tensor = torch.FloatTensor(whale_labels_train).to(self.device)
        whale_val_tensor = torch.FloatTensor(whale_labels_val).to(self.device)
        
        # Initialize model
        input_dim = X_train.shape[1]
        self.neural_model = ExpLTVModel(
            input_dim=input_dim,
            hidden_dim=self.nn_params['hidden_dim'],
            embedding_dim=self.nn_params['embedding_dim'],
            dropout_rate=self.nn_params['dropout_rate'],
            num_experts=3
        ).to(self.device)
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.neural_model.parameters(),
            lr=self.nn_params['learning_rate'],
            weight_decay=self.nn_params['weight_decay']
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=self.nn_params['lr_scheduler_factor'],
            patience=self.nn_params['lr_scheduler_patience'],
            min_lr=self.nn_params['min_lr']
        )
        
        # Loss functions
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, y_binary_train_tensor, whale_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor, y_binary_val_tensor, whale_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=self.nn_params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.nn_params['batch_size'], shuffle=False)
        
        # Training tracking
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for {self.nn_params['epochs']} epochs...")
        logger.info("Epoch | Train Loss | Val Loss | LR | Time")
        print("-" * 50)
        
        import time
        
        for epoch in range(self.nn_params['epochs']):
            epoch_start_time = time.time()
            
            # Training phase
            self.neural_model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_X, batch_y, batch_binary, batch_whale in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.neural_model(batch_X)
                
                # Multi-task loss with improved weighting for LTV regression
                ltv_loss = mse_loss(outputs['ltv_pred'].squeeze(), batch_y)
                payer_loss = bce_loss(outputs['payer_prob'].squeeze(), batch_binary)
                whale_loss = bce_loss(outputs['whale_prob'].squeeze(), batch_whale)
                
                # Combined loss with very high weight on LTV regression
                # Make regression the primary objective
                total_loss = 10.0 * ltv_loss + 0.1 * payer_loss + 0.05 * whale_loss
                
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.neural_model.parameters(), 
                    self.nn_params['gradient_clip_norm']
                )
                
                optimizer.step()
                
                train_loss += total_loss.item()
                train_batches += 1
            
            # Validation phase
            self.neural_model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_X, batch_y, batch_binary, batch_whale in val_loader:
                    outputs = self.neural_model(batch_X)
                    
                    ltv_loss = mse_loss(outputs['ltv_pred'].squeeze(), batch_y)
                    payer_loss = bce_loss(outputs['payer_prob'].squeeze(), batch_binary)
                    whale_loss = bce_loss(outputs['whale_prob'].squeeze(), batch_whale)
                    
                    total_loss = ltv_loss + 0.5 * payer_loss + 0.3 * whale_loss
                    val_loss += total_loss.item()
                    val_batches += 1
            
            # Calculate average losses
            avg_train_loss = train_loss / train_batches
            avg_val_loss = val_loss / val_batches
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                epoch_time = time.time() - epoch_start_time
                print(f"{epoch+1:5d} | {avg_train_loss:9.6f} | {avg_val_loss:8.6f} | {current_lr:.2e} | {epoch_time:.1f}s")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.neural_model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= self.nn_params['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.neural_model.load_state_dict(torch.load('best_model.pth'))
        
        # Save training history
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epochs_trained': len(train_losses)
        }
        
        logger.info(f"Training completed after {len(train_losses)} epochs")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        
        # Create training plot
        self._plot_training_progress(train_losses, val_losses)
        
    def _plot_training_progress(self, train_losses, val_losses):
        """Plot training progress"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
        plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Neural Network Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('output_basic_model/training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Training progress plot saved to output/training_progress.png")
    
    def _train_lightgbm_fallback(self, X: pd.DataFrame, y: np.ndarray, y_binary: np.ndarray):
        """Fallback LightGBM training"""
        
        # Create positive revenue subset for amount prediction
        positive_mask = y > 0
        X_positive = X[positive_mask]
        y_positive = y[positive_mask]
        y_log_positive = np.log1p(y_positive)
        
        # Train binary classifier
        logger.info("Training binary classifier...")
        self.binary_classifier = lgb.LGBMClassifier(**self.binary_params)
        self.binary_classifier.fit(X, y_binary)
        
        # Store feature importance
        self.feature_importance_binary = dict(
            zip(X.columns, self.binary_classifier.feature_importances_)
        )
        
        # Train amount regressor
        if len(X_positive) > 10:
            logger.info("Training amount regressor...")
            self.amount_regressor = lgb.LGBMRegressor(**self.regression_params)
            self.amount_regressor.fit(X_positive, y_log_positive)
            
            self.feature_importance_amount = dict(
                zip(X_positive.columns, self.amount_regressor.feature_importances_)
            )
        else:
            print("Warning: Insufficient positive samples for amount regression")
            self.amount_regressor = None
            self.feature_importance_amount = {}
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict using enhanced ZILN approach (neural network or LightGBM)
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Tuple of (expected_pltv, payer_probabilities, conditional_amounts)
        """
        
        if self.neural_model is not None:
            # Use neural network predictions
            return self._predict_neural_network(X)
        elif self.binary_classifier is not None:
            # Use LightGBM fallback
            return self._predict_lightgbm(X)
        else:
            raise ValueError("Model not fitted. Call fit() first.")
    
    def _predict_neural_network(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Neural network predictions"""
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.neural_model.eval()
        with torch.no_grad():
            # Get predictions in batches to handle large datasets
            batch_size = 1000
            all_ltv_preds = []
            all_payer_probs = []
            all_whale_probs = []
            
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                outputs = self.neural_model(batch)
                
                all_ltv_preds.append(outputs['ltv_pred'].squeeze().cpu().numpy())
                all_payer_probs.append(outputs['payer_prob'].squeeze().cpu().numpy())
                all_whale_probs.append(outputs['whale_prob'].squeeze().cpu().numpy())
            
            # Concatenate all predictions
            ltv_preds = np.concatenate(all_ltv_preds)
            payer_probs = np.concatenate(all_payer_probs)
            whale_probs = np.concatenate(all_whale_probs)
        
        # Transform predictions back to original scale
        # Inverse transform from scaled space
        ltv_preds_unscaled = self.target_scaler.inverse_transform(ltv_preds.reshape(-1, 1)).flatten()
        # Inverse log1p transformation: exp(y) - 1
        ltv_preds = np.expm1(ltv_preds_unscaled)
        
        # Ensure non-negative LTV predictions
        ltv_preds = np.maximum(ltv_preds, 0)
        
        # Expected pLTV = P(payer) * LTV_pred
        expected_pltv = payer_probs * ltv_preds
        
        return expected_pltv, payer_probs, ltv_preds
        
    def _predict_lightgbm(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """LightGBM fallback predictions"""
        
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
        
        # First calculate optimal threshold using precision-recall curve
        from sklearn.metrics import precision_recall_curve
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_binary_test, payer_probs)
        
        # Find optimal threshold (max F1 score)
        f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
        optimal_threshold_idx = np.argmax(f1_scores)
        
        # Use a more sensitive threshold for imbalanced data
        if len(thresholds) > optimal_threshold_idx:
            optimal_threshold = thresholds[optimal_threshold_idx]
            # For very imbalanced data, ensure threshold isn't too high
            optimal_threshold = min(optimal_threshold, 0.3)
        else:
            # Fallback: use actual payer rate as threshold for severely imbalanced data
            optimal_threshold = np.mean(y_binary_test) * 2  # 2x the base rate
            optimal_threshold = max(0.1, min(optimal_threshold, 0.5))  # Keep between 0.1 and 0.5
        
        # Use optimal threshold for binary predictions instead of 0.5
        payer_predictions = (payer_probs >= optimal_threshold).astype(int)
        
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
        
        # Use already calculated precision-recall curve values
        avg_precision = np.mean(precision_curve)
        avg_recall = np.mean(recall_curve)
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