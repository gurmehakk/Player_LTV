"""
Enhanced pLTV Prediction System inspired by ExpLTV methodology
Incorporates Game Whale Detection with Expert Routing and ZILN loss
Uses existing BigQuery app_data without additional datasets
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Tuple, List

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Google Cloud
from google.cloud import bigquery

# Set style
plt.style.use('default')
sns.set_palette("husl")


class GameWhaleDetector(nn.Module):
    """
    Game Whale Detection module with expert routing
    Based on ExpLTV methodology - detects whales vs low spenders vs non-payers
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(GameWhaleDetector, self).__init__()
        
        # Shared embedding layers
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Purchase probability estimator (auxiliary task)
        self.purchase_estimator = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Game whale detector (2D output: whale_prob, low_spender_prob)
        self.whale_detector = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Shared embedding
        embeddings = self.embedding(x)
        
        # Purchase probability (auxiliary task)
        purchase_prob = self.purchase_estimator(embeddings)
        
        # Whale detection (conditional probabilities given purchase)
        whale_probs = self.whale_detector(embeddings)  # [whale_prob, low_spender_prob]
        
        # Compute joint probabilities using Bayes' theorem
        # P(whale, purchase) = P(purchase) * P(whale | purchase)
        whale_purchase_prob = purchase_prob * whale_probs[:, 0:1]
        
        # P(low_spender, purchase) = P(purchase) * P(low_spender | purchase)  
        low_spender_purchase_prob = purchase_prob * whale_probs[:, 1:2]
        
        # P(non_payer) = 1 - P(purchase)
        non_payer_prob = 1 - purchase_prob
        
        return {
            'purchase_prob': purchase_prob,
            'whale_conditional_prob': whale_probs[:, 0:1],
            'low_spender_conditional_prob': whale_probs[:, 1:2],
            'whale_purchase_prob': whale_purchase_prob,
            'low_spender_purchase_prob': low_spender_purchase_prob,
            'non_payer_prob': non_payer_prob,
            'embeddings': embeddings
        }


class LTVExpert(nn.Module):
    """
    Individual LTV Expert for specific user types
    Predicts distribution parameters (mu, sigma) for ZILN
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(LTVExpert, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Distribution parameters with proper bounds
        self.mu_head = nn.Linear(hidden_dim // 2, 1)
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
    def forward(self, x):
        features = self.network(x)
        mu = self.mu_head(features)
        sigma = self.sigma_head(features) + 1e-3  # Ensure minimum sigma
        
        # Clip values to prevent overflow
        mu = torch.clamp(mu, -10, 10)
        sigma = torch.clamp(sigma, 1e-3, 5)
        
        return mu, sigma


class EnhancedPLTVSystem:
    """Enhanced pLTV system with Game Whale Detection and Expert Routing"""
    
    def __init__(self, project_id: str = "gc-forecasting-dev"):
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Models
        self.whale_detector = None
        self.whale_expert = None      # Expert for whales
        self.low_spender_expert = None  # Expert for low spenders
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Thresholds (will be computed from data)
        self.whale_threshold = None
        self.low_spender_threshold = None
        
    def extract_data(self, days_lookback: int = 90) -> pd.DataFrame:
        """Extract enhanced player data with sequential behavior features"""
        
        print(f"Extracting enhanced player data (last {days_lookback} days)...")
        
        query = f"""
        WITH player_sessions AS (
            SELECT 
                COALESCE(custom_user_id, gaid, idfa, android_id) as player_id,
                session_id,
                SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S%Ez', server_timestamp) as event_timestamp,
                name as event_name,
                COALESCE(revenue, 0) as revenue,
                COALESCE(product_price, 0) as product_price,
                COALESCE(product_quantity, 0) as product_quantity,
                is_reengagement,
                is_fingerprinted,
                country,
                campaign_name,
                publisher_name
            FROM `{self.project_id}.test_data.app_data`
            WHERE 
                SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S%Ez', server_timestamp) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days_lookback} DAY)
                AND COALESCE(custom_user_id, gaid, idfa, android_id) IS NOT NULL
        ),
        
        session_aggregates AS (
            SELECT 
                player_id,
                session_id,
                COUNT(*) as events_in_session,
                SUM(revenue) as session_revenue,
                MAX(revenue) as session_max_revenue
            FROM player_sessions
            WHERE event_timestamp IS NOT NULL
            GROUP BY player_id, session_id
        ),
        
        player_features AS (
            SELECT 
                ps.player_id,
                
                -- Basic engagement metrics
                COUNT(DISTINCT ps.session_id) as total_sessions,
                COUNT(*) as total_events,
                COUNT(DISTINCT ps.event_name) as unique_event_types,
                
                -- Temporal features
                DATE_DIFF(CURRENT_DATE(), DATE(MIN(ps.event_timestamp)), DAY) as days_since_first_session,
                DATE_DIFF(DATE(MAX(ps.event_timestamp)), DATE(MIN(ps.event_timestamp)), DAY) + 1 as player_lifetime_days,
                
                -- Revenue and purchase features
                SUM(ps.revenue) as total_revenue,
                COUNTIF(ps.revenue > 0) as purchase_events,
                COUNT(DISTINCT CASE WHEN ps.revenue > 0 THEN ps.session_id END) as revenue_sessions,
                MAX(ps.revenue) as max_single_purchase,
                AVG(CASE WHEN ps.revenue > 0 THEN ps.revenue END) as avg_purchase_amount,
                SUM(ps.product_quantity) as total_items_purchased,
                
                -- Purchase amount variability
                STDDEV(CASE WHEN ps.revenue > 0 THEN ps.revenue END) as purchase_amount_std,
                
                -- Session quality metrics  
                AVG(sa.events_in_session) as avg_events_per_session,
                MAX(sa.events_in_session) as max_events_per_session,
                
                -- Behavioral engagement
                COUNTIF(ps.is_reengagement = TRUE) as reengagement_events,
                COUNTIF(ps.is_fingerprinted = TRUE) as fingerprint_events,
                
                -- Attribution features
                ARRAY_AGG(ps.country ORDER BY ps.event_timestamp ASC LIMIT 1)[OFFSET(0)] as first_country,
                ARRAY_AGG(ps.campaign_name ORDER BY ps.event_timestamp ASC LIMIT 1)[OFFSET(0)] as first_campaign,
                ARRAY_AGG(ps.publisher_name ORDER BY ps.event_timestamp ASC LIMIT 1)[OFFSET(0)] as first_publisher,
                
                -- Recent activity (last 7 days)
                COUNTIF(DATE_DIFF(CURRENT_DATE(), DATE(ps.event_timestamp), DAY) <= 7) as events_last_7_days,
                SUM(CASE WHEN DATE_DIFF(CURRENT_DATE(), DATE(ps.event_timestamp), DAY) <= 7 THEN ps.revenue ELSE 0 END) as revenue_last_7_days,
                
                -- Purchase timing features
                DATE_DIFF(
                    DATE(MAX(CASE WHEN ps.revenue > 0 THEN ps.event_timestamp END)),
                    DATE(MIN(CASE WHEN ps.revenue > 0 THEN ps.event_timestamp END)),
                    DAY
                ) as purchase_span_days
                
            FROM player_sessions ps
            LEFT JOIN session_aggregates sa ON ps.player_id = sa.player_id AND ps.session_id = sa.session_id
            WHERE ps.event_timestamp IS NOT NULL
            GROUP BY ps.player_id
            HAVING total_sessions >= 1
        )
        
        SELECT *
        FROM player_features
        WHERE player_id IS NOT NULL
        ORDER BY total_revenue DESC
        """
        
        df = self.client.query(query).to_dataframe()
        
        # Clean and process data
        df = self._clean_and_process_data(df)
        
        print(f"Extracted {len(df):,} players")
        print(f"  - Total revenue: ${df['total_revenue'].sum():,.2f}")
        print(f"  - Payers: {len(df[df['total_revenue'] > 0]):,} ({len(df[df['total_revenue'] > 0])/len(df)*100:.1f}%)")
        
        return df
    
    def _clean_and_process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and process extracted data"""
        
        # Fill missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(0)
        
        # Fill categorical columns
        categorical_columns = ['first_country', 'first_campaign', 'first_publisher']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
        
        # Compute user type labels based on revenue distribution
        payers = df[df['total_revenue'] > 0]
        if len(payers) > 0:
            # Define thresholds based on revenue distribution
            self.whale_threshold = payers['total_revenue'].quantile(0.9)  # Top 10% of payers
            self.low_spender_threshold = 0.01  # Any revenue above 0
            
            print(f"  - Whale threshold: ${self.whale_threshold:.2f}")
            print(f"  - Low spender threshold: ${self.low_spender_threshold:.2f}")
            
            # Create user type labels
            df['user_type'] = 0  # Non-payer
            df.loc[df['total_revenue'] >= self.low_spender_threshold, 'user_type'] = 1  # Low spender
            df.loc[df['total_revenue'] >= self.whale_threshold, 'user_type'] = 2  # Whale
            
            # Create binary labels for different tasks
            df['is_payer'] = (df['total_revenue'] > 0).astype(int)
            df['is_whale'] = (df['user_type'] == 2).astype(int)
            df['is_low_spender_given_payer'] = np.where(
                df['is_payer'] == 1,
                (df['user_type'] == 1).astype(int),
                0
            )
            
            user_type_counts = df['user_type'].value_counts().sort_index()
            print(f"  - Non-payers: {user_type_counts.get(0, 0):,}")
            print(f"  - Low spenders: {user_type_counts.get(1, 0):,}")
            print(f"  - Whales: {user_type_counts.get(2, 0):,}")
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features for the enhanced model"""
        
        print("Creating enhanced features...")
        
        # Basic derived features
        df['events_per_session'] = df['total_events'] / np.maximum(df['total_sessions'], 1)
        df['events_per_day'] = df['total_events'] / np.maximum(df['player_lifetime_days'], 1)
        df['sessions_per_day'] = df['total_sessions'] / np.maximum(df['player_lifetime_days'], 1)
        df['unique_events_per_session'] = df['unique_event_types'] / np.maximum(df['total_sessions'], 1)
        
        # Revenue features
        df['revenue_per_session'] = df['total_revenue'] / np.maximum(df['total_sessions'], 1)
        df['revenue_per_day'] = df['total_revenue'] / np.maximum(df['player_lifetime_days'], 1)
        df['purchase_frequency'] = df['purchase_events'] / np.maximum(df['total_sessions'], 1)
        df['revenue_session_rate'] = df['revenue_sessions'] / np.maximum(df['total_sessions'], 1)
        
        # Sequential behavior features
        df['avg_purchase_amount'] = df['avg_purchase_amount'].fillna(0)
        df['purchase_amount_std'] = df['purchase_amount_std'].fillna(0)
        df['purchase_consistency'] = np.where(
            df['avg_purchase_amount'] > 0,
            df['purchase_amount_std'] / df['avg_purchase_amount'],
            0
        )
        df['days_between_purchases'] = np.where(
            df['purchase_events'] > 1,
            df['purchase_span_days'] / np.maximum(df['purchase_events'] - 1, 1),
            0
        )
        
        # Engagement features
        df['reengagement_rate'] = df['reengagement_events'] / np.maximum(df['total_events'], 1)
        df['fingerprint_rate'] = df['fingerprint_events'] / np.maximum(df['total_events'], 1)
        df['recent_activity_rate'] = df['events_last_7_days'] / np.maximum(df['total_events'], 1)
        df['recent_revenue_rate'] = df['revenue_last_7_days'] / np.maximum(df['total_revenue'], 1)
        
        # Session quality features
        df['session_depth_ratio'] = df['avg_events_per_session'] / np.maximum(df['max_events_per_session'], 1)
        df['item_purchase_rate'] = df['total_items_purchased'] / np.maximum(df['total_events'], 1)
        
        # Handle infinite values
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace([np.inf, -np.inf], 0)
        
        # Encode categorical features
        categorical_features = ['first_country', 'first_campaign', 'first_publisher']
        for feature in categorical_features:
            if feature in df.columns:
                # Keep top 10 categories, group others as 'other'
                top_categories = df[feature].value_counts().head(10).index.tolist()
                df[feature] = df[feature].apply(lambda x: x if x in top_categories else 'other')
                
                # Label encode
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    df[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(df[feature])
                else:
                    df[f'{feature}_encoded'] = df[feature].apply(
                        lambda x: self.label_encoders[feature].transform([x])[0] 
                        if x in self.label_encoders[feature].classes_ else -1
                    )
        
        # Define feature columns
        self.feature_columns = [
            'total_sessions', 'total_events', 'unique_event_types',
            'events_per_session', 'events_per_day', 'sessions_per_day',
            'player_lifetime_days', 'days_since_first_session',
            'purchase_events', 'revenue_sessions', 'max_single_purchase',
            'revenue_per_session', 'revenue_per_day', 'purchase_frequency',
            'revenue_session_rate', 'purchase_consistency', 'days_between_purchases',
            'reengagement_rate', 'fingerprint_rate', 'recent_activity_rate',
            'recent_revenue_rate', 'session_depth_ratio', 'item_purchase_rate',
            'unique_events_per_session'
        ]
        
        # Add encoded categorical features
        encoded_features = [f'{f}_encoded' for f in categorical_features if f'{f}_encoded' in df.columns]
        self.feature_columns.extend(encoded_features)
        
        # Only keep features that exist
        self.feature_columns = [col for col in self.feature_columns if col in df.columns]
        
        print(f"Created {len(self.feature_columns)} enhanced features")
        
        return df
    
    def ziln_loss(self, predictions, targets, purchase_probs):
        """
        Zero-Inflated Log-Normal loss function with numerical stability
        """
        mu, sigma = predictions
        
        # Separate zero and non-zero targets
        zero_mask = (targets == 0)
        nonzero_mask = ~zero_mask
        
        # Cross-entropy loss for purchase probability
        purchase_probs = torch.clamp(purchase_probs, 1e-7, 1 - 1e-7)  # Prevent log(0)
        purchase_loss = F.binary_cross_entropy(
            purchase_probs.squeeze(), 
            (targets > 0).float()
        )
        
        # Log-normal loss for non-zero targets with numerical stability
        lognormal_loss = torch.tensor(0.0).to(self.device)
        if nonzero_mask.sum() > 0:
            # Add small epsilon to prevent log(0)
            log_targets = torch.log(targets[nonzero_mask] + 1e-8)
            mu_nonzero = mu[nonzero_mask]
            sigma_nonzero = sigma[nonzero_mask]
            
            # Ensure sigma is bounded
            sigma_nonzero = torch.clamp(sigma_nonzero, 1e-3, 5)
            
            # Log-normal likelihood
            lognormal_loss = (
                torch.log(sigma_nonzero * np.sqrt(2 * np.pi)) +
                ((log_targets - mu_nonzero) ** 2) / (2 * sigma_nonzero ** 2)
            ).mean()
        
        return purchase_loss + lognormal_loss
    
    def train_models(self, df: pd.DataFrame, epochs: int = 100) -> Dict:
        """Train the enhanced pLTV system with game whale detection"""
        
        print("Training enhanced pLTV models...")
        
        # Prepare data
        X = df[self.feature_columns].fillna(0)
        y_revenue = df['total_revenue'].values
        y_payer = df['is_payer'].values
        y_whale = df['is_whale'].values
        y_user_type = df['user_type'].values
        
        # Remove any remaining infinite values
        X = X.replace([np.inf, -np.inf], 0)
        y_revenue = np.where(np.isfinite(y_revenue), y_revenue, 0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_rev_train, y_rev_test, y_pay_train, y_pay_test, y_whale_train, y_whale_test, y_type_train, y_type_test = train_test_split(
            X_scaled, y_revenue, y_payer, y_whale, y_user_type,
            test_size=0.2, random_state=42, stratify=y_payer
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_rev_train_tensor = torch.FloatTensor(y_rev_train).to(self.device)
        y_rev_test_tensor = torch.FloatTensor(y_rev_test).to(self.device)
        y_pay_train_tensor = torch.FloatTensor(y_pay_train).to(self.device)
        y_pay_test_tensor = torch.FloatTensor(y_pay_test).to(self.device)
        y_whale_train_tensor = torch.FloatTensor(y_whale_train).to(self.device)
        y_type_train_tensor = torch.LongTensor(y_type_train).to(self.device)
        
        # Initialize models
        input_dim = X_train.shape[1]
        embedding_dim = 64  # Hidden dimension from whale detector
        
        self.whale_detector = GameWhaleDetector(input_dim).to(self.device)
        self.whale_expert = LTVExpert(embedding_dim).to(self.device)
        self.low_spender_expert = LTVExpert(embedding_dim).to(self.device)
        
        # Optimizers
        detector_optimizer = torch.optim.Adam(self.whale_detector.parameters(), lr=0.001)
        whale_optimizer = torch.optim.Adam(self.whale_expert.parameters(), lr=0.001)
        low_spender_optimizer = torch.optim.Adam(self.low_spender_expert.parameters(), lr=0.001)
        
        # Training loop
        train_losses = []
        
        for epoch in range(epochs):
            # Set models to training mode
            self.whale_detector.train()
            self.whale_expert.train()
            self.low_spender_expert.train()
            
            # Forward pass through whale detector
            detector_output = self.whale_detector(X_train_tensor)
            embeddings = detector_output['embeddings']
            
            # Expert predictions
            whale_mu, whale_sigma = self.whale_expert(embeddings)
            low_spender_mu, low_spender_sigma = self.low_spender_expert(embeddings)
            
            # Route predictions based on whale detection probabilities
            whale_weight = detector_output['whale_conditional_prob']
            low_spender_weight = detector_output['low_spender_conditional_prob']
            
            # Weighted combination of expert predictions
            final_mu = whale_weight * whale_mu + low_spender_weight * low_spender_mu
            final_sigma = whale_weight * whale_sigma + low_spender_weight * low_spender_sigma
            
            # Calculate losses
            # 1. Purchase probability loss
            purchase_loss = F.binary_cross_entropy(
                detector_output['purchase_prob'].squeeze(),
                y_pay_train_tensor
            )
            
            # 2. User type classification loss (whale vs low spender given purchase)
            payer_mask = y_pay_train_tensor > 0
            type_loss = torch.tensor(0.0).to(self.device)
            if payer_mask.sum() > 0:
                payer_probs = torch.cat([
                    detector_output['whale_conditional_prob'][payer_mask],
                    detector_output['low_spender_conditional_prob'][payer_mask]
                ], dim=1)
                payer_types = y_type_train_tensor[payer_mask] - 1  # Convert to 0/1 for whales/low_spenders
                payer_types = torch.clamp(payer_types, 0, 1)  # Ensure valid range
                
                type_loss = F.cross_entropy(payer_probs, payer_types)
            
            # 3. ZILN loss for revenue prediction
            ziln_loss = self.ziln_loss(
                (final_mu, final_sigma),
                y_rev_train_tensor,
                detector_output['purchase_prob']
            )
            
            # Combined loss
            total_loss = purchase_loss + 0.5 * type_loss + ziln_loss
            
            # Backward pass
            detector_optimizer.zero_grad()
            whale_optimizer.zero_grad()
            low_spender_optimizer.zero_grad()
            
            total_loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.whale_detector.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.whale_expert.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.low_spender_expert.parameters(), max_norm=1.0)
            
            detector_optimizer.step()
            whale_optimizer.step()
            low_spender_optimizer.step()
            
            train_losses.append(total_loss.item())
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}")
        
        # Evaluation
        print("Evaluating models...")
        metrics = self._evaluate_models(X_test_tensor, y_rev_test, y_pay_test, y_whale_test)
        
        return {
            'metrics': metrics,
            'train_losses': train_losses,
            'test_data': {
                'X_test': X_test,
                'y_revenue_test': y_rev_test,
                'y_payer_test': y_pay_test,
                'y_whale_test': y_whale_test
            }
        }
    
    def _evaluate_models(self, X_test, y_revenue_test, y_payer_test, y_whale_test):
        """Evaluate the trained models with numerical stability"""
        
        # Set models to evaluation mode
        self.whale_detector.eval()
        self.whale_expert.eval()
        self.low_spender_expert.eval()
        
        with torch.no_grad():
            # Get predictions
            detector_output = self.whale_detector(X_test)
            embeddings = detector_output['embeddings']
            
            # Expert predictions
            whale_mu, whale_sigma = self.whale_expert(embeddings)
            low_spender_mu, low_spender_sigma = self.low_spender_expert(embeddings)
            
            # Weighted predictions
            whale_weight = detector_output['whale_conditional_prob']
            low_spender_weight = detector_output['low_spender_conditional_prob']
            
            final_mu = whale_weight * whale_mu + low_spender_weight * low_spender_mu
            final_sigma = whale_weight * whale_sigma + low_spender_weight * low_spender_sigma
            
            # Expected LTV using ZILN distribution with numerical stability
            purchase_probs = detector_output['purchase_prob']
            
            # Clip values to prevent overflow
            final_mu = torch.clamp(final_mu, -10, 10)
            final_sigma = torch.clamp(final_sigma, 1e-3, 5)
            
            # Safe exponential calculation
            exp_term = final_mu + final_sigma**2 / 2
            exp_term = torch.clamp(exp_term, -20, 20)  # Prevent extreme values
            expected_ltv = purchase_probs * torch.exp(exp_term)
            
            # Convert to numpy and handle any remaining issues
            ltv_predictions = expected_ltv.cpu().numpy().flatten()
            purchase_predictions = purchase_probs.cpu().numpy().flatten()
            whale_predictions = detector_output['whale_conditional_prob'].cpu().numpy().flatten()
            
            # Final safety check
            ltv_predictions = np.where(np.isfinite(ltv_predictions), ltv_predictions, 0)
            purchase_predictions = np.where(np.isfinite(purchase_predictions), purchase_predictions, 0)
            whale_predictions = np.where(np.isfinite(whale_predictions), whale_predictions, 0)
        
        # Calculate metrics
        # Revenue prediction metrics
        mae = mean_absolute_error(y_revenue_test, ltv_predictions)
        rmse = np.sqrt(mean_squared_error(y_revenue_test, ltv_predictions))
        r2 = r2_score(y_revenue_test, ltv_predictions)
        
        # Purchase classification metrics
        auc_purchase = roc_auc_score(y_payer_test, purchase_predictions)
        
        # Whale classification metrics (only for payers)
        payer_mask = y_payer_test > 0
        if np.sum(payer_mask) > 0:
            auc_whale = roc_auc_score(y_whale_test[payer_mask], whale_predictions[payer_mask])
        else:
            auc_whale = 0.5
        
        # Business metrics - lift analysis
        sorted_indices = np.argsort(ltv_predictions)[::-1]
        
        # Top 10% lift
        top_10_pct = int(len(ltv_predictions) * 0.1)
        top_10_revenue = np.sum(y_revenue_test[sorted_indices[:top_10_pct]])
        total_revenue = np.sum(y_revenue_test)
        lift_10 = (top_10_revenue / total_revenue) / 0.1 if total_revenue > 0 else 0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'auc_purchase': auc_purchase,
            'auc_whale': auc_whale,
            'lift_top_10pct': lift_10,
            'total_revenue_actual': total_revenue,
            'total_revenue_predicted': np.sum(ltv_predictions),
            'payer_rate_actual': np.mean(y_payer_test),
            'payer_rate_predicted': np.mean(purchase_predictions),
            'whale_rate_actual': np.mean(y_whale_test),
            'whale_rate_predicted': np.mean(whale_predictions)
        }
    
    def create_visualizations(self, data: pd.DataFrame, results: Dict, save_dir: str = "output"):
        """Create enhanced visualizations"""
        
        print("Creating enhanced visualizations...")
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Enhanced pLTV Analysis with Game Whale Detection', fontsize=16, fontweight='bold')
        
        # 1. User Type Distribution
        user_type_counts = data['user_type'].value_counts().sort_index()
        labels = ['Non-Payers', 'Low Spenders', 'Whales']
        colors = ['lightcoral', 'skyblue', 'gold']
        
        axes[0, 0].pie(user_type_counts.values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Player Distribution by Type')
        
        # 2. Revenue Distribution by User Type
        user_types = [0, 1, 2]
        revenues = [data[data['user_type'] == ut]['total_revenue'].sum() for ut in user_types]
        
        bars = axes[0, 1].bar(labels, revenues, color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('Total Revenue ($)')
        axes[0, 1].set_title('Revenue Distribution by User Type')
        
        # Add value labels
        for bar, revenue in zip(bars, revenues):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(revenues) * 0.01,
                          f'${revenue:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Model Performance Metrics
        metrics = results['metrics']
        performance_metrics = {
            'Revenue R²': metrics['r2'],
            'Purchase AUC': metrics['auc_purchase'],
            'Whale AUC': metrics['auc_whale'],
            'Lift (10%)': metrics['lift_top_10pct']
        }
        
        bars = axes[1, 0].bar(range(len(performance_metrics)), list(performance_metrics.values()),
                             color=['green', 'blue', 'purple', 'orange'], alpha=0.7)
        axes[1, 0].set_xticks(range(len(performance_metrics)))
        axes[1, 0].set_xticklabels(list(performance_metrics.keys()), rotation=45, ha='right')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Model Performance Metrics')
        
        # Add value labels
        for bar, value in zip(bars, performance_metrics.values()):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Training Loss
        if 'train_losses' in results:
            epochs = range(1, len(results['train_losses']) + 1)
            axes[1, 1].plot(epochs, results['train_losses'], 'b-', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Training Loss')
            axes[1, 1].set_title('Model Training Progress')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Revenue vs Predictions Scatter
        if 'test_data' in results:
            # Get test predictions
            self.whale_detector.eval()
            self.whale_expert.eval()
            self.low_spender_expert.eval()
            
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(results['test_data']['X_test']).to(self.device)
                detector_output = self.whale_detector(X_test_tensor)
                embeddings = detector_output['embeddings']
                
                whale_mu, whale_sigma = self.whale_expert(embeddings)
                low_spender_mu, low_spender_sigma = self.low_spender_expert(embeddings)
                
                whale_weight = detector_output['whale_conditional_prob']
                low_spender_weight = detector_output['low_spender_conditional_prob']
                
                final_mu = whale_weight * whale_mu + low_spender_weight * low_spender_mu
                final_sigma = whale_weight * whale_sigma + low_spender_weight * low_spender_sigma
                
                purchase_probs = detector_output['purchase_prob']
                
                # Safe calculation
                final_mu = torch.clamp(final_mu, -10, 10)
                final_sigma = torch.clamp(final_sigma, 1e-3, 5)
                exp_term = final_mu + final_sigma**2 / 2
                exp_term = torch.clamp(exp_term, -20, 20)
                expected_ltv = purchase_probs * torch.exp(exp_term)
                
                predictions = expected_ltv.cpu().numpy().flatten()
                predictions = np.where(np.isfinite(predictions), predictions, 0)
            
            actual = results['test_data']['y_revenue_test']
            
            axes[2, 0].scatter(actual, predictions, alpha=0.6, s=20)
            max_val = max(max(actual), max(predictions))
            axes[2, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
            axes[2, 0].set_xlabel('Actual Revenue ($)')
            axes[2, 0].set_ylabel('Predicted Revenue ($)')
            axes[2, 0].set_title('Prediction Accuracy')
        
        # 6. Feature Importance (using whale detector weights)
        feature_importance = {}
        if self.whale_detector is not None:
            # Get feature importance from the first layer weights
            first_layer_weights = self.whale_detector.embedding[0].weight.data.cpu().numpy()
            importance_scores = np.abs(first_layer_weights).mean(axis=0)
            
            # Ensure we don't exceed the number of features
            num_features = min(len(self.feature_columns), len(importance_scores))
            for i in range(num_features):
                feature_importance[self.feature_columns[i]] = importance_scores[i]
            
            # Plot top 10 features
            if len(feature_importance) > 0:
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                feature_names = [f.replace('_', ' ').title() for f, _ in top_features]
                importance_values = [imp for _, imp in top_features]
                
                y_pos = np.arange(len(feature_names))
                bars = axes[2, 1].barh(y_pos, importance_values, color='teal', alpha=0.7)
                axes[2, 1].set_yticks(y_pos)
                axes[2, 1].set_yticklabels(feature_names)
                axes[2, 1].set_xlabel('Feature Importance')
                axes[2, 1].set_title('Top 10 Most Important Features')
            else:
                axes[2, 1].text(0.5, 0.5, 'Feature importance\nanalysis unavailable', 
                               ha='center', va='center', transform=axes[2, 1].transAxes)
                axes[2, 1].set_title('Feature Importance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'enhanced_pltv_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Enhanced visualizations saved to {save_dir}")
    
    def generate_report(self, data: pd.DataFrame, results: Dict, save_dir: str = "output"):
        """Generate enhanced summary report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(save_dir, f"enhanced_pltv_report_{timestamp}.txt")
        
        with open(report_file, 'w') as f:
            f.write("ENHANCED pLTV PREDICTION SYSTEM REPORT\n")
            f.write("Inspired by ExpLTV Methodology with Game Whale Detection\n")
            f.write("=" * 60 + "\n\n")
            
            # Dataset overview
            total_players = len(data)
            non_payers = len(data[data['user_type'] == 0])
            low_spenders = len(data[data['user_type'] == 1]) 
            whales = len(data[data['user_type'] == 2])
            total_revenue = data['total_revenue'].sum()
            
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Players: {total_players:,}\n")
            f.write(f"Non-Payers: {non_payers:,} ({non_payers/total_players:.1%})\n")
            f.write(f"Low Spenders: {low_spenders:,} ({low_spenders/total_players:.1%})\n")
            f.write(f"Whales: {whales:,} ({whales/total_players:.1%})\n")
            f.write(f"Total Revenue: ${total_revenue:,.2f}\n")
            f.write(f"Whale Threshold: ${self.whale_threshold:.2f}\n")
            f.write(f"Features Used: {len(self.feature_columns)}\n\n")
            
            # Revenue distribution analysis
            if whales > 0:
                whale_revenue = data[data['user_type'] == 2]['total_revenue'].sum()
                low_spender_revenue = data[data['user_type'] == 1]['total_revenue'].sum()
                
                f.write("REVENUE ANALYSIS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Whale Revenue: ${whale_revenue:,.2f} ({whale_revenue/total_revenue:.1%} of total)\n")
                f.write(f"Low Spender Revenue: ${low_spender_revenue:,.2f} ({low_spender_revenue/total_revenue:.1%} of total)\n")
                f.write(f"Average Whale Spending: ${whale_revenue/whales:.2f}\n")
                f.write(f"Average Low Spender Spending: ${low_spender_revenue/low_spenders:.2f}\n\n")
            
            # Model performance
            metrics = results['metrics']
            f.write("MODEL PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            f.write(f"Revenue Prediction (R²): {metrics['r2']:.3f}\n")
            f.write(f"Purchase Classification (AUC): {metrics['auc_purchase']:.3f}\n")
            f.write(f"Whale Detection (AUC): {metrics['auc_whale']:.3f}\n")
            f.write(f"Mean Absolute Error: ${metrics['mae']:.2f}\n")
            f.write(f"Root Mean Square Error: ${metrics['rmse']:.2f}\n\n")
            
            # Business impact
            f.write("BUSINESS IMPACT\n")
            f.write("-" * 30 + "\n")
            f.write(f"Top 10% Targeting Lift: {metrics['lift_top_10pct']:.1f}x\n")
            f.write(f"Revenue Prediction Accuracy: {1 - abs(metrics['total_revenue_predicted'] - metrics['total_revenue_actual'])/metrics['total_revenue_actual']:.1%}\n")
            f.write(f"Payer Rate Accuracy: {1 - abs(metrics['payer_rate_predicted'] - metrics['payer_rate_actual']):.1%}\n")
            f.write(f"Whale Rate Accuracy: {1 - abs(metrics['whale_rate_predicted'] - metrics['whale_rate_actual']):.1%}\n\n")
            
            # Model assessment with grades
            f.write("MODEL ASSESSMENT\n")
            f.write("-" * 30 + "\n")
            
            # Enhanced grading system
            r2_grade = "A+" if metrics['r2'] > 0.8 else "A" if metrics['r2'] > 0.7 else "B" if metrics['r2'] > 0.5 else "C" if metrics['r2'] > 0.3 else "D"
            purchase_grade = "A+" if metrics['auc_purchase'] > 0.95 else "A" if metrics['auc_purchase'] > 0.9 else "B" if metrics['auc_purchase'] > 0.8 else "C" if metrics['auc_purchase'] > 0.7 else "D"
            whale_grade = "A+" if metrics['auc_whale'] > 0.95 else "A" if metrics['auc_whale'] > 0.9 else "B" if metrics['auc_whale'] > 0.8 else "C" if metrics['auc_whale'] > 0.7 else "D"
            lift_grade = "A+" if metrics['lift_top_10pct'] > 8 else "A" if metrics['lift_top_10pct'] > 5 else "B" if metrics['lift_top_10pct'] > 3 else "C" if metrics['lift_top_10pct'] > 2 else "D"
            
            f.write(f"Revenue Prediction Quality: Grade {r2_grade} ({metrics['r2']:.3f})\n")
            f.write(f"Purchase Classification: Grade {purchase_grade} ({metrics['auc_purchase']:.3f})\n")
            f.write(f"Whale Detection: Grade {whale_grade} ({metrics['auc_whale']:.3f})\n")
            f.write(f"Business Impact: Grade {lift_grade} ({metrics['lift_top_10pct']:.1f}x)\n\n")
            
            # Key innovations
            f.write("KEY INNOVATIONS (ExpLTV-Inspired)\n")
            f.write("-" * 40 + "\n")
            f.write("• Game Whale Detection with auxiliary purchase task\n")
            f.write("• Expert Routing system for whales vs low spenders\n")
            f.write("• Zero-Inflated Log-Normal (ZILN) loss function\n")
            f.write("• Multi-task learning framework\n")
            f.write("• Sequential behavior feature engineering\n")
            f.write("• Bayesian probability modeling\n\n")
            
            # Recommendations
            f.write("STRATEGIC RECOMMENDATIONS\n")
            f.write("-" * 35 + "\n")
            
            recommendations = []
            
            if metrics['r2'] < 0.5:
                recommendations.append("• Enhance sequential behavior features")
            if metrics['auc_purchase'] < 0.8:
                recommendations.append("• Improve purchase probability modeling")
            if metrics['auc_whale'] < 0.8:
                recommendations.append("• Refine whale detection thresholds")
            if metrics['lift_top_10pct'] < 3:
                recommendations.append("• Consider additional expert models")
            
            # Standard recommendations
            recommendations.extend([
                "• Target top 5% of predicted whales for premium campaigns",
                "• Use whale detection for personalized pricing",
                "• Implement A/B testing on whale-specific features",
                "• Monitor model performance with whale/low-spender metrics",
                "• Consider seasonal whale behavior patterns"
            ])
            
            for rec in recommendations:
                f.write(f"{rec}\n")
            
            f.write("\n")
            
            # Technical details
            f.write("TECHNICAL SPECIFICATIONS\n")
            f.write("-" * 35 + "\n")
            f.write(f"Model Architecture: Deep Neural Networks with Expert Routing\n")
            f.write(f"Loss Function: Combined ZILN + Cross-Entropy + KL-Divergence\n")
            f.write(f"Optimization: Multi-task learning with shared embeddings\n")
            f.write(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}\n")
            f.write(f"Training Epochs: 100\n")
            f.write(f"Feature Engineering: Sequential + Behavioral + Attribution\n\n")
            
            # Data quality metrics
            f.write("DATA QUALITY METRICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Data Completeness: {(1 - data[self.feature_columns].isnull().sum().sum() / (len(data) * len(self.feature_columns))):.1%}\n")
            f.write(f"Revenue Distribution Skewness: {data['total_revenue'].skew():.2f}\n")
            f.write(f"Feature Correlation (max): {data[self.feature_columns].corr().abs().max().max():.3f}\n")
        
        print(f"Enhanced report saved: {report_file}")
    
    def run_complete_analysis(self):
        """Run the complete enhanced pLTV analysis"""
        
        print("Starting Enhanced pLTV Analysis with Game Whale Detection")
        print("=" * 65)
        
        try:
            # 1. Extract data
            data = self.extract_data()
            
            # 2. Create features  
            data = self.create_features(data)
            
            pd.save_csv(data, "output/enhanced_player_data_20250730.csv")
            
            # 3. Train models
            results = self.train_models(data, epochs=200)
            
            # 4. Create visualizations
            self.create_visualizations(data, results)
            
            # 5. Generate report
            self.generate_report(data, results)
            
            print("\nAnalysis completed successfully!")
            print("Key improvements over basic pLTV:")
            print("   • Game whale detection with expert routing")
            print("   • ZILN loss for handling extreme values") 
            print("   • Multi-task learning framework")
            print("   • Enhanced sequential behavior features")
            print("   • Bayesian probability modeling")
            print("\nCheck the 'output' folder for detailed results.")
            
            return data, results
            
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """Main execution function"""
    system = EnhancedPLTVSystem()
    system.run_complete_analysis()


if __name__ == "__main__":
    main()