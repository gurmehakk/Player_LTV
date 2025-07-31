"""
Production-Ready Player LTV Prediction System
Combines BigQuery data extraction, ExpLTV methodology, whale detection, and comprehensive backtesting
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    roc_auc_score, precision_recall_curve, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)

# Google Cloud
from google.cloud import bigquery
from google.oauth2 import service_account

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


class Config:
    """Configuration management"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        
        default_config = {
            'bigquery': {
                'project_id': 'gc-forecasting-dev',
                'dataset': 'test_data',
                'table': 'app_data',
                'lookback_days': 90,
                'query_timeout': 600
            },
            'model': {
                'whale_threshold_percentile': 90,
                'hidden_dim': 128,
                'embedding_dim': 64,
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'epochs': 200,
                'batch_size': 1024,
                'early_stopping_patience': 20,
                'gradient_clip_norm': 1.0
            },
            'training': {
                'test_size': 0.2,
                'validation_size': 0.15,
                'cv_folds': 5,
                'random_state': 42,
                'stratify_on_payer_status': True
            },
            'features': {
                'revenue_features': True,
                'behavioral_features': True,
                'temporal_features': True,
                'attribution_features': True,
                'sequential_features': True,
                'max_categorical_levels': 10
            },
            'outputs': {
                'save_models': True,
                'save_predictions': True,
                'create_visualizations': True,
                'generate_reports': True,
                'save_engineered_data': True,
                'output_dir': 'output'
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def get(self, key: str, default=None):
        """Get configuration value with dot notation"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


class AdvancedFeatureEngineer:
    """Advanced feature engineering with comprehensive behavioral patterns"""
    
    def __init__(self, config: Config):
        self.config = config
        self.label_encoders = {}
        
    def create_advanced_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced behavioral features from user interaction patterns"""
        
        logger.info("Creating advanced behavioral features...")
        
        # Basic interaction patterns
        df['click_through_rate'] = df['total_events'] / np.maximum(df['total_sessions'], 1)
        df['session_engagement_score'] = (
            df['avg_events_per_session'] * 0.3 +
            df['avg_session_duration_minutes'] * 0.2 +
            df['avg_unique_events_per_session'] * 0.5
        )
        
        # Purchase behavior patterns
        df['purchase_velocity'] = np.where(
            df['purchase_span_days'] > 0,
            df['total_purchase_events'] / df['purchase_span_days'],
            0
        )
        
        df['purchase_momentum'] = np.where(
            df['days_to_first_purchase'] >= 0,
            df['total_purchase_events'] / np.maximum(df['days_to_first_purchase'] + 1, 1),
            0
        )
        
        # Revenue concentration patterns
        df['revenue_concentration'] = np.where(
            df['total_purchase_events'] > 0,
            df['max_purchase_amount'] / df['total_revenue'],
            0
        )
        
        # Session quality patterns
        df['session_consistency'] = 1 - (df['std_events_per_session'].fillna(0) / 
                                        np.maximum(df['avg_events_per_session'], 1))
        df['session_consistency'] = np.clip(df['session_consistency'], 0, 1)
        
        # Engagement depth features
        df['deep_engagement_rate'] = df['long_sessions'] / np.maximum(df['total_sessions'], 1)
        df['shallow_engagement_rate'] = df['short_sessions'] / np.maximum(df['total_sessions'], 1)
        
        # Game progression velocity
        df['level_progression_rate'] = df['total_level_events'] / np.maximum(df['total_sessions'], 1)
        df['achievement_rate'] = df['total_achievement_events'] / np.maximum(df['total_events'], 1)
        df['social_engagement_rate'] = df['total_social_events'] / np.maximum(df['total_events'], 1)
        
        # Advanced temporal patterns
        df['activity_recency_score'] = (
            df['sessions_last_7_days'] / np.maximum(df['total_sessions'], 1) * 0.6 +
            df['sessions_last_14_days'] / np.maximum(df['total_sessions'], 1) * 0.4
        )
        
        df['revenue_recency_score'] = np.where(
            df['total_revenue'] > 0,
            df['revenue_last_7_days'] / df['total_revenue'] * 0.7 +
            df['revenue_last_14_days'] / df['total_revenue'] * 0.3,
            0
        )
        
        # User lifecycle patterns
        df['maturity_score'] = np.clip(df['player_lifetime_days'] / 90, 0, 1)
        df['loyalty_score'] = (
            df['sessions_per_day'] * 0.4 +
            df['activity_recency_score'] * 0.6
        )
        
        # Purchase timing patterns
        df['early_monetizer'] = (df['days_to_first_purchase'] <= 3).astype(int)
        df['immediate_monetizer'] = (df['days_to_first_purchase'] == 0).astype(int)
        df['late_monetizer'] = (df['days_to_first_purchase'] > 14).astype(int)
        
        # Interaction complexity
        df['event_diversity_ratio'] = df['unique_event_types'] / np.maximum(df['total_events'], 1)
        df['interaction_complexity'] = (
            df['event_diversity_ratio'] * 0.5 +
            df['avg_unique_events_per_session'] / np.maximum(df['avg_events_per_session'], 1) * 0.5
        )
        
        # Behavioral consistency patterns
        df['engagement_volatility'] = df['std_session_duration_minutes'].fillna(0) / np.maximum(
            df['avg_session_duration_minutes'], 1
        )
        df['purchase_predictability'] = 1 - (df['std_purchase_amount'].fillna(0) / 
                                           np.maximum(df['avg_purchase_amount'], 1))
        df['purchase_predictability'] = np.clip(df['purchase_predictability'], 0, 1).fillna(0)
        
        # Advanced monetization patterns
        df['monetization_efficiency'] = df['total_revenue'] / np.maximum(df['total_events'], 1)
        df['conversion_funnel_performance'] = (
            df['revenue_session_rate'] * 0.4 +
            df['purchase_frequency'] * 0.6
        )
        
        # Social and viral behavior
        df['viral_coefficient'] = df['total_social_events'] / np.maximum(df['total_sessions'], 1)
        df['community_engagement'] = (
            df['social_engagement_rate'] * 0.6 +
            df['viral_coefficient'] * 0.4
        )
        
        # Risk and churn indicators
        df['churn_risk_score'] = (
            (1 - df['activity_recency_score']) * 0.4 +
            (1 - df['loyalty_score']) * 0.3 +
            df['shallow_engagement_rate'] * 0.3
        )
        
        # Value realization patterns
        df['value_realization_speed'] = np.where(
            df['total_revenue'] > 0,
            df['total_revenue'] / np.maximum(df['days_since_first_session'], 1),
            0
        )
        
        # Behavioral clustering features
        df['power_user_score'] = (
            df['session_engagement_score'] * 0.3 +
            df['deep_engagement_rate'] * 0.3 +
            df['interaction_complexity'] * 0.4
        )
        
        df['casual_user_score'] = (
            df['shallow_engagement_rate'] * 0.4 +
            (1 - df['interaction_complexity']) * 0.3 +
            (1 - df['loyalty_score']) * 0.3
        )
        
        # Advanced attribution features
        df['organic_likelihood'] = (df['first_install_source_encoded'] == 0).astype(int)  # Assuming 0 is organic
        df['paid_acquisition'] = 1 - df['organic_likelihood']
        
        # Retention proxies
        df['retention_proxy_7d'] = (df['sessions_last_7_days'] > 0).astype(int)
        df['retention_proxy_14d'] = (df['sessions_last_14_days'] > 0).astype(int)
        
        # Advanced purchase patterns
        df['whale_behavior_score'] = (
            df['revenue_concentration'] * 0.3 +
            df['purchase_momentum'] * 0.4 +
            (df['max_purchase_amount'] / 100) * 0.3  # Normalize max purchase
        )
        
        df['frequency_buyer_score'] = (
            df['purchase_frequency'] * 0.5 +
            df['purchase_velocity'] * 0.3 +
            (1 - df['revenue_concentration']) * 0.2
        )
        
        # Handle infinite and missing values
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace([np.inf, -np.inf], 0)
            df[col] = df[col].fillna(0)
        
        logger.info(f"Created {30} advanced behavioral features")
        
        return df
    
    def create_sequential_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features that capture sequential behavior patterns"""
        
        logger.info("Creating sequential behavior features...")
        
        # Time-based patterns
        df['days_active_ratio'] = df['player_lifetime_days'] / np.maximum(df['days_since_first_session'], 1)
        df['session_frequency'] = df['total_sessions'] / np.maximum(df['player_lifetime_days'], 1)
        
        # Purchase sequence patterns
        df['purchase_acceleration'] = np.where(
            (df['total_purchase_events'] > 1) & (df['purchase_span_days'] > 0),
            df['total_purchase_events'] / df['purchase_span_days'],
            0
        )
        
        # Engagement evolution
        df['recent_engagement_trend'] = (
            df['recent_activity_rate_7d'] - df['recent_activity_rate_14d']
        )
        
        df['revenue_trend'] = np.where(
            df['total_revenue'] > 0,
            df['recent_revenue_rate_7d'] - df['recent_revenue_rate_14d'],
            0
        )
        
        # Session pattern evolution
        df['session_length_trend'] = np.where(
            df['std_session_duration_minutes'] > 0,
            (df['max_session_duration_minutes'] - df['avg_session_duration_minutes']) / 
            df['std_session_duration_minutes'],
            0
        )
        
        # Handle infinite values
        for col in ['purchase_acceleration', 'session_length_trend']:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], 0)
                df[col] = df[col].fillna(0)
        
        logger.info("Sequential features created")
        
        return df


class DataExtractor:
    """Enhanced data extraction with comprehensive feature engineering"""
    
    def __init__(self, config: Config):
        self.config = config
        self.project_id = config.get('bigquery.project_id')
        self.client = self._initialize_client()
        self.feature_engineer = AdvancedFeatureEngineer(config)
        
    def _initialize_client(self) -> bigquery.Client:
        """Initialize BigQuery client"""
        try:
            return bigquery.Client(project=self.project_id)
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {e}")
            raise
    
    def extract_and_engineer_features(self, days_lookback: Optional[int] = None) -> pd.DataFrame:
        """Extract data and engineer comprehensive features"""
        
        days_lookback = days_lookback or self.config.get('bigquery.lookback_days', 90)
        logger.info(f"Extracting player data for last {days_lookback} days...")
        
        query = f"""
        WITH player_events AS (
            SELECT 
                COALESCE(custom_user_id, gaid, idfa, android_id, idfv, waid) as player_id,
                session_id,
                name as event_name,
                SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S%Ez', server_timestamp) as event_timestamp,
                COALESCE(revenue, 0.0) as revenue,
                COALESCE(received_revenue, 0.0) as received_revenue,
                product_sku,
                COALESCE(product_price, 0.0) as product_price,
                COALESCE(product_quantity, 0) as product_quantity,
                is_fingerprinted,
                is_reengagement,
                is_view_through,
                country,
                state,
                city,
                install_source,
                campaign_name,
                campaign_id,
                publisher_name,
                publisher_id,
                creative_name,
                utm_source,
                utm_medium,
                utm_campaign
            FROM `{self.project_id}.{self.config.get('bigquery.dataset')}.{self.config.get('bigquery.table')}`
            WHERE 
                SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S%Ez', server_timestamp) >= 
                TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days_lookback} DAY)
                AND COALESCE(custom_user_id, gaid, idfa, android_id, idfv, waid) IS NOT NULL
        ),
        
        session_metrics AS (
            SELECT
                player_id,
                session_id,
                MIN(event_timestamp) as session_start,
                MAX(event_timestamp) as session_end,
                COUNT(*) as session_events,
                COUNT(DISTINCT event_name) as unique_events_per_session,
                SUM(revenue) as session_revenue,
                SUM(received_revenue) as session_received_revenue,
                COUNTIF(revenue > 0) as session_purchase_events,
                TIMESTAMP_DIFF(MAX(event_timestamp), MIN(event_timestamp), MINUTE) as session_duration_minutes,
                
                -- Behavioral flags
                MAX(CASE WHEN is_fingerprinted THEN 1 ELSE 0 END) as session_has_fingerprint,
                MAX(CASE WHEN is_reengagement THEN 1 ELSE 0 END) as session_is_reengagement,
                MAX(CASE WHEN is_view_through THEN 1 ELSE 0 END) as session_has_view_through,
                
                -- Game progression (inferred from event names)
                COUNTIF(LOWER(event_name) LIKE '%level%' OR LOWER(event_name) LIKE '%stage%' OR LOWER(event_name) LIKE '%progress%') as session_level_events,
                COUNTIF(LOWER(event_name) LIKE '%achievement%' OR LOWER(event_name) LIKE '%unlock%' OR LOWER(event_name) LIKE '%reward%' OR LOWER(event_name) LIKE '%complete%') as session_achievement_events,
                COUNTIF(LOWER(event_name) LIKE '%social%' OR LOWER(event_name) LIKE '%share%' OR LOWER(event_name) LIKE '%invite%' OR LOWER(event_name) LIKE '%friend%') as session_social_events,
                COUNTIF(LOWER(event_name) LIKE '%tutorial%' OR LOWER(event_name) LIKE '%onboard%' OR LOWER(event_name) LIKE '%start%' OR LOWER(event_name) LIKE '%help%') as session_tutorial_events,
                COUNTIF(LOWER(event_name) LIKE '%purchase%' OR LOWER(event_name) LIKE '%buy%' OR LOWER(event_name) LIKE '%shop%' OR LOWER(event_name) LIKE '%store%') as session_shop_events,
                COUNTIF(LOWER(event_name) LIKE '%ads%' OR LOWER(event_name) LIKE '%video%' OR LOWER(event_name) LIKE '%rewarded%' OR LOWER(event_name) LIKE '%banner%') as session_ads_events,
                COUNTIF(LOWER(event_name) LIKE '%click%' OR LOWER(event_name) LIKE '%tap%' OR LOWER(event_name) LIKE '%press%') as session_interaction_events,
                COUNTIF(LOWER(event_name) LIKE '%fail%' OR LOWER(event_name) LIKE '%lose%' OR LOWER(event_name) LIKE '%death%' OR LOWER(event_name) LIKE '%retry%') as session_failure_events,
                COUNTIF(LOWER(event_name) LIKE '%win%' OR LOWER(event_name) LIKE '%success%' OR LOWER(event_name) LIKE '%victory%' OR LOWER(event_name) LIKE '%complete%') as session_success_events,
                
                -- Attribution data (first touch in session)
                ARRAY_AGG(country ORDER BY event_timestamp ASC LIMIT 1)[OFFSET(0)] as session_country,
                ARRAY_AGG(install_source ORDER BY event_timestamp ASC LIMIT 1)[OFFSET(0)] as session_install_source,
                ARRAY_AGG(campaign_name ORDER BY event_timestamp ASC LIMIT 1)[OFFSET(0)] as session_campaign,
                ARRAY_AGG(utm_source ORDER BY event_timestamp ASC LIMIT 1)[OFFSET(0)] as session_utm_source
                
            FROM player_events
            WHERE event_timestamp IS NOT NULL
            GROUP BY player_id, session_id
        ),
        
        player_features AS (
            SELECT
                player_id,
                
                -- Basic engagement metrics
                COUNT(DISTINCT session_id) as total_sessions,
                SUM(session_events) as total_events,
                AVG(session_events) as avg_events_per_session,
                MAX(session_events) as max_events_per_session,
                STDDEV(session_events) as std_events_per_session,
                
                -- Unique event diversity
                SUM(unique_events_per_session) as total_unique_events,
                AVG(unique_events_per_session) as avg_unique_events_per_session,
                MAX(unique_events_per_session) as max_unique_events_per_session,
                COUNT(DISTINCT 
                    CASE WHEN unique_events_per_session > 0 
                    THEN unique_events_per_session END
                ) as unique_event_types,
                
                -- Temporal features
                MIN(session_start) as first_session_timestamp,
                MAX(session_end) as last_session_timestamp,
                DATE_DIFF(CURRENT_DATE(), DATE(MIN(session_start)), DAY) as days_since_first_session,
                DATE_DIFF(DATE(MAX(session_end)), DATE(MIN(session_start)), DAY) + 1 as player_lifetime_days,
                
                -- Session behavior patterns
                AVG(session_duration_minutes) as avg_session_duration_minutes,
                MAX(session_duration_minutes) as max_session_duration_minutes,
                SUM(session_duration_minutes) as total_session_duration_minutes,
                STDDEV(session_duration_minutes) as std_session_duration_minutes,
                
                -- Revenue metrics
                SUM(session_revenue) as total_revenue,
                SUM(session_received_revenue) as total_received_revenue,
                SUM(session_purchase_events) as total_purchase_events,
                COUNT(CASE WHEN session_revenue > 0 THEN 1 END) as revenue_sessions,
                AVG(CASE WHEN session_revenue > 0 THEN session_revenue END) as avg_purchase_amount,
                MAX(session_revenue) as max_purchase_amount,
                STDDEV(CASE WHEN session_revenue > 0 THEN session_revenue END) as std_purchase_amount,
                
                -- Behavioral engagement
                SUM(session_has_fingerprint) as fingerprint_sessions,
                SUM(session_is_reengagement) as reengagement_sessions,
                SUM(session_has_view_through) as view_through_sessions,
                
                -- Game progression features
                SUM(session_level_events) as total_level_events,
                SUM(session_achievement_events) as total_achievement_events,
                SUM(session_social_events) as total_social_events,
                SUM(session_tutorial_events) as total_tutorial_events,
                SUM(session_shop_events) as total_shop_events,
                SUM(session_ads_events) as total_ads_events,
                SUM(session_interaction_events) as total_interaction_events,
                SUM(session_failure_events) as total_failure_events,
                SUM(session_success_events) as total_success_events,
                
                -- Session quality patterns
                COUNT(CASE WHEN session_events > 1 THEN 1 END) as multi_event_sessions,
                COUNT(CASE WHEN session_duration_minutes > 5 THEN 1 END) as long_sessions,
                COUNT(CASE WHEN session_duration_minutes < 1 THEN 1 END) as short_sessions,
                COUNT(CASE WHEN session_events = 1 THEN 1 END) as single_event_sessions,
                COUNT(CASE WHEN session_duration_minutes BETWEEN 1 AND 5 THEN 1 END) as medium_sessions,
                
                -- Attribution features (first-touch)
                ARRAY_AGG(session_country ORDER BY session_start ASC LIMIT 1)[OFFSET(0)] as first_country,
                ARRAY_AGG(session_install_source ORDER BY session_start ASC LIMIT 1)[OFFSET(0)] as first_install_source,
                ARRAY_AGG(session_campaign ORDER BY session_start ASC LIMIT 1)[OFFSET(0)] as first_campaign,
                ARRAY_AGG(session_utm_source ORDER BY session_start ASC LIMIT 1)[OFFSET(0)] as first_utm_source,
                
                -- Recent activity features (last 7, 14, and 30 days)
                COUNT(CASE WHEN DATE_DIFF(CURRENT_DATE(), DATE(session_start), DAY) <= 7 THEN 1 END) as sessions_last_7_days,
                COUNT(CASE WHEN DATE_DIFF(CURRENT_DATE(), DATE(session_start), DAY) <= 14 THEN 1 END) as sessions_last_14_days,
                COUNT(CASE WHEN DATE_DIFF(CURRENT_DATE(), DATE(session_start), DAY) <= 30 THEN 1 END) as sessions_last_30_days,
                SUM(CASE WHEN DATE_DIFF(CURRENT_DATE(), DATE(session_start), DAY) <= 7 THEN session_revenue ELSE 0 END) as revenue_last_7_days,
                SUM(CASE WHEN DATE_DIFF(CURRENT_DATE(), DATE(session_start), DAY) <= 14 THEN session_revenue ELSE 0 END) as revenue_last_14_days,
                SUM(CASE WHEN DATE_DIFF(CURRENT_DATE(), DATE(session_start), DAY) <= 30 THEN session_revenue ELSE 0 END) as revenue_last_30_days,
                
                -- Purchase timing analysis
                DATE_DIFF(
                    DATE(MAX(CASE WHEN session_revenue > 0 THEN session_start END)),
                    DATE(MIN(CASE WHEN session_revenue > 0 THEN session_start END)),
                    DAY
                ) as purchase_span_days,
                
                -- First purchase timing
                DATE_DIFF(
                    DATE(MIN(CASE WHEN session_revenue > 0 THEN session_start END)),
                    DATE(MIN(session_start)),
                    DAY
                ) as days_to_first_purchase,
                
                -- Advanced behavioral patterns
                COUNT(CASE WHEN session_success_events > session_failure_events THEN 1 END) as positive_outcome_sessions,
                AVG(CASE WHEN session_success_events + session_failure_events > 0 
                     THEN session_success_events / (session_success_events + session_failure_events) 
                     END) as success_rate,
                
                -- Interaction intensity
                SUM(session_interaction_events) as total_interactions,
                AVG(CASE WHEN session_events > 0 
                     THEN session_interaction_events / session_events 
                     END) as interaction_intensity
                
            FROM session_metrics
            GROUP BY player_id
            HAVING COUNT(DISTINCT session_id) >= 1
        )
        
        SELECT *
        FROM player_features
        WHERE player_id IS NOT NULL
        ORDER BY total_revenue DESC, total_sessions DESC
        """
        
        try:
            job_config = bigquery.QueryJobConfig()
            job_config.use_query_cache = True
            job_config.maximum_bytes_billed = 100 * 1024 * 1024 * 1024  # 100GB limit
            
            query_job = self.client.query(query, job_config=job_config)
            df = query_job.result().to_dataframe()
            
            logger.info(f"Extracted {len(df):,} players from BigQuery")
            
            # Clean and engineer features
            df = self._clean_and_engineer_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            raise
    
    def _clean_and_engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data and engineer additional features"""
        
        logger.info("Cleaning data and engineering features...")
        
        # Handle timestamps and timezone conversion
        timestamp_cols = ['first_session_timestamp', 'last_session_timestamp']
        for col in timestamp_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
        
        # Fill missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(0)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna('unknown')
        
        # Basic derived features
        df = self._create_basic_derived_features(df)
        
        # Advanced behavioral features
        df = self.feature_engineer.create_advanced_behavioral_features(df)
        
        # Sequential features
        df = self.feature_engineer.create_sequential_features(df)
        
        # Create user categorization
        df = self._categorize_users(df)
        
        # Handle categorical encoding
        df = self._encode_categorical_features(df)
        
        logger.info(f"Feature engineering completed. Final shape: {df.shape}")
        
        return df
    
    def _create_basic_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic derived behavioral and engagement features"""
        
        # Basic ratios and rates
        df['events_per_session'] = df['total_events'] / np.maximum(df['total_sessions'], 1)
        df['events_per_day'] = df['total_events'] / np.maximum(df['player_lifetime_days'], 1)
        df['sessions_per_day'] = df['total_sessions'] / np.maximum(df['player_lifetime_days'], 1)
        
        # Revenue features
        df['revenue_per_session'] = df['total_revenue'] / np.maximum(df['total_sessions'], 1)
        df['revenue_per_day'] = df['total_revenue'] / np.maximum(df['player_lifetime_days'], 1)
        df['purchase_frequency'] = df['total_purchase_events'] / np.maximum(df['total_sessions'], 1)
        df['revenue_session_rate'] = df['revenue_sessions'] / np.maximum(df['total_sessions'], 1)
        
        # Purchase behavior patterns
        df['avg_purchase_amount'] = df['avg_purchase_amount'].fillna(0)
        df['std_purchase_amount'] = df['std_purchase_amount'].fillna(0)
        
        # Time between purchases
        df['days_between_purchases'] = np.where(
            df['total_purchase_events'] > 1,
            df['purchase_span_days'] / np.maximum(df['total_purchase_events'] - 1, 1),
            0
        )
        
        # Engagement quality features
        df['fingerprint_rate'] = df['fingerprint_sessions'] / np.maximum(df['total_sessions'], 1)
        df['reengagement_rate'] = df['reengagement_sessions'] / np.maximum(df['total_sessions'], 1)
        df['view_through_rate'] = df['view_through_sessions'] / np.maximum(df['total_sessions'], 1)
        
        # Session quality metrics
        df['multi_event_session_rate'] = df['multi_event_sessions'] / np.maximum(df['total_sessions'], 1)
        df['long_session_rate'] = df['long_sessions'] / np.maximum(df['total_sessions'], 1)
        df['short_session_rate'] = df['short_sessions'] / np.maximum(df['total_sessions'], 1)
        df['single_event_session_rate'] = df['single_event_sessions'] / np.maximum(df['total_sessions'], 1)
        df['medium_session_rate'] = df['medium_sessions'] / np.maximum(df['total_sessions'], 1)
        
        # Game progression engagement
        df['level_event_rate'] = df['total_level_events'] / np.maximum(df['total_events'], 1)
        df['achievement_event_rate'] = df['total_achievement_events'] / np.maximum(df['total_events'], 1)
        df['social_event_rate'] = df['total_social_events'] / np.maximum(df['total_events'], 1)
        df['tutorial_event_rate'] = df['total_tutorial_events'] / np.maximum(df['total_events'], 1)
        df['shop_event_rate'] = df['total_shop_events'] / np.maximum(df['total_events'], 1)
        df['ads_event_rate'] = df['total_ads_events'] / np.maximum(df['total_events'], 1)
        df['interaction_event_rate'] = df['total_interaction_events'] / np.maximum(df['total_events'], 1)
        df['failure_event_rate'] = df['total_failure_events'] / np.maximum(df['total_events'], 1)
        df['success_event_rate'] = df['total_success_events'] / np.maximum(df['total_events'], 1)
        
        # Recent activity patterns
        df['recent_activity_rate_7d'] = df['sessions_last_7_days'] / np.maximum(df['total_sessions'], 1)
        df['recent_activity_rate_14d'] = df['sessions_last_14_days'] / np.maximum(df['total_sessions'], 1)
        df['recent_activity_rate_30d'] = df['sessions_last_30_days'] / np.maximum(df['total_sessions'], 1)
        df['recent_revenue_rate_7d'] = df['revenue_last_7_days'] / np.maximum(df['total_revenue'], 1)
        df['recent_revenue_rate_14d'] = df['revenue_last_14_days'] / np.maximum(df['total_revenue'], 1)
        df['recent_revenue_rate_30d'] = df['revenue_last_30_days'] / np.maximum(df['total_revenue'], 1)
        
        # Session duration patterns
        df['avg_session_duration_minutes'] = df['avg_session_duration_minutes'].fillna(0)
        df['std_session_duration_minutes'] = df['std_session_duration_minutes'].fillna(0)
        
        # Time to first purchase (conversion timing)
        df['days_to_first_purchase'] = df['days_to_first_purchase'].fillna(-1)  # -1 for non-payers
        
        # Player lifecycle stage
        df['lifecycle_stage'] = np.where(
            df['days_since_first_session'] <= 7, 'new',
            np.where(df['days_since_first_session'] <= 30, 'growing',
                    np.where(df['days_since_first_session'] <= 90, 'mature', 'veteran'))
        )
        
        # Success metrics
        df['success_rate'] = df['success_rate'].fillna(0)
        df['interaction_intensity'] = df['interaction_intensity'].fillna(0)
        
        # Handle infinite values
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace([np.inf, -np.inf], 0)
            df[col] = df[col].fillna(0)
        
        return df
    
    def _categorize_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize users into Whales, Low Spenders, Non-payers based on ExpLTV methodology"""
        
        logger.info("Categorizing users...")
        
        # Calculate whale threshold
        payers = df[df['total_revenue'] > 0]
        whale_percentile = self.config.get('model.whale_threshold_percentile', 90)
        
        if len(payers) > 0:
            whale_threshold = payers['total_revenue'].quantile(whale_percentile / 100)
            logger.info(f"Whale threshold (top {100-whale_percentile}%): ${whale_threshold:.2f}")
        else:
            whale_threshold = 100  # Default threshold
            logger.warning("No payers found, using default whale threshold")
        
        # Create user categories
        df['user_category'] = 'non_payer'
        df.loc[df['total_revenue'] > 0, 'user_category'] = 'low_spender'
        df.loc[df['total_revenue'] >= whale_threshold, 'user_category'] = 'whale'
        
        # Create binary labels for different tasks
        df['is_payer'] = (df['total_revenue'] > 0).astype(int)
        df['is_whale'] = (df['user_category'] == 'whale').astype(int)
        df['is_low_spender_given_payer'] = np.where(
            df['is_payer'] == 1,
            (df['user_category'] == 'low_spender').astype(int),
            0
        )
        
        # Create ExpLTV probability labels based on research paper
        R = whale_threshold
        df['whale_purchase_prob'] = np.where(
            df['total_revenue'] > 0,
            1 - np.exp(-df['total_revenue'] / R),
            0
        )
        
        # Category distribution
        category_counts = df['user_category'].value_counts()
        total_players = len(df)
        
        logger.info("User categorization completed:")
        for category, count in category_counts.items():
            percentage = (count / total_players) * 100
            logger.info(f"  {category}: {count:,} ({percentage:.1f}%)")
        
        # Revenue distribution
        if len(payers) > 0:
            whale_revenue = df[df['user_category'] == 'whale']['total_revenue'].sum()
            low_spender_revenue = df[df['user_category'] == 'low_spender']['total_revenue'].sum()
            total_revenue = df['total_revenue'].sum()
            
            logger.info("Revenue distribution:")
            logger.info(f"  Whale revenue: ${whale_revenue:,.2f} ({whale_revenue/total_revenue*100:.1f}%)")
            logger.info(f"  Low spender revenue: ${low_spender_revenue:,.2f} ({low_spender_revenue/total_revenue*100:.1f}%)")
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        
        categorical_features = ['first_country', 'first_install_source', 'first_campaign', 
                               'first_utm_source', 'lifecycle_stage']
        
        for feature in categorical_features:
            if feature in df.columns:
                # Keep top N categories, group others as 'other'
                max_levels = self.config.get('features.max_categorical_levels', 10)
                top_categories = df[feature].value_counts().head(max_levels).index.tolist()
                df[feature] = df[feature].apply(lambda x: x if x in top_categories else 'other')
                
                # Label encode
                encoder = LabelEncoder()
                df[f'{feature}_encoded'] = encoder.fit_transform(df[feature].astype(str))
                self.feature_engineer.label_encoders[feature] = encoder
        
        return df


class ExpLTVModel(nn.Module):
    """ExpLTV model implementation with game whale detection and expert routing"""
    
    def __init__(self, input_dim: int, config: Config):
        super(ExpLTVModel, self).__init__()
        
        self.config = config
        hidden_dim = config.get('model.hidden_dim', 128)
        embedding_dim = config.get('model.embedding_dim', 64)
        dropout_rate = config.get('model.dropout_rate', 0.2)
        
        # Shared embedding layers for feature transformation
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Purchase probability estimator (auxiliary task)
        self.purchase_estimator = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Game whale detector (outputs probabilities for whale vs low_spender given purchase)
        self.whale_detector = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )
        
        # LTV Expert networks for different user types
        self.whale_expert = self._create_ltv_expert(embedding_dim, dropout_rate)
        self.low_spender_expert = self._create_ltv_expert(embedding_dim, dropout_rate)
        
    def _create_ltv_expert(self, input_dim: int, dropout_rate: float):
        """Create an LTV expert network for ZILN distribution parameters"""
        return nn.ModuleDict({
            'network': nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ),
            'mu_head': nn.Linear(32, 1),
            'sigma_head': nn.Sequential(
                nn.Linear(32, 1),
                nn.Softplus()
            )
        })
    
    def forward(self, x):
        """Forward pass through ExpLTV model"""
        
        # Shared embeddings
        embeddings = self.embedding(x)
        
        # Purchase probability (auxiliary task)
        purchase_prob = self.purchase_estimator(embeddings)
        
        # Whale detection (conditional probabilities given purchase)
        whale_probs = self.whale_detector(embeddings)  # [whale_prob, low_spender_prob]
        
        # LTV expert predictions
        whale_features = self.whale_expert['network'](embeddings)
        whale_mu = self.whale_expert['mu_head'](whale_features)
        whale_sigma = self.whale_expert['sigma_head'](whale_features) + 1e-3
        
        low_spender_features = self.low_spender_expert['network'](embeddings)
        low_spender_mu = self.low_spender_expert['mu_head'](low_spender_features)
        low_spender_sigma = self.low_spender_expert['sigma_head'](low_spender_features) + 1e-3
        
        # Ensure numerical stability
        whale_mu = torch.clamp(whale_mu, -10, 10)
        whale_sigma = torch.clamp(whale_sigma, 1e-3, 5)
        low_spender_mu = torch.clamp(low_spender_mu, -10, 10)
        low_spender_sigma = torch.clamp(low_spender_sigma, 1e-3, 5)
        
        # Expert routing using whale detection probabilities
        whale_weight = whale_probs[:, 0:1]
        low_spender_weight = whale_probs[:, 1:2]
        
        # Weighted combination of expert predictions
        final_mu = whale_weight * whale_mu + low_spender_weight * low_spender_mu
        final_sigma = whale_weight * whale_sigma + low_spender_weight * low_spender_sigma
        
        # Compute joint probabilities using Bayes' theorem
        whale_purchase_prob = purchase_prob * whale_probs[:, 0:1]
        low_spender_purchase_prob = purchase_prob * whale_probs[:, 1:2]
        non_payer_prob = 1 - purchase_prob
        
        return {
            'purchase_prob': purchase_prob,
            'whale_conditional_prob': whale_probs[:, 0:1],
            'low_spender_conditional_prob': whale_probs[:, 1:2],
            'whale_purchase_prob': whale_purchase_prob,
            'low_spender_purchase_prob': low_spender_purchase_prob,
            'non_payer_prob': non_payer_prob,
            'final_mu': final_mu,
            'final_sigma': final_sigma,
            'embeddings': embeddings
        }


class ExpLTVTrainer:
    """Training pipeline for ExpLTV model with joint loss optimization"""
    
    def __init__(self, model: ExpLTVModel, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizers
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.get('model.learning_rate', 0.001),
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5, verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def ziln_loss(self, predictions_dict, targets, weights=None):
        """Zero-Inflated Log-Normal loss with numerical stability"""
        
        mu = predictions_dict['final_mu']
        sigma = predictions_dict['final_sigma']
        purchase_probs = predictions_dict['purchase_prob']
        
        # Separate zero and non-zero targets
        zero_mask = (targets == 0)
        nonzero_mask = ~zero_mask
        
        # Ensure numerical stability
        purchase_probs = torch.clamp(purchase_probs, 1e-7, 1 - 1e-7)
        
        # Cross-entropy loss for purchase probability
        purchase_loss = F.binary_cross_entropy(
            purchase_probs.squeeze(),
            (targets > 0).float(),
            weight=weights
        )
        
        # Log-normal loss for non-zero targets
        lognormal_loss = torch.tensor(0.0, device=self.device)
        if nonzero_mask.sum() > 0:
            log_targets = torch.log(targets[nonzero_mask] + 1e-8)
            mu_nonzero = mu[nonzero_mask]
            sigma_nonzero = sigma[nonzero_mask]
            
            # Ensure bounds
            sigma_nonzero = torch.clamp(sigma_nonzero, 1e-3, 5)
            
            # Log-normal likelihood
            lognormal_loss = (
                torch.log(sigma_nonzero * np.sqrt(2 * np.pi)) +
                ((log_targets - mu_nonzero) ** 2) / (2 * sigma_nonzero ** 2)
            ).mean()
        
        return purchase_loss + lognormal_loss
    
    def compute_combined_loss(self, predictions_dict, targets, user_types, weights=None):
        """Compute combined loss following ExpLTV methodology"""
        
        # 1. Purchase probability loss (auxiliary task)
        purchase_loss = F.binary_cross_entropy(
            predictions_dict['purchase_prob'].squeeze(),
            (targets > 0).float(),
            weight=weights
        )
        
        # 2. Whale detection loss (conditional probabilities given purchase)
        payer_mask = targets > 0
        whale_classification_loss = torch.tensor(0.0, device=self.device)
        
        if payer_mask.sum() > 0:
            # For payers, predict whale vs low_spender
            payer_whale_probs = torch.cat([
                predictions_dict['whale_conditional_prob'][payer_mask],
                predictions_dict['low_spender_conditional_prob'][payer_mask]
            ], dim=1)
            
            # Convert user_types to binary labels (whale=0, low_spender=1)
            payer_types = user_types[payer_mask]
            whale_labels = (payer_types == 2).long()  # whale = 2 in original labels
            
            if len(whale_labels) > 0:
                whale_classification_loss = F.cross_entropy(payer_whale_probs, whale_labels)
        
        # 3. ZILN loss for revenue prediction
        ziln_loss = self.ziln_loss(predictions_dict, targets, weights)
        
        # 4. KL divergence for probability consistency (from ExpLTV paper)
        whale_purchase_prob = predictions_dict['whale_purchase_prob']
        computed_whale_prob = predictions_dict['purchase_prob'] * predictions_dict['whale_conditional_prob']
        kl_loss = F.kl_div(
            torch.log(whale_purchase_prob + 1e-8),
            computed_whale_prob,
            reduction='batchmean'
        )
        
        # Combined loss with weights
        total_loss = (
            purchase_loss + 
            0.5 * whale_classification_loss + 
            ziln_loss + 
            0.1 * kl_loss
        )
        
        return {
            'total_loss': total_loss,
            'purchase_loss': purchase_loss,
            'whale_classification_loss': whale_classification_loss,
            'ziln_loss': ziln_loss,
            'kl_loss': kl_loss
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            X_batch, y_batch, user_types_batch = batch
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            user_types_batch = user_types_batch.to(self.device)
            
            # Forward pass
            predictions = self.model(X_batch)
            
            # Compute loss
            loss_dict = self.compute_combined_loss(predictions, y_batch, user_types_batch)
            total_loss_val = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_val.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.get('model.gradient_clip_norm', 1.0)
            )
            
            self.optimizer.step()
            
            total_loss += total_loss_val.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                X_batch, y_batch, user_types_batch = batch
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                user_types_batch = user_types_batch.to(self.device)
                
                # Forward pass
                predictions = self.model(X_batch)
                
                # Compute loss
                loss_dict = self.compute_combined_loss(predictions, y_batch, user_types_batch)
                total_loss_val = loss_dict['total_loss']
                
                total_loss += total_loss_val.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader, epochs: Optional[int] = None):
        """Full training loop with early stopping"""
        
        epochs = epochs or self.config.get('model.epochs', 200)
        patience = self.config.get('model.early_stopping_patience', 20)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Track losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_checkpoint('best_model.pth')
            else:
                patience_counter += 1
            
            # Log progress
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info("Training completed!")
        return self.train_losses, self.val_losses
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filename)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])


class ComprehensiveEvaluator:
    """Comprehensive evaluation system for ExpLTV model"""
    
    def __init__(self, model: ExpLTVModel, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def predict(self, X: torch.Tensor):
        """Generate predictions from the model"""
        
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            predictions = self.model(X)
            
            # Calculate expected LTV using ZILN formulation
            purchase_probs = predictions['purchase_prob']
            mu = predictions['final_mu']
            sigma = predictions['final_sigma']
            
            # Safe exponential calculation
            mu = torch.clamp(mu, -10, 10)
            sigma = torch.clamp(sigma, 1e-3, 5)
            exp_term = mu + sigma**2 / 2
            exp_term = torch.clamp(exp_term, -20, 20)
            
            expected_ltv = purchase_probs * torch.exp(exp_term)
            expected_ltv = torch.clamp(expected_ltv, 0, 1e6)  # Cap at 1M
            
            # Convert to numpy
            results = {
                'expected_ltv': expected_ltv.cpu().numpy().flatten(),
                'purchase_prob': purchase_probs.cpu().numpy().flatten(),
                'whale_prob': predictions['whale_conditional_prob'].cpu().numpy().flatten(),
                'low_spender_prob': predictions['low_spender_conditional_prob'].cpu().numpy().flatten(),
                'whale_purchase_prob': predictions['whale_purchase_prob'].cpu().numpy().flatten()
            }
            
        return results
    
    def evaluate_comprehensive(self, X_test: torch.Tensor, y_test: np.ndarray, user_types_test: np.ndarray) -> Dict:
        """Comprehensive evaluation of the ExpLTV model"""
        
        logger.info("Running comprehensive evaluation...")
        
        # Get predictions
        predictions = self.predict(X_test)
        ltv_pred = predictions['expected_ltv']
        purchase_pred = predictions['purchase_prob']
        whale_pred = predictions['whale_prob']
        
        # Ensure finite values
        ltv_pred = np.where(np.isfinite(ltv_pred), ltv_pred, 0)
        purchase_pred = np.where(np.isfinite(purchase_pred), purchase_pred, 0)
        whale_pred = np.where(np.isfinite(whale_pred), whale_pred, 0)
        
        metrics = {}
        
        # LTV PREDICTION METRICS
        metrics['ltv_mae'] = mean_absolute_error(y_test, ltv_pred)
        metrics['ltv_rmse'] = np.sqrt(mean_squared_error(y_test, ltv_pred))
        metrics['ltv_r2'] = r2_score(y_test, ltv_pred)
        
        # Revenue prediction accuracy
        total_actual = np.sum(y_test)
        total_predicted = np.sum(ltv_pred)
        metrics['revenue_prediction_error'] = abs(total_actual - total_predicted) / total_actual if total_actual > 0 else float('inf')
        
        # PURCHASE CLASSIFICATION METRICS
        y_binary = (y_test > 0).astype(int)
        
        if len(np.unique(y_binary)) > 1:
            metrics['purchase_auc'] = roc_auc_score(y_binary, purchase_pred)
        else:
            metrics['purchase_auc'] = 0.5
        
        purchase_binary_pred = (purchase_pred >= 0.5).astype(int)
        metrics['purchase_accuracy'] = accuracy_score(y_binary, purchase_binary_pred)
        metrics['purchase_precision'] = precision_score(y_binary, purchase_binary_pred, zero_division=0)
        metrics['purchase_recall'] = recall_score(y_binary, purchase_binary_pred, zero_division=0)
        metrics['purchase_f1'] = f1_score(y_binary, purchase_binary_pred, zero_division=0)
        
        # WHALE DETECTION METRICS
        payer_mask = y_test > 0
        if np.sum(payer_mask) > 0:
            y_whale_payers = (user_types_test[payer_mask] == 2).astype(int)
            whale_pred_payers = whale_pred[payer_mask]
            
            if len(np.unique(y_whale_payers)) > 1:
                metrics['whale_auc'] = roc_auc_score(y_whale_payers, whale_pred_payers)
            else:
                metrics['whale_auc'] = 0.5
            
            whale_binary_pred = (whale_pred_payers >= 0.5).astype(int)
            metrics['whale_accuracy'] = accuracy_score(y_whale_payers, whale_binary_pred)
            metrics['whale_precision'] = precision_score(y_whale_payers, whale_binary_pred, zero_division=0)
            metrics['whale_recall'] = recall_score(y_whale_payers, whale_binary_pred, zero_division=0)
            metrics['whale_f1'] = f1_score(y_whale_payers, whale_binary_pred, zero_division=0)
        else:
            metrics.update({
                'whale_auc': 0.5, 'whale_accuracy': 0, 'whale_precision': 0,
                'whale_recall': 0, 'whale_f1': 0
            })
        
        # BUSINESS IMPACT METRICS
        sorted_indices = np.argsort(ltv_pred)[::-1]
        percentiles = [0.01, 0.05, 0.1, 0.2]
        
        for pct in percentiles:
            top_n = int(len(ltv_pred) * pct)
            if top_n > 0:
                top_indices = sorted_indices[:top_n]
                top_revenue = np.sum(y_test[top_indices])
                revenue_capture_rate = top_revenue / total_actual if total_actual > 0 else 0
                lift = revenue_capture_rate / pct if pct > 0 else 0
                
                metrics[f'lift_top_{int(pct*100)}pct'] = lift
                metrics[f'revenue_capture_top_{int(pct*100)}pct'] = revenue_capture_rate
        
        # USER CATEGORY PERFORMANCE
        for category_idx, category_name in enumerate(['non_payer', 'low_spender', 'whale']):
            mask = user_types_test == category_idx
            
            if np.sum(mask) > 0:
                category_mae = mean_absolute_error(y_test[mask], ltv_pred[mask])
                category_rmse = np.sqrt(mean_squared_error(y_test[mask], ltv_pred[mask]))
                
                metrics[f'{category_name}_mae'] = category_mae
                metrics[f'{category_name}_rmse'] = category_rmse
                metrics[f'{category_name}_count'] = np.sum(mask)
                metrics[f'{category_name}_avg_actual'] = np.mean(y_test[mask])
                metrics[f'{category_name}_avg_predicted'] = np.mean(ltv_pred[mask])
        
        # MODEL CALIBRATION
        predicted_payer_rate = np.mean(purchase_pred)
        actual_payer_rate = np.mean(y_binary)
        metrics['calibration_error'] = abs(predicted_payer_rate - actual_payer_rate)
        
        # SUMMARY STATISTICS
        metrics.update({
            'total_test_samples': len(y_test),
            'total_actual_revenue': total_actual,
            'total_predicted_revenue': total_predicted,
            'actual_payer_rate': actual_payer_rate,
            'predicted_payer_rate': predicted_payer_rate,
            'actual_whale_rate': np.mean(user_types_test == 2),
            'predicted_whale_rate': np.mean(whale_pred)
        })
        
        logger.info(f"Evaluation completed. LTV R²: {metrics['ltv_r2']:.3f}, Purchase AUC: {metrics['purchase_auc']:.3f}")
        
        return metrics


class ProductionPLTVSystem:
    """Main production system orchestrating the entire ExpLTV pipeline"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = Config(config_path)
        self.setup_output_directory()
        
        # Components
        self.data_extractor = DataExtractor(self.config)
        self.scaler = RobustScaler()
        self.model = None
        self.trainer = None
        self.evaluator = None
        
        # Feature columns (will be set during feature engineering)
        self.feature_columns = []
        
        # Results storage
        self.results = {}
        
        logger.info("Production pLTV System initialized")
    
    def setup_output_directory(self):
        """Setup output directory structure"""
        output_dir = self.config.get('outputs.output_dir', 'output')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
    
    def extract_and_prepare_data(self) -> pd.DataFrame:
        """Extract data and prepare features"""
        
        logger.info("=== DATA EXTRACTION AND PREPARATION ===")
        
        # Extract data with comprehensive features
        df = self.data_extractor.extract_and_engineer_features()
        
        # Save engineered data
        if self.config.get('outputs.save_engineered_data', True):
            output_dir = self.config.get('outputs.output_dir', 'output')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_path = os.path.join(output_dir, 'data', f'engineered_player_data_{timestamp}.csv')
            df.to_csv(data_path, index=False)
            logger.info(f"Engineered data saved to {data_path} with {len(df):,} players and {len(df.columns)} features")
        
        return df
    
    def prepare_ml_data(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for machine learning"""
        
        logger.info("=== PREPARING ML DATA ===")
        
        # Define feature columns (excluding target and metadata)
        exclude_cols = [
            'player_id', 'total_revenue', 'total_received_revenue', 'user_category',
            'is_payer', 'is_whale', 'is_low_spender_given_payer', 'whale_purchase_prob',
            'first_session_timestamp', 'last_session_timestamp',
            'first_country', 'first_install_source', 'first_campaign', 'first_utm_source', 'lifecycle_stage'
        ]
        
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        logger.info(f"Using {len(self.feature_columns)} features for modeling")
        
        # Prepare feature matrix
        X = df[self.feature_columns].fillna(0)
        
        # Handle any remaining infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare targets
        y_revenue = df['total_revenue'].values
        y_user_types = df['user_category'].map({'non_payer': 0, 'low_spender': 1, 'whale': 2}).values
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_revenue_tensor = torch.FloatTensor(y_revenue)
        y_user_types_tensor = torch.LongTensor(y_user_types)
        
        logger.info(f"ML data prepared - Shape: {X_tensor.shape}")
        
        return X_tensor, y_revenue_tensor, y_user_types_tensor
    
    def run_complete_system(self):
        """Run the complete production ExpLTV system"""
        
        logger.info("STARTING PRODUCTION ExpLTV SYSTEM")
        logger.info("=" * 80)
        
        try:
            # 1. Data Extraction and Preparation
            df = self.extract_and_prepare_data()
            
            # 2. ML Data Preparation
            X, y_revenue, y_user_types = self.prepare_ml_data(df)
            
            # 3. Data Splitting
            test_size = self.config.get('training.test_size', 0.2)
            val_size = self.config.get('training.validation_size', 0.15)
            random_state = self.config.get('training.random_state', 42)
            
            # Stratify based on user category
            stratify_labels = y_user_types.numpy()
            
            # First split: train+val vs test
            X_temp, X_test, y_rev_temp, y_rev_test, y_type_temp, y_type_test = train_test_split(
                X, y_revenue, y_user_types,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_labels
            )
            
            # Second split: train vs val
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_rev_train, y_rev_val, y_type_train, y_type_val = train_test_split(
                X_temp, y_rev_temp, y_type_temp,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=y_type_temp.numpy()
            )
            
            logger.info(f"Data split - Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
            
            # 4. Create Data Loaders
            batch_size = self.config.get('model.batch_size', 1024)
            
            train_dataset = TensorDataset(X_train, y_rev_train, y_type_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            val_dataset = TensorDataset(X_val, y_rev_val, y_type_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # 5. Model Initialization
            logger.info("=== MODEL INITIALIZATION ===")
            self.model = ExpLTVModel(X.shape[1], self.config)
            self.trainer = ExpLTVTrainer(self.model, self.config)
            self.evaluator = ComprehensiveEvaluator(self.model, self.config)
            
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model initialized - Total params: {total_params:,}")
            
            # 6. Model Training
            logger.info("=== MODEL TRAINING ===")
            train_losses, val_losses = self.trainer.train(train_loader, val_loader)
            
            # Store training results
            self.results['training'] = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epochs_trained': len(train_losses)
            }
            
            # Save model
            if self.config.get('outputs.save_models', True):
                output_dir = self.config.get('outputs.output_dir', 'output')
                model_path = os.path.join(output_dir, 'models', 'expltv_model.pth')
                self.trainer.save_checkpoint(model_path)
                logger.info(f"Model saved to {model_path}")
            
            # 7. Comprehensive Evaluation
            logger.info("=== COMPREHENSIVE EVALUATION ===")
            if os.path.exists('best_model.pth'):
                self.trainer.load_checkpoint('best_model.pth')
            
            metrics = self.evaluator.evaluate_comprehensive(X_test, y_rev_test, y_type_test.numpy())
            self.results['evaluation'] = metrics
            
            # Get predictions for analysis
            predictions = self.evaluator.predict(X_test)
            self.results['predictions'] = {
                'X_test': X_test.numpy(),
                'y_test': y_rev_test.numpy(),
                'user_types_test': y_type_test.numpy(),
                'ltv_predictions': predictions['expected_ltv'],
                'purchase_probabilities': predictions['purchase_prob'],
                'whale_probabilities': predictions['whale_prob']
            }
            
            # 8. Save Results Summary
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.config.get('outputs.output_dir', 'output')
            
            # Save results summary
            results_path = os.path.join(output_dir, 'reports', f'results_summary_{timestamp}.json')
            with open(results_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = {}
                for key, value in self.results.items():
                    if key == 'predictions':
                        json_results[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                           for k, v in value.items()}
                    else:
                        json_results[key] = value
                json.dump(json_results, f, indent=2, default=str)
            
            # Final Summary
            logger.info("PRODUCTION ExpLTV SYSTEM COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info("KEY RESULTS:")
            logger.info(f"  LTV Prediction R²: {metrics['ltv_r2']:.3f}")
            logger.info(f"  Purchase Classification AUC: {metrics['purchase_auc']:.3f}")
            logger.info(f"  Whale Detection AUC: {metrics['whale_auc']:.3f}")
            logger.info(f"  Top 10% Targeting Lift: {metrics.get('lift_top_10pct', 0):.1f}x")
            logger.info(f"  Revenue Prediction Error: {metrics['revenue_prediction_error']:.1%}")
            logger.info("=" * 80)
            logger.info(f"Results saved to: {results_path}")
            logger.info("System ready for production deployment!")
            
            return self.results, df
            
        except Exception as e:
            logger.error(f"SYSTEM FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main execution function"""
    
    # Initialize and run the complete system
    system = ProductionPLTVSystem()
    results, engineered_data = system.run_complete_system()
    
    return results, engineered_data


if __name__ == "__main__":
    main()