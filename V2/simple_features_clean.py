# simple_features_clean.py

import pandas as pd
import numpy as np
from config import WHALE_THRESHOLD


def create_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create streamlined training features without redundancy"""
    
    features_df = df.copy()
    
    # ==== SESSION QUALITY FEATURES ====
    features_df['avg_session_length'] = (
        features_df['total_playtime_mins'] / features_df['total_sessions'].clip(lower=1)
    )
    
    features_df['time_to_first_session_hours'] = (
        features_df['time_to_first_session_mins'] / 60.0
    ).clip(upper=24)  # Cap at 1 day
    
    features_df['session_consistency'] = (
        features_df['total_sessions'] / features_df['active_days'].clip(lower=1)
    )
    
    features_df['session_frequency'] = (
        features_df['total_sessions'] / (features_df['total_playtime_mins'] / 60).clip(lower=0.1)
    )
    
    # ==== STORE/PRODUCT ENGAGEMENT FEATURES ====
    features_df['store_interactions_per_session'] = (
        features_df['store_interactions'] / features_df['total_sessions'].clip(lower=1)
    )
    
    features_df['unique_products_per_interaction'] = (
        features_df['unique_products_viewed'] / features_df['store_interactions'].clip(lower=1)
    )
    
    features_df['quantity_per_interaction'] = (
        features_df['total_quantity_interactions'] / features_df['store_interactions'].clip(lower=1)
    )
    
    # Keep only sku_diversity (not its inverse)
    features_df['sku_diversity'] = (
        features_df['unique_skus_viewed'] / features_df['unique_products_viewed'].clip(lower=1)
    )
    
    # ==== TIME PATTERN FEATURES ====
    features_df['morning_event_rate'] = (
        features_df['morning_events'] / features_df['total_events'].clip(lower=1)
    )
    
    features_df['evening_event_rate'] = (
        features_df['evening_events'] / features_df['total_events'].clip(lower=1)
    )
    
    features_df['weekend_event_rate'] = (
        features_df['weekend_events'] / features_df['total_events'].clip(lower=1)
    )
    
    # Keep only one time diversity metric
    features_df['time_diversity_score'] = (
        features_df['unique_hours_active'] / features_df['active_days'].clip(lower=1)
    )
    
    # ==== ENGAGEMENT INTENSITY FEATURES ====
    features_df['events_per_hour'] = (
        features_df['total_events'] / features_df['unique_hours_active'].clip(lower=1)
    )
    
    features_df['daily_engagement_consistency'] = (
        features_df['total_events'] / (features_df['active_days'] * features_df['unique_hours_active']).clip(lower=1)
    )
    
    # ==== ATTRIBUTION FEATURES ====
    features_df['attribution_confidence'] = (
        features_df['is_fingerprinted_user'] + features_df['has_touchpoint_data']
    )
    
    features_df['is_organic_user'] = (
        features_df['channel'].str.lower().str.contains('organic|direct', na=False)
    ).astype(int) if 'channel' in features_df.columns else 0
    
    # ==== SESSION TIMING BINARY FEATURES ====
    if 'first_session_time_bucket' in features_df.columns:
        # Create binary flags for time buckets (remove categorical versions)
        for bucket in ['morning', 'afternoon', 'evening', 'night']:
            features_df[f'first_session_{bucket}'] = (
                features_df['first_session_time_bucket'] == bucket
            ).astype(int)
    
    # Weekly patterns
    if 'first_session_dayofweek' in features_df.columns:
        features_df['first_session_weekend'] = (
            features_df['first_session_dayofweek'].isin([1, 7])  # Sunday=1, Saturday=7
        ).astype(int)
    
    # ==== CLEANUP ====
    # Fill NaN values
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_cols] = features_df[numeric_cols].fillna(0)
    
    # Fill categorical NaN values (keep only essential categoricals)
    categorical_cols = ['channel', 'country', 'partner', 'campaign_name', 'publisher_name', 'os', 'user_state']
    for col in categorical_cols:
        if col in features_df.columns:
            features_df[col] = features_df[col].fillna('unknown')
    
    # Create target variables (only for training data)
    if 'ltv_30_days' in features_df.columns:
        features_df['is_whale'] = (features_df['ltv_30_days'] >= WHALE_THRESHOLD).astype(int)
    
    return features_df


def create_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create same streamlined features for prediction data (no targets)"""
    return create_training_features(df)  # Same logic, just no targets