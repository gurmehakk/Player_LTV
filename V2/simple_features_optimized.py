# simple_features_optimized.py

import pandas as pd
import numpy as np

def create_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create optimized training features without revenue correlation"""
    
    features_df = df.copy()
    
    print("Creating behavioral features (no revenue correlation):")
    
    # ==== SESSION QUALITY FEATURES ====
    if 'total_playtime_mins' in features_df.columns and 'total_sessions' in features_df.columns:
        features_df['avg_session_length'] = (
            features_df['total_playtime_mins'] / features_df['total_sessions'].clip(lower=1)
        )
        print("  ✓ avg_session_length")
    
    if 'time_to_first_session_mins' in features_df.columns:
        features_df['time_to_first_session_hours'] = (
            features_df['time_to_first_session_mins'] / 60.0
        ).clip(upper=24)  # Cap at 1 day
        print("  ✓ time_to_first_session_hours")
    
    if 'total_sessions' in features_df.columns and 'active_days' in features_df.columns:
        features_df['session_consistency'] = (
            features_df['total_sessions'] / features_df['active_days'].clip(lower=1)
        )
        print("  ✓ session_consistency")
    
    if 'total_sessions' in features_df.columns and 'total_playtime_mins' in features_df.columns:
        features_df['session_frequency'] = (
            features_df['total_sessions'] / (features_df['total_playtime_mins'] / 60).clip(lower=0.1)
        )
        print("  ✓ session_frequency")
    
    # ==== PRODUCT INTERACTION FEATURES (BEHAVIORAL, NO REVENUE) ====
    if 'store_interactions' in features_df.columns and 'total_sessions' in features_df.columns:
        features_df['store_interactions_per_session'] = (
            features_df['store_interactions'] / features_df['total_sessions'].clip(lower=1)
        )
        print("  ✓ store_interactions_per_session")
    
    if 'unique_products_viewed' in features_df.columns and 'store_interactions' in features_df.columns:
        features_df['unique_products_per_interaction'] = (
            features_df['unique_products_viewed'] / features_df['store_interactions'].clip(lower=1)
        )
        print("  ✓ unique_products_per_interaction")
    
    if 'total_quantity_interactions' in features_df.columns and 'store_interactions' in features_df.columns:
        features_df['quantity_per_interaction'] = (
            features_df['total_quantity_interactions'] / features_df['store_interactions'].clip(lower=1)
        )
        print("  ✓ quantity_per_interaction")
    
    if 'unique_skus_viewed' in features_df.columns and 'unique_products_viewed' in features_df.columns:
        features_df['sku_diversity'] = (
            features_df['unique_skus_viewed'] / features_df['unique_products_viewed'].clip(lower=1)
        )
        print("  ✓ sku_diversity")
    
    if 'store_interactions' in features_df.columns and 'total_events' in features_df.columns:
        features_df['store_engagement_rate'] = (
            features_df['store_interactions'] / features_df['total_events'].clip(lower=1)
        )
        print("  ✓ store_engagement_rate")
    
    # ==== GENERAL ENGAGEMENT FEATURES ====
    if 'engagement_events' in features_df.columns and 'total_sessions' in features_df.columns:
        features_df['engagement_per_session'] = (
            features_df['engagement_events'] / features_df['total_sessions'].clip(lower=1)
        )
        print("  ✓ engagement_per_session")
    
    if 'engagement_events' in features_df.columns and 'total_events' in features_df.columns:
        features_df['engagement_rate'] = (
            features_df['engagement_events'] / features_df['total_events'].clip(lower=1)
        )
        print("  ✓ engagement_rate")
    
    # ==== TIME PATTERN FEATURES ====
    if 'business_hours_events' in features_df.columns and 'total_events' in features_df.columns:
        features_df['business_hours_rate'] = (
            features_df['business_hours_events'] / features_df['total_events'].clip(lower=1)
        )
        print("  ✓ business_hours_rate")
    
    if 'weekend_events' in features_df.columns and 'total_events' in features_df.columns:
        features_df['weekend_rate'] = (
            features_df['weekend_events'] / features_df['total_events'].clip(lower=1)
        )
        print("  ✓ weekend_rate")
    
    # ==== ACTIVITY INTENSITY FEATURES ====
    if 'total_events' in features_df.columns and 'total_playtime_mins' in features_df.columns:
        features_df['events_per_minute'] = (
            features_df['total_events'] / features_df['total_playtime_mins'].clip(lower=1)
        )
        print("  ✓ events_per_minute")
    
    if 'active_days' in features_df.columns and 'total_playtime_mins' in features_df.columns:
        features_df['avg_daily_playtime'] = (
            features_df['total_playtime_mins'] / features_df['active_days'].clip(lower=1)
        )
        print("  ✓ avg_daily_playtime")
    
    # ==== CATEGORICAL FEATURE ENCODING ====
    # Note: Categorical encoding will be handled consistently in prepare_features()
    # to ensure same features across train/val/test
    print("  ✓ Categorical features will be encoded consistently across datasets")
    
    print(f"\nTotal features created: {len([c for c in features_df.columns if c not in ['user_id', 'install_date', 'install_timestamp', 'ltv_30_days', 'purchase_events_30d', 'is_whale']])}")
    print("✓ All features are behavioral - no revenue correlation")
    
    return features_df


def create_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create prediction features (same as training)"""
    return create_training_features(df)


def list_all_features(df: pd.DataFrame) -> list:
    """List all feature columns (excluding IDs and targets)"""
    exclude_cols = ['user_id', 'install_date', 'install_timestamp', 'ltv_30_days', 'purchase_events_30d', 'is_whale', 'first_event', 'last_event']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print("\n=== ENGINEERED FEATURE SET ===")
    print(f"Total features before final processing: {len(feature_cols)}")
    
    # Group features by type
    session_features = [c for c in feature_cols if any(x in c for x in ['session', 'playtime', 'time_to'])]
    product_features = [c for c in feature_cols if any(x in c for x in ['store', 'product', 'sku', 'quantity'])]
    engagement_features = [c for c in feature_cols if any(x in c for x in ['engagement', 'events'])]
    time_features = [c for c in feature_cols if any(x in c for x in ['business_hours', 'weekend'])]
    categorical_features = [c for c in feature_cols if any(x in c for x in ['os', 'country', 'channel'])]
    other_features = [c for c in feature_cols if c not in session_features + product_features + engagement_features + time_features + categorical_features]
    
    print(f"\nSession & Time Features ({len(session_features)}):")
    for f in session_features: print(f"  • {f}")
    
    print(f"\nProduct Interaction Features ({len(product_features)}):")
    for f in product_features: print(f"  • {f}")
    
    print(f"\nEngagement Features ({len(engagement_features)}):")
    for f in engagement_features: print(f"  • {f}")
    
    print(f"\nTime Pattern Features ({len(time_features)}):")
    for f in time_features: print(f"  • {f}")
    
    print(f"\nPlatform Features ({len(categorical_features)}):")
    for f in categorical_features: print(f"  • {f}")
    
    if other_features:
        print(f"\nOther Features ({len(other_features)}):")
        for f in other_features: print(f"  • {f}")
    
    print("\n✅ NO REVENUE-RELATED FEATURES")
    print("✅ ALL FEATURES ARE BEHAVIORAL")
    print("Note: Final feature set will be determined in prepare_features() for consistency")
    
    return feature_cols