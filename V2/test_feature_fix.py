#!/usr/bin/env python3

import pandas as pd
import numpy as np
from simple_features_optimized import create_training_features, create_prediction_features

def test_feature_consistency():
    """Test that feature engineering creates consistent features across datasets"""
    
    # Create mock datasets with different categorical distributions
    np.random.seed(42)
    
    # Training data
    train_data = pd.DataFrame({
        'user_id': range(100),
        'total_sessions': np.random.randint(1, 50, 100),
        'total_events': np.random.randint(10, 500, 100),
        'active_days': np.random.randint(1, 7, 100),
        'total_playtime_mins': np.random.randint(10, 1000, 100),
        'time_to_first_session_mins': np.random.randint(0, 60, 100),
        'store_interactions': np.random.randint(0, 50, 100),
        'unique_products_viewed': np.random.randint(0, 20, 100),
        'unique_skus_viewed': np.random.randint(0, 25, 100),
        'total_quantity_interactions': np.random.randint(0, 100, 100),
        'engagement_events': np.random.randint(5, 200, 100),
        'business_hours_events': np.random.randint(5, 200, 100),
        'weekend_events': np.random.randint(2, 100, 100),
        'os': np.random.choice(['android', 'ios'], 100),
        'country': np.random.choice(['US', 'IN', 'PK', 'ID'], 100),  # Training has these countries
        'channel': np.random.choice(['vending', 'installer', 'samsung'], 100),
        'ltv_30_days': np.random.exponential(2, 100),
        'purchase_events_30d': np.random.randint(0, 5, 100),
        'is_whale': np.random.randint(0, 2, 100)
    })
    
    # Validation data - different country distribution
    val_data = pd.DataFrame({
        'user_id': range(100, 150),
        'total_sessions': np.random.randint(1, 50, 50),
        'total_events': np.random.randint(10, 500, 50),
        'active_days': np.random.randint(1, 7, 50),
        'total_playtime_mins': np.random.randint(10, 1000, 50),
        'time_to_first_session_mins': np.random.randint(0, 60, 50),
        'store_interactions': np.random.randint(0, 50, 50),
        'unique_products_viewed': np.random.randint(0, 20, 50),
        'unique_skus_viewed': np.random.randint(0, 25, 50),
        'total_quantity_interactions': np.random.randint(0, 100, 50),
        'engagement_events': np.random.randint(5, 200, 50),
        'business_hours_events': np.random.randint(5, 200, 50),
        'weekend_events': np.random.randint(2, 100, 50),
        'os': np.random.choice(['android', 'ios'], 50),
        'country': np.random.choice(['US', 'PH', 'BD'], 50),  # Validation has PH, BD instead of IN, PK
        'channel': np.random.choice(['vending', 'xiaomi', 'vivo'], 50),  # Different channels
        'ltv_30_days': np.random.exponential(2, 50),
        'purchase_events_30d': np.random.randint(0, 5, 50),
        'is_whale': np.random.randint(0, 2, 50)
    })
    
    # Test data - yet different distribution
    test_data = pd.DataFrame({
        'user_id': range(150, 180),
        'total_sessions': np.random.randint(1, 50, 30),
        'total_events': np.random.randint(10, 500, 30),
        'active_days': np.random.randint(1, 7, 30),
        'total_playtime_mins': np.random.randint(10, 1000, 30),
        'time_to_first_session_mins': np.random.randint(0, 60, 30),
        'store_interactions': np.random.randint(0, 50, 30),
        'unique_products_viewed': np.random.randint(0, 20, 30),
        'unique_skus_viewed': np.random.randint(0, 25, 30),
        'total_quantity_interactions': np.random.randint(0, 100, 30),
        'engagement_events': np.random.randint(5, 200, 30),
        'business_hours_events': np.random.randint(5, 200, 30),
        'weekend_events': np.random.randint(2, 100, 30),
        'os': np.random.choice(['android', 'ios'], 30),
        'country': np.random.choice(['CA', 'UK'], 30),  # Completely different countries
        'channel': np.random.choice(['playstore', 'appstore'], 30),  # Different channels
        'ltv_30_days': np.random.exponential(2, 30),
        'purchase_events_30d': np.random.randint(0, 5, 30),
        'is_whale': np.random.randint(0, 2, 30)
    })
    
    print("Testing feature engineering consistency...")
    print(f"Train countries: {train_data['country'].unique()}")
    print(f"Val countries: {val_data['country'].unique()}")
    print(f"Test countries: {test_data['country'].unique()}")
    
    # Apply feature engineering
    train_engineered = create_training_features(train_data)
    val_engineered = create_prediction_features(val_data)
    test_engineered = create_prediction_features(test_data)
    
    print(f"\nFeature counts:")
    print(f"Train: {len(train_engineered.columns)} columns")
    print(f"Val: {len(val_engineered.columns)} columns")
    print(f"Test: {len(test_engineered.columns)} columns")
    
    # Check for common columns (this was the issue before)
    common_cols = set(train_engineered.columns) & set(val_engineered.columns) & set(test_engineered.columns)
    print(f"\nCommon columns across all datasets: {len(common_cols)}")
    
    # Test the prepare_features logic
    from main import prepare_features
    try:
        (X_train, X_val, X_test, 
         y_ltv_train, y_ltv_val, y_ltv_test,
         y_whale_train, y_whale_val, y_whale_test,
         feature_cols) = prepare_features(train_engineered, val_engineered, test_engineered)
        
        print(f"\n✅ prepare_features successful!")
        print(f"Final feature count: {len(feature_cols)}")
        print(f"Train shape: {X_train.shape}")
        print(f"Val shape: {X_val.shape}")
        print(f"Test shape: {X_test.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ prepare_features failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_feature_consistency()
    if success:
        print("\n🎉 Feature consistency fix works!")
    else:
        print("\n❌ Still has issues")