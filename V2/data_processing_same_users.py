# data_processing_same_users.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import TRAIN_SAMPLES, VAL_SAMPLES, TEST_SAMPLES, TRAIN_CSV, VAL_CSV, TEST_CSV
from load_data_batched import load_prediction_data_batched
from simple_features_clean import create_prediction_features


def prepare_datasets_same_users(training_df: pd.DataFrame):
    """Use SAME users for all three sets, just different time periods"""
    
    # Remove rows with missing targets
    training_df = training_df.dropna(subset=['ltv_30_days', 'is_whale'])
    
    # Get user list - use SAME users for all sets
    user_ids = training_df['user_id'].unique()
    print(f"   Total available users: {len(user_ids):,}")
    # Sample users if needed (but same users go to all three sets)
    max_users = max(TRAIN_SAMPLES, VAL_SAMPLES, TEST_SAMPLES) 
    print(f"   Total available users: {max_users:,}") 
    if len(user_ids) > max_users:
        sampled_user_ids = np.random.choice(user_ids, size=max_users, replace=False)
    else:
        sampled_user_ids = user_ids
    
    print(f"   Using SAME {len(sampled_user_ids):,} users for all three sets")
    
    # ===== TRAIN DATA: Same users with D0-D3 features =====
    train_data = training_df[training_df['user_id'].isin(sampled_user_ids)].copy()
    print("train data: ", train_data)
    # Limit to TRAIN_SAMPLES if needed
    if len(train_data) > TRAIN_SAMPLES:
        train_data = train_data.sample(n=TRAIN_SAMPLES, random_state=42)
    
    print(f"   Train: {len(train_data):,} users with D0-D3 features")
    
    # ===== VAL + TEST DATA: Same users with D4-D33 features =====
    print(f"   Loading D4-D33 features for validation and test...")
    
    # Load D4-D33 features for the same users
    prediction_df = load_prediction_data_batched(sampled_user_ids)
    print("sampled users:",sampled_user_ids)
    if len(prediction_df) > 0:
        prediction_df = create_prediction_features(prediction_df)
        print("prediction df: ", len(prediction_df))
        # Get targets for same users
        targets = training_df[training_df['user_id'].isin(sampled_user_ids)][['user_id', 'ltv_30_days', 'is_whale']]
        full_prediction_data = prediction_df.merge(targets, on='user_id', how='left')
        print("full prediction data: ", len(full_prediction_data))
        
        # ===== SPLIT D4-D33 data into VAL and TEST =====
        # Use the same users but split them randomly into val/test
        val_data, test_data = train_test_split(
            full_prediction_data,
            test_size=0.5,  # Split exactly in half
            random_state=42,
            stratify=full_prediction_data['is_whale'] if full_prediction_data['is_whale'].sum() > 1 else None
        )
        
        # Limit sizes if needed
        if len(val_data) > VAL_SAMPLES:
            val_data = val_data.sample(n=VAL_SAMPLES, random_state=42)
        if len(test_data) > TEST_SAMPLES:
            test_data = test_data.sample(n=TEST_SAMPLES, random_state=42)
            
    else:
        print("   Warning: No D4-D33 data found for prediction")
        val_data = pd.DataFrame()
        test_data = pd.DataFrame()
    
    print(f"   Val: {len(val_data):,} users with D4-D33 features")
    print(f"   Test: {len(test_data):,} users with D4-D33 features")
    
    # Handle categorical features
    label_encoders = {}
    categorical_features = ['channel', 'country', 'partner', 'os', 'campaign_name', 'publisher_name']
    
    # Collect all unique values first
    all_datasets = [train_data, val_data, test_data]
    for col in categorical_features:
        all_values = []
        for df in all_datasets:
            if len(df) > 0 and col in df.columns:
                all_values.extend(df[col].astype(str).unique())
        
        if all_values:
            le = LabelEncoder()
            le.fit(list(set(all_values)))
            label_encoders[col] = le
            
            # Apply to all datasets
            for df in all_datasets:
                if len(df) > 0 and col in df.columns:
                    df[col] = df[col].astype(str)
                    df[col] = le.transform(df[col])
    
    # Fill missing values
    for dataset in all_datasets:
        if len(dataset) > 0:
            numeric_cols = dataset.select_dtypes(include=[np.number]).columns
            dataset[numeric_cols] = dataset[numeric_cols].fillna(0)
    
    return train_data, val_data, test_data, label_encoders


def save_datasets(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame, label_encoders: dict = None):
    """Save datasets to CSV files"""
    
    print(f"Saving train data ({len(train_data):,} rows) to {TRAIN_CSV}")
    train_data.to_csv(TRAIN_CSV, index=False)
    
    print(f"Saving validation data ({len(val_data):,} rows) to {VAL_CSV}")
    val_data.to_csv(VAL_CSV, index=False)
    
    print(f"Saving test data ({len(test_data):,} rows) to {TEST_CSV}")
    test_data.to_csv(TEST_CSV, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("SAME USERS, DIFFERENT TIME PERIODS")
    print("="*60)
    
    for name, data in [("Train", train_data), ("Validation", val_data), ("Test", test_data)]:
        if len(data) > 0:
            whale_count = data['is_whale'].sum()
            whale_pct = whale_count / len(data) * 100
            avg_ltv = data['ltv_30_days'].mean()
            
            print(f"\n{name} Set:")
            print(f"  Users: {len(data):,}")
            print(f"  Whales: {whale_count:,} ({whale_pct:.1f}%)")
            print(f"  Average LTV: ${avg_ltv:.2f}")
            print(f"  Feature Period: {'D0-D3' if name == 'Train' else 'D4-D33'}")
            print(f"  Target Period: D4-D33 (same for all)")
    
    print(f"\nKey Insight:")
    print(f"- Train: Early behavior (D0-D3) predicting D4-D33 LTV")
    print(f"- Val/Test: Later behavior (D4-D33) predicting same D4-D33 LTV")  
    print(f"- This tests: 'Is early behavior as predictive as later behavior?'")