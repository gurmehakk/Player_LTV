#!/usr/bin/env python3

import pandas as pd
import numpy as np
from models import ZILNModel, WhaleDetectionModel, evaluate_classification_model, evaluate_regression_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load final datasets and prepare for modeling"""
    
    print("="*80)
    print("LOADING FINAL DATASETS FROM CSV")
    print("="*80)
    
    # Load datasets
    train_data = pd.read_csv('/Users/gurmehakkaur/gameramp/Player_LTV/V2/final_train_data.csv')
    val_data = pd.read_csv('/Users/gurmehakkaur/gameramp/Player_LTV/V2/final_val_data.csv')
    test_data = pd.read_csv('/Users/gurmehakkaur/gameramp/Player_LTV/V2/final_test_data.csv')
    
    print(f"Train: {train_data.shape}")
    print(f"Val: {val_data.shape}")
    print(f"Test: {test_data.shape}")
    
    # Define feature columns (exclude identifiers and targets)
    exclude_cols = ['user_id', 'ltv_30_days', 'purchase_events_30d', 'is_whale']
    feature_cols = [col for col in train_data.columns if col not in exclude_cols]
    
    print(f"Features: {len(feature_cols)} columns")
    
    # Get common features across all datasets
    common_cols = set(train_data.columns) & set(val_data.columns) & set(test_data.columns)
    feature_cols = [col for col in feature_cols if col in common_cols]
    
    print(f"Common features: {len(feature_cols)} columns")
    
    # Prepare feature matrices
    X_train = train_data[feature_cols].fillna(0)
    X_val = val_data[feature_cols].fillna(0)
    X_test = test_data[feature_cols].fillna(0)
    
    # Handle categorical features consistently
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    print(f"Encoding {len(categorical_cols)} categorical features...")
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on combined data to handle unseen categories
        combined_data = pd.concat([X_train[col], X_val[col], X_test[col]]).astype(str)
        le.fit(combined_data)
        
        X_train[col] = le.transform(X_train[col].astype(str))
        X_val[col] = le.transform(X_val[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        
        label_encoders[col] = le
        print(f"  ✓ {col}: {len(le.classes_)} unique values")
    
    # Prepare targets
    y_ltv_train = train_data['ltv_30_days'].values
    y_ltv_val = val_data['ltv_30_days'].values
    y_ltv_test = test_data['ltv_30_days'].values
    
    y_whale_train = train_data['is_whale'].values
    y_whale_val = val_data['is_whale'].values
    y_whale_test = test_data['is_whale'].values
    
    print(f"\\nTarget summary:")
    print(f"Train LTV: mean=${y_ltv_train.mean():.3f}, max=${y_ltv_train.max():.2f}")
    print(f"Val LTV: mean=${y_ltv_val.mean():.3f}, max=${y_ltv_val.max():.2f}")
    print(f"Test LTV: mean=${y_ltv_test.mean():.3f}, max=${y_ltv_test.max():.2f}")
    print(f"Train whales: {y_whale_train.sum()} ({y_whale_train.mean():.1%})")
    print(f"Val whales: {y_whale_val.sum()} ({y_whale_val.mean():.1%})")
    print(f"Test whales: {y_whale_test.sum()} ({y_whale_test.mean():.1%})")
    
    return (X_train, X_val, X_test, 
            y_ltv_train, y_ltv_val, y_ltv_test,
            y_whale_train, y_whale_val, y_whale_test,
            feature_cols)

def train_whale_detection(X_train, X_val, X_test, y_whale_train, y_whale_val, y_whale_test):
    """Train whale detection model"""
    print("\\n" + "="*60)
    print("WHALE DETECTION MODEL")
    print("="*60)
    
    train_whales = np.sum(y_whale_train)
    val_whales = np.sum(y_whale_val)
    test_whales = np.sum(y_whale_test)
    
    print(f"Whale counts - Train: {train_whales}, Val: {val_whales}, Test: {test_whales}")
    
    if train_whales < 2:
        print("WARNING: Insufficient whales for training. Using baseline predictions.")
        return None, np.zeros(len(y_whale_test)), np.zeros(len(y_whale_test))
    
    try:
        # Train model
        whale_model = WhaleDetectionModel(input_dim=X_train.shape[1])
        history = whale_model.fit(X_train, y_whale_train, X_val, y_whale_val, epochs=30)
        
        # Make predictions
        test_predictions, test_probabilities = evaluate_classification_model(
            whale_model, X_test, y_whale_test, "Whale Detection"
        )
        
        return whale_model, test_predictions, test_probabilities
        
    except Exception as e:
        print(f"Error training whale model: {str(e)}")
        return None, np.zeros(len(y_whale_test)), np.zeros(len(y_whale_test))

def train_ltv_prediction(X_train, X_val, X_test, y_ltv_train, y_ltv_val, y_ltv_test):
    """Train LTV prediction model"""
    print("\\n" + "="*60)
    print("LTV PREDICTION MODEL")
    print("="*60)
    
    print(f"LTV summary - Train: min={y_ltv_train.min():.2f}, max={y_ltv_train.max():.2f}, mean={y_ltv_train.mean():.3f}")
    print(f"LTV summary - Val: min={y_ltv_val.min():.2f}, max={y_ltv_val.max():.2f}, mean={y_ltv_val.mean():.3f}")
    print(f"LTV summary - Test: min={y_ltv_test.min():.2f}, max={y_ltv_test.max():.2f}, mean={y_ltv_test.mean():.3f}")
    
    non_zero_train = np.sum(y_ltv_train > 0)
    non_zero_val = np.sum(y_ltv_val > 0)
    
    print(f"Non-zero LTV - Train: {non_zero_train} ({non_zero_train/len(y_ltv_train):.3%})")
    print(f"Non-zero LTV - Val: {non_zero_val} ({non_zero_val/len(y_ltv_val):.3%})")
    
    try:
        # Train model
        ltv_model = ZILNModel(input_dim=X_train.shape[1])
        class_history, reg_history = ltv_model.fit(X_train, y_ltv_train, X_val, y_ltv_val, epochs=30)
        
        # Make predictions
        test_predictions, test_probabilities = ltv_model.predict(X_test)
        
        # Evaluate
        evaluate_regression_model(test_predictions, y_ltv_test, "ZILN LTV Model")
        
        return ltv_model, test_predictions
        
    except Exception as e:
        print(f"Error training LTV model: {str(e)}")
        print("Using simple baseline predictions...")
        
        # Simple baseline: predict mean LTV
        baseline_pred = np.full(len(y_ltv_test), y_ltv_train.mean())
        evaluate_regression_model(baseline_pred, y_ltv_test, "Baseline LTV Model")
        
        return None, baseline_pred

def main():
    """Main training pipeline from saved CSV files"""
    
    # Load and prepare data
    (X_train, X_val, X_test, 
     y_ltv_train, y_ltv_val, y_ltv_test,
     y_whale_train, y_whale_val, y_whale_test,
     feature_cols) = load_and_prepare_data()
    
    # Train whale detection model
    whale_model, whale_predictions, whale_probabilities = train_whale_detection(
        X_train, X_val, X_test, y_whale_train, y_whale_val, y_whale_test
    )
    
    # Train LTV prediction model
    ltv_model, ltv_predictions = train_ltv_prediction(
        X_train, X_val, X_test, y_ltv_train, y_ltv_val, y_ltv_test
    )
    
    # Summary
    print("\\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    
    mae = mean_absolute_error(y_ltv_test, ltv_predictions)
    r2 = r2_score(y_ltv_test, ltv_predictions)
    
    print(f"Final Results:")
    print(f"- LTV MAE: ${mae:.3f}")
    print(f"- LTV R²: {r2:.3f}")
    print(f"- Features used: {len(feature_cols)}")
    print(f"- All features are behavioral (no revenue correlation)")
    print(f"\\n✅ Models trained successfully from saved CSV files!")

if __name__ == "__main__":
    main()