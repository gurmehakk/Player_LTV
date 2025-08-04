#!/usr/bin/env python3

import pandas as pd
import numpy as np
from google.cloud import bigquery
from simple_features_optimized import create_training_features, create_prediction_features, list_all_features
from config import PROJECT_ID, DATASET, TABLE, FEATURE_DAYS, PREDICTION_DAYS

def load_and_save_final_datasets():
    """Load datasets, apply feature engineering, and save as final CSVs"""
    
    print("="*80)
    print("LOADING AND SAVING FINAL DATASETS")
    print("="*80)
    
    client = bigquery.Client(project=PROJECT_ID)
    table_path = f"{PROJECT_ID}.{DATASET}.{TABLE}"
    
    # Base query template
    base_query = f"""
    WITH install_dates AS (
      SELECT
        COALESCE(gaid, idfa, android_id, waid, idfv) AS user_id,
        DATE(MIN(attribution_event_timestamp)) AS install_date,
        MIN(attribution_event_timestamp) AS install_timestamp
      FROM `{table_path}`
      WHERE DATE(attribution_event_timestamp) >= '{{start_date}}'
        AND DATE(attribution_event_timestamp) <= '{{end_date}}'
        AND (gaid IS NOT NULL OR idfa IS NOT NULL OR android_id IS NOT NULL)
      GROUP BY user_id
      HAVING COUNT(*) >= 5
      {{limit_clause}}
    ),
    
    features AS (
      SELECT
        COALESCE(e.gaid, e.idfa, e.android_id, e.waid, e.idfv) AS user_id,
        i.install_date,
        i.install_timestamp,
        
        -- Core behavioral features
        COUNT(DISTINCT e.session_id) AS total_sessions,
        COUNT(*) AS total_events,
        COUNT(DISTINCT DATE(e.attribution_event_timestamp)) AS active_days,
        
        -- Product interaction features (behavioral only, no revenue)
        COUNT(CASE WHEN e.product_name IS NOT NULL THEN 1 END) AS store_interactions,
        COUNT(DISTINCT e.product_name) AS unique_products_viewed,
        COUNT(DISTINCT e.product_sku) AS unique_skus_viewed,
        AVG(SAFE_CAST(e.product_quantity AS FLOAT64)) AS avg_quantity_per_interaction,
        SUM(SAFE_CAST(e.product_quantity AS FLOAT64)) AS total_quantity_interactions,
        
        -- General engagement features
        COUNT(CASE WHEN e.session_id IS NOT NULL THEN 1 END) AS engagement_events,
        
        -- Platform and timing
        MAX(CASE WHEN e.gaid IS NOT NULL THEN 'android' 
                WHEN e.idfa IS NOT NULL THEN 'ios' 
                ELSE 'unknown' END) AS os,
        MAX(e.country) AS country,
        MAX(e.install_source) AS channel,
        
        -- Time patterns
        COUNTIF(EXTRACT(HOUR FROM e.attribution_event_timestamp) BETWEEN 9 AND 17) AS business_hours_events,
        COUNTIF(EXTRACT(DAYOFWEEK FROM e.attribution_event_timestamp) IN (1, 7)) AS weekend_events
                
      FROM `{table_path}` e
      JOIN install_dates i ON COALESCE(e.gaid, e.idfa, e.android_id, e.waid, e.idfv) = i.user_id
      WHERE {{feature_date_filter}}
      GROUP BY user_id, i.install_date, i.install_timestamp
    ),
    
    targets AS (
      SELECT
        COALESCE(e.gaid, e.idfa, e.android_id, e.waid, e.idfv) AS user_id,
        COALESCE(SUM(SAFE_CAST(e.converted_revenue AS FLOAT64)), 0) AS ltv_30_days,
        COUNT(CASE WHEN SAFE_CAST(e.converted_revenue AS FLOAT64) > 0 THEN 1 END) AS purchase_events_30d
      FROM `{table_path}` e
      JOIN install_dates i ON COALESCE(e.gaid, e.idfa, e.android_id, e.waid, e.idfv) = i.user_id
      WHERE {{target_date_filter}}
      GROUP BY user_id
    )
    
    SELECT f.*, 
      -- Time-based derived features (from timestamps, will be converted to numeric)
      TIMESTAMP_DIFF(TIMESTAMP_ADD(f.install_timestamp, INTERVAL {FEATURE_DAYS} DAY), 
                     f.install_timestamp, MINUTE) AS total_window_mins,
      
      -- Derived features (behavioral only, no revenue amounts)
      SAFE_DIVIDE(f.total_events, f.total_sessions) AS events_per_session,
      SAFE_DIVIDE(f.engagement_events, f.total_sessions) AS engagement_per_session,
      SAFE_DIVIDE(f.store_interactions, f.total_sessions) AS store_interactions_per_session,
      SAFE_DIVIDE(f.unique_products_viewed, f.store_interactions) AS product_diversity,
      SAFE_DIVIDE(f.store_interactions, f.total_events) AS store_engagement_rate,
      SAFE_DIVIDE(f.total_events, f.active_days) AS events_per_active_day,
      SAFE_DIVIDE(f.business_hours_events, f.total_events) AS business_hours_ratio,
      SAFE_DIVIDE(f.weekend_events, f.total_events) AS weekend_ratio,
      
      COALESCE(t.ltv_30_days, 0) AS ltv_30_days,
      COALESCE(t.purchase_events_30d, 0) AS purchase_events_30d
    FROM features f
    LEFT JOIN targets t ON f.user_id = t.user_id
    WHERE f.total_sessions > 0
    ORDER BY RAND()
    """
    
    # Training data: Oct 2024 - Mar 2025 users, D0-D3 features
    print("\\n1. Loading Training Data...")
    train_query = base_query.format(
        start_date='2024-10-01',
        end_date='2025-04-01', 
        limit_clause='LIMIT 50000',  # Limit for faster processing
        feature_date_filter=f'DATE(e.attribution_event_timestamp) BETWEEN i.install_date AND DATE_ADD(i.install_date, INTERVAL {FEATURE_DAYS} DAY)',
        target_date_filter=f'DATE(e.attribution_event_timestamp) BETWEEN DATE_ADD(i.install_date, INTERVAL {FEATURE_DAYS + 1} DAY) AND DATE_ADD(i.install_date, INTERVAL {FEATURE_DAYS + PREDICTION_DAYS} DAY)'
    )
    
    train_data = client.query(train_query).result().to_dataframe()
    print(f"   Training data loaded: {len(train_data):,} users")
    
    # Validation data: Apr - May 2025 users
    print("\\n2. Loading Validation Data...")
    val_query = base_query.format(
        start_date='2025-04-01',
        end_date='2025-05-01',
        limit_clause='LIMIT 20000',
        feature_date_filter=f'DATE(e.attribution_event_timestamp) BETWEEN DATE_ADD(i.install_date, INTERVAL {FEATURE_DAYS + 1} DAY) AND DATE_ADD(i.install_date, INTERVAL {FEATURE_DAYS + PREDICTION_DAYS} DAY)',
        target_date_filter=f'DATE(e.attribution_event_timestamp) BETWEEN DATE_ADD(i.install_date, INTERVAL {FEATURE_DAYS + 1} DAY) AND DATE_ADD(i.install_date, INTERVAL {FEATURE_DAYS + PREDICTION_DAYS} DAY)'
    )
    
    val_data = client.query(val_query).result().to_dataframe()
    print(f"   Validation data loaded: {len(val_data):,} users")
    
    # Test data: May+ 2025 users  
    print("\\n3. Loading Test Data...")
    test_query = base_query.format(
        start_date='2025-05-01',
        end_date='2025-12-31',
        limit_clause='LIMIT 20000',
        feature_date_filter=f'DATE(e.attribution_event_timestamp) BETWEEN DATE_ADD(i.install_date, INTERVAL {FEATURE_DAYS + 1} DAY) AND DATE_ADD(i.install_date, INTERVAL {FEATURE_DAYS + PREDICTION_DAYS} DAY)',
        target_date_filter=f'DATE(e.attribution_event_timestamp) BETWEEN DATE_ADD(i.install_date, INTERVAL {FEATURE_DAYS + 1} DAY) AND DATE_ADD(i.install_date, INTERVAL {FEATURE_DAYS + PREDICTION_DAYS} DAY)'
    )
    
    test_data = client.query(test_query).result().to_dataframe()
    print(f"   Test data loaded: {len(test_data):,} users")
    
    # Add whale classification with proper thresholds
    def add_whale_classification(df, name):
        whale_threshold = max(df['ltv_30_days'].quantile(0.99), 1.0)
        df['is_whale'] = (df['ltv_30_days'] >= whale_threshold).astype(int)
        whale_count = df['is_whale'].sum()
        whale_pct = whale_count / len(df) * 100
        print(f"   {name} whale threshold: ${whale_threshold:.2f}, Whales: {whale_count} ({whale_pct:.2f}%)")
        return df
    
    train_data = add_whale_classification(train_data, "Train")
    val_data = add_whale_classification(val_data, "Val")  
    test_data = add_whale_classification(test_data, "Test")
    
    # Apply feature engineering
    print("\\n" + "="*50)
    print("FEATURE ENGINEERING")
    print("="*50)
    train_data = create_training_features(train_data)
    val_data = create_prediction_features(val_data)
    test_data = create_prediction_features(test_data)
    
    # Drop datetime columns that can't be used in ML
    datetime_cols = ['install_date', 'install_timestamp']
    for col in datetime_cols:
        if col in train_data.columns:
            train_data = train_data.drop(columns=[col])
            val_data = val_data.drop(columns=[col])
            test_data = test_data.drop(columns=[col])
    
    print(f"\\nFinal dataset shapes:")
    print(f"Train: {train_data.shape}")
    print(f"Val: {val_data.shape}")
    print(f"Test: {test_data.shape}")
    
    # Save datasets
    print(f"\\nSaving final datasets...")
    train_data.to_csv('/Users/gurmehakkaur/gameramp/Player_LTV/V2/final_train_data.csv', index=False)
    val_data.to_csv('/Users/gurmehakkaur/gameramp/Player_LTV/V2/final_val_data.csv', index=False)
    test_data.to_csv('/Users/gurmehakkaur/gameramp/Player_LTV/V2/final_test_data.csv', index=False)
    
    print(f"✅ Saved final_train_data.csv ({len(train_data):,} rows, {len(train_data.columns)} cols)")
    print(f"✅ Saved final_val_data.csv ({len(val_data):,} rows, {len(val_data.columns)} cols)")
    print(f"✅ Saved final_test_data.csv ({len(test_data):,} rows, {len(test_data.columns)} cols)")
    
    # Summary
    print(f"\\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Train LTV: min=${train_data['ltv_30_days'].min():.2f}, max=${train_data['ltv_30_days'].max():.2f}, mean=${train_data['ltv_30_days'].mean():.2f}")
    print(f"Val LTV: min=${val_data['ltv_30_days'].min():.2f}, max=${val_data['ltv_30_days'].max():.2f}, mean=${val_data['ltv_30_days'].mean():.2f}")
    print(f"Test LTV: min=${test_data['ltv_30_days'].min():.2f}, max=${test_data['ltv_30_days'].max():.2f}, mean=${test_data['ltv_30_days'].mean():.2f}")
    
    print(f"\\nFeatures: {len(train_data.columns) - 3} behavioral features (no revenue correlation)")
    print(f"✅ Ready for ML model training!")

if __name__ == "__main__":
    load_and_save_final_datasets()