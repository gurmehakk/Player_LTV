#!/usr/bin/env python3

import pandas as pd
import numpy as np
from google.cloud import bigquery
from config import PROJECT_ID, DATASET, TABLE, FEATURE_DAYS, PREDICTION_DAYS
from simple_features_optimized import create_training_features, list_all_features

def test_feature_engineering():
    """Test the optimized feature engineering"""
    
    client = bigquery.Client(project=PROJECT_ID)
    table_path = f"{PROJECT_ID}.{DATASET}.{TABLE}"
    
    # Small test query
    test_query = f"""
    WITH install_dates AS (
      SELECT
        COALESCE(gaid, idfa, android_id, waid, idfv) AS user_id,
        DATE(MIN(attribution_event_timestamp)) AS install_date,
        MIN(attribution_event_timestamp) AS install_timestamp
      FROM `{table_path}`
      WHERE DATE(attribution_event_timestamp) >= '2025-01-01'
        AND DATE(attribution_event_timestamp) <= '2025-04-01'
        AND (gaid IS NOT NULL OR idfa IS NOT NULL OR android_id IS NOT NULL)
      GROUP BY user_id
      HAVING COUNT(*) >= 5
      LIMIT 100
    ),
    
    training_features AS (
      SELECT
        COALESCE(e.gaid, e.idfa, e.android_id, e.waid, e.idfv) AS user_id,
        i.install_date,
        i.install_timestamp,
        
        COUNT(DISTINCT e.session_id) AS total_sessions,
        COUNT(*) AS total_events,
        COUNT(DISTINCT DATE(e.attribution_event_timestamp)) AS active_days,
        MIN(e.attribution_event_timestamp) AS first_event,
        MAX(e.attribution_event_timestamp) AS last_event,
        
        -- Product interaction features (behavioral only, no revenue)
        COUNT(CASE WHEN e.product_name IS NOT NULL THEN 1 END) AS store_interactions,
        COUNT(DISTINCT e.product_name) AS unique_products_viewed,
        COUNT(DISTINCT e.product_sku) AS unique_skus_viewed,
        AVG(SAFE_CAST(e.product_quantity AS FLOAT64)) AS avg_quantity_per_interaction,
        SUM(SAFE_CAST(e.product_quantity AS FLOAT64)) AS total_quantity_interactions,
        
        COUNT(CASE WHEN e.session_id IS NOT NULL THEN 1 END) AS engagement_events,
        
        MAX(CASE WHEN e.gaid IS NOT NULL THEN 'android' 
                WHEN e.idfa IS NOT NULL THEN 'ios' 
                ELSE 'unknown' END) AS os,
        MAX(e.country) AS country,
        MAX(e.install_source) AS channel,
        
        COUNTIF(EXTRACT(HOUR FROM e.attribution_event_timestamp) BETWEEN 9 AND 17) AS business_hours_events,
        COUNTIF(EXTRACT(DAYOFWEEK FROM e.attribution_event_timestamp) IN (1, 7)) AS weekend_events
                
      FROM `{table_path}` e
      JOIN install_dates i ON COALESCE(e.gaid, e.idfa, e.android_id, e.waid, e.idfv) = i.user_id
      WHERE DATE(e.attribution_event_timestamp) BETWEEN i.install_date 
        AND DATE_ADD(i.install_date, INTERVAL {FEATURE_DAYS} DAY)
      GROUP BY user_id, i.install_date, i.install_timestamp
    ),
    
    training_targets AS (
      SELECT
        COALESCE(e.gaid, e.idfa, e.android_id, e.waid, e.idfv) AS user_id,
        COALESCE(SUM(SAFE_CAST(e.converted_revenue AS FLOAT64)), 0) AS ltv_30_days,
        COUNT(CASE WHEN SAFE_CAST(e.converted_revenue AS FLOAT64) > 0 THEN 1 END) AS purchase_events_30d
      FROM `{table_path}` e
      JOIN install_dates i ON COALESCE(e.gaid, e.idfa, e.android_id, e.waid, e.idfv) = i.user_id
      WHERE DATE(e.attribution_event_timestamp) BETWEEN DATE_ADD(i.install_date, INTERVAL {FEATURE_DAYS + 1} DAY)
        AND DATE_ADD(i.install_date, INTERVAL {FEATURE_DAYS + PREDICTION_DAYS} DAY)
      GROUP BY user_id
    )
    
    SELECT f.*, 
      TIMESTAMP_DIFF(f.last_event, f.first_event, MINUTE) AS total_playtime_mins,
      TIMESTAMP_DIFF(f.first_event, f.install_timestamp, MINUTE) AS time_to_first_session_mins,
      
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
    FROM training_features f
    LEFT JOIN training_targets t ON f.user_id = t.user_id
    WHERE f.total_sessions > 0
    """
    
    print("Testing optimized feature engineering...")
    
    try:
        # Load test data
        test_data = client.query(test_query).result().to_dataframe()
        print(f"✓ Query successful! Sample size: {len(test_data)} users")
        
        # Add whale classification
        whale_threshold = test_data['ltv_30_days'].quantile(0.99)
        test_data['is_whale'] = (test_data['ltv_30_days'] >= whale_threshold).astype(int)
        
        # Apply feature engineering
        print("\nApplying feature engineering...")
        engineered_data = create_training_features(test_data)
        
        # List all features
        feature_cols = list_all_features(engineered_data)
        
        print(f"\n✅ Feature engineering successful!")
        print(f"   Original columns: {len(test_data.columns)}")
        print(f"   Final columns: {len(engineered_data.columns)}")
        print(f"   Added features: {len(engineered_data.columns) - len(test_data.columns)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Feature engineering failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_feature_engineering()
    if success:
        print("\n🎉 Feature engineering ready!")
    else:
        print("\n❌ Feature engineering needs fixes")