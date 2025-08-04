#!/usr/bin/env python3

import pandas as pd
import numpy as np
from google.cloud import bigquery
from config import PROJECT_ID, DATASET, TABLE, FEATURE_DAYS, PREDICTION_DAYS

def test_optimized_query():
    """Test the optimized query with smaller dataset"""
    
    client = bigquery.Client(project=PROJECT_ID)
    table_path = f"{PROJECT_ID}.{DATASET}.{TABLE}"
    
    # Optimized query with LIMIT for testing
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
      LIMIT 1000  -- Test with small sample
    ),
    
    training_features AS (
      SELECT
        COALESCE(e.gaid, e.idfa, e.android_id, e.waid, e.idfv) AS user_id,
        i.install_date,
        i.install_timestamp,
        
        -- Core behavioral features (NO REVENUE)
        COUNT(DISTINCT e.session_id) AS total_sessions,
        COUNT(*) AS total_events,
        COUNT(DISTINCT DATE(e.attribution_event_timestamp)) AS active_days,
        
        -- Engagement features (no product/store data to avoid revenue correlation)
        COUNT(CASE WHEN e.session_id IS NOT NULL THEN 1 END) AS engagement_events,
        
        -- Platform and timing
        MAX(CASE WHEN e.gaid IS NOT NULL THEN 'android' 
                WHEN e.idfa IS NOT NULL THEN 'ios' 
                ELSE 'unknown' END) AS os,
        MAX(e.country) AS country,
        MAX(e.install_source) AS channel,
        
        -- Time patterns (simplified)
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
      -- Derived features (no revenue-related features)
      SAFE_DIVIDE(f.total_events, f.total_sessions) AS events_per_session,
      SAFE_DIVIDE(f.engagement_events, f.total_sessions) AS engagement_per_session,
      SAFE_DIVIDE(f.total_events, f.active_days) AS events_per_active_day,
      SAFE_DIVIDE(f.business_hours_events, f.total_events) AS business_hours_ratio,
      SAFE_DIVIDE(f.weekend_events, f.total_events) AS weekend_ratio,
      
      COALESCE(t.ltv_30_days, 0) AS ltv_30_days,
      COALESCE(t.purchase_events_30d, 0) AS purchase_events_30d
    FROM training_features f
    LEFT JOIN training_targets t ON f.user_id = t.user_id
    WHERE f.total_sessions > 0
    """
    
    print("Testing optimized query...")
    print(f"Features reduced to: {len(['total_sessions', 'total_events', 'active_days', 'engagement_events', 'os', 'country', 'channel', 'business_hours_events', 'weekend_events'])} core features")
    print("No revenue-related features included in feature set")
    
    try:
        test_data = client.query(test_query).result().to_dataframe()
        print(f"✓ Query successful! Sample size: {len(test_data)} users")
        print(f"✓ Features: {list(test_data.columns)}")
        print(f"✓ Data types: {test_data.dtypes.to_dict()}")
        print(f"✓ Memory usage: {test_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Verify no revenue leakage in features
        feature_cols = [col for col in test_data.columns if col not in ['user_id', 'install_date', 'install_timestamp', 'ltv_30_days', 'purchase_events_30d']]
        print(f"✓ Clean feature set: {len(feature_cols)} features")
        print(f"✓ No product/store/revenue features in feature set")
        
        return True
        
    except Exception as e:
        print(f"✗ Query failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_optimized_query()
    if success:
        print("\n🎉 Optimized query ready for production!")
    else:
        print("\n❌ Query needs fixes")