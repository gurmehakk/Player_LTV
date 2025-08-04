# load_data_batched.py

from google.cloud import bigquery
import pandas as pd
from config import PROJECT_ID, DATASET, TABLE, MIN_DATE, FEATURE_DAYS, PREDICTION_DAYS, TRAIN_SAMPLES, VAL_SAMPLES, TEST_SAMPLES


def load_user_data():
    """Load user data with install dates and feature/target periods"""
    client = bigquery.Client(project=PROJECT_ID)
    table_path = f"{PROJECT_ID}.{DATASET}.{TABLE}"

    query = f"""
    WITH install_dates AS (
      SELECT
        COALESCE(gaid, idfa, android_id, waid, idfv) AS user_id,
        DATE(MIN(attribution_event_timestamp)) AS install_date,
        MIN(attribution_event_timestamp) AS install_timestamp
      FROM `{table_path}`
      WHERE DATE(attribution_event_timestamp) >= '{MIN_DATE}'
        AND (gaid IS NOT NULL OR idfa IS NOT NULL OR android_id IS NOT NULL OR waid IS NOT NULL OR idfv IS NOT NULL)
      GROUP BY user_id
      HAVING COUNT(*) >= 10  -- Filter users with minimal activity
    ),
    
    feature_period_events AS (
      SELECT
        e.*,
        i.install_date,
        i.install_timestamp,
        COALESCE(e.gaid, e.idfa, e.android_id, e.waid, e.idfv) AS user_id,
        DATE(e.attribution_event_timestamp) AS event_date,
        DATE_DIFF(DATE(e.attribution_event_timestamp), i.install_date, DAY) AS days_since_install
      FROM `{table_path}` e
      JOIN install_dates i ON COALESCE(e.gaid, e.idfa, e.android_id, e.waid, e.idfv) = i.user_id
      WHERE DATE(e.attribution_event_timestamp) BETWEEN i.install_date 
        AND DATE_ADD(i.install_date, INTERVAL {FEATURE_DAYS} DAY)
    ),
    
    ltv_period_events AS (
      SELECT
        COALESCE(e.gaid, e.idfa, e.android_id, e.waid, e.idfv) AS user_id,
        i.install_date,
        COALESCE(SUM(SAFE_CAST(e.converted_revenue AS FLOAT64)), 0) AS ltv_30_days,
        COUNT(CASE WHEN SAFE_CAST(e.converted_revenue AS FLOAT64) > 0 THEN 1 END) AS purchase_events
      FROM `{table_path}` e
      JOIN install_dates i ON COALESCE(e.gaid, e.idfa, e.android_id, e.waid, e.idfv) = i.user_id
      WHERE DATE(e.attribution_event_timestamp) BETWEEN DATE_ADD(i.install_date, INTERVAL {FEATURE_DAYS + 1} DAY)
        AND DATE_ADD(i.install_date, INTERVAL {FEATURE_DAYS + PREDICTION_DAYS} DAY)
      GROUP BY user_id, i.install_date
    )
    
    SELECT 
      f.*,
      COALESCE(l.ltv_30_days, 0) AS ltv_30_days,
      COALESCE(l.purchase_events, 0) AS purchase_events
    FROM feature_period_events f
    LEFT JOIN ltv_period_events l ON f.user_id = l.user_id
    """

    df = client.query(query).result().to_dataframe()
    return df


def load_training_data():
    """Load training features (D0-D3) and targets (D4-D33) for model training"""
    client = bigquery.Client(project=PROJECT_ID)
    table_path = f"{PROJECT_ID}.{DATASET}.{TABLE}"
    
    query = f"""
    WITH install_dates AS (
      SELECT
        COALESCE(gaid, idfa, android_id, waid, idfv) AS user_id,
        DATE(MIN(attribution_event_timestamp)) AS install_date,
        MIN(attribution_event_timestamp) AS install_timestamp
      FROM `{table_path}`
      WHERE DATE(attribution_event_timestamp) >= '{MIN_DATE}'
        AND (gaid IS NOT NULL OR idfa IS NOT NULL OR android_id IS NOT NULL OR waid IS NOT NULL OR idfv IS NOT NULL)
      GROUP BY user_id
      HAVING COUNT(*) >= 5
    ),
    
    training_features AS (
      SELECT
        COALESCE(e.gaid, e.idfa, e.android_id, e.waid, e.idfv) AS user_id,
        i.install_date,
        i.install_timestamp,
        
        -- Session features (NO REVENUE FEATURES)
        COUNT(DISTINCT e.session_id) AS total_sessions,
        COUNT(*) AS total_events,
        MIN(e.attribution_event_timestamp) AS first_event,
        MAX(e.attribution_event_timestamp) AS last_event,
        
        -- Session timing features
        ARRAY_AGG(e.attribution_event_timestamp ORDER BY e.attribution_event_timestamp)[OFFSET(0)] AS first_session_start,
        ARRAY_AGG(e.attribution_event_timestamp ORDER BY e.attribution_event_timestamp DESC)[OFFSET(0)] AS last_session_end,
        
        -- Product engagement features (without revenue) - Store/Offer interactions
        COUNT(CASE WHEN e.product_name IS NOT NULL THEN 1 END) AS store_interactions,
        COUNT(DISTINCT e.product_name) AS unique_products_viewed,
        COUNT(DISTINCT e.product_sku) AS unique_skus_viewed,
        AVG(SAFE_CAST(e.product_quantity AS FLOAT64)) AS avg_quantity_per_interaction,
        SUM(SAFE_CAST(e.product_quantity AS FLOAT64)) AS total_quantity_interactions,
        
        -- Navigation/UI interaction patterns
        COUNT(CASE WHEN e.product_name IS NOT NULL THEN 1 END) AS store_button_clicks,
        
        -- Attribution features
        MAX(e.install_source) AS channel,
        MAX(e.country) AS country,
        MAX(e.partner) AS partner,
        MAX(e.campaign_name) AS campaign_name,
        MAX(e.publisher_name) AS publisher_name,
        MAX(CASE WHEN e.gaid IS NOT NULL THEN 'android' 
                WHEN e.idfa IS NOT NULL THEN 'ios' 
                ELSE 'unknown' END) AS os,
                
        -- Behavioral and timing patterns
        COUNT(DISTINCT DATE(e.attribution_event_timestamp)) AS active_days,
        COUNT(DISTINCT EXTRACT(HOUR FROM e.attribution_event_timestamp)) AS unique_hours_active,
        
        -- Session timing and gaps
        COUNTIF(EXTRACT(HOUR FROM e.attribution_event_timestamp) BETWEEN 6 AND 11) AS morning_events,
        COUNTIF(EXTRACT(HOUR FROM e.attribution_event_timestamp) BETWEEN 12 AND 17) AS afternoon_events,
        COUNTIF(EXTRACT(HOUR FROM e.attribution_event_timestamp) BETWEEN 18 AND 23) AS evening_events,
        COUNTIF(EXTRACT(HOUR FROM e.attribution_event_timestamp) NOT BETWEEN 6 AND 23) AS night_events,
        
        -- Weekend vs weekday activity
        COUNTIF(EXTRACT(DAYOFWEEK FROM e.attribution_event_timestamp) IN (1, 7)) AS weekend_events,
        COUNTIF(EXTRACT(DAYOFWEEK FROM e.attribution_event_timestamp) NOT IN (1, 7)) AS weekday_events,
        
        -- Attribution diversity features
        MAX(e.state) AS user_state,
        MAX(CASE WHEN e.touchpoint_timestamp IS NOT NULL THEN 1 ELSE 0 END) AS has_touchpoint_data,
        MAX(CASE WHEN e.is_fingerprinted IS TRUE THEN 1 ELSE 0 END) AS is_fingerprinted_user,
        MAX(CASE WHEN e.is_reengagement IS TRUE THEN 1 ELSE 0 END) AS is_reengagement_user,
        MAX(CASE WHEN e.is_view_through IS TRUE THEN 1 ELSE 0 END) AS is_view_through_user
                
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
    
    SELECT 
      f.*,
      -- Session timing calculations
      TIMESTAMP_DIFF(f.last_event, f.first_event, MINUTE) AS total_playtime_mins,
      TIMESTAMP_DIFF(f.first_session_start, f.install_timestamp, MINUTE) AS time_to_first_session_mins,
      
      -- Session hour patterns
      EXTRACT(HOUR FROM f.first_session_start) AS first_session_hour,
      EXTRACT(HOUR FROM f.last_session_end) AS last_session_hour,
      EXTRACT(DAYOFWEEK FROM f.first_session_start) AS first_session_dayofweek,
      
      -- Session timing buckets
      CASE 
        WHEN EXTRACT(HOUR FROM f.first_session_start) BETWEEN 6 AND 11 THEN 'morning'
        WHEN EXTRACT(HOUR FROM f.first_session_start) BETWEEN 12 AND 17 THEN 'afternoon' 
        WHEN EXTRACT(HOUR FROM f.first_session_start) BETWEEN 18 AND 23 THEN 'evening'
        ELSE 'night'
      END AS first_session_time_bucket,
      
      CASE 
        WHEN EXTRACT(HOUR FROM f.last_session_end) BETWEEN 6 AND 11 THEN 'morning'
        WHEN EXTRACT(HOUR FROM f.last_session_end) BETWEEN 12 AND 17 THEN 'afternoon' 
        WHEN EXTRACT(HOUR FROM f.last_session_end) BETWEEN 18 AND 23 THEN 'evening'
        ELSE 'night'
      END AS last_session_time_bucket,
      
      -- Derived metrics
      SAFE_DIVIDE(f.total_events, f.total_sessions) AS events_per_session,
      SAFE_DIVIDE(f.store_interactions, f.total_sessions) AS store_interactions_per_session,
      SAFE_DIVIDE(f.unique_products_viewed, f.store_interactions) AS product_diversity,
      SAFE_DIVIDE(f.total_events, f.active_days) AS events_per_active_day,
      SAFE_DIVIDE(f.store_interactions, f.total_events) AS store_engagement_rate,
      
      -- Targets
      COALESCE(t.ltv_30_days, 0) AS ltv_30_days,
      COALESCE(t.purchase_events_30d, 0) AS purchase_events_30d
    FROM training_features f
    LEFT JOIN training_targets t ON f.user_id = t.user_id
    WHERE f.total_sessions > 0  -- Ensure users had at least one session
    ORDER BY RAND()
    LIMIT {TRAIN_SAMPLES + VAL_SAMPLES + TEST_SAMPLES}
    """
    
    df = client.query(query).result().to_dataframe()
    return df


def load_prediction_data_batched(user_ids):
    """Load D4-D33 features for specific users in batches"""
    if len(user_ids) == 0:
        return pd.DataFrame()
        
    client = bigquery.Client(project=PROJECT_ID)
    table_path = f"{PROJECT_ID}.{DATASET}.{TABLE}"
    
    # Process in batches to avoid query size limits
    batch_size = 1000  # Process 1000 users at a time
    all_results = []
    
    print(f"   Processing {len(user_ids):,} users in batches of {batch_size:,}...")
    
    for i in range(0, len(user_ids), batch_size):
        batch_users = user_ids[i:i+batch_size]
        user_list = "', '".join(str(uid) for uid in batch_users)
        
        print(f"   Batch {i//batch_size + 1}/{(len(user_ids)-1)//batch_size + 1}: {len(batch_users):,} users")
        
        query = f"""
        WITH requested_user_ids AS (
          SELECT uid as user_id FROM UNNEST(['{user_list.replace("', '", "', '")}']) as uid
        ),
        
        user_installs AS (
          SELECT
            COALESCE(gaid, idfa, android_id, waid, idfv) AS user_id,
            DATE(MIN(attribution_event_timestamp)) AS install_date,
            MIN(attribution_event_timestamp) AS install_timestamp
          FROM `{table_path}`
          WHERE COALESCE(gaid, idfa, android_id, waid, idfv) IN ('{user_list}')
          GROUP BY user_id
        ),
        
        all_users AS (
          SELECT 
            r.user_id,
            COALESCE(i.install_date, DATE('1900-01-01')) as install_date,
            COALESCE(i.install_timestamp, TIMESTAMP('1900-01-01')) as install_timestamp
          FROM requested_user_ids r
          LEFT JOIN user_installs i ON r.user_id = i.user_id
        ),
        
        prediction_features AS (
          SELECT
            u.user_id,
            u.install_date,
            u.install_timestamp,
            
            -- Same features as training (NO REVENUE) but from D4-D33
            COUNT(DISTINCT e.session_id) AS total_sessions,
            COUNT(*) AS total_events,
            MIN(e.attribution_event_timestamp) AS first_event,
            MAX(e.attribution_event_timestamp) AS last_event,
            
            -- Session timing features
            ARRAY_AGG(e.attribution_event_timestamp ORDER BY e.attribution_event_timestamp)[OFFSET(0)] AS first_session_start,
            ARRAY_AGG(e.attribution_event_timestamp ORDER BY e.attribution_event_timestamp DESC)[OFFSET(0)] AS last_session_end,
            
            -- Product engagement features (without revenue) - Store/Offer interactions
            COUNT(CASE WHEN e.product_name IS NOT NULL THEN 1 END) AS store_interactions,
            COUNT(DISTINCT e.product_name) AS unique_products_viewed,
            COUNT(DISTINCT e.product_sku) AS unique_skus_viewed,
            AVG(SAFE_CAST(e.product_quantity AS FLOAT64)) AS avg_quantity_per_interaction,
            SUM(SAFE_CAST(e.product_quantity AS FLOAT64)) AS total_quantity_interactions,
            
            -- Navigation/UI interaction patterns
            COUNT(CASE WHEN e.product_name IS NOT NULL THEN 1 END) AS store_button_clicks,
            
            MAX(e.install_source) AS channel,
            MAX(e.country) AS country,
            MAX(e.partner) AS partner,
            MAX(e.campaign_name) AS campaign_name,
            MAX(e.publisher_name) AS publisher_name,
            MAX(CASE WHEN e.gaid IS NOT NULL THEN 'android' 
                    WHEN e.idfa IS NOT NULL THEN 'ios' 
                    ELSE 'unknown' END) AS os,
                    
            -- Behavioral and timing patterns
            COUNT(DISTINCT DATE(e.attribution_event_timestamp)) AS active_days,
            COUNT(DISTINCT EXTRACT(HOUR FROM e.attribution_event_timestamp)) AS unique_hours_active,
            
            -- Session timing and gaps
            COUNTIF(EXTRACT(HOUR FROM e.attribution_event_timestamp) BETWEEN 6 AND 11) AS morning_events,
            COUNTIF(EXTRACT(HOUR FROM e.attribution_event_timestamp) BETWEEN 12 AND 17) AS afternoon_events,
            COUNTIF(EXTRACT(HOUR FROM e.attribution_event_timestamp) BETWEEN 18 AND 23) AS evening_events,
            COUNTIF(EXTRACT(HOUR FROM e.attribution_event_timestamp) NOT BETWEEN 6 AND 23) AS night_events,
            
            -- Weekend vs weekday activity
            COUNTIF(EXTRACT(DAYOFWEEK FROM e.attribution_event_timestamp) IN (1, 7)) AS weekend_events,
            COUNTIF(EXTRACT(DAYOFWEEK FROM e.attribution_event_timestamp) NOT IN (1, 7)) AS weekday_events,
            
            -- Attribution diversity features
            MAX(e.state) AS user_state,
            MAX(CASE WHEN e.touchpoint_timestamp IS NOT NULL THEN 1 ELSE 0 END) AS has_touchpoint_data,
            MAX(CASE WHEN e.is_fingerprinted IS TRUE THEN 1 ELSE 0 END) AS is_fingerprinted_user,
            MAX(CASE WHEN e.is_reengagement IS TRUE THEN 1 ELSE 0 END) AS is_reengagement_user,
            MAX(CASE WHEN e.is_view_through IS TRUE THEN 1 ELSE 0 END) AS is_view_through_user
                    
          FROM all_users u
          LEFT JOIN `{table_path}` e ON COALESCE(e.gaid, e.idfa, e.android_id, e.waid, e.idfv) = u.user_id
            AND (u.install_date = DATE('1900-01-01') OR DATE(e.attribution_event_timestamp) BETWEEN DATE_ADD(u.install_date, INTERVAL {FEATURE_DAYS + 1} DAY)
            AND DATE_ADD(u.install_date, INTERVAL {FEATURE_DAYS + PREDICTION_DAYS} DAY))
          GROUP BY u.user_id, u.install_date, u.install_timestamp
        )
        
        SELECT 
          f.*,
          -- Session timing calculations
          TIMESTAMP_DIFF(f.last_event, f.first_event, MINUTE) AS total_playtime_mins,
          TIMESTAMP_DIFF(f.first_session_start, f.install_timestamp, MINUTE) AS time_to_first_session_mins,
          
          -- Session hour patterns
          EXTRACT(HOUR FROM f.first_session_start) AS first_session_hour,
          EXTRACT(HOUR FROM f.last_session_end) AS last_session_hour,
          EXTRACT(DAYOFWEEK FROM f.first_session_start) AS first_session_dayofweek,
          
          -- Session timing buckets
          CASE 
            WHEN EXTRACT(HOUR FROM f.first_session_start) BETWEEN 6 AND 11 THEN 'morning'
            WHEN EXTRACT(HOUR FROM f.first_session_start) BETWEEN 12 AND 17 THEN 'afternoon' 
            WHEN EXTRACT(HOUR FROM f.first_session_start) BETWEEN 18 AND 23 THEN 'evening'
            ELSE 'night'
          END AS first_session_time_bucket,
          
          CASE 
            WHEN EXTRACT(HOUR FROM f.last_session_end) BETWEEN 6 AND 11 THEN 'morning'
            WHEN EXTRACT(HOUR FROM f.last_session_end) BETWEEN 12 AND 17 THEN 'afternoon' 
            WHEN EXTRACT(HOUR FROM f.last_session_end) BETWEEN 18 AND 23 THEN 'evening'
            ELSE 'night'
          END AS last_session_time_bucket,
          
          -- Derived metrics
          SAFE_DIVIDE(f.total_events, f.total_sessions) AS events_per_session,
          SAFE_DIVIDE(f.store_interactions, f.total_sessions) AS store_interactions_per_session,
          SAFE_DIVIDE(f.unique_products_viewed, f.store_interactions) AS product_diversity,
          SAFE_DIVIDE(f.total_events, f.active_days) AS events_per_active_day,
          SAFE_DIVIDE(f.store_interactions, f.total_events) AS store_engagement_rate
        FROM prediction_features f
        """
        
        try:
            batch_df = client.query(query).result().to_dataframe()
            if len(batch_df) > 0:
                all_results.append(batch_df)
                print(f"     Retrieved {len(batch_df):,} users from batch")
            else:
                print(f"     No data found for this batch")
        except Exception as e:
            print(f"     Error in batch {i//batch_size + 1}: {str(e)}")
            continue
    
    # Combine all results
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        print(f"   Total users retrieved: {len(final_df):,}")
        return final_df
    else:
        print(f"   No data retrieved for any users")
        return pd.DataFrame()