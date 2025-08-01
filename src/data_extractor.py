"""
Data extraction module for BigQuery app_data table
Handles session-level behavioral tracking and game progression events
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.oauth2 import service_account
from typing import Optional


class DataExtractor:
    """Handles data extraction from BigQuery with comprehensive feature extraction"""
    
    def __init__(self, project_id: str, credentials_path: Optional[str] = None):
        self.project_id = project_id
        self.client = self._initialize_bigquery_client(credentials_path)
        
    def _initialize_bigquery_client(self, credentials_path: Optional[str]) -> bigquery.Client:
        """Initialize BigQuery client with optional service account"""
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            return bigquery.Client(credentials=credentials, project=self.project_id)
        else:
            return bigquery.Client(project=self.project_id)
    
    def extract_player_data(self, observation_days: int = 60, prediction_days: int = 30, 
                          min_total_days: int = 90) -> pd.DataFrame:
        """Extract player data with simplified, faster query
        
        Args:
            observation_days: Days to use for feature calculation (60)
            prediction_days: Days to predict LTV for (30) 
            min_total_days: Minimum total days of data needed (90)
        """
        
        # SIMPLE 3-DAY OBSERVATION: Last 3 days of May for observation, June for prediction
        observation_start = "2025-05-28"  # 3 days before May 31
        observation_end = "2025-05-31"    # End of May
        prediction_end = "2025-06-30"     # End of June (1 month prediction)
        
        query = f"""
        WITH base_data AS (
            SELECT 
                COALESCE(NULLIF(custom_user_id, ''), gaid, idfa, android_id, idfv, waid) as player_id,
                session_id,
                name as event_name,
                PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S%Ez', server_timestamp) as event_timestamp,
                COALESCE(converted_revenue, 0.0) as revenue,
                country,
                install_source,
                is_fingerprinted,
                is_reengagement
                
            FROM `{self.project_id}.test_data.app_data`
            WHERE 
                server_timestamp IS NOT NULL
                AND server_timestamp >= '{observation_start}T00:00:00Z'
                AND server_timestamp < '{prediction_end}T00:00:00Z'
                AND COALESCE(NULLIF(custom_user_id, ''), gaid, idfa, android_id, idfv, waid) IS NOT NULL
        ),
        
        observation_data AS (
            SELECT 
                player_id,
                COUNT(DISTINCT session_id) as total_sessions,
                COUNT(*) as total_events,
                SUM(revenue) as total_revenue,
                COUNTIF(revenue > 0) as purchase_events,
                MIN(event_timestamp) as first_session,
                ANY_VALUE(country) as country
                
            FROM base_data
            WHERE event_timestamp < TIMESTAMP('{observation_end}')
            GROUP BY player_id
        ),
        
        prediction_ltv AS (
            SELECT 
                player_id,
                SUM(revenue) as ltv_target
                
            FROM base_data
            WHERE event_timestamp >= TIMESTAMP('{observation_end}')
            GROUP BY player_id
        )
        
        SELECT
            obs.player_id,
            obs.total_sessions,
            obs.total_events,
            obs.total_revenue,
            obs.purchase_events,
            obs.country as first_country,
            
            DATE_DIFF(CURRENT_DATE(), DATE(obs.first_session), DAY) as days_since_first_session,
            
            COALESCE(pred.ltv_target, 0.0) as ltv_target,
            
            CASE WHEN pred.ltv_target > 0 THEN 1 ELSE 0 END as will_pay_in_future,
            CASE WHEN obs.total_revenue > 0 THEN 1 ELSE 0 END as historical_payer
            
        FROM observation_data obs
        LEFT JOIN prediction_ltv pred ON obs.player_id = pred.player_id
        WHERE obs.total_sessions >= 1
        ORDER BY ltv_target DESC
        """
        
        try:
            # Execute query with proper configuration - increase timeout for complex query
            job_config = bigquery.QueryJobConfig()
            job_config.use_query_cache = True
            job_config.maximum_bytes_billed = 50 * 1024 * 1024 * 1024  # 50GB limit
            job_config.job_timeout_ms = 600000  # 10 minutes
            
            query_job = self.client.query(query, job_config=job_config)
            df = query_job.result().to_dataframe()
            
            # Data cleaning and validation
            df = self._clean_and_validate_data(df)
            

            # Save raw data to CSV
            df.to_csv('output/raw_data.csv', index=True)
            
            print(f"Extracted {len(df)} players with LTV targets")
            if len(df) > 0:
                historical_payer_rate = len(df[df['total_revenue'] > 0]) / len(df) * 100
                future_payer_rate = len(df[df['ltv_target'] > 0]) / len(df) * 100
                print(f"Historical Revenue (Observation): ${df['total_revenue'].min():.2f} - ${df['total_revenue'].max():.2f}")
                print(f"LTV Target (Prediction): ${df['ltv_target'].min():.2f} - ${df['ltv_target'].max():.2f}")
                print(f"Historical Payer Rate: {historical_payer_rate:.1f}%")
                print(f"Future Payer Rate: {future_payer_rate:.1f}%")
                print(f"Observation Period: {observation_days} days, Prediction Period: {prediction_days} days")
            
            return df
            
        except Exception as e:
            print(f"Query execution failed: {str(e)}")
            raise
    
    def _clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate extracted data"""
        
        # Convert timestamp columns and handle timezone issues
        timestamp_cols = ['first_session_timestamp', 'last_session_timestamp']
        for col in timestamp_cols:
            if col in df.columns:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Handle timezone - convert to UTC if timezone-aware, or localize if naive
                if df[col].dt.tz is None:
                    # If timezone-naive, assume UTC
                    df[col] = df[col].dt.tz_localize('UTC', errors='coerce')
                else:
                    # Convert to UTC if different timezone
                    df[col] = df[col].dt.tz_convert('UTC')
        
        # Fill missing values with appropriate defaults
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(0)
        
        # Fill categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna('unknown')
        
        # Validate data quality
        assert len(df) > 0, "No data after cleaning"
        assert 'player_id' in df.columns, "Missing player_id column"
        assert 'ltv_target' in df.columns, "Missing ltv_target column"
        assert 'total_revenue' in df.columns, "Missing total_revenue column"
        
        return df