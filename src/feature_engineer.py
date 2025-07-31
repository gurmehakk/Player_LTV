"""
Streamlined feature engineering for pLTV prediction
Focuses on essential features derivable from event-level data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple
from datetime import datetime, timedelta


class FeatureEngineer:
    """Streamlined feature engineering for gaming pLTV prediction"""
    
    def __init__(self):
        self.feature_columns = []
        self.numeric_features = []
        self.categorical_features = []
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.cluster_model = KMeans(n_clusters=5, random_state=42)
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create essential feature set for pLTV prediction"""
        
        df_features = df.copy()
        
        # 1. Basic activity features
        df_features = self._create_basic_features(df_features)
        
        # 2. Revenue features
        df_features = self._create_revenue_features(df_features)
        
        # 3. Session pattern features
        df_features = self._create_session_features(df_features)
        
        # 4. Attribution features
        df_features = self._create_attribution_features(df_features)
        
        # 5. Temporal features
        df_features = self._create_temporal_features(df_features)
        
        # 6. User segmentation features
        df_features = self._create_clustering_features(df_features)
        
        # 7. Encode categorical features
        df_features = self._encode_categorical_features(df_features)
        
        # 8. Create key interaction features
        df_features = self._create_interaction_features(df_features)
        
        # 9. Finalize feature columns
        self._finalize_feature_columns(df_features)
        
        print(f"Created {len(self.feature_columns)} total features:")
        print(f"- Numeric features: {len(self.numeric_features)}")
        print(f"- Categorical features: {len(self.categorical_features)}")
        
        return df_features
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic activity and engagement features"""
        
        # Core activity metrics (assuming these are pre-aggregated)
        if 'total_sessions' in df.columns and 'player_lifetime_days' in df.columns:
            df['sessions_per_day'] = df['total_sessions'] / np.maximum(df['player_lifetime_days'], 1)
        else:
            df['sessions_per_day'] = 0
            
        if 'total_events' in df.columns:
            if 'total_sessions' in df.columns:
                df['events_per_session'] = df['total_events'] / np.maximum(df['total_sessions'], 1)
            if 'player_lifetime_days' in df.columns:
                df['events_per_day'] = df['total_events'] / np.maximum(df['player_lifetime_days'], 1)
        
        # Session quality
        if 'avg_session_duration_minutes' in df.columns and 'total_sessions' in df.columns:
            df['session_efficiency'] = df['total_events'] / np.maximum(df['avg_session_duration_minutes'] * df['total_sessions'], 1)
        
        # Engagement intensity
        if 'avg_events_per_session' in df.columns and 'avg_session_duration_minutes' in df.columns:
            df['avg_events_per_minute'] = df['avg_events_per_session'] / np.maximum(df['avg_session_duration_minutes'], 1)
        
        return df
    
    def _create_revenue_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create revenue and monetization features from HISTORICAL data only"""
        
        # Use historical revenue (observation period) for features - NOT the LTV target
        revenue_col = 'total_revenue'  # This is historical revenue from observation period
        
        if revenue_col in df.columns:
            # Basic revenue indicators (based on historical behavior)
            df['is_historical_payer'] = (df[revenue_col] > 0).astype(int)
            df['is_high_value_historical_payer'] = (df[revenue_col] > df[revenue_col].quantile(0.9)).astype(int)
            
            # Revenue efficiency metrics (historical)
            if 'total_sessions' in df.columns:
                df['revenue_per_session'] = df[revenue_col] / np.maximum(df['total_sessions'], 1)
            if 'player_lifetime_days' in df.columns:
                df['revenue_per_day'] = df[revenue_col] / np.maximum(df['player_lifetime_days'], 1)
            if 'total_events' in df.columns:
                df['revenue_per_event'] = df[revenue_col] / np.maximum(df['total_events'], 1)
        else:
            # Create default values if revenue column not found
            df['is_historical_payer'] = 0
            df['is_high_value_historical_payer'] = 0
            df['revenue_per_session'] = 0
            df['revenue_per_day'] = 0
            df['revenue_per_event'] = 0
        
        # Purchase behavior (if available)
        if 'total_purchase_events' in df.columns and 'total_sessions' in df.columns:
            df['purchase_frequency'] = df['total_purchase_events'] / np.maximum(df['total_sessions'], 1)
        
        if 'avg_purchase_amount' in df.columns:
            df['avg_purchase_amount'] = df['avg_purchase_amount'].fillna(0)
        
        return df
    
    def _create_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create session pattern features"""
        
        # Session length patterns
        if 'long_sessions' in df.columns and 'total_sessions' in df.columns:
            df['long_session_rate'] = df['long_sessions'] / np.maximum(df['total_sessions'], 1)
        if 'short_sessions' in df.columns and 'total_sessions' in df.columns:
            df['short_session_rate'] = df['short_sessions'] / np.maximum(df['total_sessions'], 1)
        
        # Engagement patterns
        if 'multi_event_sessions' in df.columns and 'total_sessions' in df.columns:
            df['multi_event_session_rate'] = df['multi_event_sessions'] / np.maximum(df['total_sessions'], 1)
        
        # Recent activity (if available)
        if 'sessions_last_7_days' in df.columns:
            if 'total_sessions' in df.columns:
                df['recent_activity_rate'] = df['sessions_last_7_days'] / np.maximum(df['total_sessions'], 1)
            df['is_recently_active'] = (df['sessions_last_7_days'] > 0).astype(int)
        
        # Session consistency
        if 'avg_session_duration_minutes' in df.columns and 'max_session_duration_minutes' in df.columns:
            df['session_consistency'] = df['avg_session_duration_minutes'] / np.maximum(df['max_session_duration_minutes'], 1)
        
        return df
    
    def _create_attribution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create attribution and acquisition features"""
        
        # Geographic features
        if 'country' in df.columns:
            df['has_country_info'] = (df['country'].notna() & (df['country'] != 'unknown')).astype(int)
        elif 'first_country' in df.columns:
            df['has_country_info'] = (df['first_country'].notna() & (df['first_country'] != 'unknown')).astype(int)
        else:
            df['has_country_info'] = 0
        
        # Attribution source
        if 'install_source' in df.columns:
            df['has_install_source'] = (df['install_source'].notna() & (df['install_source'] != 'unknown')).astype(int)
        elif 'first_install_source' in df.columns:
            df['has_install_source'] = (df['first_install_source'].notna() & (df['first_install_source'] != 'unknown')).astype(int)
        else:
            df['has_install_source'] = 0
        
        # Campaign info
        if 'campaign_name' in df.columns:
            df['has_campaign_info'] = (df['campaign_name'].notna() & (df['campaign_name'] != 'unknown')).astype(int)
        elif 'first_campaign' in df.columns:
            df['has_campaign_info'] = (df['first_campaign'].notna() & (df['first_campaign'] != 'unknown')).astype(int)
        else:
            df['has_campaign_info'] = 0
        
        # Attribution quality indicators
        attribution_cols = ['is_fingerprinted', 'is_reengagement', 'is_view_through']
        for col in attribution_cols:
            if col in df.columns:
                df[f'{col}_rate'] = df[col].astype(int) if df[col].dtype == bool else df[col].fillna(0)
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal and lifecycle features"""
        
        # Player lifecycle stage
        if 'player_lifetime_days' in df.columns:
            df['is_new_player'] = (df['player_lifetime_days'] <= 7).astype(int)
            df['is_mature_player'] = (df['player_lifetime_days'] > 30).astype(int)
            df['lifetime_weeks'] = df['player_lifetime_days'] / 7.0
        
        # Activity recency
        if 'days_since_last_session' in df.columns:
            df['is_recently_active'] = (df['days_since_last_session'] <= 3).astype(int)
            df['is_at_risk'] = (df['days_since_last_session'] > 7).astype(int)
        elif 'last_session_timestamp' in df.columns:
            # Calculate recency from timestamp
            try:
                current_time = pd.Timestamp.now(tz='UTC')
                if df['last_session_timestamp'].dt.tz is None:
                    df['last_session_timestamp'] = df['last_session_timestamp'].dt.tz_localize('UTC')
                else:
                    df['last_session_timestamp'] = df['last_session_timestamp'].dt.tz_convert('UTC')
                
                df['days_since_last_session'] = (current_time - df['last_session_timestamp']).dt.days
                df['is_recently_active'] = (df['days_since_last_session'] <= 3).astype(int)
                df['is_at_risk'] = (df['days_since_last_session'] > 7).astype(int)
            except:
                df['is_recently_active'] = 1
                df['is_at_risk'] = 0
        
        return df
    
    def _create_clustering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create player segments based on key behavioral patterns"""
        
        # Select core features for clustering (avoid direct revenue to prevent bias)
        base_features = ['total_sessions', 'total_events', 'player_lifetime_days']
        optional_features = ['avg_session_duration_minutes', 'sessions_per_day', 'events_per_session']
        
        cluster_features = [f for f in base_features if f in df.columns]
        cluster_features.extend([f for f in optional_features if f in df.columns])
        
        if len(cluster_features) >= 3:
            # Prepare clustering data
            cluster_data = df[cluster_features].fillna(0)
            cluster_data = cluster_data.replace([np.inf, -np.inf], 0)
            
            # Scale and cluster
            cluster_data_scaled = self.scaler.fit_transform(cluster_data)
            clusters = self.cluster_model.fit_predict(cluster_data_scaled)
            df['player_segment'] = clusters
            
            # Create segment indicators
            for i in range(5):
                df[f'is_segment_{i}'] = (df['player_segment'] == i).astype(int)
        else:
            # Fallback: simple activity-based segments
            if 'total_sessions' in df.columns:
                df['player_segment'] = pd.cut(df['total_sessions'], 
                                            bins=5, labels=range(5), duplicates='drop').astype(int)
                for i in range(5):
                    df[f'is_segment_{i}'] = (df['player_segment'] == i).astype(int)
            else:
                df['player_segment'] = 0
                df['is_segment_0'] = 1
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode key categorical features"""
        
        # Define categorical columns that might exist
        potential_categorical = [
            'country', 'first_country', 
            'install_source', 'first_install_source',
            'campaign_name', 'first_campaign'
        ]
        
        for col in potential_categorical:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
                
                # Group low-frequency categories (keep top 10)
                value_counts = df[col].value_counts()
                if len(value_counts) > 10:
                    top_categories = value_counts.head(10).index.tolist()
                    df[col] = df[col].apply(lambda x: x if x in top_categories else 'other')
                
                # Label encode
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
                else:
                    # Handle unseen categories
                    df[f'{col}_encoded'] = df[col].apply(
                        lambda x: self.label_encoders[col].transform([x])[0] 
                        if x in self.label_encoders[col].classes_ else -1
                    )
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create key interaction features"""
        
        # Revenue-engagement interactions (using historical revenue only)
        revenue_col = 'total_revenue'  # Historical revenue, not LTV target
        if revenue_col in df.columns and 'events_per_session' in df.columns:
            df['revenue_engagement_score'] = df[revenue_col] * df['events_per_session']
        
        # Session quality score
        if 'avg_session_duration_minutes' in df.columns and 'events_per_session' in df.columns:
            df['session_quality_score'] = df['avg_session_duration_minutes'] * df['events_per_session']
        
        # Activity consistency score
        if 'sessions_per_day' in df.columns and 'session_consistency' in df.columns:
            df['activity_consistency_score'] = df['sessions_per_day'] * df['session_consistency']
        
        # Lifecycle value indicator (using historical revenue)
        if revenue_col in df.columns and 'player_lifetime_days' in df.columns:
            df['historical_lifecycle_score'] = df[revenue_col] * np.log1p(df['player_lifetime_days'])
        
        return df
    
    def _finalize_feature_columns(self, df: pd.DataFrame) -> None:
        """Finalize and organize feature columns"""
        
        # Core numeric features
        self.numeric_features = [
            # Basic activity
            'total_sessions', 'total_events', 'player_lifetime_days',
            'sessions_per_day', 'events_per_session', 'events_per_day',
            
            # Session patterns
            'avg_session_duration_minutes', 'session_efficiency', 'avg_events_per_minute',
            'long_session_rate', 'short_session_rate', 'multi_event_session_rate',
            'session_consistency', 'recent_activity_rate',
            
            # Revenue features
            'total_revenue', 'revenue_per_session', 'revenue_per_day', 'revenue_per_event',
            'purchase_frequency', 'avg_purchase_amount',
            
            # Temporal features
            'lifetime_weeks', 'days_since_last_session',
            
            # Interaction features
            'revenue_engagement_score', 'session_quality_score', 
            'activity_consistency_score', 'historical_lifecycle_score'
        ]
        
        # Core categorical features
        self.categorical_features = [
            # Revenue indicators (historical only)
            'is_historical_payer', 'is_high_value_historical_payer',
            
            # Attribution indicators
            'has_country_info', 'has_install_source', 'has_campaign_info',
            
            # Lifecycle indicators
            'is_new_player', 'is_mature_player', 'is_recently_active', 'is_at_risk',
            
            # Segments
            'player_segment'
        ]
        
        # Add attribution rate features if they exist
        attribution_rate_features = [col for col in df.columns if col.endswith('_rate') and col.startswith('is_')]
        self.numeric_features.extend(attribution_rate_features)
        
        # Add encoded categorical features
        encoded_features = [col for col in df.columns if col.endswith('_encoded')]
        self.categorical_features.extend(encoded_features)
        
        # Add segment features
        segment_features = [col for col in df.columns if col.startswith('is_segment_')]
        self.categorical_features.extend(segment_features)
        
        # Filter features that actually exist in dataframe
        self.numeric_features = [f for f in self.numeric_features if f in df.columns]
        self.categorical_features = [f for f in self.categorical_features if f in df.columns]
        
        # Combine all features
        self.feature_columns = self.numeric_features + self.categorical_features
        
        # Fill any remaining NaN values
        for col in self.feature_columns:
            if col in df.columns:
                if col in self.numeric_features:
                    df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)
                else:
                    df[col] = df[col].fillna(0)
    
    def get_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get feature matrix for modeling"""
        return df[self.feature_columns].fillna(0)
    
    def create_user_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user categories based on FUTURE LTV (whale detection for prediction)"""
        
        df_with_categories = df.copy()
        
        # Use LTV target for future whale categorization 
        ltv_col = 'ltv_target'
        historical_revenue_col = 'total_revenue'
        
        if ltv_col not in df.columns:
            print("Warning: No LTV target found, using historical revenue for categories")
            ltv_col = historical_revenue_col
        
        # Calculate LTV thresholds for whale detection
        future_payers = df_with_categories[df_with_categories[ltv_col] > 0]
        
        if len(future_payers) == 0:
            # No future payers in dataset
            df_with_categories['future_user_category'] = 'Non-payer'
            df_with_categories['will_be_whale'] = 0
            df_with_categories['will_be_medium_spender'] = 0
            df_with_categories['will_be_low_spender'] = 0
            df_with_categories['will_be_non_payer'] = 1
        else:
            # Define thresholds based on future LTV
            whale_threshold = future_payers[ltv_col].quantile(0.95)  # Top 5% of future payers
            medium_threshold = future_payers[ltv_col].quantile(0.7)   # Top 30% of future payers
            
            # Create future categories
            def categorize_future_user(ltv):
                if ltv == 0:
                    return 'Non-payer'
                elif ltv >= whale_threshold:
                    return 'Whale'
                elif ltv >= medium_threshold:
                    return 'Medium_Spender'
                else:
                    return 'Low_Spender'
            
            df_with_categories['future_user_category'] = df_with_categories[ltv_col].apply(categorize_future_user)
            
            # Create binary indicators for future behavior
            df_with_categories['will_be_whale'] = (df_with_categories['future_user_category'] == 'Whale').astype(int)
            df_with_categories['will_be_medium_spender'] = (df_with_categories['future_user_category'] == 'Medium_Spender').astype(int)
            df_with_categories['will_be_low_spender'] = (df_with_categories['future_user_category'] == 'Low_Spender').astype(int)
            df_with_categories['will_be_non_payer'] = (df_with_categories['future_user_category'] == 'Non-payer').astype(int)
            
            # Store thresholds
            self.whale_threshold = whale_threshold
            self.medium_threshold = medium_threshold
            
            # Also create historical categories for comparison
            if historical_revenue_col in df.columns:
                historical_payers = df_with_categories[df_with_categories[historical_revenue_col] > 0]
                if len(historical_payers) > 0:
                    hist_whale_threshold = historical_payers[historical_revenue_col].quantile(0.95)
                    df_with_categories['historical_user_category'] = df_with_categories[historical_revenue_col].apply(
                        lambda x: 'Whale' if x >= hist_whale_threshold else ('Payer' if x > 0 else 'Non-payer')
                    )
            
            # Print distribution
            category_counts = df_with_categories['future_user_category'].value_counts()
            total_users = len(df_with_categories)
            
            print("Future User Category Distribution (LTV-based):")
            for category, count in category_counts.items():
                percentage = (count / total_users) * 100
                avg_ltv = df_with_categories[df_with_categories['future_user_category'] == category][ltv_col].mean()
                avg_historical = df_with_categories[df_with_categories['future_user_category'] == category][historical_revenue_col].mean()
                print(f"  {category}: {count:,} users ({percentage:.1f}%) - Avg LTV: ${avg_ltv:.2f} - Avg Historical: ${avg_historical:.2f}")
            
            print(f"\nFuture LTV Thresholds:")
            print(f"  Future Whale threshold (95th percentile): ${whale_threshold:.2f}")
            print(f"  Future Medium spender threshold (70th percentile): ${medium_threshold:.2f}")
        
        return df_with_categories
    
    def get_whale_features(self, df: pd.DataFrame) -> List[str]:
        """Get features most predictive of whale behavior"""
        
        whale_predictive_features = [
            'total_revenue', 'revenue_per_session', 'revenue_per_day',
            'avg_purchase_amount', 'purchase_frequency',
            'session_quality_score', 'lifecycle_value_score',
            'revenue_engagement_score', 'long_session_rate',
            'events_per_session', 'sessions_per_day',
            'player_lifetime_days', 'is_mature_player'
        ]
        
        # Filter to only include features that exist in the dataframe
        available_features = [f for f in whale_predictive_features if f in df.columns]
        
        return available_features
    
    def get_feature_importance_names(self) -> List[str]:
        """Get list of all feature names for importance analysis"""
        return self.feature_columns.copy()
    
    def print_feature_summary(self, df: pd.DataFrame) -> None:
        """Print summary of created features"""
        
        print(f"\n=== Feature Engineering Summary ===")
        print(f"Total Features: {len(self.feature_columns)}")
        print(f"Numeric Features: {len(self.numeric_features)}")
        print(f"Categorical Features: {len(self.categorical_features)}")
        
        print(f"\n=== Key Feature Categories ===")
        print(f"Activity Features: {len([f for f in self.numeric_features if any(x in f for x in ['session', 'event', 'day'])])}")
        print(f"Revenue Features: {len([f for f in self.feature_columns if 'revenue' in f or 'purchase' in f])}")
        print(f"Attribution Features: {len([f for f in self.feature_columns if any(x in f for x in ['country', 'source', 'campaign'])])}")
        print(f"Temporal Features: {len([f for f in self.feature_columns if any(x in f for x in ['lifetime', 'days_since', 'recent'])])}")
        
        # Check data quality
        missing_data = df[self.feature_columns].isnull().sum()
        if missing_data.sum() > 0:
            print(f"\n=== Data Quality Issues ===")
            print(f"Features with missing data: {(missing_data > 0).sum()}")
            print("Top missing features:")
            print(missing_data[missing_data > 0].head())
        else:
            print(f"\n=== Data Quality ===")
            print("✓ No missing data in feature matrix")