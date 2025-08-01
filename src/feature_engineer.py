"""
Streamlined feature engineering for pLTV prediction
Focuses on essential features derivable from event-level data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Streamlined feature engineering for gaming pLTV prediction"""
    
    def __init__(self):
        self.feature_columns = []
        self.numeric_features = []
        self.categorical_features = []
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.cluster_model = KMeans(n_clusters=5, random_state=42)
        self.use_smote = False
        self.smote = SMOTE(random_state=42)
        
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
        """Create enhanced activity and engagement features"""
        
        # Core activity metrics (assuming these are pre-aggregated)
        if 'total_sessions' in df.columns and 'player_lifetime_days' in df.columns:
            df['sessions_per_day'] = df['total_sessions'] / np.maximum(df['player_lifetime_days'], 1)
            
            # Enhanced activity patterns
            df['session_frequency_score'] = np.log1p(df['sessions_per_day'])
            df['is_daily_player'] = (df['sessions_per_day'] >= 1.0).astype(int)
            df['is_casual_player'] = (df['sessions_per_day'] < 0.5).astype(int)
            df['is_hardcore_player'] = (df['sessions_per_day'] > 3.0).astype(int)
        else:
            df['sessions_per_day'] = 0
            df['session_frequency_score'] = 0
            df['is_daily_player'] = 0
            df['is_casual_player'] = 1
            df['is_hardcore_player'] = 0
            
        if 'total_events' in df.columns:
            if 'total_sessions' in df.columns:
                df['events_per_session'] = df['total_events'] / np.maximum(df['total_sessions'], 1)
                
                # Enhanced engagement metrics
                df['engagement_intensity'] = np.log1p(df['events_per_session'])
                df['is_high_engagement'] = (df['events_per_session'] > df['events_per_session'].quantile(0.75)).astype(int)
                df['is_low_engagement'] = (df['events_per_session'] < df['events_per_session'].quantile(0.25)).astype(int)
                
            if 'player_lifetime_days' in df.columns:
                df['events_per_day'] = df['total_events'] / np.maximum(df['player_lifetime_days'], 1)
                df['activity_velocity'] = np.log1p(df['events_per_day'])
        
        # Session quality and efficiency
        if 'total_events' in df.columns and 'total_sessions' in df.columns:
            total_session_time = df.get('total_session_duration_minutes', df['total_sessions'] * 10)  # Default 10 min per session
            df['session_efficiency'] = df['total_events'] / np.maximum(total_session_time, 1)
            df['efficiency_score'] = np.log1p(df['session_efficiency'])
        
        # Engagement depth
        if 'total_events' in df.columns and 'total_sessions' in df.columns:
            df['engagement_depth'] = df['total_events'] * np.log1p(df['total_sessions'])
            df['activity_score'] = np.sqrt(df['total_events'] * df['total_sessions'])
        
        # Add important new features from ChatGPT suggestions
        # 1. First session engagement quality (proxy for immediate user intent)
        if 'total_sessions' in df.columns and 'total_events' in df.columns:
            # Estimate first session quality based on events per session
            df['first_session_quality'] = df['total_events'] / np.maximum(df['total_sessions'], 1)
            df['is_high_first_engagement'] = (df['first_session_quality'] > df['first_session_quality'].quantile(0.8)).astype(int)
        
        # 2. Purchase intent signals (based on revenue behavior)
        if 'purchase_events' in df.columns:
            df['has_purchase_intent'] = (df['purchase_events'] > 0).astype(int)
            # Purchase conversion rate from events
            if 'total_events' in df.columns:
                df['purchase_conversion_rate'] = df['purchase_events'] / np.maximum(df['total_events'], 1)
                df['is_high_purchase_intent'] = (df['purchase_conversion_rate'] > 0).astype(int)
        
        return df
    
    def _create_revenue_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced revenue and monetization features from HISTORICAL data only"""
        
        # Use historical revenue (observation period) for features - NOT the LTV target
        revenue_col = 'total_revenue'  # This is historical revenue from observation period
        
        if revenue_col in df.columns:
            # Basic revenue indicators (based on historical behavior)
            df['is_historical_payer'] = (df[revenue_col] > 0).astype(int)
            
            # Enhanced revenue categorization
            payer_revenues = df[df[revenue_col] > 0][revenue_col]
            if len(payer_revenues) > 0:
                whale_threshold = payer_revenues.quantile(0.95)
                high_value_threshold = payer_revenues.quantile(0.8)
                medium_value_threshold = payer_revenues.quantile(0.5)
                
                df['is_whale_historical'] = (df[revenue_col] >= whale_threshold).astype(int)
                df['is_high_value_historical_payer'] = (df[revenue_col] >= high_value_threshold).astype(int)
                df['is_medium_value_historical_payer'] = (df[revenue_col] >= medium_value_threshold).astype(int)
                df['is_low_value_historical_payer'] = ((df[revenue_col] > 0) & (df[revenue_col] < medium_value_threshold)).astype(int)
            else:
                df['is_whale_historical'] = 0
                df['is_high_value_historical_payer'] = 0
                df['is_medium_value_historical_payer'] = 0
                df['is_low_value_historical_payer'] = 0
            
            # Revenue efficiency metrics (historical)
            if 'total_sessions' in df.columns:
                df['revenue_per_session'] = df[revenue_col] / np.maximum(df['total_sessions'], 1)
                df['revenue_session_score'] = np.log1p(df['revenue_per_session'])
                
            if 'player_lifetime_days' in df.columns:
                df['revenue_per_day'] = df[revenue_col] / np.maximum(df['player_lifetime_days'], 1)
                df['revenue_velocity'] = np.log1p(df['revenue_per_day'])
                
            if 'total_events' in df.columns:
                df['revenue_per_event'] = df[revenue_col] / np.maximum(df['total_events'], 1)
                df['monetization_efficiency'] = np.log1p(df['revenue_per_event'])
            
            # Revenue patterns
            df['revenue_log'] = np.log1p(df[revenue_col])
            df['revenue_sqrt'] = np.sqrt(df[revenue_col])
            
            # Revenue concentration (how quickly they spent)
            if 'days_since_first_session' in df.columns:
                df['revenue_concentration'] = df[revenue_col] / np.maximum(df['days_since_first_session'], 1)
                df['early_monetizer'] = ((df[revenue_col] > 0) & (df['days_since_first_session'] <= 7)).astype(int)
                
        else:
            # Create default values if revenue column not found
            default_revenue_features = [
                'is_historical_payer', 'is_whale_historical', 'is_high_value_historical_payer',
                'is_medium_value_historical_payer', 'is_low_value_historical_payer',
                'revenue_per_session', 'revenue_per_day', 'revenue_per_event',
                'revenue_session_score', 'revenue_velocity', 'monetization_efficiency',
                'revenue_log', 'revenue_sqrt', 'revenue_concentration', 'early_monetizer'
            ]
            for feature in default_revenue_features:
                df[feature] = 0
        
        # Enhanced purchase behavior (if available)
        if 'purchase_events' in df.columns and 'total_sessions' in df.columns:
            df['purchase_frequency'] = df['purchase_events'] / np.maximum(df['total_sessions'], 1)
            df['purchase_propensity'] = np.log1p(df['purchase_frequency'])
            df['is_frequent_purchaser'] = (df['purchase_frequency'] > 0.1).astype(int)
        
        # Purchase value patterns
        if 'total_revenue' in df.columns and 'purchase_events' in df.columns:
            df['avg_purchase_value'] = df['total_revenue'] / np.maximum(df['purchase_events'], 1)
            df['purchase_value_score'] = np.log1p(df['avg_purchase_value'])
        
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
        
        # Enhanced numeric features
        self.numeric_features = [
            # Basic activity
            'total_sessions', 'total_events', 'player_lifetime_days',
            'sessions_per_day', 'events_per_session', 'events_per_day',
            
            # Enhanced activity patterns
            'session_frequency_score', 'engagement_intensity', 'activity_velocity',
            'engagement_depth', 'activity_score', 'efficiency_score',
            
            # Session patterns
            'session_efficiency', 'long_session_rate', 'short_session_rate', 
            'multi_event_session_rate', 'session_consistency', 'recent_activity_rate',
            
            # Enhanced revenue features
            'total_revenue', 'revenue_per_session', 'revenue_per_day', 'revenue_per_event',
            'revenue_session_score', 'revenue_velocity', 'monetization_efficiency',
            'revenue_log', 'revenue_sqrt', 'revenue_concentration',
            'purchase_frequency', 'purchase_propensity', 'avg_purchase_value', 'purchase_value_score',
            
            # Temporal features
            'lifetime_weeks', 'days_since_last_session',
            
            # Interaction features
            'revenue_engagement_score', 'session_quality_score', 
            'activity_consistency_score', 'historical_lifecycle_score'
        ]
        
        # Enhanced categorical features
        self.categorical_features = [
            # Enhanced revenue indicators (historical only)
            'is_historical_payer', 'is_whale_historical', 'is_high_value_historical_payer',
            'is_medium_value_historical_payer', 'is_low_value_historical_payer', 'early_monetizer',
            
            # Enhanced player type indicators
            'is_daily_player', 'is_casual_player', 'is_hardcore_player',
            'is_high_engagement', 'is_low_engagement', 'is_frequent_purchaser',
            
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
    
    def get_feature_matrix(self, df: pd.DataFrame, apply_smote: bool = None) -> pd.DataFrame:
        """Get feature matrix for modeling with optional SMOTE"""
        
        X = df[self.feature_columns].fillna(0)
        
        # Apply SMOTE if requested and we have target variable
        # Explicit apply_smote=False overrides self.use_smote
        should_apply_smote = (apply_smote if apply_smote is not None else self.use_smote)
        if should_apply_smote and 'ltv_target' in df.columns:
            try:
                # Create binary target for SMOTE (payer vs non-payer)
                y_binary = (df['ltv_target'] > 0).astype(int)
                
                # Only apply SMOTE if we have both classes
                if len(np.unique(y_binary)) > 1:
                    logger.info(f"Applying SMOTE. Before: {len(X)} samples")
                    # Ensure X is numeric and properly formatted for SMOTE
                    X_numeric = X.astype(float)
                    X_resampled, y_resampled = self.smote.fit_resample(X_numeric, y_binary)
                    
                    # Create new dataframe with resampled data
                    X_df = pd.DataFrame(X_resampled, columns=X.columns)
                    
                    # Estimate LTV for new synthetic samples
                    # For synthetic non-payers: LTV = 0
                    # For synthetic payers: Use median LTV of real payers
                    real_payer_ltv = df[df['ltv_target'] > 0]['ltv_target'].median()
                    synthetic_ltv = np.where(y_resampled == 1, real_payer_ltv, 0)
                    
                    # Add LTV target to resampled data
                    X_df['ltv_target'] = synthetic_ltv
                    
                    logger.info(f"SMOTE applied. After: {len(X_df)} samples, Payer ratio: {y_resampled.mean():.3f}")
                    return X_df
                else:
                    logger.warning("SMOTE skipped: Only one class present in target")
            except Exception as e:
                logger.warning(f"SMOTE failed: {e}. Using original data.")
                
        return X
    
    def create_user_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """SIMPLE USER CATEGORIZATION - Easy to understand whale detection"""
        
        df_with_categories = df.copy()
        
        # STEP 1: Get the future LTV values (what we want to predict)
        future_ltv = df_with_categories['ltv_target'].values
        
        # STEP 2: Find players who will spend money in the future
        future_payers = future_ltv[future_ltv > 0]  # Only players with LTV > $0
        
        if len(future_payers) == 0:
            print("WARNING: No future payers found in dataset")
            df_with_categories['future_user_category'] = 'Non-payer'
            df_with_categories['will_be_whale'] = 0
            return df_with_categories
        
        # STEP 3: SIMPLE THRESHOLDS - Easy to understand
        # Instead of percentiles, use simple dollar amounts
        print(f"Future payers analysis:")
        print(f"   Players who will spend: {len(future_payers):,}")
        print(f"   Average future spending: ${future_payers.mean():.2f}")
        print(f"   Maximum future spending: ${future_payers.max():.2f}")
        
        # SIMPLE WHALE DEFINITION: Top spenders (above average + 2 standard deviations)
        avg_spending = future_payers.mean()
        std_spending = future_payers.std()
        
        # Simple thresholds:
        whale_threshold = avg_spending + (2 * std_spending)  # High spenders
        medium_threshold = avg_spending                       # Average spenders
        
        print(f"\nSIMPLE THRESHOLDS:")
        print(f"   Whale threshold: ${whale_threshold:.2f} (High spenders)")  
        print(f"   Medium threshold: ${medium_threshold:.2f} (Average spenders)")
        print(f"   Low threshold: $0.01 (Any spenders)")
        print(f"\nNote: Using SMOTE under/oversampling to balance whale vs non-whale data for training")
        
        # STEP 4: Categorize each player
        def simple_categorize(ltv_value):
            if ltv_value == 0:
                return 'Non-payer'         # Will spend $0
            elif ltv_value >= whale_threshold:
                return 'Whale'             # Will spend a lot (above avg + 2*std)
            elif ltv_value >= medium_threshold:
                return 'Medium_Spender'    # Will spend average amount
            else:
                return 'Low_Spender'       # Will spend below average
        
        # Apply categorization
        df_with_categories['future_user_category'] = df_with_categories['ltv_target'].apply(simple_categorize)
        
        # Create simple binary flags
        df_with_categories['will_be_whale'] = (df_with_categories['future_user_category'] == 'Whale').astype(int)
        df_with_categories['will_be_medium_spender'] = (df_with_categories['future_user_category'] == 'Medium_Spender').astype(int)
        df_with_categories['will_be_low_spender'] = (df_with_categories['future_user_category'] == 'Low_Spender').astype(int)
        df_with_categories['will_be_non_payer'] = (df_with_categories['future_user_category'] == 'Non-payer').astype(int)
        
        # Store thresholds for later use
        self.whale_threshold = whale_threshold
        self.medium_threshold = medium_threshold
        
        # STEP 5: Print results
        category_counts = df_with_categories['future_user_category'].value_counts()
        total_users = len(df_with_categories)
        
        print(f"\nCATEGORIZATION RESULTS:")
        for category, count in category_counts.items():
            percentage = (count / total_users) * 100
            avg_ltv = df_with_categories[df_with_categories['future_user_category'] == category]['ltv_target'].mean()
            print(f"   {category}: {count:,} users ({percentage:.1f}%) - Avg Future LTV: ${avg_ltv:.2f}")
        
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
        
        print(f"\nFeature Engineering Summary")
        print(f"Total Features: {len(self.feature_columns)}")
        print(f"Numeric Features: {len(self.numeric_features)}")
        print(f"Categorical Features: {len(self.categorical_features)}")
        
        print(f"\nKey Feature Categories")
        print(f"Activity Features: {len([f for f in self.numeric_features if any(x in f for x in ['session', 'event', 'day'])])}")
        print(f"Revenue Features: {len([f for f in self.feature_columns if 'revenue' in f or 'purchase' in f])}")
        print(f"Attribution Features: {len([f for f in self.feature_columns if any(x in f for x in ['country', 'source', 'campaign'])])}")
        print(f"Temporal Features: {len([f for f in self.feature_columns if any(x in f for x in ['lifetime', 'days_since', 'recent'])])}")
        
        # Check data quality
        missing_data = df[self.feature_columns].isnull().sum()
        if missing_data.sum() > 0:
            print(f"\nData Quality Issues")
            print(f"Features with missing data: {(missing_data > 0).sum()}")
            print("Top missing features:")
            print(missing_data[missing_data > 0].head())
        else:
            print(f"\nData Quality")
            print("No missing data in feature matrix")