import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Global variable to collect plot data
PLOT_DATA = []

def collect_evaluation_data(y_true, y_pred, model_name, feature_importance=None):
    """Collect evaluation data for comprehensive plotting"""
    from scipy import stats
    
    # Calculate comprehensive metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE only for non-zero actual values
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) > 0:
        mape = mean_absolute_percentage_error(y_true[non_zero_mask], y_pred[non_zero_mask])
    else:
        mape = float('inf')
    
    # Revenue capture metrics
    total_actual_revenue = np.sum(y_true)
    total_predicted_revenue = np.sum(y_pred)
    revenue_capture_rate = total_predicted_revenue / total_actual_revenue if total_actual_revenue > 0 else 0
    
    # Spender identification metrics
    actual_spenders = np.sum(y_true > 0)
    predicted_spenders = np.sum(y_pred > 0)
    
    # Top percentile analysis
    top_5_pct_idx = np.argsort(y_pred)[-int(0.05 * len(y_pred)):]
    top_10_pct_idx = np.argsort(y_pred)[-int(0.10 * len(y_pred)):]
    
    top_5_pct_actual = np.sum(y_true[top_5_pct_idx])
    top_10_pct_actual = np.sum(y_true[top_10_pct_idx])
    
    top_5_pct_capture = top_5_pct_actual / total_actual_revenue if total_actual_revenue > 0 else 0
    top_10_pct_capture = top_10_pct_actual / total_actual_revenue if total_actual_revenue > 0 else 0
    
    # Correlation analysis
    correlation = stats.pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else 0
    spearman_corr = stats.spearmanr(y_true, y_pred)[0] if len(y_true) > 1 else 0
    
    plot_info = {
        'model_name': model_name,
        'y_true': y_true,
        'y_pred': y_pred,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'correlation': correlation,
        'spearman_corr': spearman_corr,
        'revenue_capture_rate': revenue_capture_rate,
        'total_actual_revenue': total_actual_revenue,
        'total_predicted_revenue': total_predicted_revenue,
        'actual_spenders': actual_spenders,
        'predicted_spenders': predicted_spenders,
        'top_5_pct_capture': top_5_pct_capture,
        'top_10_pct_capture': top_10_pct_capture,
        'feature_importance': feature_importance
    }
    
    PLOT_DATA.append(plot_info)
    print(f"Collected evaluation data for {model_name}")

def plot_comprehensive_evaluation():
    """Create comprehensive evaluation plots for all models"""
    if not PLOT_DATA:
        print("No evaluation data collected")
        return
    
    n_models = len(PLOT_DATA)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(40, 30))
    
    # Create a grid layout
    gs = fig.add_gridspec(4, n_models * 2, hspace=0.3, wspace=0.3)
    
    for i, data in enumerate(PLOT_DATA):
        col_start = i * 2
        
        # 1. Predicted vs Actual scatter plot
        ax1 = fig.add_subplot(gs[0, col_start:col_start+2])
        
        # Sample data if too large for plotting
        plot_indices = np.random.choice(len(data['y_true']), 
                                      min(1000, len(data['y_true'])), 
                                      replace=False)
        y_true_sample = data['y_true'][plot_indices]
        y_pred_sample = data['y_pred'][plot_indices]
        
        ax1.scatter(y_true_sample, y_pred_sample, alpha=0.6, s=20)
        max_val = max(np.max(y_true_sample), np.max(y_pred_sample))
        ax1.plot([0, max_val], [0, max_val], 'r--', lw=2, alpha=0.8)
        ax1.set_xlabel('Actual LTV')
        ax1.set_ylabel('Predicted LTV')
        ax1.set_title(f'{data["model_name"]}\nPredicted vs Actual (R² = {data["r2"]:.4f})')
        ax1.grid(True, alpha=0.3)
        
        # Add correlation text
        ax1.text(0.05, 0.95, f'Pearson: {data["correlation"]:.3f}\nSpearman: {data["spearman_corr"]:.3f}', 
                transform=ax1.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Residuals plot
        ax2 = fig.add_subplot(gs[1, col_start:col_start+2])
        residuals = y_pred_sample - y_true_sample
        ax2.scatter(y_pred_sample, residuals, alpha=0.6, s=20)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted LTV')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'{data["model_name"]}\nResiduals Plot')
        ax2.grid(True, alpha=0.3)
        
        # 3. Revenue capture analysis
        ax3 = fig.add_subplot(gs[2, col_start])
        
        # Revenue metrics
        metrics = ['Total Revenue\nCapture', 'Top 5%\nCapture', 'Top 10%\nCapture']
        values = [data['revenue_capture_rate'], data['top_5_pct_capture'], data['top_10_pct_capture']]
        colors = ['skyblue', 'lightgreen', 'lightcoral']
        
        bars = ax3.bar(metrics, values, color=colors, alpha=0.7)
        ax3.set_ylabel('Capture Rate')
        ax3.set_title(f'{data["model_name"]}\nRevenue Capture Analysis')
        ax3.set_ylim(0, 1.2)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Model performance metrics
        ax4 = fig.add_subplot(gs[2, col_start+1])
        
        # Performance metrics
        perf_metrics = ['MAE', 'RMSE', 'MAPE']
        perf_values = [data['mae'], data['rmse'], 
                      data['mape'] if data['mape'] != float('inf') else 0]
        
        # Normalize values for better visualization
        if max(perf_values) > 0:
            perf_values_norm = [v / max(perf_values) for v in perf_values]
        else:
            perf_values_norm = perf_values
            
        bars = ax4.bar(perf_metrics, perf_values_norm, color=['orange', 'purple', 'brown'], alpha=0.7)
        ax4.set_ylabel('Normalized Error')
        ax4.set_title(f'{data["model_name"]}\nError Metrics (Normalized)')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add actual values as text
        for bar, actual_val in zip(bars, perf_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{actual_val:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 5. Feature importance (if available)
        if data['feature_importance'] is not None:
            ax5 = fig.add_subplot(gs[3, col_start:col_start+2])
            
            # Get top 10 features
            importance_data = data['feature_importance']
            if len(importance_data['importance']) > 10:
                top_indices = np.argsort(importance_data['importance'])[-10:]
                top_features = [importance_data['feature_names'][i] for i in top_indices]
                top_importance = [importance_data['importance'][i] for i in top_indices]
            else:
                top_features = importance_data['feature_names']
                top_importance = importance_data['importance']
            
            y_pos = np.arange(len(top_features))
            ax5.barh(y_pos, top_importance, alpha=0.7)
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels(top_features)
            ax5.set_xlabel('Feature Importance')
            ax5.set_title(f'{data["model_name"]}\nTop Feature Importance')
            ax5.grid(True, alpha=0.3, axis='x')
        else:
            # Model summary instead of feature importance
            ax5 = fig.add_subplot(gs[3, col_start:col_start+2])
            ax5.axis('off')
            
            summary_text = f"""
            {data["model_name"]} - PERFORMANCE SUMMARY
            
            Accuracy Metrics:
            • MAE: {data['mae']:.4f}
            • RMSE: {data['rmse']:.4f}
            • R² Score: {data['r2']:.4f}
            • MAPE: {data['mape']:.2f}% (non-zero only)
            
            Business Metrics:
            • Total Revenue Capture: {data['revenue_capture_rate']*100:.1f}%
            • Predicted Revenue: ${data['total_predicted_revenue']:.2f}
            • Actual Revenue: ${data['total_actual_revenue']:.2f}
            
            User Targeting:
            • Actual Spenders: {data['actual_spenders']}
            • Predicted Spenders: {data['predicted_spenders']}
            
            Top Percentile Performance:
            • Top 5% Revenue Capture: {data['top_5_pct_capture']*100:.1f}%
            • Top 10% Revenue Capture: {data['top_10_pct_capture']*100:.1f}%
            """
            
            ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Comprehensive LTV Model Evaluation', fontsize=20, y=0.98)
    plt.savefig('comprehensive_ltv_evaluation.png', dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive evaluation plot with {n_models} models")
    plt.close()
    
    # Clear collected data
    PLOT_DATA.clear()

def load_temporal_datasets():
    """Load temporal datasets using only behavioral features (no revenue features)"""
    try:
        from google.cloud import bigquery
    except ImportError:
        print("Warning: google-cloud-bigquery not installed. Skipping BigQuery functionality.")
        return None, None, None
    
    print("="*80)
    print("LOADING TEMPORAL DATASETS - BEHAVIORAL FEATURES ONLY")
    print("="*80)
    print("Train: Oct 2024 - Mar 2025 users, D0-D3 behavioral features → D4-D33 LTV targets")
    print("Val: Apr 2025 users, D0-D3 behavioral features → D4-D33 LTV targets")
    print("Test: May+ 2025 users, D0-D3 behavioral features → D4-D33 LTV targets")
    print("="*80)
    
    PROJECT_ID = "gc-forecasting-dev"  # Replace with your project ID
    DATASET = "test_data"        # Replace with your dataset
    TABLE = "app_data"           # Replace with your table
    FEATURE_DAYS = 3               # D0-D3 (4 days)
    PREDICTION_DAYS = 30           # D4-D33 (30 days)
    
    client = bigquery.Client(project=PROJECT_ID)
    table_path = f"{PROJECT_ID}.{DATASET}.{TABLE}"
    
    # Base query for behavioral features (NO REVENUE FIELDS)
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
    ),
    
    behavioral_features AS (
      SELECT
        COALESCE(e.gaid, e.idfa, e.android_id, e.waid, e.idfv) AS user_id,
        i.install_date,
        i.install_timestamp,
        
        -- Session and engagement metrics (behavioral only)
        COUNT(DISTINCT e.session_id) AS total_sessions,
        COUNT(*) AS total_events,
        COUNT(DISTINCT DATE(e.attribution_event_timestamp)) AS active_days,
        
        -- Timing patterns
        MIN(e.attribution_event_timestamp) AS first_event,
        MAX(e.attribution_event_timestamp) AS last_event,
        
        -- Product interaction behavioral patterns (NO REVENUE DATA)
        COUNT(CASE WHEN e.product_name IS NOT NULL THEN 1 END) AS product_interactions,
        COUNT(DISTINCT e.product_name) AS unique_products_viewed,
        COUNT(DISTINCT e.product_sku) AS unique_skus_viewed,
        AVG(SAFE_CAST(e.product_quantity AS FLOAT64)) AS avg_quantity_per_interaction,
        SUM(SAFE_CAST(e.product_quantity AS FLOAT64)) AS total_quantity_interactions,
        
        -- Platform and attribution
        MAX(CASE WHEN e.gaid IS NOT NULL THEN 'android' 
                WHEN e.idfa IS NOT NULL THEN 'ios' 
                ELSE 'unknown' END) AS platform,
        MAX(e.country) AS country,
        MAX(e.install_source) AS install_source,
        MAX(e.city) AS city,
        MAX(e.state) AS state,
        
        -- Campaign and attribution features (behavioral)
        MAX(e.campaign_name) AS campaign_name,
        MAX(e.partner) AS partner,
        MAX(e.publisher_name) AS publisher_name,
        COUNTIF(e.is_fingerprinted = true) AS fingerprinted_events,
        COUNTIF(e.is_reengagement = true) AS reengagement_events,
        COUNTIF(e.is_view_through = true) AS view_through_events,
        
        -- Time-based behavioral patterns
        COUNTIF(EXTRACT(HOUR FROM e.attribution_event_timestamp) BETWEEN 9 AND 17) AS business_hours_events,
        COUNTIF(EXTRACT(DAYOFWEEK FROM e.attribution_event_timestamp) IN (1, 7)) AS weekend_events,
        COUNTIF(EXTRACT(HOUR FROM e.attribution_event_timestamp) BETWEEN 18 AND 23) AS evening_events,
        COUNTIF(EXTRACT(HOUR FROM e.attribution_event_timestamp) BETWEEN 0 AND 6) AS late_night_events,
        
        -- Session depth indicators
        COUNT(DISTINCT e.session_id) / COUNT(DISTINCT DATE(e.attribution_event_timestamp)) AS avg_sessions_per_day,
        COUNT(*) / COUNT(DISTINCT e.session_id) AS avg_events_per_session
                
      FROM `{table_path}` e
      JOIN install_dates i ON COALESCE(e.gaid, e.idfa, e.android_id, e.waid, e.idfv) = i.user_id
      WHERE DATE(e.attribution_event_timestamp) BETWEEN i.install_date 
        AND DATE_ADD(i.install_date, INTERVAL {FEATURE_DAYS} DAY)
      GROUP BY user_id, i.install_date, i.install_timestamp
    ),
    
    ltv_targets AS (
      SELECT
        COALESCE(e.gaid, e.idfa, e.android_id, e.waid, e.idfv) AS user_id,
        -- Only use revenue fields for TARGET calculation (not features)
        COALESCE(SUM(SAFE_CAST(e.revenue AS FLOAT64)), 0) AS ltv_30_days,
        COALESCE(SUM(SAFE_CAST(e.received_revenue AS FLOAT64)), 0) AS received_ltv_30_days,
        COUNT(CASE WHEN SAFE_CAST(e.revenue AS FLOAT64) > 0 THEN 1 END) AS purchase_events
      FROM `{table_path}` e
      JOIN install_dates i ON COALESCE(e.gaid, e.idfa, e.android_id, e.waid, e.idfv) = i.user_id
      WHERE DATE(e.attribution_event_timestamp) BETWEEN DATE_ADD(i.install_date, INTERVAL {FEATURE_DAYS + 1} DAY)
        AND DATE_ADD(i.install_date, INTERVAL {FEATURE_DAYS + PREDICTION_DAYS} DAY)
      GROUP BY user_id
    )
    
    SELECT 
      f.*,
      -- Derived behavioral features
      TIMESTAMP_DIFF(f.last_event, f.first_event, MINUTE) AS session_duration_minutes,
      TIMESTAMP_DIFF(f.first_event, f.install_timestamp, MINUTE) AS time_to_first_session_minutes,
      
      -- Engagement intensity features
      SAFE_DIVIDE(f.total_events, f.total_sessions) AS events_per_session,
      SAFE_DIVIDE(f.product_interactions, f.total_sessions) AS product_interactions_per_session,
      SAFE_DIVIDE(f.unique_products_viewed, NULLIF(f.product_interactions, 0)) AS product_diversity_rate,
      SAFE_DIVIDE(f.product_interactions, f.total_events) AS product_engagement_rate,
      SAFE_DIVIDE(f.total_events, f.active_days) AS events_per_active_day,
      SAFE_DIVIDE(f.business_hours_events, f.total_events) AS business_hours_ratio,
      SAFE_DIVIDE(f.weekend_events, f.total_events) AS weekend_activity_ratio,
      SAFE_DIVIDE(f.evening_events, f.total_events) AS evening_activity_ratio,
      SAFE_DIVIDE(f.late_night_events, f.total_events) AS late_night_ratio,
      SAFE_DIVIDE(f.fingerprinted_events, f.total_events) AS fingerprinted_ratio,
      SAFE_DIVIDE(f.reengagement_events, f.total_events) AS reengagement_ratio,
      
      -- Target variable (revenue-based but only for labels)
      COALESCE(t.ltv_30_days, 0) AS ltv_30_days,
      COALESCE(t.received_ltv_30_days, 0) AS received_ltv_30_days,
      COALESCE(t.purchase_events, 0) AS purchase_events
      
    FROM behavioral_features f
    LEFT JOIN ltv_targets t ON f.user_id = t.user_id
    WHERE f.total_sessions > 0 AND f.active_days > 0
    ORDER BY RAND()
    """
    
    # Load Training Data (Oct 2024 - Mar 2025)
    print("\n1. Loading Training Cohort (Oct 2024 - Mar 2025)...")
    train_query = base_query.format(start_date='2024-10-01', end_date='2025-03-31')
    train_data = client.query(train_query).result().to_dataframe()
    print(f"   Training cohort: {len(train_data):,} users")
    
    # Load Validation Data (Apr 2025)
    print("\n2. Loading Validation Cohort (Apr 2025)...")
    val_query = base_query.format(start_date='2025-04-01', end_date='2025-04-30')
    val_data = client.query(val_query).result().to_dataframe()
    print(f"   Validation cohort: {len(val_data):,} users")
    
    # Load Test Data (May+ 2025)
    print("\n3. Loading Test Cohort (May+ 2025)...")
    test_query = base_query.format(start_date='2025-05-01', end_date='2025-08-01')
    test_data = client.query(test_query).result().to_dataframe()
    print(f"   Test cohort: {len(test_data):,} users")
    
    print(f"\nDataset Summary:")
    print(f"Train: {len(train_data):,} users, Avg LTV: ${train_data['ltv_30_days'].mean():.2f}")
    print(f"Val: {len(val_data):,} users, Avg LTV: ${val_data['ltv_30_days'].mean():.2f}")
    print(f"Test: {len(test_data):,} users, Avg LTV: ${test_data['ltv_30_days'].mean():.2f}")
    
    return train_data, val_data, test_data

def prepare_features(train_data, val_data, test_data):
    """Prepare features ensuring no revenue data leakage"""
    
    # Define columns to exclude (identifiers, targets, and revenue-related)
    exclude_cols = [
        'user_id', 'install_date', 'install_timestamp', 
        'first_event', 'last_event',  # datetime columns
        'ltv_30_days', 'received_ltv_30_days', 'purchase_events',  # target columns
        'purchase_events_30d',  # additional target columns
        'revenue', 'received_revenue', 'product_price',  # revenue columns to prevent leakage
        'is_revenue_receipt_included', 'is_revenue_valid'  # revenue-related flags
    ]
    
    # Get behavioral feature columns only
    all_cols = set(train_data.columns) & set(val_data.columns) & set(test_data.columns)
    feature_cols = [col for col in all_cols if col not in exclude_cols]
    
    print(f"Using {len(feature_cols)} behavioral features (no revenue data)")
    
    # Extract features
    X_train = train_data[feature_cols].copy()
    X_val = val_data[feature_cols].copy()
    X_test = test_data[feature_cols].copy()
    
    # Handle missing values
    for df in [X_train, X_val, X_test]:
        df.fillna(0, inplace=True)
        # Replace infinite values
        df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Encode categorical features consistently
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
    
    # Create binary features for top categories
    high_cardinality_cols = ['country', 'channel', 'engagement_bucket']  # Updated column names
    for col in high_cardinality_cols:
        if col in X_train.columns:
            # Get top 5 categories from training data
            combined_col = pd.concat([train_data[col], val_data[col], test_data[col]])
            top_categories = combined_col.value_counts().head(5).index
            
            for cat in top_categories:
                X_train[f'{col}_is_{cat}'] = (train_data[col] == cat).astype(int)
                X_val[f'{col}_is_{cat}'] = (val_data[col] == cat).astype(int)
                X_test[f'{col}_is_{cat}'] = (test_data[col] == cat).astype(int)
    
    # Extract targets - handle different possible target column names
    target_col = 'ltv_30_days'
    if target_col not in train_data.columns:
        print("Warning: ltv_30_days column not found, checking for alternatives...")
        possible_targets = ['ltv_30_days', 'ltv_7_days', 'revenue_30d', 'total_revenue']
        for col in possible_targets:
            if col in train_data.columns:
                target_col = col
                print(f"Using {target_col} as target variable")
                break
        else:
            raise ValueError("No suitable target column found")
    
    y_train = train_data[target_col].values
    y_val = val_data[target_col].values
    y_test = test_data[target_col].values
    
    # Create additional derived features
    print("Creating advanced behavioral features...")
    
    for df, data in zip([X_train, X_val, X_test], [train_data, val_data, test_data]):
        # Ensure we have the required columns for calculations
        session_col = 'total_sessions'
        events_col = 'total_events'
        active_days_col = 'active_days'
        
        # Basic engagement score (handle missing columns gracefully)
        engagement_components = []
        if session_col in df.columns:
            engagement_components.append(df[session_col] * 0.3)
        if events_col in df.columns:
            engagement_components.append(df[events_col] * 0.2)
        if active_days_col in df.columns:
            engagement_components.append(df[active_days_col] * 0.2)
        
        # Add store/product interaction components if available
        if 'store_interactions' in df.columns:
            engagement_components.append(df['store_interactions'] * 0.2)
        if 'unique_products_viewed' in df.columns:
            engagement_components.append(df['unique_products_viewed'] * 0.1)
        
        if engagement_components:
            df['engagement_score'] = sum(engagement_components)
        else:
            df['engagement_score'] = 0
        
        # User behavior pattern features (with safe calculations)
        if events_col in df.columns:
            df['is_heavy_user'] = (df[events_col] > df[events_col].quantile(0.8)).astype(int)
        else:
            df['is_heavy_user'] = 0
            
        if 'unique_products_viewed' in df.columns:
            df['is_product_explorer'] = (df['unique_products_viewed'] > df['unique_products_viewed'].quantile(0.7)).astype(int)
        else:
            df['is_product_explorer'] = 0
            
        if active_days_col in df.columns:
            df['is_consistent_user'] = (df[active_days_col] >= 3).astype(int)
        else:
            df['is_consistent_user'] = 0
            
        if 'weekend_ratio' in df.columns:
            df['is_weekend_user'] = (df['weekend_ratio'] > 0.3).astype(int)
        else:
            df['is_weekend_user'] = 0
            
        if 'business_hours_ratio' in df.columns:
            df['is_business_hours_user'] = (df['business_hours_ratio'] > 0.5).astype(int)
        else:
            df['is_business_hours_user'] = 0
        
        # Interaction features (with safe calculations)
        sessions_val = df.get(session_col, pd.Series([0] * len(df)))
        products_val = df.get('unique_products_viewed', pd.Series([0] * len(df)))
        events_val = df.get(events_col, pd.Series([0] * len(df)))
        active_days_val = df.get(active_days_col, pd.Series([0] * len(df)))
        store_interactions_val = df.get('store_interactions', pd.Series([0] * len(df)))
        
        df['sessions_x_products'] = sessions_val * products_val
        df['events_x_active_days'] = events_val * active_days_val
        df['store_interactions_x_sessions'] = store_interactions_val * sessions_val
    
    # Update feature columns
    feature_cols = list(X_train.columns)
    
    print(f"Final feature set: {len(feature_cols)} features")
    print(f"Target distribution - Train: ${y_train.mean():.2f} ± ${y_train.std():.2f}")
    print(f"Target distribution - Val: ${y_val.mean():.2f} ± ${y_val.std():.2f}")
    print(f"Target distribution - Test: ${y_test.mean():.2f} ± ${y_test.std():.2f}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

def train_ensemble_ltv_models(X_train, X_val, X_test, y_train, y_val, y_test, feature_cols):
    """Train multiple LTV prediction models and ensemble them"""
    
    print("\n" + "="*80)
    print("TRAINING ENSEMBLE LTV PREDICTION MODELS")
    print("="*80)
    
    # Scale features using RobustScaler (less sensitive to outliers)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    predictions = {}
    
    # 1. Random Forest Regressor
    print("\n1. Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_pred = np.maximum(rf_pred, 0)  # Ensure non-negative predictions
    
    models['Random Forest'] = rf_model
    predictions['Random Forest'] = rf_pred
    
    # Feature importance for Random Forest
    rf_importance = {
        'feature_names': feature_cols,
        'importance': rf_model.feature_importances_
    }
    
    # Evaluate Random Forest
    print(f"   Random Forest Results:")
    print(f"   MAE: {mean_absolute_error(y_test, rf_pred):.4f}")
    print(f"   RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred)):.4f}")
    print(f"   R²: {r2_score(y_test, rf_pred):.4f}")
    
    # Collect evaluation data
    collect_evaluation_data(y_test, rf_pred, "Random Forest", rf_importance)
    
    # 2. Gradient Boosting Regressor
    print("\n2. Training Gradient Boosting Regressor...")
    gb_model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.1,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42
    )
    
    gb_model.fit(X_train_scaled, y_train)
    gb_pred = gb_model.predict(X_test_scaled)
    gb_pred = np.maximum(gb_pred, 0)  # Ensure non-negative predictions
    
    models['Gradient Boosting'] = gb_model
    predictions['Gradient Boosting'] = gb_pred
    
    # Feature importance for Gradient Boosting
    gb_importance = {
        'feature_names': feature_cols,
        'importance': gb_model.feature_importances_
    }
    
    # Evaluate Gradient Boosting
    print(f"   Gradient Boosting Results:")
    print(f"   MAE: {mean_absolute_error(y_test, gb_pred):.4f}")
    print(f"   RMSE: {np.sqrt(mean_squared_error(y_test, gb_pred)):.4f}")
    print(f"   R²: {r2_score(y_test, gb_pred):.4f}")
    
    # Collect evaluation data
    collect_evaluation_data(y_test, gb_pred, "Gradient Boosting", gb_importance)
    
    # 4. Ensemble Model (weighted average)
    print("\n3. Creating Ensemble Model...")
    
    # Calculate weights based on validation performance
    val_predictions = {}
    val_scores = {}
    
    for name, model in models.items():
        val_pred = model.predict(X_val_scaled)
        val_pred = np.maximum(val_pred, 0)
        val_predictions[name] = val_pred
        val_scores[name] = r2_score(y_val, val_pred)
    
    # Calculate ensemble weights (higher weight for better R² scores)
    total_score = sum(max(0, score) for score in val_scores.values())
    if total_score > 0:
        weights = {name: max(0, score) / total_score for name, score in val_scores.items()}
    else:
        weights = {name: 1/len(models) for name in models.keys()}
    
    print(f"   Ensemble weights: {weights}")
    
    # Create ensemble prediction
    ensemble_pred = np.zeros(len(y_test))
    for name, weight in weights.items():
        ensemble_pred += weight * predictions[name]
    
    predictions['Ensemble'] = ensemble_pred
    
    # Evaluate Ensemble
    print(f"   Ensemble Results:")
    print(f"   MAE: {mean_absolute_error(y_test, ensemble_pred):.4f}")
    print(f"   RMSE: {np.sqrt(mean_squared_error(y_test, ensemble_pred)):.4f}")
    print(f"   R²: {r2_score(y_test, ensemble_pred):.4f}")
    
    # Collect evaluation data for ensemble
    collect_evaluation_data(y_test, ensemble_pred, "Ensemble", None)
    
    # 5. Two-Stage Model (Classification + Regression)
    print("\n4. Training Two-Stage Model (Classification + Regression)...")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    
    # Stage 1: Binary classification (spender vs non-spender)
    y_binary_train = (y_train > 0).astype(int)
    y_binary_test = (y_test > 0).astype(int)
    
    binary_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    )
    
    binary_classifier.fit(X_train_scaled, y_binary_train)
    binary_pred = binary_classifier.predict(X_test_scaled)
    binary_proba = binary_classifier.predict_proba(X_test_scaled)[:, 1]
    
    print(f"   Binary Classification Results:")
    print(f"   Actual spenders: {np.sum(y_binary_test)}")
    print(f"   Predicted spenders: {np.sum(binary_pred)}")
    
    # Stage 2: Regression for predicted spenders
    spender_mask_train = y_train > 0
    two_stage_pred = np.zeros(len(y_test))
    
    if np.sum(spender_mask_train) > 10:
        X_spenders = X_train_scaled[spender_mask_train]
        y_spenders = y_train[spender_mask_train]
        
        # Use log transformation for better regression on positive values
        y_spenders_log = np.log1p(y_spenders)
        
        regression_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        regression_model.fit(X_spenders, y_spenders_log)
        
        # Predict for all users, but weight by spender probability
        log_predictions = regression_model.predict(X_test_scaled)
        raw_predictions = np.expm1(log_predictions)
        
        # Weight predictions by spender probability
        two_stage_pred = raw_predictions * binary_proba
        two_stage_pred = np.maximum(two_stage_pred, 0)
        
        print(f"   Regression trained on {len(y_spenders)} spenders")
    else:
        print("   Not enough spenders for regression training")
        two_stage_pred = binary_pred * np.mean(y_train[y_train > 0]) if np.sum(y_train > 0) > 0 else binary_pred
    
    predictions['Two-Stage'] = two_stage_pred
    
    # Evaluate Two-Stage
    print(f"   Two-Stage Results:")
    print(f"   MAE: {mean_absolute_error(y_test, two_stage_pred):.4f}")
    print(f"   RMSE: {np.sqrt(mean_squared_error(y_test, two_stage_pred)):.4f}")
    print(f"   R²: {r2_score(y_test, two_stage_pred):.4f}")
    
    # Collect evaluation data
    collect_evaluation_data(y_test, two_stage_pred, "Two-Stage", None)
    
    # Cross-validation analysis
    print("\n" + "="*60)
    print("CROSS-VALIDATION ANALYSIS")
    print("="*60)
    
    cv_scores = {}
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in [('Random Forest', rf_model), ('Gradient Boosting', gb_model)]:
        scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, 
                               scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_scores[name] = -scores.mean()
        print(f"{name} CV MAE: {cv_scores[name]:.4f} (±{scores.std():.4f})")
    
    # Model comparison summary
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    comparison_metrics = {}
    for name, pred in predictions.items():
        comparison_metrics[name] = {
            'MAE': mean_absolute_error(y_test, pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
            'R²': r2_score(y_test, pred),
            'Revenue_Capture': np.sum(pred) / np.sum(y_test) if np.sum(y_test) > 0 else 0,
            'Mean_Prediction': np.mean(pred),
            'Std_Prediction': np.std(pred)
        }
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_metrics).T
    comparison_df = comparison_df.round(4)
    print(comparison_df)
    
    # Business impact analysis
    print("\n" + "="*60)
    print("BUSINESS IMPACT ANALYSIS")
    print("="*60)
    
    actual_total_revenue = np.sum(y_test)
    actual_spenders = np.sum(y_test > 0)
    
    for name, pred in predictions.items():
        predicted_total_revenue = np.sum(pred)
        predicted_spenders = np.sum(pred > 0)
        
        # Top 10% analysis
        top_10_pct_users = int(0.1 * len(pred))
        top_10_pct_idx = np.argsort(pred)[-top_10_pct_users:]
        top_10_pct_actual_revenue = np.sum(y_test[top_10_pct_idx])
        top_10_pct_capture_rate = top_10_pct_actual_revenue / actual_total_revenue if actual_total_revenue > 0 else 0
        
        print(f"\n{name} Model:")
        print(f"  Revenue Capture Rate: {predicted_total_revenue/actual_total_revenue*100:.1f}%")
        print(f"  Predicted Spenders: {predicted_spenders} (Actual: {actual_spenders})")
        print(f"  Top 10% Revenue Capture: {top_10_pct_capture_rate*100:.1f}%")
        print(f"  Average Predicted LTV: ${np.mean(pred):.2f}")
        
        if predicted_spenders > 0:
            efficiency = predicted_total_revenue / predicted_spenders
            print(f"  Revenue per Targeted User: ${efficiency:.2f}")
    
    return models, predictions, comparison_df

def main():
    print("="*80)
    print("LTV PREDICTION PIPELINE")
    print("="*80)
    print("Loading datasets...")
    try:
        train_data = pd.read_csv("train_data.csv")
        val_data = pd.read_csv("val_data.csv") 
        test_data = pd.read_csv("test_data.csv")
        print("Loaded from CSV files")
    except FileNotFoundError:
        print("CSV files not found. Using load_temporal_datasets()...")
        datasets = load_temporal_datasets()
        if datasets[0] is None:
            print("ERROR: Cannot load datasets. Please ensure CSV files exist or BigQuery is configured.")
            return None, None, None, None
        
        train_data, val_data, test_data = datasets
        
        # Save for future use
        train_data.to_csv("train_data.csv", index=False)
        val_data.to_csv("val_data.csv", index=False)
        test_data.to_csv("test_data.csv", index=False)
        print("Saved datasets to CSV files")
    
    print(f"Train: {train_data.shape}")
    print(f"Val: {val_data.shape}")
    print(f"Test: {test_data.shape}")
    
    # Prepare features (ensuring no revenue data leakage)
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_features(
        train_data, val_data, test_data
    )
    
    print(f"\nDataset Summary:")
    print(f"Features: {len(feature_cols)}")
    print(f"Features: {feature_cols}")
    print(f"Train samples: {len(X_train):,}")
    print(f"Val samples: {len(X_val):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Average LTV - Train: ${y_train.mean():.2f}, Test: ${y_test.mean():.2f}")
    print(f"LTV std dev - Train: ${y_train.std():.2f}, Test: ${y_test.std():.2f}")
    print(f"Spender rate - Train: {(y_train > 0).mean()*100:.1f}%, Test: {(y_test > 0).mean()*100:.1f}%")
    
    # Train ensemble models
    models, predictions, comparison_df = train_ensemble_ltv_models(
        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols
    )
    
    # Generate comprehensive evaluation plots
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE EVALUATION PLOTS")
    print("="*60)
    plot_comprehensive_evaluation()
    
    # Final recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR IMPROVEMENT")
    print("="*80)
    
    best_model = comparison_df['R²'].idxmax()
    best_r2 = comparison_df.loc[best_model, 'R²']
    best_mae = comparison_df.loc[best_model, 'MAE']
    
    print(f"Best performing model: {best_model}")
    print(f"Best R² score: {best_r2:.4f}")
    print(f"Best MAE: {best_mae:.4f}")
    
    print(f"\nNext steps to improve LTV prediction:")
    print(f"1. Feature Engineering:")
    print(f"   - Add more temporal features (hour of day patterns, day of week)")
    print(f"   - Create user cohort features (install week, seasonal effects)")
    print(f"   - Add interaction features between behavioral metrics")
    print(f"   - Include sequence-based features (session patterns over time)")
    
    print(f"\n2. Advanced Modeling:")
    print(f"   - Try XGBoost or LightGBM for better gradient boosting")
    print(f"   - Implement neural networks for complex pattern recognition")
    print(f"   - Use time series models for sequential behavior")
    print(f"   - Add stacking ensemble with meta-learner")
    
    print(f"\n3. Data Quality:")
    print(f"   - Increase feature collection period (D0-D7 instead of D0-D3)")
    print(f"   - Add external data sources (device info, app store data)")
    print(f"   - Implement feature importance analysis for feature selection")
    print(f"   - Handle class imbalance with advanced sampling techniques")
    
    print(f"\n4. Evaluation Improvements:")
    print(f"   - Use business-specific metrics (profit optimization)")
    print(f"   - Implement time-based validation (forward chaining)")
    print(f"   - Add confidence intervals for predictions")
    print(f"   - Create segment-specific models (high/low engagement users)")
    
    if best_r2 < 0.3:
        print(f"\nCurrent R² is low ({best_r2:.3f}). Consider:")
        print(f"   - Collecting more behavioral features")
        print(f"   - Extending feature collection period")
        print(f"   - Using external data sources")
        print(f"   - Focusing on user segments with predictable behavior")
    elif best_r2 < 0.5:
        print(f"\nModerate predictive power ({best_r2:.3f}). Improvements possible:")
        print(f"   - Fine-tune hyperparameters")
        print(f"   - Add feature interactions")
        print(f"   - Try advanced ensemble methods")
    else:
        print(f"\nGood predictive power ({best_r2:.3f})! Focus on:")
        print(f"   - Model deployment and monitoring")
        print(f"   - A/B testing with business impact")
        print(f"   - Regular model retraining")
    
    print(f"\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Models trained: {len(models)}")
    print(f"Evaluation plots saved: comprehensive_ltv_evaluation.png")
    print(f"Best model: {best_model} (R² = {best_r2:.4f})")
    
    return models, predictions, comparison_df, (X_train, X_val, X_test, y_train, y_val, y_test)

if __name__ == "__main__":
    models, predictions, comparison_df, data = main()