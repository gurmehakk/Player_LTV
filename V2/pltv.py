import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso, HuberRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Advanced ML models - handle import errors gracefully
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    print("CatBoost not available. Install with: pip install catboost")
    CATBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential, layers, callbacks
    TENSORFLOW_AVAILABLE = True
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
except ImportError:
    print("TensorFlow not available. Install with: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False

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
    """Create comprehensive evaluation plots for good models only (R² >= 0.1) including AUC, ROC, RMSE, residual, actual vs predicted, accuracy, feature importance"""
    if not PLOT_DATA:
        print("No evaluation data collected")
        return
    
    # Filter out models with R² < 0.1
    good_models = [data for data in PLOT_DATA if data['r2'] >= 0.1]
    
    if not good_models:
        print("No models with R² >= 0.1 found")
        return
    
    print(f"Creating plots for {len(good_models)} models with R² >= 0.1:")
    for data in good_models:
        print(f"  - {data['model_name']}: R² = {data['r2']:.4f}")
    
    n_models = len(good_models)
    
    # Create single comprehensive figure with 6 subplots
    fig = plt.figure(figsize=(24, 20))
    colors = plt.cm.Set3(np.linspace(0, 1, n_models))
    
    # 1. ROC Curves (top left)
    ax1 = plt.subplot(3, 2, 1)
    for i, data in enumerate(good_models):
        # Convert regression to binary classification for ROC
        y_true_binary = (data['y_true'] > 0).astype(int)
        y_pred_binary = (data['y_pred'] > 0).astype(int)
        
        if len(np.unique(y_true_binary)) > 1:
            try:
                fpr, tpr, _ = roc_curve(y_true_binary, data['y_pred'])
                auc_score = auc(fpr, tpr)
                ax1.plot(fpr, tpr, color=colors[i], linewidth=2, 
                        label=f"{data['model_name']} (AUC: {auc_score:.3f})")
            except:
                pass
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves - Spender Detection')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curves (top right)
    ax2 = plt.subplot(3, 2, 2)
    for i, data in enumerate(good_models):
        y_true_binary = (data['y_true'] > 0).astype(int)
        
        if len(np.unique(y_true_binary)) > 1:
            try:
                precision, recall, _ = precision_recall_curve(y_true_binary, data['y_pred'])
                pr_auc = auc(recall, precision)
                ax2.plot(recall, precision, color=colors[i], linewidth=2,
                        label=f"{data['model_name']} (AUC: {pr_auc:.3f})")
            except:
                pass
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance Summary (middle left)
    ax3 = plt.subplot(3, 2, 3)
    model_names = [data['model_name'] for data in good_models]
    r2_scores = [data['r2'] for data in good_models]
    mae_scores = [data['mae'] for data in good_models]
    
    x_pos = np.arange(len(model_names))
    
    # Create grouped bar chart
    width = 0.35
    ax3_twin = ax3.twinx()
    
    bars1 = ax3.bar(x_pos - width/2, r2_scores, width, color=colors[:len(good_models)], 
                   alpha=0.8, label='R² Score')
    bars2 = ax3_twin.bar(x_pos + width/2, mae_scores, width, color='red', 
                        alpha=0.6, label='MAE')
    
    ax3.set_xlabel('Models')
    ax3.set_ylabel('R² Score', color='blue')
    ax3_twin.set_ylabel('MAE', color='red')
    ax3.set_title('Performance Summary: R² vs MAE')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars1, r2_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. RMSE Comparison (middle right)
    ax4 = plt.subplot(3, 2, 4)
    rmse_scores = [data['rmse'] for data in good_models]
    bars = ax4.bar(model_names, rmse_scores, color=colors[:len(good_models)], alpha=0.8)
    ax4.set_ylabel('RMSE')
    ax4.set_title('RMSE Comparison')
    ax4.set_xticklabels(model_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, rmse_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{score:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Feature Importance (bottom left) - only for tree-based models
    ax5 = plt.subplot(3, 2, 5)
    feature_importance_data = [data for data in good_models if data['feature_importance'] is not None]
    
    if feature_importance_data:
        # Use the best performing model's feature importance
        best_model = max(feature_importance_data, key=lambda x: x['r2'])
        importance = best_model['feature_importance']['importance']
        feature_names = best_model['feature_importance']['feature_names']
        
        # Get top 15 features
        top_indices = np.argsort(importance)[-15:]
        top_importance = importance[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        bars = ax5.barh(range(15), top_importance, color='skyblue', alpha=0.8)
        ax5.set_yticks(range(15))
        ax5.set_yticklabels(top_names, fontsize=8)
        ax5.set_xlabel('Feature Importance')
        ax5.set_title(f'Top Features - {best_model["model_name"]}')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No feature importance available', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Feature Importance')
    
    # 6. Revenue Capture Analysis (bottom right)
    ax6 = plt.subplot(3, 2, 6)
    capture_rates = [data['top_10_pct_capture'] * 100 for data in good_models]  # Convert to percentage
    revenue_ratios = [data['total_predicted_revenue'] / max(data['total_actual_revenue'], 1) 
                     for data in good_models]
    
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax6.bar(x_pos - width/2, capture_rates, width, color='green', 
                   alpha=0.7, label='Top 10% Capture (%)')
    ax6_twin = ax6.twinx()
    bars2 = ax6_twin.bar(x_pos + width/2, revenue_ratios, width, color='orange', 
                        alpha=0.7, label='Pred/Actual Revenue Ratio')
    
    ax6.set_xlabel('Models')
    ax6.set_ylabel('Top 10% Capture (%)', color='green')
    ax6_twin.set_ylabel('Revenue Ratio', color='orange')
    ax6.set_title('Business Metrics: Revenue Capture')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(model_names, rotation=45, ha='right')
    ax6.grid(True, alpha=0.3)
    
    # Add legends
    ax6.legend(loc='upper left')
    ax6_twin.legend(loc='upper right')
    
    plt.suptitle('LTV Prediction Models - Comprehensive Analysis (R² ≥ 0.1)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('ltv_models_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nComprehensive analysis plot saved as 'ltv_models_comprehensive_analysis.png'")
    print(f"Models included: {', '.join([data['model_name'] for data in good_models])}")
    print(f"Best model: {max(good_models, key=lambda x: x['r2'])['model_name']} (R² = {max(good_models, key=lambda x: x['r2'])['r2']:.4f})")

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
        COUNT(*) / COUNT(DISTINCT e.session_id) AS avg_events_per_session,
        
        -- Advanced ad engagement features (behavioral only, no revenue)
        -- Note: event_name field not available, using alternative engagement metrics
        0 AS ad_interaction_events,
        0 AS ad_click_events,
        0 AS ad_view_events,
        0 AS video_ad_events,
        0 AS store_browse_events,
        
        -- Social and sharing features (engagement indicators)
        -- Note: event_name field not available, using alternative engagement metrics
        0 AS social_share_events,
        0 AS social_invite_events,
        0 AS tutorial_completion_events,
        0 AS game_progress_events,
        0 AS achievement_events,
        
        -- App quality and technical engagement features
        -- Note: event_name field not available, using alternative engagement metrics
        0 AS error_events,
        0 AS notification_events,
        0 AS settings_interaction_events,
        0 AS search_events,
        
        -- Deep engagement features
        1 AS unique_event_types,
        COUNT(DISTINCT EXTRACT(HOUR FROM e.attribution_event_timestamp)) AS active_hours_spread,
        COUNT(DISTINCT EXTRACT(DAYOFWEEK FROM e.attribution_event_timestamp)) AS active_days_of_week,
        
        -- Retention and re-engagement patterns
        DATE_DIFF(MAX(DATE(e.attribution_event_timestamp)), MIN(DATE(e.attribution_event_timestamp)), DAY) AS activity_span_days,
        COUNT(DISTINCT DATE(e.attribution_event_timestamp)) / (DATE_DIFF(MAX(DATE(e.attribution_event_timestamp)), MIN(DATE(e.attribution_event_timestamp)), DAY) + 1) AS activity_consistency_ratio
                
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
      
      -- Advanced ad engagement ratios
      SAFE_DIVIDE(f.ad_interaction_events, f.total_events) AS ad_interaction_ratio,
      SAFE_DIVIDE(f.ad_click_events, NULLIF(f.ad_view_events, 0)) AS ad_click_through_rate,
      SAFE_DIVIDE(f.video_ad_events, NULLIF(f.ad_interaction_events, 0)) AS video_ad_preference_ratio,
      SAFE_DIVIDE(f.store_browse_events, f.total_events) AS store_browse_ratio,
      
      -- Social engagement ratios
      SAFE_DIVIDE(f.social_share_events + f.social_invite_events, f.total_events) AS social_engagement_ratio,
      SAFE_DIVIDE(f.tutorial_completion_events, f.total_sessions) AS tutorial_completion_rate,
      SAFE_DIVIDE(f.game_progress_events, f.total_events) AS game_progress_ratio,
      SAFE_DIVIDE(f.achievement_events, f.total_events) AS achievement_ratio,
      
      -- App quality metrics
      SAFE_DIVIDE(f.error_events, f.total_events) AS error_rate,
      SAFE_DIVIDE(f.notification_events, f.total_events) AS notification_engagement_ratio,
      SAFE_DIVIDE(f.settings_interaction_events, f.total_sessions) AS customization_ratio,
      SAFE_DIVIDE(f.search_events, f.total_events) AS search_behavior_ratio,
      
      -- Deep engagement metrics
      SAFE_DIVIDE(f.unique_event_types, f.total_events) AS event_diversity_ratio,
      SAFE_DIVIDE(f.active_hours_spread, 24) AS time_diversity_ratio,
      SAFE_DIVIDE(f.active_days_of_week, 7) AS weekly_consistency_ratio,
      
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
    
    # Extract targets - handle different possible target column names
    target_col = 'ltv_30_days'
    if target_col not in train_data.columns:
        print("Warning: ltv_30_days column not found, checking for alternatives...")
        possible_targets = ['ltv_30_days', 'ltv_7_days', 'revenue_30d', 'total_revenue']
        for col in possible_targets:
            if col in train_data.columns:
                target_col = col
                print(f"Using target column: {target_col}")
                break
        else:
            raise ValueError("No valid target column found!")
    
    y_train = train_data[target_col].copy()
    y_val = val_data[target_col].copy()
    y_test = test_data[target_col].copy()
    
    # Handle missing values - fill with 0 for behavioral features
    print("\nHandling missing values...")
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)
    
    # Encode categorical variables
    categorical_cols = []
    for col in feature_cols:
        if X_train[col].dtype == 'object':
            categorical_cols.append(col)
    
    if categorical_cols:
        print(f"Encoding {len(categorical_cols)} categorical columns: {categorical_cols}")
        from sklearn.preprocessing import LabelEncoder
        
        for col in categorical_cols:
            le = LabelEncoder()
            
            # Combine all values to ensure consistent encoding
            all_values = pd.concat([X_train[col], X_val[col], X_test[col]]).astype(str)
            le.fit(all_values)
            
            # Transform each dataset
            X_train[col] = le.transform(X_train[col].astype(str))
            X_val[col] = le.transform(X_val[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
    
    print(f"\nFinal dataset shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    print(f"\nTarget statistics:")
    print(f"Train - Mean: ${y_train.mean():.2f}, Std: ${y_train.std():.2f}, Spenders: {(y_train > 0).sum()}")
    print(f"Val - Mean: ${y_val.mean():.2f}, Std: ${y_val.std():.2f}, Spenders: {(y_val > 0).sum()}")
    print(f"Test - Mean: ${y_test.mean():.2f}, Std: ${y_test.std():.2f}, Spenders: {(y_test > 0).sum()}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

# Note: The actual train_ensemble_ltv_models function is defined later in the file

def create_comprehensive_plots():
    """Create comprehensive comparison plots for all models"""
    import os
    os.makedirs('plots', exist_ok=True)
    
    if len(PLOT_DATA) == 0:
        print("No plot data available")
        return
    
    # Model comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    # Collect metrics for all models
    model_names = [data['model_name'] for data in PLOT_DATA]
    rmse_values = []
    r2_values = []
    mae_values = []
    
    for data in PLOT_DATA:
        y_true = data['y_true']
        y_pred = data['y_pred']
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        rmse_values.append(rmse)
        r2_values.append(r2)
        mae_values.append(mae)
    
    # Plot 1: RMSE comparison
    axes[0,0].bar(range(len(model_names)), rmse_values)
    axes[0,0].set_xticks(range(len(model_names)))
    axes[0,0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0,0].set_title('RMSE Comparison')
    axes[0,0].set_ylabel('RMSE ($)')
    
    # Plot 2: R² comparison
    axes[0,1].bar(range(len(model_names)), r2_values)
    axes[0,1].set_xticks(range(len(model_names)))
    axes[0,1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0,1].set_title('R² Comparison')
    axes[0,1].set_ylabel('R² Score')
    
    # Plot 3: Actual vs Predicted for best model
    best_idx = np.argmax(r2_values)
    best_data = PLOT_DATA[best_idx]
    y_true = best_data['y_true']
    y_pred = best_data['y_pred']
    
    axes[1,0].scatter(y_true, y_pred, alpha=0.6)
    max_val = max(y_true.max(), y_pred.max())
    axes[1,0].plot([0, max_val], [0, max_val], 'r--')
    axes[1,0].set_xlabel('Actual LTV ($)')
    axes[1,0].set_ylabel('Predicted LTV ($)')
    axes[1,0].set_title(f'Best Model: {best_data["model_name"]}')
    
    # Plot 4: Feature importance for best model
    if best_data['feature_importance'] is not None:
        importance = best_data['feature_importance']['importance']
        feature_names = best_data['feature_importance']['feature_names']
        top_indices = np.argsort(importance)[-10:]
        
        axes[1,1].barh(range(10), importance[top_indices])
        axes[1,1].set_yticks(range(10))
        axes[1,1].set_yticklabels([feature_names[i] for i in top_indices])
        axes[1,1].set_title(f'Top Features - {best_data["model_name"]}')
        axes[1,1].set_xlabel('Importance')
    else:
        axes[1,1].text(0.5, 0.5, 'No feature importance available', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Feature Importance')
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison_comprehensive.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Additional comparison plot - Error distribution
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for data in PLOT_DATA[:5]:  # Show top 5 models
        y_true = data['y_true']
        y_pred = data['y_pred']
        residuals = y_pred - y_true
        
        ax.hist(residuals, alpha=0.6, bins=30, label=data['model_name'], density=True)
    
    ax.set_xlabel('Prediction Error ($)')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution Comparison (Top 5 Models)')
    ax.legend()
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('plots/error_distribution_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comprehensive comparison plots to 'plots/' directory")

def create_roc_pr_curves():
    """Create ROC and Precision-Recall curves for all models"""
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os

    os.makedirs('plots', exist_ok=True)

    if len(PLOT_DATA) == 0:
        print("No plot data available for ROC/PR curves")
        return

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Colors for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(PLOT_DATA)))

    print("\nCalculating ROC and PR curves for LTV prediction (binary: LTV > 0)...")

    roc_results = []
    pr_results = []

    for i, data in enumerate(PLOT_DATA):
        model_name = data['model_name']
        y_true = data['y_true']
        y_pred = data['y_pred']

        # Convert to binary classification (LTV > 0 vs LTV = 0)
        y_true_binary = (y_true > 0).astype(int)

        # Use predictions as scores (higher prediction = more likely to have LTV > 0)
        y_scores = y_pred

        try:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
            roc_auc = auc(fpr, tpr)

            # Precision-Recall Curve
            precision, recall, thresholds = precision_recall_curve(y_true_binary, y_scores)
            pr_auc = average_precision_score(y_true_binary, y_scores)

            # Interpolate precision at recall = 0.90
            precision_at_90_recall = "N/A"
            try:
                if np.any(recall >= 0.9):
                    recall_reversed = recall[::-1]
                    precision_reversed = precision[::-1]
                    interp_func = interp1d(recall_reversed, precision_reversed, kind='linear', fill_value="extrapolate")
                    precision_val = float(interp_func(0.9))
                    precision_at_90_recall = f"{precision_val:.3f}"
            except Exception as e:
                print(f"    Could not interpolate precision@0.9 recall for {model_name}: {e}")

            # Store results
            roc_results.append({
                'model': model_name,
                'roc_auc': roc_auc,
                'fpr': fpr,
                'tpr': tpr,
                'precision_at_90_recall': precision_at_90_recall
            })

            pr_results.append({
                'model': model_name,
                'pr_auc': pr_auc,
                'precision': precision,
                'recall': recall
            })

            # Plot ROC curve
            ax1.plot(fpr, tpr, color=colors[i], linewidth=2,
                     label=f'{model_name} (AUC = {roc_auc:.3f})')

            # Plot PR curve
            ax2.plot(recall, precision, color=colors[i], linewidth=2,
                     label=f'{model_name} (AP = {pr_auc:.3f})')

            print(f"  {model_name}: ROC-AUC = {roc_auc:.3f}, PR-AUC = {pr_auc:.3f}, Precision@0.9Recall = {precision_at_90_recall}")

        except Exception as e:
            print(f"  Error calculating curves for {model_name}: {e}")
            continue

    # Format ROC plot
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves - LTV Prediction (LTV > 0)')
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Format PR plot
    # Calculate baseline precision from first model's data
    first_y_true = PLOT_DATA[0]['y_true']
    y_true_binary_baseline = (first_y_true > 0).astype(int)
    baseline_precision = np.sum(y_true_binary_baseline) / len(y_true_binary_baseline)
    ax2.axhline(y=baseline_precision, color='k', linestyle='--', linewidth=1,
                label=f'Random (AP = {baseline_precision:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves - LTV Prediction (LTV > 0)')
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/roc_pr_curves_all_models.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Create summary table
    summary_data = []
    for roc_res, pr_res in zip(roc_results, pr_results):
        summary_data.append({
            'Model': roc_res['model'],
            'ROC_AUC': roc_res['roc_auc'],
            'PR_AUC': pr_res['pr_auc'],
            'Precision_at_90_Recall': roc_res['precision_at_90_recall']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('ROC_AUC', ascending=False)

    print(f"\nLTV > 0 Detection Performance Summary:")
    print("=" * 60)
    print(summary_df.round(3))

    # Save summary
    summary_df.to_csv('ltv_detection_performance.csv', index=False)

    print(f"\nSaved ROC/PR curves to 'plots/roc_pr_curves_all_models.png'")
    print(f"Saved LTV detection summary to 'ltv_detection_performance.csv'")

    return summary_df

def create_individual_model_plots():
    """Create individual detailed plots for each model"""
    import os
    os.makedirs('plots/individual_models', exist_ok=True)
    
    for data in PLOT_DATA:
        # Create individual model figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{data["model_name"]} - Detailed Analysis', fontsize=14)
        
        # Simple plots for each model
        y_true = data['y_true']
        y_pred = data['y_pred']
        
        # Plot 1: Actual vs Predicted
        axes[0,0].scatter(y_true, y_pred, alpha=0.6)
        axes[0,0].plot([0, max(y_true.max(), y_pred.max())], [0, max(y_true.max(), y_pred.max())], 'r--')
        axes[0,0].set_xlabel('Actual LTV')
        axes[0,0].set_ylabel('Predicted LTV')
        axes[0,0].set_title('Actual vs Predicted')
        
        # Plot 2: Residuals
        residuals = y_pred - y_true
        axes[0,1].scatter(y_pred, residuals, alpha=0.6)
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        axes[0,1].set_xlabel('Predicted LTV')
        axes[0,1].set_ylabel('Residuals')
        axes[0,1].set_title('Residuals Plot')
        
        # Plot 3: Distribution
        axes[1,0].hist(y_true[y_true > 0], alpha=0.7, label='Actual', bins=20)
        axes[1,0].hist(y_pred[y_pred > 0], alpha=0.7, label='Predicted', bins=20)
        axes[1,0].set_xlabel('LTV')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('LTV Distribution')
        axes[1,0].legend()
        
        # Plot 4: Feature Importance (if available)
        if data['feature_importance'] is not None:
            importance = data['feature_importance']['importance']
            feature_names = data['feature_importance']['feature_names']
            top_indices = np.argsort(importance)[-10:]
            axes[1,1].barh(range(10), importance[top_indices])
            axes[1,1].set_yticks(range(10))
            axes[1,1].set_yticklabels([feature_names[i] for i in top_indices])
            axes[1,1].set_title('Top 10 Features')
        else:
            axes[1,1].text(0.5, 0.5, 'No feature importance available', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Feature Importance')
        
        plt.tight_layout()
        plt.savefig(f'plots/individual_models/{data["model_name"].replace(" ", "_")}_analysis.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved individual model plots for {len(PLOT_DATA)} models")

# Load data and train models functions are defined later in the file

def train_ensemble_ltv_models(X_train, X_val, X_test, y_train, y_val, y_test, feature_cols):
    """Train multiple LTV prediction models and ensemble them"""
    
    print("\n" + "="*80)
    print("TRAINING ENSEMBLE LTV PREDICTION MODELS")
    print("="*80)
    
    # Clear any previous plot data
    global PLOT_DATA
    PLOT_DATA.clear()
    
    # Feature scaling
    from sklearn.preprocessing import RobustScaler
    print("\nScaling features using RobustScaler...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    predictions = {}
    results = []
    
    # Import all necessary libraries
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    import xgboost as xgb
    from catboost import CatBoostRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import joblib
    import os
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Define high-performing models only (removed KNN, SVR, Neural Network due to poor performance)
    model_configs = {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='rmse'
        ),
        'CatBoost': CatBoostRegressor(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=False
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ),
        'AdaBoost': AdaBoostRegressor(
            n_estimators=100,
            learning_rate=1.0,
            random_state=42
        ),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5),
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42)
    }
    
    print(f"\nTraining {len(model_configs)} different models...")
    
    for name, model in model_configs.items():
        print(f"\n--- Training {name} ---")
        try:
            # Use scaled data for models that benefit from it
            use_scaled = name in ['Ridge Regression', 'Lasso Regression', 'Elastic Net', 'Linear Regression']
            
            if use_scaled:
                X_tr, X_v, X_te = X_train_scaled, X_val_scaled, X_test_scaled
            else:
                X_tr, X_v, X_te = X_train, X_val, X_test
            
            # Train model
            model.fit(X_tr, y_train)
            
            # Make predictions
            train_pred = model.predict(X_tr)
            val_pred = model.predict(X_v)
            test_pred = model.predict(X_te)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            train_mae = mean_absolute_error(y_train, train_pred)
            train_r2 = r2_score(y_train, train_pred)
            
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            test_mae = mean_absolute_error(y_test, test_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            # Store results
            results.append({
                'Model': name,
                'Train_RMSE': train_rmse,
                'Train_MAE': train_mae,
                'Train_R2': train_r2,
                'Val_RMSE': val_rmse,
                'Val_MAE': val_mae,
                'Val_R2': val_r2,
                'Test_RMSE': test_rmse,
                'Test_MAE': test_mae,
                'Test_R2': test_r2
            })
            
            # Store model and predictions
            models[name] = model
            predictions[name] = {
                'train': train_pred,
                'val': val_pred,
                'test': test_pred
            }
            
            # Save model
            joblib.dump(model, f'models/{name.replace(" ", "_").lower()}_model.pkl')
            
            # Get feature importance if available
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = {
                    'importance': model.feature_importances_,
                    'feature_names': feature_cols
                }
            elif hasattr(model, 'coef_'):
                feature_importance = {
                    'importance': np.abs(model.coef_),
                    'feature_names': feature_cols
                }
            
            # Store plot data
            PLOT_DATA.append({
                'model_name': name,
                'y_true': y_test,
                'y_pred': test_pred,
                'feature_importance': feature_importance,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'test_r2': test_r2
            })
            
            print(f"   Train RMSE: ${train_rmse:.2f}, Val RMSE: ${val_rmse:.2f}, Test RMSE: ${test_rmse:.2f}")
            print(f"   Train R²: {train_r2:.3f}, Val R²: {val_r2:.3f}, Test R²: {test_r2:.3f}")
            
        except Exception as e:
            print(f"   Error training {name}: {str(e)}")
            continue
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    
    if len(comparison_df) > 0:
        # Sort by validation RMSE
        comparison_df = comparison_df.sort_values('Val_RMSE').reset_index(drop=True)
        
        print(f"\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        print(comparison_df.round(3))
        
        # Save comparison results
        comparison_df.to_csv('model_comparison_results.csv', index=False)
        print(f"\nSaved model comparison results to 'model_comparison_results.csv'")
        
        # Save scaler
        joblib.dump(scaler, 'models/feature_scaler.pkl')
        print(f"Saved feature scaler to 'models/feature_scaler.pkl'")
        
        # Create ensemble prediction (simple average of top 3 models)
        top_3_models = comparison_df.head(3)['Model'].tolist()
        if len(top_3_models) >= 3:
            ensemble_pred_test = np.mean([predictions[model]['test'] for model in top_3_models], axis=0)
            ensemble_pred_val = np.mean([predictions[model]['val'] for model in top_3_models], axis=0)
            ensemble_pred_train = np.mean([predictions[model]['train'] for model in top_3_models], axis=0)
            
            ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred_test))
            ensemble_mae = mean_absolute_error(y_test, ensemble_pred_test)
            ensemble_r2 = r2_score(y_test, ensemble_pred_test)
            
            predictions['Ensemble (Top 3)'] = {
                'test': ensemble_pred_test,
                'val': ensemble_pred_val,
                'train': ensemble_pred_train
            }
            
            # Add ensemble to plot data for ROC/PR curves
            PLOT_DATA.append({
                'model_name': 'Ensemble (Top 3)',
                'y_true': y_test,
                'y_pred': ensemble_pred_test,
                'feature_importance': None,  # No feature importance for ensemble
                'train_rmse': None,
                'val_rmse': None,
                'test_rmse': ensemble_rmse,
                'train_r2': None,
                'val_r2': None,
                'test_r2': ensemble_r2
            })
            
            print(f"\nEnsemble Model (Top 3 average):")
            print(f"   Test RMSE: ${ensemble_rmse:.2f}, Test MAE: ${ensemble_mae:.2f}, Test R²: {ensemble_r2:.3f}")
            print(f"   Models used: {', '.join(top_3_models)}")
    
    return models, predictions, comparison_df

def save_datasets(train_data, val_data, test_data, X_train, X_val, X_test, y_train, y_val, y_test, feature_cols):
    """Save all datasets and processed features"""
    import os
    import joblib
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    print("\nSaving datasets...")
    
    # Save raw datasets
    train_data.to_csv('data/train_data_raw.csv', index=False)
    val_data.to_csv('data/val_data_raw.csv', index=False)
    test_data.to_csv('data/test_data_raw.csv', index=False)
    
    # Save processed features
    pd.DataFrame(X_train, columns=feature_cols).to_csv('data/X_train.csv', index=False)
    pd.DataFrame(X_val, columns=feature_cols).to_csv('data/X_val.csv', index=False)
    pd.DataFrame(X_test, columns=feature_cols).to_csv('data/X_test.csv', index=False)
    
    # Save targets
    pd.DataFrame({'ltv_30_days': y_train}).to_csv('data/y_train.csv', index=False)
    pd.DataFrame({'ltv_30_days': y_val}).to_csv('data/y_val.csv', index=False)
    pd.DataFrame({'ltv_30_days': y_test}).to_csv('data/y_test.csv', index=False)
    
    # Save feature column names
    joblib.dump(feature_cols, 'data/feature_columns.pkl')
    
    # Save dataset summary
    summary = {
        'train_shape': train_data.shape,
        'val_shape': val_data.shape,
        'test_shape': test_data.shape,
        'num_features': len(feature_cols),
        'feature_names': feature_cols,
        'train_ltv_stats': {
            'mean': float(y_train.mean()),
            'std': float(y_train.std()),
            'spenders': int((y_train > 0).sum()),
            'spender_rate': float((y_train > 0).mean())
        },
        'val_ltv_stats': {
            'mean': float(y_val.mean()),
            'std': float(y_val.std()),
            'spenders': int((y_val > 0).sum()),
            'spender_rate': float((y_val > 0).mean())
        },
        'test_ltv_stats': {
            'mean': float(y_test.mean()),
            'std': float(y_test.std()),
            'spenders': int((y_test > 0).sum()),
            'spender_rate': float((y_test > 0).mean())
        }
    }
    
    import json
    with open('data/dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved datasets to 'data/' directory:")
    print(f"  - Raw datasets: train_data_raw.csv, val_data_raw.csv, test_data_raw.csv")
    print(f"  - Features: X_train.csv, X_val.csv, X_test.csv")
    print(f"  - Targets: y_train.csv, y_val.csv, y_test.csv")
    print(f"  - Feature names: feature_columns.pkl")
    print(f"  - Summary: dataset_summary.json")

def load_existing_datasets():
    """Load existing processed datasets if they exist"""
    import os
    import joblib
    
    try:
        if (os.path.exists('data/X_train.csv') and 
            os.path.exists('data/X_val.csv') and 
            os.path.exists('data/X_test.csv') and
            os.path.exists('data/y_train.csv') and
            os.path.exists('data/y_val.csv') and
            os.path.exists('data/y_test.csv') and
            os.path.exists('data/feature_columns.pkl')):
            
            print("Found existing processed datasets. Loading...")
            
            # Load processed features
            X_train = pd.read_csv('data/X_train.csv')
            X_val = pd.read_csv('data/X_val.csv') 
            X_test = pd.read_csv('data/X_test.csv')
            
            # Load targets
            y_train = pd.read_csv('data/y_train.csv')['ltv_30_days'].values
            y_val = pd.read_csv('data/y_val.csv')['ltv_30_days'].values
            y_test = pd.read_csv('data/y_test.csv')['ltv_30_days'].values
            
            # Load feature columns
            feature_cols = joblib.load('data/feature_columns.pkl')
            
            # Load raw data if available
            train_data = None
            val_data = None
            test_data = None
            
            if (os.path.exists('data/train_data_raw.csv') and
                os.path.exists('data/val_data_raw.csv') and
                os.path.exists('data/test_data_raw.csv')):
                train_data = pd.read_csv('data/train_data_raw.csv')
                val_data = pd.read_csv('data/val_data_raw.csv')
                test_data = pd.read_csv('data/test_data_raw.csv')
            
            print(f"Loaded existing datasets:")
            print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
            print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
            print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
            print(f"  Features: {len(feature_cols)}")
            
            return train_data, val_data, test_data, X_train, X_val, X_test, y_train, y_val, y_test, feature_cols
            
    except Exception as e:
        print(f"Error loading existing datasets: {e}")
        print("Will create new datasets...")
    
    return None

def main():
    """Main execution function"""
    print("LTV Prediction Pipeline - Enhanced Version")
    
    # Try to load existing datasets first
    existing_data = load_existing_datasets()
    
    if existing_data is not None:
        train_data, val_data, test_data, X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = existing_data
    else:
        # Load and prepare data from scratch
        print("\nLoading datasets from BigQuery...")
        train_data, val_data, test_data = load_temporal_datasets()
        
        if train_data is None:
            print("No data loaded. Exiting.")
            return None, None, None, None
        
        # Prepare features
        print("Preparing features...")
        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_features(
            train_data, val_data, test_data
        )
        
        # Save datasets
        save_datasets(train_data, val_data, test_data, X_train, X_val, X_test, y_train, y_val, y_test, feature_cols)
    
    # Train models
    print("Training models...")
    models, predictions, comparison_df = train_ensemble_ltv_models(
        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols
    )
    
    # Create comprehensive plots
    if len(PLOT_DATA) > 0:
        print("\nCreating visualizations...")
        create_comprehensive_plots()
        create_roc_pr_curves()
        create_individual_model_plots()
    
    print("Pipeline completed successfully!")
    return models, predictions, comparison_df, test_data

if __name__ == "__main__":
    models, predictions, comparison_df, data = main()
