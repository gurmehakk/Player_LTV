from matplotlib import table
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
    PREDICTION_DAYS = 60           # D4-D33 (30 days)
    
    client = bigquery.Client(project=PROJECT_ID)
    table_path = f"{PROJECT_ID}.{DATASET}.{TABLE}"
    
    train_query = f"""  
        WITH install_cohort AS (
        -- Get install date for Feb-March installs
        SELECT 
            COALESCE(gaid, idfa, android_id, custom_user_id) as user_id,
            DATE(MIN(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64))))) as install_date,
            MIN(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64)))) as install_timestamp
        FROM {table_path}
        WHERE DATE(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64)))) >= '2024-08-01'
            AND DATE(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64)))) <= '2025-01-01'
            AND COALESCE(gaid, idfa, android_id, custom_user_id) IS NOT NULL
            AND COALESCE(gaid, idfa, android_id, custom_user_id) != ''  -- Remove empty user IDs
        GROUP BY COALESCE(gaid, idfa, android_id, custom_user_id)
        ),

        user_demographics AS (
        -- Basic demographics and attribution info
        SELECT 
            ic.user_id,
            ic.install_date,
            ic.install_timestamp,
            
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.country END) as country,
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.city END) as city,
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.state END) as state,
            
            -- Derive platform
            ANY_VALUE(CASE 
            WHEN e.idfa IS NOT NULL OR e.idfa_md5 IS NOT NULL OR e.idfv IS NOT NULL THEN 'ios'
            WHEN e.gaid IS NOT NULL OR e.android_id IS NOT NULL THEN 'android'
            ELSE 'unknown'
            END) as platform,
            
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.os_version END) as os_version,
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.app_version END) as app_version,
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.install_source END) as install_source,
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.campaign_name END) as campaign_name,
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.partner END) as partner,
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.publisher_name END) as publisher_name
            
        FROM install_cohort ic
        LEFT JOIN `{table_path}` e
            ON ic.user_id = COALESCE(e.gaid, e.idfa, e.android_id, e.custom_user_id)
            AND DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date
        GROUP BY ic.user_id, ic.install_date, ic.install_timestamp
        ),

        -- D0-D3 FEATURE WINDOW for training
        feature_window_events AS (
        SELECT 
            ic.user_id,
            COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64))) as event_timestamp,
            e.session_id,
            e.name,
            e.arguments,
            e.product_name,
            e.product_sku,
            e.product_category,
            e.product_price,
            e.product_quantity,
            COALESCE(e.is_fingerprinted, false) as is_fingerprinted,
            COALESCE(e.is_reengagement, false) as is_reengagement,
            COALESCE(e.is_view_through, false) as is_view_through,
            DATE_DIFF(DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))), ic.install_date, DAY) as day_offset
        FROM install_cohort ic
        LEFT JOIN {table_path} e
            ON ic.user_id = COALESCE(e.gaid, e.idfa, e.android_id, e.custom_user_id)
            AND DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) BETWEEN ic.install_date AND DATE_ADD(ic.install_date, INTERVAL {FEATURE_DAYS} DAY)
        WHERE e.name IS NOT NULL
        ),

        -- TARGET: D4-D33 LTV CALCULATION
        ltv_target AS (
        SELECT 
            ic.user_id,
            SUM(COALESCE(e.converted_revenue, 0)) as ltv_target,
            SUM(COALESCE(e.converted_revenue, 0)) as received_ltv_target,
            COUNTIF(COALESCE(e.converted_revenue, 0) > 0) as purchase_events_target
        FROM install_cohort ic
        LEFT JOIN {table_path} e
            ON ic.user_id = COALESCE(e.gaid, e.idfa, e.android_id, e.custom_user_id)
            AND DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) BETWEEN DATE_ADD(ic.install_date, INTERVAL {FEATURE_DAYS + 1} DAY) AND DATE_ADD(ic.install_date, INTERVAL {FEATURE_DAYS + PREDICTION_DAYS} DAY)
        GROUP BY ic.user_id
        ),

        -- FEATURE ENGINEERING (Generic names)
        basic_engagement_metrics AS (
        SELECT 
            user_id,
            COUNT(*) as total_events,
            COUNT(DISTINCT session_id) as total_sessions,
            COUNT(DISTINCT DATE(event_timestamp)) as active_days,
            
            -- Period-based events (generic names)
            COUNTIF(day_offset = 0) as period_1_events,        -- Day 0 events
            COUNTIF(day_offset = 1) as period_2_events,        -- Day 1 events  
            COUNTIF(day_offset = 2) as period_3_events,        -- Day 2 events
            COUNTIF(day_offset = 3) as period_4_events,        -- Day 3 events
            
            -- Period-based sessions (generic names)
            COUNT(DISTINCT CASE WHEN day_offset = 0 THEN session_id END) as period_1_sessions,
            COUNT(DISTINCT CASE WHEN day_offset = 1 THEN session_id END) as period_2_sessions,
            COUNT(DISTINCT CASE WHEN day_offset = 2 THEN session_id END) as period_3_sessions,
            COUNT(DISTINCT CASE WHEN day_offset = 3 THEN session_id END) as period_4_sessions,
            
            -- Retention flags (generic names)
            CASE WHEN COUNT(DISTINCT CASE WHEN day_offset >= 1 THEN DATE(event_timestamp) END) >= 1 THEN 1 ELSE 0 END as next_period_retained,
            CASE WHEN COUNT(DISTINCT CASE WHEN day_offset >= 2 THEN DATE(event_timestamp) END) >= 1 THEN 1 ELSE 0 END as period_plus_2_retained,
            CASE WHEN COUNT(DISTINCT CASE WHEN day_offset >= 3 THEN DATE(event_timestamp) END) >= 1 THEN 1 ELSE 0 END as period_plus_3_retained,
            
            MIN(event_timestamp) as first_event_timestamp,
            MAX(event_timestamp) as last_event_timestamp
        FROM feature_window_events
        WHERE user_id IS NOT NULL
        GROUP BY user_id
        ),

        time_pattern_features AS (
        SELECT 
            user_id,
            COUNTIF(EXTRACT(HOUR FROM event_timestamp) BETWEEN 9 AND 17) as business_hours_events,
            COUNTIF(EXTRACT(HOUR FROM event_timestamp) BETWEEN 18 AND 22) as evening_events,
            COUNTIF(EXTRACT(HOUR FROM event_timestamp) >= 23 OR EXTRACT(HOUR FROM event_timestamp) <= 6) as late_night_events,
            COUNTIF(EXTRACT(DAYOFWEEK FROM event_timestamp) IN (1, 7)) as weekend_events,
            COUNT(DISTINCT EXTRACT(HOUR FROM event_timestamp)) as active_hours_spread,
            COUNT(DISTINCT EXTRACT(DAYOFWEEK FROM event_timestamp)) as active_days_of_week,
            COUNTIF(is_fingerprinted = true) as fingerprinted_events,
            COUNTIF(is_reengagement = true) as reengagement_events,
            COUNTIF(is_view_through = true) as view_through_events
        FROM feature_window_events
        WHERE user_id IS NOT NULL
        GROUP BY user_id
        ),

        game_specific_features AS (
        SELECT 
            user_id,
            COUNTIF(LOWER(name) LIKE '%ftue_completed%' OR LOWER(name) LIKE '%tutorial%complete%') as tutorial_completions,
            CASE WHEN COUNTIF(LOWER(name) LIKE '%ftue_completed%' OR LOWER(name) LIKE '%tutorial%complete%') > 0 THEN 1 ELSE 0 END as tutorial_completed_flag,
            MIN(CASE WHEN LOWER(name) LIKE '%ftue_completed%' OR LOWER(name) LIKE '%tutorial%complete%' THEN event_timestamp END) as tutorial_completion_timestamp,
            COUNTIF(LOWER(name) LIKE '%level_progress%' OR LOWER(name) LIKE '%level%complete%') as levels_completed,
            MAX(CASE 
            WHEN LOWER(name) LIKE '%level_progress%' OR LOWER(name) LIKE '%level%complete%'
            THEN SAFE_CAST(COALESCE(
                JSON_EXTRACT_SCALAR(arguments, '$.content_level_number'),
                JSON_EXTRACT_SCALAR(arguments, '$.level'),
                JSON_EXTRACT_SCALAR(arguments, '$.level_number')
            ) AS INT64) 
            END) as max_level_reached,
            COUNTIF(LOWER(name) LIKE '%ad%' OR LOWER(name) LIKE '%admon%') as total_ads_viewed,
            COUNTIF(
            (LOWER(name) LIKE '%ad%' OR LOWER(name) LIKE '%admon%') AND 
            (LOWER(JSON_EXTRACT_SCALAR(arguments, '$.ad_placement_name')) LIKE '%reward%' OR
            LOWER(JSON_EXTRACT_SCALAR(arguments, '$.ad_type')) LIKE '%reward%' OR
            LOWER(name) LIKE '%reward%')
            ) as rewarded_ads_viewed,
            COUNTIF(
            (LOWER(name) LIKE '%ad%' OR LOWER(name) LIKE '%admon%') AND 
            (LOWER(JSON_EXTRACT_SCALAR(arguments, '$.ad_placement_name')) LIKE '%interstitial%' OR
            LOWER(JSON_EXTRACT_SCALAR(arguments, '$.ad_type')) LIKE '%interstitial%' OR
            LOWER(name) LIKE '%interstitial%')
            ) as interstitial_ads_viewed,
            COUNTIF(LOWER(name) LIKE '%store%' OR LOWER(name) LIKE '%shop%') as store_views,
            COUNTIF(LOWER(name) LIKE '%purchase%intent%' OR LOWER(name) LIKE '%iap%click%') as purchase_intents,
            COUNTIF(LOWER(name) = 'currency_earned') as currency_earned_events,
            COUNTIF(LOWER(name) = 'currency_spent') as currency_spent_events,
            COUNTIF(product_name IS NOT NULL AND COALESCE(product_price, 0) = 0) as product_interactions_non_revenue,
            COUNT(DISTINCT CASE WHEN product_name IS NOT NULL AND COALESCE(product_price, 0) = 0 THEN product_name END) as unique_products_viewed,
            COUNT(DISTINCT CASE WHEN product_sku IS NOT NULL AND COALESCE(product_price, 0) = 0 THEN product_sku END) as unique_skus_viewed,
            COUNTIF(LOWER(name) LIKE '%social%' OR LOWER(name) LIKE '%share%' OR LOWER(name) LIKE '%invite%') as social_events,
            COUNTIF(LOWER(name) LIKE '%achievement%' OR LOWER(name) LIKE '%trophy%' OR LOWER(name) LIKE '%leaderboard%') as achievement_events
        FROM feature_window_events
        WHERE user_id IS NOT NULL
        GROUP BY user_id
        ),

        session_timing_features AS (
        SELECT 
            user_id,
            AVG(session_duration_minutes) as avg_session_length,
            SUM(session_duration_minutes) as total_playtime_mins,
            MAX(session_duration_minutes) as max_session_length,
            MIN(CASE WHEN session_rank = 1 THEN session_start_hour END) as first_session_hour,
            MIN(CASE WHEN session_rank = 1 THEN session_day_of_week END) as first_session_day_of_week
        FROM (
            SELECT 
            user_id,
            session_id,
            MIN(event_timestamp) as session_start,
            MAX(event_timestamp) as session_end,
            EXTRACT(HOUR FROM MIN(event_timestamp)) as session_start_hour,
            EXTRACT(DAYOFWEEK FROM MIN(event_timestamp)) as session_day_of_week,
            TIMESTAMP_DIFF(MAX(event_timestamp), MIN(event_timestamp), MINUTE) as session_duration_minutes,
            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY MIN(event_timestamp)) as session_rank
            FROM feature_window_events
            WHERE session_id IS NOT NULL AND user_id IS NOT NULL
            GROUP BY user_id, session_id
        )
        GROUP BY user_id
        )

        -- FINAL TRAIN DATA OUTPUT
        SELECT 
        ud.user_id,
        ud.install_date,
        ud.install_timestamp,
        
        -- Demographics
        ud.country,
        ud.city,
        ud.state,
        ud.platform,
        ud.os_version,
        ud.app_version,
        ud.install_source,
        ud.campaign_name,
        ud.partner,
        ud.publisher_name,
        
        -- Basic engagement metrics (generic names)
        COALESCE(bem.total_events, 0) as total_events,
        COALESCE(bem.total_sessions, 0) as total_sessions,
        COALESCE(bem.active_days, 0) as active_days,
        COALESCE(bem.period_1_events, 0) as period_1_events,
        COALESCE(bem.period_2_events, 0) as period_2_events,
        COALESCE(bem.period_3_events, 0) as period_3_events,
        COALESCE(bem.period_4_events, 0) as period_4_events,
        COALESCE(bem.period_1_sessions, 0) as period_1_sessions,
        COALESCE(bem.period_2_sessions, 0) as period_2_sessions,
        COALESCE(bem.period_3_sessions, 0) as period_3_sessions,
        COALESCE(bem.period_4_sessions, 0) as period_4_sessions,
        COALESCE(bem.next_period_retained, 0) as next_period_retained,
        COALESCE(bem.period_plus_2_retained, 0) as period_plus_2_retained,
        COALESCE(bem.period_plus_3_retained, 0) as period_plus_3_retained,
        
        -- Time patterns
        COALESCE(tpf.business_hours_events, 0) as business_hours_events,
        COALESCE(tpf.evening_events, 0) as evening_events,
        COALESCE(tpf.late_night_events, 0) as late_night_events,
        COALESCE(tpf.weekend_events, 0) as weekend_events,
        COALESCE(tpf.active_hours_spread, 0) as active_hours_spread,
        COALESCE(tpf.active_days_of_week, 0) as active_days_of_week,
        COALESCE(tpf.fingerprinted_events, 0) as fingerprinted_events,
        COALESCE(tpf.reengagement_events, 0) as reengagement_events,
        COALESCE(tpf.view_through_events, 0) as view_through_events,
        
        -- Game features
        COALESCE(gsf.tutorial_completed_flag, 0) as tutorial_completed_flag,
        gsf.tutorial_completion_timestamp,
        COALESCE(gsf.levels_completed, 0) as levels_completed,
        COALESCE(gsf.max_level_reached, 0) as max_level_reached,
        COALESCE(gsf.total_ads_viewed, 0) as total_ads_viewed,
        COALESCE(gsf.rewarded_ads_viewed, 0) as rewarded_ads_viewed,
        COALESCE(gsf.interstitial_ads_viewed, 0) as interstitial_ads_viewed,
        COALESCE(gsf.store_views, 0) as store_views,
        COALESCE(gsf.purchase_intents, 0) as purchase_intents,
        COALESCE(gsf.currency_earned_events, 0) as currency_earned_events,
        COALESCE(gsf.currency_spent_events, 0) as currency_spent_events,
        COALESCE(gsf.product_interactions_non_revenue, 0) as product_interactions_non_revenue,
        COALESCE(gsf.unique_products_viewed, 0) as unique_products_viewed,
        COALESCE(gsf.unique_skus_viewed, 0) as unique_skus_viewed,
        COALESCE(gsf.social_events, 0) as social_events,
        COALESCE(gsf.achievement_events, 0) as achievement_events,
        
        -- Session timing
        COALESCE(stf.avg_session_length, 0) as avg_session_length,
        COALESCE(stf.total_playtime_mins, 0) as total_playtime_mins,
        COALESCE(stf.max_session_length, 0) as max_session_length,
        stf.first_session_hour,
        stf.first_session_day_of_week,
        
        -- Install timing
        EXTRACT(HOUR FROM ud.install_timestamp) as install_hour,
        EXTRACT(DAYOFWEEK FROM ud.install_timestamp) as install_day_of_week,
        EXTRACT(MONTH FROM ud.install_timestamp) as install_month,
        TIMESTAMP_DIFF(bem.first_event_timestamp, ud.install_timestamp, MINUTE) as minutes_install_to_first_event,
        TIMESTAMP_DIFF(gsf.tutorial_completion_timestamp, ud.install_timestamp, MINUTE) as minutes_install_to_tutorial,
        
        -- Advanced categorical features
        CASE 
            WHEN EXTRACT(HOUR FROM ud.install_timestamp) BETWEEN 0 AND 5 THEN 'late_night_install'
            WHEN EXTRACT(HOUR FROM ud.install_timestamp) BETWEEN 6 AND 11 THEN 'morning_install' 
            WHEN EXTRACT(HOUR FROM ud.install_timestamp) BETWEEN 12 AND 17 THEN 'afternoon_install'
            WHEN EXTRACT(HOUR FROM ud.install_timestamp) BETWEEN 18 AND 23 THEN 'evening_install'
        END as install_time_segment,
        
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 1 THEN 1 ELSE 0 END as install_sunday,
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 2 THEN 1 ELSE 0 END as install_monday,
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 3 THEN 1 ELSE 0 END as install_tuesday,
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 4 THEN 1 ELSE 0 END as install_wednesday,
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 5 THEN 1 ELSE 0 END as install_thursday,
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 6 THEN 1 ELSE 0 END as install_friday,
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 7 THEN 1 ELSE 0 END as install_saturday,
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) IN (1, 7) THEN 1 ELSE 0 END as weekend_install,
        
        -- OS sophistication
        CASE 
            WHEN ud.platform = 'android' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 13 THEN 'android_premium'
            WHEN ud.platform = 'android' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 11 THEN 'android_modern'
            WHEN ud.platform = 'android' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 9 THEN 'android_standard'
            WHEN ud.platform = 'android' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 7 THEN 'android_legacy'
            WHEN ud.platform = 'android' THEN 'android_ancient'
            WHEN ud.platform = 'ios' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 16 THEN 'ios_premium'
            WHEN ud.platform = 'ios' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 14 THEN 'ios_modern'
            WHEN ud.platform = 'ios' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 12 THEN 'ios_standard'
            ELSE 'ios_legacy'
        END as os_sophistication_tier,
        
        -- Country economic tier
        CASE 
            WHEN ud.country IN ('US', 'GB', 'DE', 'CA', 'AU', 'NL', 'CH', 'NO', 'SE', 'DK') THEN 'tier_1_economy'
            WHEN ud.country IN ('JP', 'KR', 'FR', 'IT', 'ES', 'BE', 'AT', 'FI', 'IE', 'NZ') THEN 'tier_1b_economy'
            WHEN ud.country IN ('CN', 'SG', 'HK', 'TW', 'AE', 'QA', 'KW', 'SA', 'BH', 'OM') THEN 'tier_2_economy'
            WHEN ud.country IN ('BR', 'MX', 'AR', 'CL', 'RU', 'TR', 'PL', 'CZ', 'HU', 'GR') THEN 'tier_3_economy'
            WHEN ud.country IN ('IN', 'ID', 'TH', 'MY', 'PH', 'VN', 'ZA', 'EG', 'CO', 'PE') THEN 'tier_4_economy'
            ELSE 'tier_5_economy'
        END as economic_tier,
        
        -- Device tier
        CASE 
            WHEN ud.platform = 'ios' THEN 'premium_platform'
            WHEN ud.platform = 'android' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 12 THEN 'premium_android'
            WHEN ud.platform = 'android' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 9 THEN 'mid_tier_android'
            ELSE 'budget_android'
        END as device_tier,
        
        -- Ad behavior
        CASE 
            WHEN COALESCE(gsf.rewarded_ads_viewed, 0) >= 5 AND COALESCE(gsf.interstitial_ads_viewed, 0) <= 2 THEN 'reward_seeker'
            WHEN COALESCE(gsf.interstitial_ads_viewed, 0) >= 3 AND COALESCE(gsf.rewarded_ads_viewed, 0) <= 1 THEN 'ad_tolerant'
            WHEN COALESCE(gsf.total_ads_viewed, 0) >= 8 THEN 'high_ad_engagement'
            WHEN COALESCE(gsf.total_ads_viewed, 0) >= 3 THEN 'moderate_ad_engagement'
            WHEN COALESCE(gsf.total_ads_viewed, 0) >= 1 THEN 'low_ad_engagement'
            ELSE 'ad_avoider'
        END as ad_behavior_profile,
        
        -- Tutorial completion speed
        CASE 
            WHEN COALESCE(gsf.tutorial_completed_flag, 0) = 1 AND TIMESTAMP_DIFF(gsf.tutorial_completion_timestamp, ud.install_timestamp, MINUTE) <= 5 THEN 'instant_tutorial'
            WHEN COALESCE(gsf.tutorial_completed_flag, 0) = 1 AND TIMESTAMP_DIFF(gsf.tutorial_completion_timestamp, ud.install_timestamp, MINUTE) <= 15 THEN 'fast_tutorial'
            WHEN COALESCE(gsf.tutorial_completed_flag, 0) = 1 AND TIMESTAMP_DIFF(gsf.tutorial_completion_timestamp, ud.install_timestamp, MINUTE) <= 60 THEN 'slow_tutorial'
            WHEN COALESCE(gsf.tutorial_completed_flag, 0) = 1 THEN 'very_slow_tutorial'
            ELSE 'no_tutorial'
        END as tutorial_completion_speed,
        
        -- Progression tier
        CASE 
            WHEN COALESCE(gsf.levels_completed, 0) >= 10 THEN 'fast_progressor'
            WHEN COALESCE(gsf.levels_completed, 0) >= 5 THEN 'moderate_progressor'
            WHEN COALESCE(gsf.levels_completed, 0) >= 2 THEN 'slow_progressor'
            WHEN COALESCE(gsf.levels_completed, 0) >= 1 THEN 'minimal_progressor'
            ELSE 'non_progressor'
        END as progression_tier,
        
        -- Calculated ratios (generic names)
        SAFE_DIVIDE(COALESCE(bem.total_events, 0), NULLIF(COALESCE(bem.total_sessions, 0), 0)) as avg_events_per_session,
        SAFE_DIVIDE(COALESCE(bem.total_events, 0), NULLIF(COALESCE(bem.active_days, 0), 0)) as avg_events_per_active_day,
        SAFE_DIVIDE(COALESCE(bem.total_sessions, 0), NULLIF(COALESCE(bem.active_days, 0), 0)) as avg_sessions_per_day,
        SAFE_DIVIDE(COALESCE(tpf.business_hours_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as business_hours_ratio,
        SAFE_DIVIDE(COALESCE(tpf.evening_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as evening_activity_ratio,
        SAFE_DIVIDE(COALESCE(tpf.weekend_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as weekend_activity_ratio,
        SAFE_DIVIDE(COALESCE(tpf.late_night_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as late_night_ratio,
        SAFE_DIVIDE(COALESCE(gsf.rewarded_ads_viewed, 0), NULLIF(COALESCE(gsf.total_ads_viewed, 0), 0)) as rewarded_ad_ratio,
        SAFE_DIVIDE(COALESCE(gsf.total_ads_viewed, 0), NULLIF(COALESCE(bem.total_sessions, 0), 0)) as ads_per_session,
        SAFE_DIVIDE(COALESCE(gsf.total_ads_viewed, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as ad_engagement_rate,
        SAFE_DIVIDE(COALESCE(gsf.product_interactions_non_revenue, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as product_interest_ratio,
        SAFE_DIVIDE(COALESCE(gsf.store_views, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as store_engagement_ratio,
        SAFE_DIVIDE(COALESCE(gsf.purchase_intents, 0), NULLIF(COALESCE(gsf.store_views, 0), 0)) as store_conversion_intent_rate,
        SAFE_DIVIDE(COALESCE(gsf.levels_completed, 0), NULLIF(COALESCE(bem.active_days, 0), 0)) as levels_per_day,
        SAFE_DIVIDE(COALESCE(gsf.levels_completed, 0), NULLIF(COALESCE(bem.period_1_sessions, 0), 0)) as levels_per_session,
        
        -- Period growth ratios (generic names)
        SAFE_DIVIDE(COALESCE(bem.period_2_events, 0), NULLIF(COALESCE(bem.period_1_events, 0), 0)) as period_2_to_1_growth_ratio,
        SAFE_DIVIDE(COALESCE(bem.period_4_events, 0), NULLIF(COALESCE(bem.period_1_events, 0), 0)) as period_4_to_1_growth_ratio,
        SAFE_DIVIDE(COALESCE(bem.period_2_sessions, 0), NULLIF(COALESCE(bem.period_1_sessions, 0), 0)) as session_growth_period_1_to_2,
        
        SAFE_DIVIDE(COALESCE(bem.active_days, 0), 4) as activity_consistency_ratio,
        SAFE_DIVIDE(COALESCE(tpf.active_hours_spread, 0), 24) as time_diversity_ratio,
        SAFE_DIVIDE(COALESCE(tpf.active_days_of_week, 0), 7) as weekly_consistency_ratio,
        SAFE_DIVIDE(COALESCE(tpf.fingerprinted_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as fingerprinted_ratio,
        SAFE_DIVIDE(COALESCE(tpf.reengagement_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as reengagement_ratio,
        SAFE_DIVIDE(COALESCE(tpf.view_through_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as view_through_ratio,
        
        -- Platform × Country combo
        CONCAT(
            CASE WHEN ud.platform = 'ios' THEN 'ios' ELSE 'android' END,
            '_',
            CASE 
            WHEN ud.country IN ('US', 'GB', 'DE', 'CA', 'AU') THEN 'premium_geo'
            WHEN ud.country IN ('JP', 'KR', 'FR', 'IT', 'ES') THEN 'good_geo'
            WHEN ud.country IN ('CN', 'BR', 'RU', 'IN', 'MX') THEN 'medium_geo'
            ELSE 'other_geo'
            END
        ) as platform_geo_combo,
        
        -- Monetization potential
        CASE 
            WHEN ud.country IN ('US', 'GB', 'DE', 'CA', 'AU', 'NL', 'CH', 'NO', 'SE', 'DK') 
                AND ud.platform = 'ios' 
                AND COALESCE(gsf.tutorial_completed_flag, 0) = 1 
                AND COALESCE(gsf.total_ads_viewed, 0) >= 3 THEN 'high_monetization_potential'
            WHEN ud.country IN ('US', 'GB', 'DE', 'CA', 'AU', 'JP', 'KR', 'FR', 'IT', 'ES') 
                AND COALESCE(gsf.tutorial_completed_flag, 0) = 1 
                AND COALESCE(bem.period_1_sessions, 0) >= 3 THEN 'medium_monetization_potential'
            WHEN COALESCE(gsf.tutorial_completed_flag, 0) = 1 OR COALESCE(bem.period_1_sessions, 0) >= 2 THEN 'low_monetization_potential'
            ELSE 'minimal_monetization_potential'
        END as monetization_potential_tier,
        
        -- ==========================================
        -- TARGET VARIABLES (D4-D33 LTV)
        -- ==========================================
        COALESCE(lt.ltv_target, 0) as ltv_target,
        COALESCE(lt.received_ltv_target, 0) as received_ltv_target,
        COALESCE(lt.purchase_events_target, 0) as purchase_events_target

        FROM user_demographics ud
        LEFT JOIN basic_engagement_metrics bem ON ud.user_id = bem.user_id
        LEFT JOIN time_pattern_features tpf ON ud.user_id = tpf.user_id
        LEFT JOIN game_specific_features gsf ON ud.user_id = gsf.user_id
        LEFT JOIN session_timing_features stf ON ud.user_id = stf.user_id
        LEFT JOIN ltv_target lt ON ud.user_id = lt.user_id
        WHERE COALESCE(bem.total_events, 0) >= 1  -- At least some activity
        ORDER BY ud.install_date, ud.user_id;

    """
    val_query = f"""
        WITH install_cohort AS (
        -- Get install date for April installs
        SELECT 
            COALESCE(gaid, idfa, android_id, custom_user_id) as user_id,
            DATE(MIN(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64))))) as install_date,
            MIN(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64)))) as install_timestamp
        FROM {table_path}
        WHERE DATE(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64)))) >= '2025-01-01'
            AND DATE(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64)))) <= '2025-02-28'
            AND COALESCE(gaid, idfa, android_id, custom_user_id) IS NOT NULL
            AND COALESCE(gaid, idfa, android_id, custom_user_id) != ''  -- Remove empty user IDs
        GROUP BY COALESCE(gaid, idfa, android_id, custom_user_id)
        ),

        user_demographics AS (
        -- Basic demographics and attribution info
        SELECT 
            ic.user_id,
            ic.install_date,
            ic.install_timestamp,
            
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.country END) as country,
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.city END) as city,
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.state END) as state,
            
            -- Derive platform
            ANY_VALUE(CASE 
            WHEN e.idfa IS NOT NULL OR e.idfa_md5 IS NOT NULL OR e.idfv IS NOT NULL THEN 'ios'
            WHEN e.gaid IS NOT NULL OR e.android_id IS NOT NULL THEN 'android'
            ELSE 'unknown'
            END) as platform,
            
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.os_version END) as os_version,
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.app_version END) as app_version,
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.install_source END) as install_source,
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.campaign_name END) as campaign_name,
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.partner END) as partner,
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.publisher_name END) as publisher_name
            
        FROM install_cohort ic
        LEFT JOIN {table_path} e
            ON ic.user_id = COALESCE(e.gaid, e.idfa, e.android_id, e.custom_user_id)
            AND DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date
        GROUP BY ic.user_id, ic.install_date, ic.install_timestamp
        ),

        -- D4-D33 FEATURE WINDOW for test data (maps to generic period names)
        feature_window_events AS (
        SELECT 
            ic.user_id,
            COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64))) as event_timestamp,
            e.session_id,
            e.name,
            e.arguments,
            e.product_name,
            e.product_sku,
            e.product_category,
            e.product_price,
            e.product_quantity,
            COALESCE(e.is_fingerprinted, false) as is_fingerprinted,
            COALESCE(e.is_reengagement, false) as is_reengagement,
            COALESCE(e.is_view_through, false) as is_view_through,
            DATE_DIFF(DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))), ic.install_date, DAY) as day_offset
        FROM install_cohort ic
        LEFT JOIN {table_path} e
            ON ic.user_id = COALESCE(e.gaid, e.idfa, e.android_id, e.custom_user_id)
            AND DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) BETWEEN DATE_ADD(ic.install_date, INTERVAL {FEATURE_DAYS + 1} DAY) AND DATE_ADD(ic.install_date, INTERVAL {FEATURE_DAYS + PREDICTION_DAYS} DAY)
        WHERE e.name IS NOT NULL
        ),

        -- TARGET: D4-D33 LTV CALCULATION (same window as features for test data)
        ltv_target AS (
        SELECT 
            ic.user_id,
            SUM(COALESCE(e.converted_revenue, 0)) as ltv_target,
            SUM(COALESCE(e.converted_revenue, 0)) as received_ltv_target,
            COUNTIF(COALESCE(e.converted_revenue, 0) > 0) as purchase_events_target
        FROM install_cohort ic
        LEFT JOIN {table_path} e
            ON ic.user_id = COALESCE(e.gaid, e.idfa, e.android_id, e.custom_user_id)
            AND DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) BETWEEN DATE_ADD(ic.install_date, INTERVAL {FEATURE_DAYS + 1} DAY) AND DATE_ADD(ic.install_date, INTERVAL {FEATURE_DAYS + PREDICTION_DAYS} DAY)
        GROUP BY ic.user_id
        ),

        -- FEATURE ENGINEERING (D4-D33 mapped to generic period names)
        basic_engagement_metrics AS (
        SELECT 
            user_id,
            COUNT(*) as total_events,
            COUNT(DISTINCT session_id) as total_sessions,
            COUNT(DISTINCT DATE(event_timestamp)) as active_days,
            
            -- Map D4-D33 to generic period names (same as train data structure)
            COUNTIF(day_offset BETWEEN 4 AND 10) as period_1_events,      -- Days 4-10 → Period 1
            COUNTIF(day_offset BETWEEN 11 AND 17) as period_2_events,     -- Days 11-17 → Period 2
            COUNTIF(day_offset BETWEEN 18 AND 25) as period_3_events,     -- Days 18-25 → Period 3
            COUNTIF(day_offset BETWEEN 26 AND 33) as period_4_events,     -- Days 26-33 → Period 4
            
            -- Session breakdown (same generic names)
            COUNT(DISTINCT CASE WHEN day_offset BETWEEN 4 AND 10 THEN session_id END) as period_1_sessions,
            COUNT(DISTINCT CASE WHEN day_offset BETWEEN 11 AND 17 THEN session_id END) as period_2_sessions,
            COUNT(DISTINCT CASE WHEN day_offset BETWEEN 18 AND 25 THEN session_id END) as period_3_sessions,
            COUNT(DISTINCT CASE WHEN day_offset BETWEEN 26 AND 33 THEN session_id END) as period_4_sessions,
            
            -- Retention flags (same generic names)
            CASE WHEN COUNT(DISTINCT CASE WHEN day_offset BETWEEN 11 AND 33 THEN DATE(event_timestamp) END) >= 1 THEN 1 ELSE 0 END as next_period_retained,
            CASE WHEN COUNT(DISTINCT CASE WHEN day_offset BETWEEN 18 AND 33 THEN DATE(event_timestamp) END) >= 1 THEN 1 ELSE 0 END as period_plus_2_retained,
            CASE WHEN COUNT(DISTINCT CASE WHEN day_offset BETWEEN 26 AND 33 THEN DATE(event_timestamp) END) >= 1 THEN 1 ELSE 0 END as period_plus_3_retained,
            
            MIN(event_timestamp) as first_event_timestamp,
            MAX(event_timestamp) as last_event_timestamp
        FROM feature_window_events
        WHERE user_id IS NOT NULL
        GROUP BY user_id
        ),

        time_pattern_features AS (
        SELECT 
            user_id,
            COUNTIF(EXTRACT(HOUR FROM event_timestamp) BETWEEN 9 AND 17) as business_hours_events,
            COUNTIF(EXTRACT(HOUR FROM event_timestamp) BETWEEN 18 AND 22) as evening_events,
            COUNTIF(EXTRACT(HOUR FROM event_timestamp) >= 23 OR EXTRACT(HOUR FROM event_timestamp) <= 6) as late_night_events,
            COUNTIF(EXTRACT(DAYOFWEEK FROM event_timestamp) IN (1, 7)) as weekend_events,
            COUNT(DISTINCT EXTRACT(HOUR FROM event_timestamp)) as active_hours_spread,
            COUNT(DISTINCT EXTRACT(DAYOFWEEK FROM event_timestamp)) as active_days_of_week,
            COUNTIF(is_fingerprinted = true) as fingerprinted_events,
            COUNTIF(is_reengagement = true) as reengagement_events,
            COUNTIF(is_view_through = true) as view_through_events
        FROM feature_window_events
        WHERE user_id IS NOT NULL
        GROUP BY user_id
        ),

        game_specific_features AS (
        SELECT 
            user_id,
            COUNTIF(LOWER(name) LIKE '%ftue_completed%' OR LOWER(name) LIKE '%tutorial%complete%') as tutorial_completions,
            CASE WHEN COUNTIF(LOWER(name) LIKE '%ftue_completed%' OR LOWER(name) LIKE '%tutorial%complete%') > 0 THEN 1 ELSE 0 END as tutorial_completed_flag,
            MIN(CASE WHEN LOWER(name) LIKE '%ftue_completed%' OR LOWER(name) LIKE '%tutorial%complete%' THEN event_timestamp END) as tutorial_completion_timestamp,
            COUNTIF(LOWER(name) LIKE '%level_progress%' OR LOWER(name) LIKE '%level%complete%') as levels_completed,
            MAX(CASE 
            WHEN LOWER(name) LIKE '%level_progress%' OR LOWER(name) LIKE '%level%complete%'
            THEN SAFE_CAST(COALESCE(
                JSON_EXTRACT_SCALAR(arguments, '$.content_level_number'),
                JSON_EXTRACT_SCALAR(arguments, '$.level'),
                JSON_EXTRACT_SCALAR(arguments, '$.level_number')
            ) AS INT64) 
            END) as max_level_reached,
            COUNTIF(LOWER(name) LIKE '%ad%' OR LOWER(name) LIKE '%admon%') as total_ads_viewed,
            COUNTIF(
            (LOWER(name) LIKE '%ad%' OR LOWER(name) LIKE '%admon%') AND 
            (LOWER(JSON_EXTRACT_SCALAR(arguments, '$.ad_placement_name')) LIKE '%reward%' OR
            LOWER(JSON_EXTRACT_SCALAR(arguments, '$.ad_type')) LIKE '%reward%' OR
            LOWER(name) LIKE '%reward%')
            ) as rewarded_ads_viewed,
            COUNTIF(
            (LOWER(name) LIKE '%ad%' OR LOWER(name) LIKE '%admon%') AND 
            (LOWER(JSON_EXTRACT_SCALAR(arguments, '$.ad_placement_name')) LIKE '%interstitial%' OR
            LOWER(JSON_EXTRACT_SCALAR(arguments, '$.ad_type')) LIKE '%interstitial%' OR
            LOWER(name) LIKE '%interstitial%')
            ) as interstitial_ads_viewed,
            COUNTIF(LOWER(name) LIKE '%store%' OR LOWER(name) LIKE '%shop%') as store_views,
            COUNTIF(LOWER(name) LIKE '%purchase%intent%' OR LOWER(name) LIKE '%iap%click%') as purchase_intents,
            COUNTIF(LOWER(name) = 'currency_earned') as currency_earned_events,
            COUNTIF(LOWER(name) = 'currency_spent') as currency_spent_events,
            COUNTIF(product_name IS NOT NULL AND COALESCE(product_price, 0) = 0) as product_interactions_non_revenue,
            COUNT(DISTINCT CASE WHEN product_name IS NOT NULL AND COALESCE(product_price, 0) = 0 THEN product_name END) as unique_products_viewed,
            COUNT(DISTINCT CASE WHEN product_sku IS NOT NULL AND COALESCE(product_price, 0) = 0 THEN product_sku END) as unique_skus_viewed,
            COUNTIF(LOWER(name) LIKE '%social%' OR LOWER(name) LIKE '%share%' OR LOWER(name) LIKE '%invite%') as social_events,
            COUNTIF(LOWER(name) LIKE '%achievement%' OR LOWER(name) LIKE '%trophy%' OR LOWER(name) LIKE '%leaderboard%') as achievement_events
        FROM feature_window_events
        WHERE user_id IS NOT NULL
        GROUP BY user_id
        ),

        session_timing_features AS (
        SELECT 
            user_id,
            AVG(session_duration_minutes) as avg_session_length,
            SUM(session_duration_minutes) as total_playtime_mins,
            MAX(session_duration_minutes) as max_session_length,
            MIN(CASE WHEN session_rank = 1 THEN session_start_hour END) as first_session_hour,
            MIN(CASE WHEN session_rank = 1 THEN session_day_of_week END) as first_session_day_of_week
        FROM (
            SELECT 
            user_id,
            session_id,
            MIN(event_timestamp) as session_start,
            MAX(event_timestamp) as session_end,
            EXTRACT(HOUR FROM MIN(event_timestamp)) as session_start_hour,
            EXTRACT(DAYOFWEEK FROM MIN(event_timestamp)) as session_day_of_week,
            TIMESTAMP_DIFF(MAX(event_timestamp), MIN(event_timestamp), MINUTE) as session_duration_minutes,
            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY MIN(event_timestamp)) as session_rank
            FROM feature_window_events
            WHERE session_id IS NOT NULL AND user_id IS NOT NULL
            GROUP BY user_id, session_id
        )
        GROUP BY user_id
        )

        -- FINAL TEST DATA OUTPUT - IDENTICAL STRUCTURE TO TRAIN DATA
        SELECT 
        ud.user_id,
        ud.install_date,
        ud.install_timestamp,
        
        -- Demographics
        ud.country,
        ud.city,
        ud.state,
        ud.platform,
        ud.os_version,
        ud.app_version,
        ud.install_source,
        ud.campaign_name,
        ud.partner,
        ud.publisher_name,
        
        -- Basic engagement metrics (SAME COLUMN NAMES as train data)
        COALESCE(bem.total_events, 0) as total_events,
        COALESCE(bem.total_sessions, 0) as total_sessions,
        COALESCE(bem.active_days, 0) as active_days,
        COALESCE(bem.period_1_events, 0) as period_1_events,
        COALESCE(bem.period_2_events, 0) as period_2_events,
        COALESCE(bem.period_3_events, 0) as period_3_events,
        COALESCE(bem.period_4_events, 0) as period_4_events,
        COALESCE(bem.period_1_sessions, 0) as period_1_sessions,
        COALESCE(bem.period_2_sessions, 0) as period_2_sessions,
        COALESCE(bem.period_3_sessions, 0) as period_3_sessions,
        COALESCE(bem.period_4_sessions, 0) as period_4_sessions,
        COALESCE(bem.next_period_retained, 0) as next_period_retained,
        COALESCE(bem.period_plus_2_retained, 0) as period_plus_2_retained,
        COALESCE(bem.period_plus_3_retained, 0) as period_plus_3_retained,
        
        -- Time patterns
        COALESCE(tpf.business_hours_events, 0) as business_hours_events,
        COALESCE(tpf.evening_events, 0) as evening_events,
        COALESCE(tpf.late_night_events, 0) as late_night_events,
        COALESCE(tpf.weekend_events, 0) as weekend_events,
        COALESCE(tpf.active_hours_spread, 0) as active_hours_spread,
        COALESCE(tpf.active_days_of_week, 0) as active_days_of_week,
        COALESCE(tpf.fingerprinted_events, 0) as fingerprinted_events,
        COALESCE(tpf.reengagement_events, 0) as reengagement_events,
        COALESCE(tpf.view_through_events, 0) as view_through_events,
        
        -- Game features
        COALESCE(gsf.tutorial_completed_flag, 0) as tutorial_completed_flag,
        gsf.tutorial_completion_timestamp,
        COALESCE(gsf.levels_completed, 0) as levels_completed,
        COALESCE(gsf.max_level_reached, 0) as max_level_reached,
        COALESCE(gsf.total_ads_viewed, 0) as total_ads_viewed,
        COALESCE(gsf.rewarded_ads_viewed, 0) as rewarded_ads_viewed,
        COALESCE(gsf.interstitial_ads_viewed, 0) as interstitial_ads_viewed,
        COALESCE(gsf.store_views, 0) as store_views,
        COALESCE(gsf.purchase_intents, 0) as purchase_intents,
        COALESCE(gsf.currency_earned_events, 0) as currency_earned_events,
        COALESCE(gsf.currency_spent_events, 0) as currency_spent_events,
        COALESCE(gsf.product_interactions_non_revenue, 0) as product_interactions_non_revenue,
        COALESCE(gsf.unique_products_viewed, 0) as unique_products_viewed,
        COALESCE(gsf.unique_skus_viewed, 0) as unique_skus_viewed,
        COALESCE(gsf.social_events, 0) as social_events,
        COALESCE(gsf.achievement_events, 0) as achievement_events,
        
        -- Session timing
        COALESCE(stf.avg_session_length, 0) as avg_session_length,
        COALESCE(stf.total_playtime_mins, 0) as total_playtime_mins,
        COALESCE(stf.max_session_length, 0) as max_session_length,
        stf.first_session_hour,
        stf.first_session_day_of_week,
        
        -- Install timing
        EXTRACT(HOUR FROM ud.install_timestamp) as install_hour,
        EXTRACT(DAYOFWEEK FROM ud.install_timestamp) as install_day_of_week,
        EXTRACT(MONTH FROM ud.install_timestamp) as install_month,
        TIMESTAMP_DIFF(bem.first_event_timestamp, ud.install_timestamp, MINUTE) as minutes_install_to_first_event,
        TIMESTAMP_DIFF(gsf.tutorial_completion_timestamp, ud.install_timestamp, MINUTE) as minutes_install_to_tutorial,
        
        -- Advanced categorical features (IDENTICAL to train data)
        CASE 
            WHEN EXTRACT(HOUR FROM ud.install_timestamp) BETWEEN 0 AND 5 THEN 'late_night_install'
            WHEN EXTRACT(HOUR FROM ud.install_timestamp) BETWEEN 6 AND 11 THEN 'morning_install' 
            WHEN EXTRACT(HOUR FROM ud.install_timestamp) BETWEEN 12 AND 17 THEN 'afternoon_install'
            WHEN EXTRACT(HOUR FROM ud.install_timestamp) BETWEEN 18 AND 23 THEN 'evening_install'
        END as install_time_segment,
        
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 1 THEN 1 ELSE 0 END as install_sunday,
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 2 THEN 1 ELSE 0 END as install_monday,
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 3 THEN 1 ELSE 0 END as install_tuesday,
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 4 THEN 1 ELSE 0 END as install_wednesday,
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 5 THEN 1 ELSE 0 END as install_thursday,
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 6 THEN 1 ELSE 0 END as install_friday,
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 7 THEN 1 ELSE 0 END as install_saturday,
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) IN (1, 7) THEN 1 ELSE 0 END as weekend_install,
        
        -- OS sophistication (IDENTICAL logic)
        CASE 
            WHEN ud.platform = 'android' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 11 THEN 'android_modern'
            WHEN ud.platform = 'android' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 9 THEN 'android_standard'
            WHEN ud.platform = 'android' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 7 THEN 'android_legacy'
            WHEN ud.platform = 'android' THEN 'android_ancient'
            WHEN ud.platform = 'ios' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 16 THEN 'ios_premium'
            WHEN ud.platform = 'ios' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 14 THEN 'ios_modern'
            WHEN ud.platform = 'ios' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 12 THEN 'ios_standard'
            ELSE 'ios_legacy'
        END as os_sophistication_tier,
        
        -- Country economic tier (IDENTICAL logic)
        CASE 
            WHEN ud.country IN ('US', 'GB', 'DE', 'CA', 'AU', 'NL', 'CH', 'NO', 'SE', 'DK') THEN 'tier_1_economy'
            WHEN ud.country IN ('JP', 'KR', 'FR', 'IT', 'ES', 'BE', 'AT', 'FI', 'IE', 'NZ') THEN 'tier_1b_economy'
            WHEN ud.country IN ('CN', 'SG', 'HK', 'TW', 'AE', 'QA', 'KW', 'SA', 'BH', 'OM') THEN 'tier_2_economy'
            WHEN ud.country IN ('BR', 'MX', 'AR', 'CL', 'RU', 'TR', 'PL', 'CZ', 'HU', 'GR') THEN 'tier_3_economy'
            WHEN ud.country IN ('IN', 'ID', 'TH', 'MY', 'PH', 'VN', 'ZA', 'EG', 'CO', 'PE') THEN 'tier_4_economy'
            ELSE 'tier_5_economy'
        END as economic_tier,
        
        -- Device tier (IDENTICAL logic)
        CASE 
            WHEN ud.platform = 'ios' THEN 'premium_platform'
            WHEN ud.platform = 'android' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 12 THEN 'premium_android'
            WHEN ud.platform = 'android' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 9 THEN 'mid_tier_android'
            ELSE 'budget_android'
        END as device_tier,
        
        -- Ad behavior (IDENTICAL logic)
        CASE 
            WHEN COALESCE(gsf.rewarded_ads_viewed, 0) >= 5 AND COALESCE(gsf.interstitial_ads_viewed, 0) <= 2 THEN 'reward_seeker'
            WHEN COALESCE(gsf.interstitial_ads_viewed, 0) >= 3 AND COALESCE(gsf.rewarded_ads_viewed, 0) <= 1 THEN 'ad_tolerant'
            WHEN COALESCE(gsf.total_ads_viewed, 0) >= 8 THEN 'high_ad_engagement'
            WHEN COALESCE(gsf.total_ads_viewed, 0) >= 3 THEN 'moderate_ad_engagement'
            WHEN COALESCE(gsf.total_ads_viewed, 0) >= 1 THEN 'low_ad_engagement'
            ELSE 'ad_avoider'
        END as ad_behavior_profile,
        
        -- Tutorial completion speed (IDENTICAL logic)
        CASE 
            WHEN COALESCE(gsf.tutorial_completed_flag, 0) = 1 AND TIMESTAMP_DIFF(gsf.tutorial_completion_timestamp, ud.install_timestamp, MINUTE) <= 5 THEN 'instant_tutorial'
            WHEN COALESCE(gsf.tutorial_completed_flag, 0) = 1 AND TIMESTAMP_DIFF(gsf.tutorial_completion_timestamp, ud.install_timestamp, MINUTE) <= 15 THEN 'fast_tutorial'
            WHEN COALESCE(gsf.tutorial_completed_flag, 0) = 1 AND TIMESTAMP_DIFF(gsf.tutorial_completion_timestamp, ud.install_timestamp, MINUTE) <= 60 THEN 'slow_tutorial'
            WHEN COALESCE(gsf.tutorial_completed_flag, 0) = 1 THEN 'very_slow_tutorial'
            ELSE 'no_tutorial'
        END as tutorial_completion_speed,
        
        -- Progression tier (IDENTICAL logic)
        CASE 
            WHEN COALESCE(gsf.levels_completed, 0) >= 10 THEN 'fast_progressor'
            WHEN COALESCE(gsf.levels_completed, 0) >= 5 THEN 'moderate_progressor'
            WHEN COALESCE(gsf.levels_completed, 0) >= 2 THEN 'slow_progressor'
            WHEN COALESCE(gsf.levels_completed, 0) >= 1 THEN 'minimal_progressor'
            ELSE 'non_progressor'
        END as progression_tier,
        
        -- Calculated ratios (IDENTICAL column names and logic)
        SAFE_DIVIDE(COALESCE(bem.total_events, 0), NULLIF(COALESCE(bem.total_sessions, 0), 0)) as avg_events_per_session,
        SAFE_DIVIDE(COALESCE(bem.total_events, 0), NULLIF(COALESCE(bem.active_days, 0), 0)) as avg_events_per_active_day,
        SAFE_DIVIDE(COALESCE(bem.total_sessions, 0), NULLIF(COALESCE(bem.active_days, 0), 0)) as avg_sessions_per_day,
        SAFE_DIVIDE(COALESCE(tpf.business_hours_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as business_hours_ratio,
        SAFE_DIVIDE(COALESCE(tpf.evening_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as evening_activity_ratio,
        SAFE_DIVIDE(COALESCE(tpf.weekend_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as weekend_activity_ratio,
        SAFE_DIVIDE(COALESCE(tpf.late_night_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as late_night_ratio,
        SAFE_DIVIDE(COALESCE(gsf.rewarded_ads_viewed, 0), NULLIF(COALESCE(gsf.total_ads_viewed, 0), 0)) as rewarded_ad_ratio,
        SAFE_DIVIDE(COALESCE(gsf.total_ads_viewed, 0), NULLIF(COALESCE(bem.total_sessions, 0), 0)) as ads_per_session,
        SAFE_DIVIDE(COALESCE(gsf.total_ads_viewed, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as ad_engagement_rate,
        SAFE_DIVIDE(COALESCE(gsf.product_interactions_non_revenue, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as product_interest_ratio,
        SAFE_DIVIDE(COALESCE(gsf.store_views, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as store_engagement_ratio,
        SAFE_DIVIDE(COALESCE(gsf.purchase_intents, 0), NULLIF(COALESCE(gsf.store_views, 0), 0)) as store_conversion_intent_rate,
        SAFE_DIVIDE(COALESCE(gsf.levels_completed, 0), NULLIF(COALESCE(bem.active_days, 0), 0)) as levels_per_day,
        SAFE_DIVIDE(COALESCE(gsf.levels_completed, 0), NULLIF(COALESCE(bem.period_1_sessions, 0), 0)) as levels_per_session,
        
        -- Period growth ratios (IDENTICAL column names)
        SAFE_DIVIDE(COALESCE(bem.period_2_events, 0), NULLIF(COALESCE(bem.period_1_events, 0), 0)) as period_2_to_1_growth_ratio,
        SAFE_DIVIDE(COALESCE(bem.period_4_events, 0), NULLIF(COALESCE(bem.period_1_events, 0), 0)) as period_4_to_1_growth_ratio,
        SAFE_DIVIDE(COALESCE(bem.period_2_sessions, 0), NULLIF(COALESCE(bem.period_1_sessions, 0), 0)) as session_growth_period_1_to_2,
        
        SAFE_DIVIDE(COALESCE(bem.active_days, 0), 4) as activity_consistency_ratio,
        SAFE_DIVIDE(COALESCE(tpf.active_hours_spread, 0), 24) as time_diversity_ratio,
        SAFE_DIVIDE(COALESCE(tpf.active_days_of_week, 0), 7) as weekly_consistency_ratio,
        SAFE_DIVIDE(COALESCE(tpf.fingerprinted_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as fingerprinted_ratio,
        SAFE_DIVIDE(COALESCE(tpf.reengagement_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as reengagement_ratio,
        SAFE_DIVIDE(COALESCE(tpf.view_through_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as view_through_ratio,
        
        -- Platform × Country combo (IDENTICAL logic)
        CONCAT(
            CASE WHEN ud.platform = 'ios' THEN 'ios' ELSE 'android' END,
            '_',
            CASE 
            WHEN ud.country IN ('US', 'GB', 'DE', 'CA', 'AU') THEN 'premium_geo'
            WHEN ud.country IN ('JP', 'KR', 'FR', 'IT', 'ES') THEN 'good_geo'
            WHEN ud.country IN ('CN', 'BR', 'RU', 'IN', 'MX') THEN 'medium_geo'
            ELSE 'other_geo'
            END
        ) as platform_geo_combo,
        
        -- Monetization potential (IDENTICAL logic)
        CASE 
            WHEN ud.country IN ('US', 'GB', 'DE', 'CA', 'AU', 'NL', 'CH', 'NO', 'SE', 'DK') 
                AND ud.platform = 'ios' 
                AND COALESCE(gsf.tutorial_completed_flag, 0) = 1 
                AND COALESCE(gsf.total_ads_viewed, 0) >= 3 THEN 'high_monetization_potential'
            WHEN ud.country IN ('US', 'GB', 'DE', 'CA', 'AU', 'JP', 'KR', 'FR', 'IT', 'ES') 
                AND COALESCE(gsf.tutorial_completed_flag, 0) = 1 
                AND COALESCE(bem.period_1_sessions, 0) >= 3 THEN 'medium_monetization_potential'
            WHEN COALESCE(gsf.tutorial_completed_flag, 0) = 1 OR COALESCE(bem.period_1_sessions, 0) >= 2 THEN 'low_monetization_potential'
            ELSE 'minimal_monetization_potential'
        END as monetization_potential_tier,
        
        -- ==========================================
        -- TARGET VARIABLES (IDENTICAL column names)
        -- ==========================================
        COALESCE(lt.ltv_target, 0) as ltv_target,
        COALESCE(lt.received_ltv_target, 0) as received_ltv_target,
        COALESCE(lt.purchase_events_target, 0) as purchase_events_target

        FROM user_demographics ud
        LEFT JOIN basic_engagement_metrics bem ON ud.user_id = bem.user_id
        LEFT JOIN time_pattern_features tpf ON ud.user_id = tpf.user_id
        LEFT JOIN game_specific_features gsf ON ud.user_id = gsf.user_id
        LEFT JOIN session_timing_features stf ON ud.user_id = stf.user_id
        LEFT JOIN ltv_target lt ON ud.user_id = lt.user_id
        WHERE COALESCE(bem.total_events, 0) >= 1  -- At least some activity in D4-D33 window
        ORDER BY ud.install_date, ud.user_id;
    """

    test_query = f"""
        WITH install_cohort AS (
        -- Get install date for April installs
        SELECT 
            COALESCE(gaid, idfa, android_id, custom_user_id) as user_id,
            DATE(MIN(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64))))) as install_date,
            MIN(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64)))) as install_timestamp
        FROM {table_path}
        WHERE DATE(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64)))) >= '2025-03-01'
            AND DATE(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64)))) <= '2025-07-30'
            AND COALESCE(gaid, idfa, android_id, custom_user_id) IS NOT NULL
            AND COALESCE(gaid, idfa, android_id, custom_user_id) != ''  -- Remove empty user IDs
        GROUP BY COALESCE(gaid, idfa, android_id, custom_user_id)
        ),

        user_demographics AS (
        -- Basic demographics and attribution info
        SELECT 
            ic.user_id,
            ic.install_date,
            ic.install_timestamp,
            
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.country END) as country,
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.city END) as city,
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.state END) as state,
            
            -- Derive platform
            ANY_VALUE(CASE 
            WHEN e.idfa IS NOT NULL OR e.idfa_md5 IS NOT NULL OR e.idfv IS NOT NULL THEN 'ios'
            WHEN e.gaid IS NOT NULL OR e.android_id IS NOT NULL THEN 'android'
            ELSE 'unknown'
            END) as platform,
            
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.os_version END) as os_version,
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.app_version END) as app_version,
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.install_source END) as install_source,
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.campaign_name END) as campaign_name,
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.partner END) as partner,
            ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.publisher_name END) as publisher_name
            
        FROM install_cohort ic
        LEFT JOIN {table_path} e
            ON ic.user_id = COALESCE(e.gaid, e.idfa, e.android_id, e.custom_user_id)
            AND DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date
        GROUP BY ic.user_id, ic.install_date, ic.install_timestamp
        ),

        -- D4-D33 FEATURE WINDOW for test data (maps to generic period names)
        feature_window_events AS (
        SELECT 
            ic.user_id,
            COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64))) as event_timestamp,
            e.session_id,
            e.name,
            e.arguments,
            e.product_name,
            e.product_sku,
            e.product_category,
            e.product_price,
            e.product_quantity,
            COALESCE(e.is_fingerprinted, false) as is_fingerprinted,
            COALESCE(e.is_reengagement, false) as is_reengagement,
            COALESCE(e.is_view_through, false) as is_view_through,
            DATE_DIFF(DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))), ic.install_date, DAY) as day_offset
        FROM install_cohort ic
        LEFT JOIN {table_path} e
            ON ic.user_id = COALESCE(e.gaid, e.idfa, e.android_id, e.custom_user_id)
            AND DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) BETWEEN DATE_ADD(ic.install_date, INTERVAL {FEATURE_DAYS + 1} DAY) AND DATE_ADD(ic.install_date, INTERVAL {FEATURE_DAYS + PREDICTION_DAYS} DAY)
        WHERE e.name IS NOT NULL
        ),

        -- TARGET: D4-D33 LTV CALCULATION (same window as features for test data)
        ltv_target AS (
        SELECT 
            ic.user_id,
            SUM(COALESCE(e.converted_revenue, 0)) as ltv_target,
            SUM(COALESCE(e.converted_revenue, 0)) as received_ltv_target,
            COUNTIF(COALESCE(e.converted_revenue, 0) > 0) as purchase_events_target
        FROM install_cohort ic
        LEFT JOIN {table_path} e
            ON ic.user_id = COALESCE(e.gaid, e.idfa, e.android_id, e.custom_user_id)
            AND DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) BETWEEN DATE_ADD(ic.install_date, INTERVAL {FEATURE_DAYS + 1} DAY) AND DATE_ADD(ic.install_date, INTERVAL {FEATURE_DAYS + PREDICTION_DAYS} DAY)
        GROUP BY ic.user_id
        ),

        -- FEATURE ENGINEERING (D4-D33 mapped to generic period names)
        basic_engagement_metrics AS (
        SELECT 
            user_id,
            COUNT(*) as total_events,
            COUNT(DISTINCT session_id) as total_sessions,
            COUNT(DISTINCT DATE(event_timestamp)) as active_days,
            
            -- Map D4-D33 to generic period names (same as train data structure)
            COUNTIF(day_offset BETWEEN 4 AND 10) as period_1_events,      -- Days 4-10 → Period 1
            COUNTIF(day_offset BETWEEN 11 AND 17) as period_2_events,     -- Days 11-17 → Period 2
            COUNTIF(day_offset BETWEEN 18 AND 25) as period_3_events,     -- Days 18-25 → Period 3
            COUNTIF(day_offset BETWEEN 26 AND 33) as period_4_events,     -- Days 26-33 → Period 4
            
            -- Session breakdown (same generic names)
            COUNT(DISTINCT CASE WHEN day_offset BETWEEN 4 AND 10 THEN session_id END) as period_1_sessions,
            COUNT(DISTINCT CASE WHEN day_offset BETWEEN 11 AND 17 THEN session_id END) as period_2_sessions,
            COUNT(DISTINCT CASE WHEN day_offset BETWEEN 18 AND 25 THEN session_id END) as period_3_sessions,
            COUNT(DISTINCT CASE WHEN day_offset BETWEEN 26 AND 33 THEN session_id END) as period_4_sessions,
            
            -- Retention flags (same generic names)
            CASE WHEN COUNT(DISTINCT CASE WHEN day_offset BETWEEN 11 AND 33 THEN DATE(event_timestamp) END) >= 1 THEN 1 ELSE 0 END as next_period_retained,
            CASE WHEN COUNT(DISTINCT CASE WHEN day_offset BETWEEN 18 AND 33 THEN DATE(event_timestamp) END) >= 1 THEN 1 ELSE 0 END as period_plus_2_retained,
            CASE WHEN COUNT(DISTINCT CASE WHEN day_offset BETWEEN 26 AND 33 THEN DATE(event_timestamp) END) >= 1 THEN 1 ELSE 0 END as period_plus_3_retained,
            
            MIN(event_timestamp) as first_event_timestamp,
            MAX(event_timestamp) as last_event_timestamp
        FROM feature_window_events
        WHERE user_id IS NOT NULL
        GROUP BY user_id
        ),

        time_pattern_features AS (
        SELECT 
            user_id,
            COUNTIF(EXTRACT(HOUR FROM event_timestamp) BETWEEN 9 AND 17) as business_hours_events,
            COUNTIF(EXTRACT(HOUR FROM event_timestamp) BETWEEN 18 AND 22) as evening_events,
            COUNTIF(EXTRACT(HOUR FROM event_timestamp) >= 23 OR EXTRACT(HOUR FROM event_timestamp) <= 6) as late_night_events,
            COUNTIF(EXTRACT(DAYOFWEEK FROM event_timestamp) IN (1, 7)) as weekend_events,
            COUNT(DISTINCT EXTRACT(HOUR FROM event_timestamp)) as active_hours_spread,
            COUNT(DISTINCT EXTRACT(DAYOFWEEK FROM event_timestamp)) as active_days_of_week,
            COUNTIF(is_fingerprinted = true) as fingerprinted_events,
            COUNTIF(is_reengagement = true) as reengagement_events,
            COUNTIF(is_view_through = true) as view_through_events
        FROM feature_window_events
        WHERE user_id IS NOT NULL
        GROUP BY user_id
        ),

        game_specific_features AS (
        SELECT 
            user_id,
            COUNTIF(LOWER(name) LIKE '%ftue_completed%' OR LOWER(name) LIKE '%tutorial%complete%') as tutorial_completions,
            CASE WHEN COUNTIF(LOWER(name) LIKE '%ftue_completed%' OR LOWER(name) LIKE '%tutorial%complete%') > 0 THEN 1 ELSE 0 END as tutorial_completed_flag,
            MIN(CASE WHEN LOWER(name) LIKE '%ftue_completed%' OR LOWER(name) LIKE '%tutorial%complete%' THEN event_timestamp END) as tutorial_completion_timestamp,
            COUNTIF(LOWER(name) LIKE '%level_progress%' OR LOWER(name) LIKE '%level%complete%') as levels_completed,
            MAX(CASE 
            WHEN LOWER(name) LIKE '%level_progress%' OR LOWER(name) LIKE '%level%complete%'
            THEN SAFE_CAST(COALESCE(
                JSON_EXTRACT_SCALAR(arguments, '$.content_level_number'),
                JSON_EXTRACT_SCALAR(arguments, '$.level'),
                JSON_EXTRACT_SCALAR(arguments, '$.level_number')
            ) AS INT64) 
            END) as max_level_reached,
            COUNTIF(LOWER(name) LIKE '%ad%' OR LOWER(name) LIKE '%admon%') as total_ads_viewed,
            COUNTIF(
            (LOWER(name) LIKE '%ad%' OR LOWER(name) LIKE '%admon%') AND 
            (LOWER(JSON_EXTRACT_SCALAR(arguments, '$.ad_placement_name')) LIKE '%reward%' OR
            LOWER(JSON_EXTRACT_SCALAR(arguments, '$.ad_type')) LIKE '%reward%' OR
            LOWER(name) LIKE '%reward%')
            ) as rewarded_ads_viewed,
            COUNTIF(
            (LOWER(name) LIKE '%ad%' OR LOWER(name) LIKE '%admon%') AND 
            (LOWER(JSON_EXTRACT_SCALAR(arguments, '$.ad_placement_name')) LIKE '%interstitial%' OR
            LOWER(JSON_EXTRACT_SCALAR(arguments, '$.ad_type')) LIKE '%interstitial%' OR
            LOWER(name) LIKE '%interstitial%')
            ) as interstitial_ads_viewed,
            COUNTIF(LOWER(name) LIKE '%store%' OR LOWER(name) LIKE '%shop%') as store_views,
            COUNTIF(LOWER(name) LIKE '%purchase%intent%' OR LOWER(name) LIKE '%iap%click%') as purchase_intents,
            COUNTIF(LOWER(name) = 'currency_earned') as currency_earned_events,
            COUNTIF(LOWER(name) = 'currency_spent') as currency_spent_events,
            COUNTIF(product_name IS NOT NULL AND COALESCE(product_price, 0) = 0) as product_interactions_non_revenue,
            COUNT(DISTINCT CASE WHEN product_name IS NOT NULL AND COALESCE(product_price, 0) = 0 THEN product_name END) as unique_products_viewed,
            COUNT(DISTINCT CASE WHEN product_sku IS NOT NULL AND COALESCE(product_price, 0) = 0 THEN product_sku END) as unique_skus_viewed,
            COUNTIF(LOWER(name) LIKE '%social%' OR LOWER(name) LIKE '%share%' OR LOWER(name) LIKE '%invite%') as social_events,
            COUNTIF(LOWER(name) LIKE '%achievement%' OR LOWER(name) LIKE '%trophy%' OR LOWER(name) LIKE '%leaderboard%') as achievement_events
        FROM feature_window_events
        WHERE user_id IS NOT NULL
        GROUP BY user_id
        ),

        session_timing_features AS (
        SELECT 
            user_id,
            AVG(session_duration_minutes) as avg_session_length,
            SUM(session_duration_minutes) as total_playtime_mins,
            MAX(session_duration_minutes) as max_session_length,
            MIN(CASE WHEN session_rank = 1 THEN session_start_hour END) as first_session_hour,
            MIN(CASE WHEN session_rank = 1 THEN session_day_of_week END) as first_session_day_of_week
        FROM (
            SELECT 
            user_id,
            session_id,
            MIN(event_timestamp) as session_start,
            MAX(event_timestamp) as session_end,
            EXTRACT(HOUR FROM MIN(event_timestamp)) as session_start_hour,
            EXTRACT(DAYOFWEEK FROM MIN(event_timestamp)) as session_day_of_week,
            TIMESTAMP_DIFF(MAX(event_timestamp), MIN(event_timestamp), MINUTE) as session_duration_minutes,
            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY MIN(event_timestamp)) as session_rank
            FROM feature_window_events
            WHERE session_id IS NOT NULL AND user_id IS NOT NULL
            GROUP BY user_id, session_id
        )
        GROUP BY user_id
        )

        -- FINAL TEST DATA OUTPUT - IDENTICAL STRUCTURE TO TRAIN DATA
        SELECT 
        ud.user_id,
        ud.install_date,
        ud.install_timestamp,
        
        -- Demographics
        ud.country,
        ud.city,
        ud.state,
        ud.platform,
        ud.os_version,
        ud.app_version,
        ud.install_source,
        ud.campaign_name,
        ud.partner,
        ud.publisher_name,
        
        -- Basic engagement metrics (SAME COLUMN NAMES as train data)
        COALESCE(bem.total_events, 0) as total_events,
        COALESCE(bem.total_sessions, 0) as total_sessions,
        COALESCE(bem.active_days, 0) as active_days,
        COALESCE(bem.period_1_events, 0) as period_1_events,
        COALESCE(bem.period_2_events, 0) as period_2_events,
        COALESCE(bem.period_3_events, 0) as period_3_events,
        COALESCE(bem.period_4_events, 0) as period_4_events,
        COALESCE(bem.period_1_sessions, 0) as period_1_sessions,
        COALESCE(bem.period_2_sessions, 0) as period_2_sessions,
        COALESCE(bem.period_3_sessions, 0) as period_3_sessions,
        COALESCE(bem.period_4_sessions, 0) as period_4_sessions,
        COALESCE(bem.next_period_retained, 0) as next_period_retained,
        COALESCE(bem.period_plus_2_retained, 0) as period_plus_2_retained,
        COALESCE(bem.period_plus_3_retained, 0) as period_plus_3_retained,
        
        -- Time patterns
        COALESCE(tpf.business_hours_events, 0) as business_hours_events,
        COALESCE(tpf.evening_events, 0) as evening_events,
        COALESCE(tpf.late_night_events, 0) as late_night_events,
        COALESCE(tpf.weekend_events, 0) as weekend_events,
        COALESCE(tpf.active_hours_spread, 0) as active_hours_spread,
        COALESCE(tpf.active_days_of_week, 0) as active_days_of_week,
        COALESCE(tpf.fingerprinted_events, 0) as fingerprinted_events,
        COALESCE(tpf.reengagement_events, 0) as reengagement_events,
        COALESCE(tpf.view_through_events, 0) as view_through_events,
        
        -- Game features
        COALESCE(gsf.tutorial_completed_flag, 0) as tutorial_completed_flag,
        gsf.tutorial_completion_timestamp,
        COALESCE(gsf.levels_completed, 0) as levels_completed,
        COALESCE(gsf.max_level_reached, 0) as max_level_reached,
        COALESCE(gsf.total_ads_viewed, 0) as total_ads_viewed,
        COALESCE(gsf.rewarded_ads_viewed, 0) as rewarded_ads_viewed,
        COALESCE(gsf.interstitial_ads_viewed, 0) as interstitial_ads_viewed,
        COALESCE(gsf.store_views, 0) as store_views,
        COALESCE(gsf.purchase_intents, 0) as purchase_intents,
        COALESCE(gsf.currency_earned_events, 0) as currency_earned_events,
        COALESCE(gsf.currency_spent_events, 0) as currency_spent_events,
        COALESCE(gsf.product_interactions_non_revenue, 0) as product_interactions_non_revenue,
        COALESCE(gsf.unique_products_viewed, 0) as unique_products_viewed,
        COALESCE(gsf.unique_skus_viewed, 0) as unique_skus_viewed,
        COALESCE(gsf.social_events, 0) as social_events,
        COALESCE(gsf.achievement_events, 0) as achievement_events,
        
        -- Session timing
        COALESCE(stf.avg_session_length, 0) as avg_session_length,
        COALESCE(stf.total_playtime_mins, 0) as total_playtime_mins,
        COALESCE(stf.max_session_length, 0) as max_session_length,
        stf.first_session_hour,
        stf.first_session_day_of_week,
        
        -- Install timing
        EXTRACT(HOUR FROM ud.install_timestamp) as install_hour,
        EXTRACT(DAYOFWEEK FROM ud.install_timestamp) as install_day_of_week,
        EXTRACT(MONTH FROM ud.install_timestamp) as install_month,
        TIMESTAMP_DIFF(bem.first_event_timestamp, ud.install_timestamp, MINUTE) as minutes_install_to_first_event,
        TIMESTAMP_DIFF(gsf.tutorial_completion_timestamp, ud.install_timestamp, MINUTE) as minutes_install_to_tutorial,
        
        -- Advanced categorical features (IDENTICAL to train data)
        CASE 
            WHEN EXTRACT(HOUR FROM ud.install_timestamp) BETWEEN 0 AND 5 THEN 'late_night_install'
            WHEN EXTRACT(HOUR FROM ud.install_timestamp) BETWEEN 6 AND 11 THEN 'morning_install' 
            WHEN EXTRACT(HOUR FROM ud.install_timestamp) BETWEEN 12 AND 17 THEN 'afternoon_install'
            WHEN EXTRACT(HOUR FROM ud.install_timestamp) BETWEEN 18 AND 23 THEN 'evening_install'
        END as install_time_segment,
        
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 1 THEN 1 ELSE 0 END as install_sunday,
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 2 THEN 1 ELSE 0 END as install_monday,
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 3 THEN 1 ELSE 0 END as install_tuesday,
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 4 THEN 1 ELSE 0 END as install_wednesday,
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 5 THEN 1 ELSE 0 END as install_thursday,
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 6 THEN 1 ELSE 0 END as install_friday,
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 7 THEN 1 ELSE 0 END as install_saturday,
        CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) IN (1, 7) THEN 1 ELSE 0 END as weekend_install,
        
        -- OS sophistication (IDENTICAL logic)
        CASE 
            WHEN ud.platform = 'android' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 11 THEN 'android_modern'
            WHEN ud.platform = 'android' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 9 THEN 'android_standard'
            WHEN ud.platform = 'android' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 7 THEN 'android_legacy'
            WHEN ud.platform = 'android' THEN 'android_ancient'
            WHEN ud.platform = 'ios' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 16 THEN 'ios_premium'
            WHEN ud.platform = 'ios' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 14 THEN 'ios_modern'
            WHEN ud.platform = 'ios' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 12 THEN 'ios_standard'
            ELSE 'ios_legacy'
        END as os_sophistication_tier,
        
        -- Country economic tier (IDENTICAL logic)
        CASE 
            WHEN ud.country IN ('US', 'GB', 'DE', 'CA', 'AU', 'NL', 'CH', 'NO', 'SE', 'DK') THEN 'tier_1_economy'
            WHEN ud.country IN ('JP', 'KR', 'FR', 'IT', 'ES', 'BE', 'AT', 'FI', 'IE', 'NZ') THEN 'tier_1b_economy'
            WHEN ud.country IN ('CN', 'SG', 'HK', 'TW', 'AE', 'QA', 'KW', 'SA', 'BH', 'OM') THEN 'tier_2_economy'
            WHEN ud.country IN ('BR', 'MX', 'AR', 'CL', 'RU', 'TR', 'PL', 'CZ', 'HU', 'GR') THEN 'tier_3_economy'
            WHEN ud.country IN ('IN', 'ID', 'TH', 'MY', 'PH', 'VN', 'ZA', 'EG', 'CO', 'PE') THEN 'tier_4_economy'
            ELSE 'tier_5_economy'
        END as economic_tier,
        
        -- Device tier (IDENTICAL logic)
        CASE 
            WHEN ud.platform = 'ios' THEN 'premium_platform'
            WHEN ud.platform = 'android' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 12 THEN 'premium_android'
            WHEN ud.platform = 'android' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 9 THEN 'mid_tier_android'
            ELSE 'budget_android'
        END as device_tier,
        
        -- Ad behavior (IDENTICAL logic)
        CASE 
            WHEN COALESCE(gsf.rewarded_ads_viewed, 0) >= 5 AND COALESCE(gsf.interstitial_ads_viewed, 0) <= 2 THEN 'reward_seeker'
            WHEN COALESCE(gsf.interstitial_ads_viewed, 0) >= 3 AND COALESCE(gsf.rewarded_ads_viewed, 0) <= 1 THEN 'ad_tolerant'
            WHEN COALESCE(gsf.total_ads_viewed, 0) >= 8 THEN 'high_ad_engagement'
            WHEN COALESCE(gsf.total_ads_viewed, 0) >= 3 THEN 'moderate_ad_engagement'
            WHEN COALESCE(gsf.total_ads_viewed, 0) >= 1 THEN 'low_ad_engagement'
            ELSE 'ad_avoider'
        END as ad_behavior_profile,
        
        -- Tutorial completion speed (IDENTICAL logic)
        CASE 
            WHEN COALESCE(gsf.tutorial_completed_flag, 0) = 1 AND TIMESTAMP_DIFF(gsf.tutorial_completion_timestamp, ud.install_timestamp, MINUTE) <= 5 THEN 'instant_tutorial'
            WHEN COALESCE(gsf.tutorial_completed_flag, 0) = 1 AND TIMESTAMP_DIFF(gsf.tutorial_completion_timestamp, ud.install_timestamp, MINUTE) <= 15 THEN 'fast_tutorial'
            WHEN COALESCE(gsf.tutorial_completed_flag, 0) = 1 AND TIMESTAMP_DIFF(gsf.tutorial_completion_timestamp, ud.install_timestamp, MINUTE) <= 60 THEN 'slow_tutorial'
            WHEN COALESCE(gsf.tutorial_completed_flag, 0) = 1 THEN 'very_slow_tutorial'
            ELSE 'no_tutorial'
        END as tutorial_completion_speed,
        
        -- Progression tier (IDENTICAL logic)
        CASE 
            WHEN COALESCE(gsf.levels_completed, 0) >= 10 THEN 'fast_progressor'
            WHEN COALESCE(gsf.levels_completed, 0) >= 5 THEN 'moderate_progressor'
            WHEN COALESCE(gsf.levels_completed, 0) >= 2 THEN 'slow_progressor'
            WHEN COALESCE(gsf.levels_completed, 0) >= 1 THEN 'minimal_progressor'
            ELSE 'non_progressor'
        END as progression_tier,
        
        -- Calculated ratios (IDENTICAL column names and logic)
        SAFE_DIVIDE(COALESCE(bem.total_events, 0), NULLIF(COALESCE(bem.total_sessions, 0), 0)) as avg_events_per_session,
        SAFE_DIVIDE(COALESCE(bem.total_events, 0), NULLIF(COALESCE(bem.active_days, 0), 0)) as avg_events_per_active_day,
        SAFE_DIVIDE(COALESCE(bem.total_sessions, 0), NULLIF(COALESCE(bem.active_days, 0), 0)) as avg_sessions_per_day,
        SAFE_DIVIDE(COALESCE(tpf.business_hours_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as business_hours_ratio,
        SAFE_DIVIDE(COALESCE(tpf.evening_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as evening_activity_ratio,
        SAFE_DIVIDE(COALESCE(tpf.weekend_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as weekend_activity_ratio,
        SAFE_DIVIDE(COALESCE(tpf.late_night_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as late_night_ratio,
        SAFE_DIVIDE(COALESCE(gsf.rewarded_ads_viewed, 0), NULLIF(COALESCE(gsf.total_ads_viewed, 0), 0)) as rewarded_ad_ratio,
        SAFE_DIVIDE(COALESCE(gsf.total_ads_viewed, 0), NULLIF(COALESCE(bem.total_sessions, 0), 0)) as ads_per_session,
        SAFE_DIVIDE(COALESCE(gsf.total_ads_viewed, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as ad_engagement_rate,
        SAFE_DIVIDE(COALESCE(gsf.product_interactions_non_revenue, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as product_interest_ratio,
        SAFE_DIVIDE(COALESCE(gsf.store_views, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as store_engagement_ratio,
        SAFE_DIVIDE(COALESCE(gsf.purchase_intents, 0), NULLIF(COALESCE(gsf.store_views, 0), 0)) as store_conversion_intent_rate,
        SAFE_DIVIDE(COALESCE(gsf.levels_completed, 0), NULLIF(COALESCE(bem.active_days, 0), 0)) as levels_per_day,
        SAFE_DIVIDE(COALESCE(gsf.levels_completed, 0), NULLIF(COALESCE(bem.period_1_sessions, 0), 0)) as levels_per_session,
        
        -- Period growth ratios (IDENTICAL column names)
        SAFE_DIVIDE(COALESCE(bem.period_2_events, 0), NULLIF(COALESCE(bem.period_1_events, 0), 0)) as period_2_to_1_growth_ratio,
        SAFE_DIVIDE(COALESCE(bem.period_4_events, 0), NULLIF(COALESCE(bem.period_1_events, 0), 0)) as period_4_to_1_growth_ratio,
        SAFE_DIVIDE(COALESCE(bem.period_2_sessions, 0), NULLIF(COALESCE(bem.period_1_sessions, 0), 0)) as session_growth_period_1_to_2,
        
        SAFE_DIVIDE(COALESCE(bem.active_days, 0), 4) as activity_consistency_ratio,
        SAFE_DIVIDE(COALESCE(tpf.active_hours_spread, 0), 24) as time_diversity_ratio,
        SAFE_DIVIDE(COALESCE(tpf.active_days_of_week, 0), 7) as weekly_consistency_ratio,
        SAFE_DIVIDE(COALESCE(tpf.fingerprinted_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as fingerprinted_ratio,
        SAFE_DIVIDE(COALESCE(tpf.reengagement_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as reengagement_ratio,
        SAFE_DIVIDE(COALESCE(tpf.view_through_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as view_through_ratio,
        
        -- Platform × Country combo (IDENTICAL logic)
        CONCAT(
            CASE WHEN ud.platform = 'ios' THEN 'ios' ELSE 'android' END,
            '_',
            CASE 
            WHEN ud.country IN ('US', 'GB', 'DE', 'CA', 'AU') THEN 'premium_geo'
            WHEN ud.country IN ('JP', 'KR', 'FR', 'IT', 'ES') THEN 'good_geo'
            WHEN ud.country IN ('CN', 'BR', 'RU', 'IN', 'MX') THEN 'medium_geo'
            ELSE 'other_geo'
            END
        ) as platform_geo_combo,
        
        -- Monetization potential (IDENTICAL logic)
        CASE 
            WHEN ud.country IN ('US', 'GB', 'DE', 'CA', 'AU', 'NL', 'CH', 'NO', 'SE', 'DK') 
                AND ud.platform = 'ios' 
                AND COALESCE(gsf.tutorial_completed_flag, 0) = 1 
                AND COALESCE(gsf.total_ads_viewed, 0) >= 3 THEN 'high_monetization_potential'
            WHEN ud.country IN ('US', 'GB', 'DE', 'CA', 'AU', 'JP', 'KR', 'FR', 'IT', 'ES') 
                AND COALESCE(gsf.tutorial_completed_flag, 0) = 1 
                AND COALESCE(bem.period_1_sessions, 0) >= 3 THEN 'medium_monetization_potential'
            WHEN COALESCE(gsf.tutorial_completed_flag, 0) = 1 OR COALESCE(bem.period_1_sessions, 0) >= 2 THEN 'low_monetization_potential'
            ELSE 'minimal_monetization_potential'
        END as monetization_potential_tier,
        
        -- ==========================================
        -- TARGET VARIABLES (IDENTICAL column names)
        -- ==========================================
        COALESCE(lt.ltv_target, 0) as ltv_target,
        COALESCE(lt.received_ltv_target, 0) as received_ltv_target,
        COALESCE(lt.purchase_events_target, 0) as purchase_events_target

        FROM user_demographics ud
        LEFT JOIN basic_engagement_metrics bem ON ud.user_id = bem.user_id
        LEFT JOIN time_pattern_features tpf ON ud.user_id = tpf.user_id
        LEFT JOIN game_specific_features gsf ON ud.user_id = gsf.user_id
        LEFT JOIN session_timing_features stf ON ud.user_id = stf.user_id
        LEFT JOIN ltv_target lt ON ud.user_id = lt.user_id
        WHERE COALESCE(bem.total_events, 0) >= 1  -- At least some activity in D4-D33 window
        ORDER BY ud.install_date, ud.user_id;
    """
    
    # Load Training Data (Oct 2024 - Mar 2025)
    print("\n1. Loading Training Cohort (Oct 2024 - Mar 2025)...")
    train_data = client.query(train_query).result().to_dataframe()
    print(f"   Training cohort: {len(train_data):,} users")
    
    # Load Validation Data (Apr 2025)
    print("\n2. Loading Validation Cohort (Apr 2025)...")
    val_data = client.query(val_query).result().to_dataframe()
    print(f"   Validation cohort: {len(val_data):,} users")
    
    # Load Test Data (May+ 2025)
    print("\n3. Loading Test Cohort (May+ 2025)...")
    test_data = client.query(test_query).result().to_dataframe()
    print(f"   Test cohort: {len(test_data):,} users")
    
    print(f"\nDataset Summary:")
    print("train data features - ",train_data.columns)
    print("val data features - ",val_data.columns)
    print("test data features - ",test_data.columns)
    print(f"Train: {len(train_data):,} users, Avg LTV: ${train_data['ltv_target'].mean():.2f}")
    print(f"Val: {len(val_data):,} users, Avg LTV: ${val_data['ltv_target'].mean():.2f}")
    print(f"Test: {len(test_data):,} users, Avg LTV: ${test_data['ltv_target'].mean():.2f}")
    
    return train_data, val_data, test_data

def prepare_features(train_data, val_data, test_data):
    """Prepare features ensuring no revenue data leakage"""
    
    # Define columns to exclude (identifiers, targets, and revenue-related)
    exclude_cols = [
        'user_id', 'install_date', 'install_timestamp', 
        'first_event', 'last_event',  # datetime columns
        'ltv_target', 'received_ltv_target', 'purchase_events',  # target columns
        'purchase_events_30d', 'purchase_events_target',  # additional target columns
        'revenue', 'received_revenue', 'product_price', 'converted_revenue',  # revenue columns to prevent leakage
        'is_revenue_receipt_included', 'is_revenue_valid',  # revenue-related flags
        'ltv_30_days', 'received_ltv_30_days',  # any LTV columns from features
        # Remove product-related features that might indicate spending behavior
        'product_interactions', 'unique_products_viewed', 'unique_skus_viewed',
        'avg_quantity_per_interaction', 'total_quantity_interactions',
        'product_interactions_per_session', 'product_diversity_rate', 'product_engagement_rate'
    ]
    
    # Get behavioral feature columns only
    all_cols = set(train_data.columns) & set(val_data.columns) & set(test_data.columns)
    print(all_cols)
    feature_cols = [col for col in all_cols if col not in exclude_cols]
    
    print(f"Using {len(feature_cols)} behavioral features (no revenue data)")
    
    # Extract features
    X_train = train_data[feature_cols].copy()
    X_val = val_data[feature_cols].copy()
    X_test = test_data[feature_cols].copy()
    
    # Extract targets - handle different possible target column names
    possible_targets = ['ltv_target', 'ltv_30_days', 'ltv_7_days', 'revenue_30d', 'total_revenue']
    target_col = None
    for col in possible_targets:
        if col in train_data.columns:
            target_col = col
            print(f"Using target column: {target_col}")
            break
    
    if target_col is None:
        print("Available columns:", sorted(train_data.columns))
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
    
    # No feature scaling needed for tree-based models
    print("\nUsing unscaled features for tree-based models...")
    
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
    
    # Define models optimized for sparse target (many zeros)
    model_configs = {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=20,  # Very low to prevent overfitting
            max_depth=2,      # Very shallow trees
            learning_rate=0.01, # Very slow learning
            subsample=0.5,    # Strong subsampling
            colsample_bytree=0.5, # Strong feature subsampling
            reg_alpha=10.0,   # Very strong L1 regularization
            reg_lambda=10.0,  # Very strong L2 regularization
            min_child_weight=20, # Require many samples per leaf
            random_state=42,
            eval_metric='rmse'
        ),
        'CatBoost': CatBoostRegressor(
            iterations=20,    # Very low
            depth=2,          # Very shallow
            learning_rate=0.01, # Very slow
            l2_leaf_reg=100,  # Very strong L2 regularization
            random_seed=42,
            verbose=False
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=20,  # Very low
            max_depth=2,      # Very shallow
            min_samples_split=50, # Require many samples to split
            min_samples_leaf=20,  # Require many samples per leaf
            max_features=0.3, # Use fewer features
            random_state=42,
            n_jobs=-1
        ),
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=20,  # Very low
            max_depth=2,      # Very shallow  
            min_samples_split=50, # Require many samples to split
            min_samples_leaf=20,  # Require many samples per leaf
            max_features=0.3, # Use fewer features
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=20,  # Very low
            max_depth=2,      # Very shallow
            learning_rate=0.01, # Very slow learning
            min_samples_split=50, # Require many samples to split
            min_samples_leaf=20,  # Require many samples per leaf
            subsample=0.5,    # Strong subsampling
            random_state=42
        ),
        'Decision Tree': DecisionTreeRegressor(
            max_depth=2,      # Very shallow
            min_samples_split=100, # Very conservative splits
            min_samples_leaf=50,   # Large leaves
            random_state=42
        )
        # Remove linear models as they're performing very poorly with scaling
    }
    
    # Data validation before training
    print(f"\nData validation:")
    print(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Val set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Check for potential data leakage - target distribution should be similar
    print(f"\nTarget distribution check:")
    print(f"Train spenders: {(y_train > 0).sum()} ({(y_train > 0).mean():.1%})")
    print(f"Val spenders: {(y_val > 0).sum()} ({(y_val > 0).mean():.1%})")
    print(f"Test spenders: {(y_test > 0).sum()} ({(y_test > 0).mean():.1%})")
    
    # Warn about small test set
    if len(y_test) < 100:
        print(f"\nWARNING: Very small test set ({len(y_test)} samples) - results may be unreliable!")
    
    print(f"\nTraining {len(model_configs)} different models...")
    
    for name, model in model_configs.items():
        print(f"\n--- Training {name} ---")
        try:
            # All models use unscaled data (tree-based only)
            X_tr, X_v, X_te = X_train, X_val, X_test
            
            # Train model
            model.fit(X_tr, y_train)
            
            # Make predictions and ensure non-negative (LTV can't be negative)
            train_pred = np.maximum(model.predict(X_tr), 0)
            val_pred = np.maximum(model.predict(X_v), 0)
            test_pred = np.maximum(model.predict(X_te), 0)
            
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
            
            # Check for overfitting
            r2_gap = train_r2 - val_r2
            if r2_gap > 0.3:
                print(f"   ⚠️  WARNING: Possible overfitting (Train-Val R² gap: {r2_gap:.3f})")
            
            if test_r2 < 0:
                print(f"   ❌ WARNING: Negative test R² ({test_r2:.3f}) - model performing worse than baseline")
            
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
        
        # Save feature column names (no scaler needed for tree models)
        with open('models/feature_columns.txt', 'w') as f:
            f.write('\n'.join(feature_cols))
        print(f"Saved feature columns to 'models/feature_columns.txt'")
        
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
    pd.DataFrame({'ltv_target': y_train}).to_csv('data/y_train.csv', index=False)
    pd.DataFrame({'ltv_target': y_val}).to_csv('data/y_val.csv', index=False)
    pd.DataFrame({'ltv_target': y_test}).to_csv('data/y_test.csv', index=False)
    
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
            y_train = pd.read_csv('data/y_train.csv')['ltv_target'].values
            y_val = pd.read_csv('data/y_val.csv')['ltv_target'].values
            y_test = pd.read_csv('data/y_test.csv')['ltv_target'].values
            
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
        print("train data features - ",train_data.columns)
        print("val data features - ",val_data.columns)
        print("test data features - ",test_data.columns)
        
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
