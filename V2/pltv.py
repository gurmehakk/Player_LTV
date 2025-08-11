from matplotlib import table
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso, HuberRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import cross_val_score, KFold, train_test_split
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

# Baseline metrics storage
BASELINE_METRICS = None

def save_baseline_metrics(model_results):
    """Save current best model performance as baseline for future accuracy loss tracking"""
    global BASELINE_METRICS
    
    best_model = max(model_results, key=lambda x: x['test_r2'])
    BASELINE_METRICS = {
        'model_name': best_model['model_name'],
        'test_r2': best_model['test_r2'],
        'test_rmse': best_model['test_rmse'],
        'val_r2': best_model['val_r2'],
        'val_rmse': best_model['val_rmse'],
        'timestamp': pd.Timestamp.now()
    }
    
    # Save to file for persistence
    import pickle
    with open('baseline_metrics.pkl', 'wb') as f:
        pickle.dump(BASELINE_METRICS, f)
    
    print(f"Baseline metrics saved: {best_model['model_name']} (Test R²: {best_model['test_r2']:.4f})")

def load_baseline_metrics():
    """Load baseline metrics from file if available"""
    global BASELINE_METRICS
    import pickle
    import os
    
    if os.path.exists('baseline_metrics.pkl'):
        try:
            with open('baseline_metrics.pkl', 'rb') as f:
                BASELINE_METRICS = pickle.load(f)
            print(f"Loaded baseline metrics: {BASELINE_METRICS['model_name']} (Test R²: {BASELINE_METRICS['test_r2']:.4f})")
        except Exception as e:
            print(f"Error loading baseline metrics: {e}")
            BASELINE_METRICS = None

def calculate_accuracy_loss():
    """Calculate accuracy loss compared to baseline metrics"""
    if not BASELINE_METRICS or not PLOT_DATA:
        print("No baseline metrics or current results available for accuracy loss calculation")
        return None
    
    # Find current best model
    current_best = max(PLOT_DATA, key=lambda x: x['test_r2'])
    
    # Calculate degradation metrics
    r2_loss = (BASELINE_METRICS['test_r2'] - current_best['test_r2']) / BASELINE_METRICS['test_r2']
    rmse_increase = (current_best['test_rmse'] - BASELINE_METRICS['test_rmse']) / BASELINE_METRICS['test_rmse']
    
    accuracy_loss = {
        'baseline_model': BASELINE_METRICS['model_name'],
        'baseline_r2': BASELINE_METRICS['test_r2'],
        'baseline_rmse': BASELINE_METRICS['test_rmse'],
        'baseline_timestamp': BASELINE_METRICS['timestamp'],
        'current_model': current_best['model_name'],
        'current_r2': current_best['test_r2'],
        'current_rmse': current_best['test_rmse'],
        'r2_loss_percent': r2_loss * 100,
        'rmse_increase_percent': rmse_increase * 100,
        'days_since_baseline': (pd.Timestamp.now() - BASELINE_METRICS['timestamp']).days,
        'needs_retraining': r2_loss > 0.15 or rmse_increase > 0.20  # Alert if >15% R² loss or >20% RMSE increase
    }
    
    return accuracy_loss

def plot_accuracy_loss_analysis():
    """Create accuracy loss visualization and monitoring dashboard"""
    import os
    
    accuracy_loss = calculate_accuracy_loss()
    
    if not accuracy_loss:
        print("Cannot create accuracy loss plot - insufficient data")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Accuracy Loss Analysis', fontsize=16)
    
    # 1. R² Comparison
    ax1 = axes[0, 0]
    models = ['Baseline', 'Current']
    r2_values = [accuracy_loss['baseline_r2'], accuracy_loss['current_r2']]
    colors = ['green' if accuracy_loss['r2_loss_percent'] <= 5 else 'orange' if accuracy_loss['r2_loss_percent'] <= 15 else 'red', 'blue']
    
    bars = ax1.bar(models, r2_values, color=colors, alpha=0.7)
    ax1.set_ylabel('R² Score')
    ax1.set_title('R² Score Comparison')
    ax1.set_ylim(0, max(r2_values) * 1.1)
    
    # Add value labels
    for bar, value in zip(bars, r2_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    # Add loss percentage
    ax1.text(0.5, max(r2_values) * 0.9, f'R² Loss: {accuracy_loss["r2_loss_percent"]:.1f}%',
            ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 2. RMSE Comparison
    ax2 = axes[0, 1]
    rmse_values = [accuracy_loss['baseline_rmse'], accuracy_loss['current_rmse']]
    colors = ['green' if accuracy_loss['rmse_increase_percent'] <= 5 else 'orange' if accuracy_loss['rmse_increase_percent'] <= 20 else 'red', 'blue']
    
    bars = ax2.bar(models, rmse_values, color=colors, alpha=0.7)
    ax2.set_ylabel('RMSE ($)')
    ax2.set_title('RMSE Comparison')
    
    for bar, value in zip(bars, rmse_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'${value:.2f}', ha='center', va='bottom')
    
    ax2.text(0.5, max(rmse_values) * 0.9, f'RMSE Increase: {accuracy_loss["rmse_increase_percent"]:.1f}%',
            ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 3. Model Performance Timeline (simulate timeline)
    ax3 = axes[1, 0]
    timeline_days = [0, accuracy_loss['days_since_baseline']]
    r2_timeline = [accuracy_loss['baseline_r2'], accuracy_loss['current_r2']]
    
    ax3.plot(timeline_days, r2_timeline, 'o-', linewidth=2, markersize=8)
    ax3.set_xlabel('Days Since Baseline')
    ax3.set_ylabel('R² Score')
    ax3.set_title('Model Performance Over Time')
    ax3.grid(True, alpha=0.3)
    
    # Add threshold line
    threshold_r2 = accuracy_loss['baseline_r2'] * 0.85  # 15% loss threshold
    ax3.axhline(y=threshold_r2, color='red', linestyle='--', alpha=0.7, label='Retraining Threshold')
    ax3.legend()
    
    # 4. Retraining Alert Dashboard
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create status box
    status_color = 'red' if accuracy_loss['needs_retraining'] else 'green'
    status_text = 'RETRAINING NEEDED' if accuracy_loss['needs_retraining'] else 'MODEL HEALTHY'
    
    ax4.text(0.5, 0.8, 'Model Health Status', ha='center', va='center', fontsize=16, weight='bold')
    ax4.text(0.5, 0.6, status_text, ha='center', va='center', fontsize=14, weight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor=status_color, alpha=0.3))
    
    # Add metrics summary
    summary_text = f"""
    Days since baseline: {accuracy_loss['days_since_baseline']}
    R² degradation: {accuracy_loss['r2_loss_percent']:.1f}%
    RMSE increase: {accuracy_loss['rmse_increase_percent']:.1f}%
    
    Baseline: {accuracy_loss['baseline_model']}
    Current: {accuracy_loss['current_model']}
    """
    ax4.text(0.5, 0.3, summary_text, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    os.makedirs('plots/monitoring', exist_ok=True)
    plt.savefig('plots/monitoring/accuracy_loss_analysis.png', dpi=300, bbox_inches='tight')
    print("Accuracy loss analysis saved to plots/monitoring/accuracy_loss_analysis.png")
    
    # Print summary
    print(f"\n📊 ACCURACY LOSS ANALYSIS")
    print(f"{'='*50}")
    print(f"Baseline Model: {accuracy_loss['baseline_model']}")
    print(f"Current Model: {accuracy_loss['current_model']}")
    print(f"Days Since Baseline: {accuracy_loss['days_since_baseline']}")
    print(f"R² Loss: {accuracy_loss['r2_loss_percent']:.1f}% ({accuracy_loss['baseline_r2']:.4f} → {accuracy_loss['current_r2']:.4f})")
    print(f"RMSE Increase: {accuracy_loss['rmse_increase_percent']:.1f}% (${accuracy_loss['baseline_rmse']:.2f} → ${accuracy_loss['current_rmse']:.2f})")
    
    if accuracy_loss['needs_retraining']:
        print(f"\n🚨 ALERT: Model performance has degraded significantly!")
        print(f"   Recommend retraining the model with fresh data.")
    else:
        print(f"\n✅ Model performance is within acceptable limits.")
    
    return accuracy_loss

def plot_comprehensive_evaluation():
    """Create comprehensive evaluation plots for good models only (R² >= 0.1) including AUC, ROC, RMSE, residual, actual vs predicted, accuracy, feature importance"""
    if not PLOT_DATA:
        print("No evaluation data collected")
        return
    
    # Filter out models with R² < 0.1
    good_models = [data for data in PLOT_DATA if data['test_r2'] >= 0.1]
    
    if not good_models:
        print("No models with R² >= 0.1 found")
        return
    
    print(f"Creating plots for {len(good_models)} models with R² >= 0.1:")
    for data in good_models:
        print(f"  - {data['model_name']}: R² = {data['test_r2']:.4f}")
    
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
    r2_scores = [data['test_r2'] for data in good_models]
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
        best_model = max(feature_importance_data, key=lambda x: x['test_r2'])
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
    print(f"Best model: {max(good_models, key=lambda x: x['test_r2'])['model_name']} (R² = {max(good_models, key=lambda x: x['test_r2'])['test_r2']:.4f})")

def load_temporal_datasets():
    """Load temporal datasets using only behavioral features (no revenue features)"""
    try:
        from google.cloud import bigquery
    except ImportError:
        print("Warning: google-cloud-bigquery not installed. Skipping BigQuery functionality.")
        return None, None, None
    
    print("="*80)
    print("LOADING DATASETS - BEHAVIORAL FEATURES ONLY")
    print("="*80)
    print("Train: 70% of users, D0-D3 behavioral features → D4-D33 LTV targets")
    print("Val: 15% of users, D0-D3 behavioral features → D4-D33 LTV targets")
    print("Test: 15% of users, D0-D3 behavioral features → D4-D33 LTV targets")
    print("Split Method: Random ratio-based (reproducible with seed=42)")
    print("="*80)
    
    PROJECT_ID = "gc-forecasting-dev"
    DATASET = "test_data"
    TABLE = "app_data"
    FEATURE_DAYS = 3  # D0-D3 (4 days)
    PREDICTION_DAYS = 30  # D4-D33 (30 days)
    
    client = bigquery.Client(project=PROJECT_ID)
    table_path = f"{PROJECT_ID}.{DATASET}.{TABLE}"
    
    # Integrated SQL query (no separate file needed)
    data_query = f"""
    WITH install_cohort AS (
    -- Get install date for Feb-March installs
    SELECT 
        COALESCE(gaid, idfa, android_id, custom_user_id) as user_id,
        DATE(MIN(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64))))) as install_date,
        MIN(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64)))) as install_timestamp
    FROM `{table_path}`
    WHERE DATE(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64)))) >= '2024-06-01'
        AND DATE(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64)))) <= '2025-08-01'
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
    LEFT JOIN `{table_path}` e
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
    LEFT JOIN `{table_path}` e
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
    
    -- Separate specific event queries instead of LIKE operations
    tutorial_events AS (
    SELECT 
        user_id,
        COUNT(*) as tutorial_completions,
        MIN(event_timestamp) as tutorial_completion_timestamp
    FROM feature_window_events
    WHERE user_id IS NOT NULL 
        AND (name IN ('ftue_completed', 'tutorial_completed', 'onboarding_completed', 'tutorial_complete')
            OR name = 'session_start' AND JSON_EXTRACT_SCALAR(arguments, '$.tutorial_completed') = 'true'
            OR name = 'level_end' AND JSON_EXTRACT_SCALAR(arguments, '$.is_tutorial') = 'true')
    GROUP BY user_id
    ),
    
    level_progression_events AS (
    SELECT 
        user_id,
        COUNT(*) as levels_completed,
        MAX(SAFE_CAST(COALESCE(
            JSON_EXTRACT_SCALAR(arguments, '$.level_number'),
            JSON_EXTRACT_SCALAR(arguments, '$.level'),
            JSON_EXTRACT_SCALAR(arguments, '$.content_level_number')
        ) AS INT64)) as max_level_reached
    FROM feature_window_events
    WHERE user_id IS NOT NULL 
        AND (name IN ('level_complete', 'level_completed', 'level_end', 'level_progress', 'level_up')
            OR name = 'session_start' AND JSON_EXTRACT_SCALAR(arguments, '$.last_completed_level') IS NOT NULL)
    GROUP BY user_id
    ),
    
    ad_engagement_events AS (
    SELECT 
        user_id,
        COUNT(*) as total_ads_viewed,
        COUNTIF(name IN ('rewarded_ad_watched', 'rewarded_video_completed', 'ad_reward_earned') 
                OR JSON_EXTRACT_SCALAR(arguments, '$.ad_type') = 'rewarded'
                OR JSON_EXTRACT_SCALAR(arguments, '$.ad_placement_name') IN ('rewarded_video', 'reward_ad')) as rewarded_ads_viewed,
        COUNTIF(name IN ('interstitial_ad_watched', 'interstitial_completed') 
                OR JSON_EXTRACT_SCALAR(arguments, '$.ad_type') = 'interstitial'
                OR JSON_EXTRACT_SCALAR(arguments, '$.ad_placement_name') IN ('interstitial', 'interstitial_ad')) as interstitial_ads_viewed
    FROM feature_window_events
    WHERE user_id IS NOT NULL 
        AND (name IN ('ad_watched', 'ad_completed', 'ad_impression', 'rewarded_ad_watched', 'interstitial_ad_watched', 
                        'rewarded_video_completed', 'interstitial_completed', 'ad_reward_earned', 'ad_started')
            OR JSON_EXTRACT_SCALAR(arguments, '$.ad_type') IS NOT NULL
            OR JSON_EXTRACT_SCALAR(arguments, '$.ad_placement_name') IS NOT NULL)
    GROUP BY user_id
    ),
    
    monetization_events AS (
    SELECT 
        user_id,
        COUNTIF(name IN ('store_opened', 'shop_opened', 'iap_store_opened', 'store_view', 'shop_view')) as store_views,
        COUNTIF(name IN ('purchase_intent', 'iap_clicked', 'purchase_started', 'item_selected', 'buy_button_clicked')) as purchase_intents
    FROM feature_window_events
    WHERE user_id IS NOT NULL 
        AND (name IN ('store_opened', 'shop_opened', 'iap_store_opened', 'store_view', 'shop_view',
                        'purchase_intent', 'iap_clicked', 'purchase_started', 'item_selected', 'buy_button_clicked')
            OR JSON_EXTRACT_SCALAR(arguments, '$.store_type') IS NOT NULL
            OR JSON_EXTRACT_SCALAR(arguments, '$.purchase_intent') = 'true')
    GROUP BY user_id
    ),
    
    currency_and_product_events AS (
    SELECT 
        user_id,
        COUNTIF(name = 'currency_earned') as currency_earned_events,
        COUNTIF(name = 'currency_spent') as currency_spent_events,
        COUNTIF(product_name IS NOT NULL AND COALESCE(product_price, 0) = 0) as product_interactions_non_revenue,
        COUNT(DISTINCT CASE WHEN product_name IS NOT NULL AND COALESCE(product_price, 0) = 0 THEN product_name END) as unique_products_viewed,
        COUNT(DISTINCT CASE WHEN product_sku IS NOT NULL AND COALESCE(product_price, 0) = 0 THEN product_sku END) as unique_skus_viewed
    FROM feature_window_events
    WHERE user_id IS NOT NULL
    GROUP BY user_id
    ),
    
    social_and_achievement_events AS (
    SELECT 
        user_id,
        COUNTIF(name IN ('social_share', 'invite_sent', 'friend_invited', 'social_connect', 'share_completed')) as social_events,
        COUNTIF(name IN ('achievement_earned', 'trophy_earned', 'leaderboard_view', 'milestone_reached', 'badge_earned')) as achievement_events
    FROM feature_window_events
    WHERE user_id IS NOT NULL 
        AND (name IN ('social_share', 'invite_sent', 'friend_invited', 'social_connect', 'share_completed',
                        'achievement_earned', 'trophy_earned', 'leaderboard_view', 'milestone_reached', 'badge_earned')
            OR JSON_EXTRACT_SCALAR(arguments, '$.social_action') IS NOT NULL
            OR JSON_EXTRACT_SCALAR(arguments, '$.achievement_type') IS NOT NULL)
    GROUP BY user_id
    ),
    
    game_specific_features AS (
    SELECT 
        base_users.user_id,
        COALESCE(te.tutorial_completions, 0) as tutorial_completions,
        CASE WHEN COALESCE(te.tutorial_completions, 0) > 0 THEN 1 ELSE 0 END as tutorial_completed_flag,
        te.tutorial_completion_timestamp,
        COALESCE(lpe.levels_completed, 0) as levels_completed,
        COALESCE(lpe.max_level_reached, 0) as max_level_reached,
        COALESCE(aee.total_ads_viewed, 0) as total_ads_viewed,
        COALESCE(aee.rewarded_ads_viewed, 0) as rewarded_ads_viewed,
        COALESCE(aee.interstitial_ads_viewed, 0) as interstitial_ads_viewed,
        COALESCE(me.store_views, 0) as store_views,
        COALESCE(me.purchase_intents, 0) as purchase_intents,
        COALESCE(cape.currency_earned_events, 0) as currency_earned_events,
        COALESCE(cape.currency_spent_events, 0) as currency_spent_events,
        COALESCE(cape.product_interactions_non_revenue, 0) as product_interactions_non_revenue,
        COALESCE(cape.unique_products_viewed, 0) as unique_products_viewed,
        COALESCE(cape.unique_skus_viewed, 0) as unique_skus_viewed,
        COALESCE(saae.social_events, 0) as social_events,
        COALESCE(saae.achievement_events, 0) as achievement_events
    FROM (SELECT DISTINCT user_id FROM feature_window_events WHERE user_id IS NOT NULL) base_users
    LEFT JOIN tutorial_events te ON base_users.user_id = te.user_id
    LEFT JOIN level_progression_events lpe ON base_users.user_id = lpe.user_id
    LEFT JOIN ad_engagement_events aee ON base_users.user_id = aee.user_id
    LEFT JOIN monetization_events me ON base_users.user_id = me.user_id
    LEFT JOIN currency_and_product_events cape ON base_users.user_id = cape.user_id
    LEFT JOIN social_and_achievement_events saae ON base_users.user_id = saae.user_id
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
    
    print(f"Using integrated SQL query ({len(data_query):,} characters)")
    
    print("\n1. Loading full dataset...")
    try:
        full_data = client.query(data_query).result().to_dataframe()
        print(f"   Full dataset: {len(full_data):,} users")
    except Exception as e:
        print(f"Error executing query: {e}")
        return None, None, None

    if len(full_data) == 0:
        print("No data returned from query!")
        return None, None, None

    # Handle datetime fields properly to avoid fillna issues
    datetime_columns = ['install_timestamp', 'tutorial_completion_timestamp']
    for col in datetime_columns:
        if col in full_data.columns:
            # Convert to string to avoid datetime fillna issues
            full_data[col] = full_data[col].astype(str)
            full_data[col] = full_data[col].replace('NaT', '')
            full_data[col] = full_data[col].replace('None', '')

    # Ratio-based train/val/test split (70/15/15)
    
    # Set random seed for reproducible splits
    random_state = 42
    
    print(f"\n2. Creating ratio-based splits (70% train, 15% val, 15% test)...")
    
    # Optional stratification based on spender status for better distribution
    spender_status = (full_data['ltv_target'] > 0).astype(int)
    spender_rate = spender_status.mean()
    print(f"   Overall spender rate: {spender_rate:.1%}")
    
    # Use stratification if we have enough spenders in both classes
    stratify_col = spender_status if spender_rate > 0.01 and spender_rate < 0.99 else None
    stratify_msg = "with spender stratification" if stratify_col is not None else "without stratification"
    print(f"   Splitting {stratify_msg}...")
    
    # First split: separate out test set (15%)
    temp_data, test_data = train_test_split(
        full_data, 
        test_size=0.15, 
        random_state=random_state,
        stratify=stratify_col
    )
    
    # Second split: divide remaining data into train (70% of total) and val (15% of total)
    # Update stratification for remaining data
    if stratify_col is not None:
        temp_stratify = (temp_data['ltv_target'] > 0).astype(int)
    else:
        temp_stratify = None
        
    train_data, val_data = train_test_split(
        temp_data, 
        test_size=0.176,  # 15/85 ≈ 0.176 to get 15% of original data
        random_state=random_state,
        stratify=temp_stratify
    )
    
    print(f"   Train set: {len(train_data):,} users ({len(train_data)/len(full_data)*100:.1f}%)")
    print(f"   Val set: {len(val_data):,} users ({len(val_data)/len(full_data)*100:.1f}%)")
    print(f"   Test set: {len(test_data):,} users ({len(test_data)/len(full_data)*100:.1f}%)")
    
    # Show spender distribution across splits
    train_spenders = (train_data['ltv_target'] > 0).sum()
    val_spenders = (val_data['ltv_target'] > 0).sum()
    test_spenders = (test_data['ltv_target'] > 0).sum()
    
    print(f"   Spender distribution:")
    print(f"     Train: {train_spenders:,} spenders ({train_spenders/len(train_data)*100:.1f}%)")
    print(f"     Val: {val_spenders:,} spenders ({val_spenders/len(val_data)*100:.1f}%)")
    print(f"     Test: {test_spenders:,} spenders ({test_spenders/len(test_data)*100:.1f}%)")
    
    print(f"\nDataset Summary:")
    print("train data features - ", train_data.columns)
    print("val data features - ", val_data.columns)
    print("test data features - ", test_data.columns)
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
    
    # Feature engineering and selection
    def add_engineered_features(df):
        """Add simplified engineered features - key behavioral indicators only"""
        df = df.copy()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Core engagement metrics (simplified)
        df['total_sessions'] = (
            df.get('d0_sessions', 0) + 
            df.get('d1_sessions', 0) + 
            df.get('d2_sessions', 0)
        )
        df['total_events'] = (
            df.get('d0_total_events', 0) + 
            df.get('d1_total_events', 0) + 
            df.get('d2_total_events', 0)
        )
        
        # Retention indicators
        df['retention_d1'] = (df.get('d1_sessions', 0) > 0).astype(int)
        df['retention_d2'] = (df.get('d2_sessions', 0) > 0).astype(int)
        df['retention_score'] = df['retention_d1'] + df['retention_d2']
        
        # Tutorial completion
        df['tutorial_completed'] = (
            df.get('d0_tutorial_completed', 0) + 
            df.get('d1_tutorial_completed', 0) + 
            df.get('d2_tutorial_completed', 0) > 0
        ).astype(int)
        
        # Progression
        df['max_level'] = df[['d0_max_level', 'd1_max_level', 'd2_max_level']].fillna(0).max(axis=1)
        
        # Activity intensity
        df['events_per_session'] = df['total_events'] / (df['total_sessions'] + 1)
        df['d0_intensity'] = df.get('d0_total_events', 0) / (df.get('d0_sessions', 1) + 1)
        
        return df
    
    # Apply feature engineering
    train_data_eng = add_engineered_features(train_data)
    val_data_eng = add_engineered_features(val_data)
    test_data_eng = add_engineered_features(test_data)
    
    # Use ALL available features from SQL query (excluding identifiers, targets, and revenue)
    all_columns = set(train_data_eng.columns)
    available_features = [col for col in all_columns if col not in exclude_cols]
    
    # Sort features for consistent ordering
    available_features = sorted(available_features)
    
    print(f"Using {len(available_features)} selected + engineered features (no revenue data)")
    print("Selected features:", available_features)
    
    # Extract features
    X_train = train_data_eng[available_features].copy()
    X_val = val_data_eng[available_features].copy()
    X_test = test_data_eng[available_features].copy()
    
    # Handle categorical encoding for non-numeric features
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    
    categorical_features = []
    numeric_features = []
    
    # Identify categorical vs numeric features
    for col in available_features:
        if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
            categorical_features.append(col)
        else:
            numeric_features.append(col)
    
    print(f"Found {len(categorical_features)} categorical features: {categorical_features}")
    print(f"Found {len(numeric_features)} numeric features")
    
    # Encode categorical features using LabelEncoder
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        
        # Fit on combined training data to ensure consistent encoding
        combined_values = pd.concat([X_train[col], X_val[col], X_test[col]]).fillna('missing')
        le.fit(combined_values)
        
        # Transform all sets
        X_train[col] = le.transform(X_train[col].fillna('missing'))
        X_val[col] = le.transform(X_val[col].fillna('missing'))  
        X_test[col] = le.transform(X_test[col].fillna('missing'))
        
        label_encoders[col] = le
    
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
    
    # All features are now numeric after encoding
    print(f"\nFeature engineering completed - {len(categorical_features)} categorical features encoded, {len(numeric_features)} numeric features")
    
    print(f"\nFinal dataset shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    print(f"\nTarget statistics:")
    print(f"Train - Mean: ${y_train.mean():.2f}, Std: ${y_train.std():.2f}, Spenders: {(y_train > 0).sum()}")
    print(f"Val - Mean: ${y_val.mean():.2f}, Std: ${y_val.std():.2f}, Spenders: {(y_val > 0).sum()}")
    print(f"Test - Mean: ${y_test.mean():.2f}, Std: ${y_test.std():.2f}, Spenders: {(y_test > 0).sum()}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, available_features

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

def create_enhanced_validation_plots():
    """Create enhanced validation plots including ROC/PR for full predictions and performance analysis"""
    import os
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    
    os.makedirs('plots/validation', exist_ok=True)
    
    if len(PLOT_DATA) == 0:
        print("No plot data available for validation plots")
        return
    
    print("Creating enhanced validation plots...")
    
    # 1. ROC/PR Curves for Full LTV Predictions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Enhanced Model Validation - Full LTV Predictions', fontsize=16)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(PLOT_DATA)))
    
    # ROC Curves using full LTV predictions as scores
    ax = axes[0, 0]
    for i, data in enumerate(PLOT_DATA):
        y_true_binary = (data['y_true'] > 0).astype(int)
        y_scores = data['y_pred']  # Use full LTV predictions as scores
        
        if len(np.unique(y_true_binary)) > 1:
            fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
            auc_score = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], linewidth=2, 
                   label=f"{data['model_name']} (AUC: {auc_score:.3f})")
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC: Payer Detection Using Full LTV Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Precision-Recall Curves using full LTV predictions
    ax = axes[0, 1]
    for i, data in enumerate(PLOT_DATA):
        y_true_binary = (data['y_true'] > 0).astype(int)
        y_scores = data['y_pred']
        
        if len(np.unique(y_true_binary)) > 1 and y_scores.max() > 0:
            precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
            pr_auc = auc(recall, precision)
            ax.plot(recall, precision, color=colors[i], linewidth=2,
                   label=f"{data['model_name']} (PR-AUC: {pr_auc:.3f})")
    
    baseline = sum(data['y_true'] > 0) / len(data['y_true'])
    ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline: {baseline:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall: Payer Detection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Residual Analysis for best model
    ax = axes[1, 0]
    # Find best model by R² score - handle different key formats
    best_model = None
    best_r2 = -999
    for data in PLOT_DATA:
        r2_score = data.get('r2', data.get('Test_R2', 0))
        if r2_score > best_r2:
            best_r2 = r2_score
            best_model = data
    residuals = best_model['y_pred'] - best_model['y_true']
    
    ax.scatter(best_model['y_pred'], residuals, alpha=0.6, s=1)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel('Predicted LTV ($)')
    ax.set_ylabel('Residuals ($)')
    ax.set_title(f'Residual Plot - {best_model["model_name"]}')
    ax.grid(True, alpha=0.3)
    
    # LTV Distribution Comparison
    ax = axes[1, 1]
    ax.hist(best_model['y_true'], bins=50, alpha=0.7, label='Actual LTV', density=True)
    ax.hist(best_model['y_pred'], bins=50, alpha=0.7, label='Predicted LTV', density=True)
    ax.set_xlabel('LTV ($)')
    ax.set_ylabel('Density')
    ax.set_title('LTV Distribution Comparison')
    ax.legend()
    ax.set_xlim(0, 50)  # Focus on main distribution
    
    plt.tight_layout()
    plt.savefig('plots/validation/enhanced_validation_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Performance Analysis by LTV Segments
    create_ltv_segment_analysis()
    
    # 3. Model Calibration Plot
    create_calibration_plot()
    
    print("Enhanced validation plots saved to 'plots/validation/' directory")

def create_ltv_segment_analysis():
    """Analyze model performance across different LTV segments"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance by LTV Segments', fontsize=16)
    
    best_model = max(PLOT_DATA, key=lambda x: x['test_r2'])
    y_true = best_model['y_true']
    y_pred = best_model['y_pred']
    
    # Define LTV segments
    segments = {
        'Non-Payers ($0)': (y_true == 0),
        'Low Spenders ($0-$5)': (y_true > 0) & (y_true <= 5),
        'Medium Spenders ($5-$20)': (y_true > 5) & (y_true <= 20),
        'High Spenders ($20+)': (y_true > 20)
    }
    
    segment_metrics = {}
    for name, mask in segments.items():
        if mask.sum() > 0:
            segment_mae = mean_absolute_error(y_true[mask], y_pred[mask])
            segment_rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
            segment_metrics[name] = {'mae': segment_mae, 'rmse': segment_rmse, 'count': mask.sum()}
    
    # MAE by segment
    ax = axes[0, 0]
    names = list(segment_metrics.keys())
    maes = [segment_metrics[name]['mae'] for name in names]
    ax.bar(names, maes)
    ax.set_title('MAE by LTV Segment')
    ax.set_ylabel('Mean Absolute Error ($)')
    ax.tick_params(axis='x', rotation=45)
    
    # RMSE by segment
    ax = axes[0, 1]
    rmses = [segment_metrics[name]['rmse'] for name in names]
    ax.bar(names, rmses)
    ax.set_title('RMSE by LTV Segment')
    ax.set_ylabel('Root Mean Square Error ($)')
    ax.tick_params(axis='x', rotation=45)
    
    # Sample counts by segment
    ax = axes[1, 0]
    counts = [segment_metrics[name]['count'] for name in names]
    ax.bar(names, counts)
    ax.set_title('Sample Count by LTV Segment')
    ax.set_ylabel('Number of Users')
    ax.tick_params(axis='x', rotation=45)
    
    # Prediction vs Actual by segment (scatter)
    ax = axes[1, 1]
    colors = ['red', 'orange', 'green', 'blue']
    for i, (name, mask) in enumerate(segments.items()):
        if mask.sum() > 0:
            ax.scatter(y_true[mask], y_pred[mask], alpha=0.6, s=10, 
                      color=colors[i], label=name)
    
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    ax.set_xlabel('Actual LTV ($)')
    ax.set_ylabel('Predicted LTV ($)')
    ax.set_title('Predictions by LTV Segment')
    ax.legend()
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    
    plt.tight_layout()
    plt.savefig('plots/validation/ltv_segment_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_calibration_plot():
    """Create calibration plot to check if predicted probabilities match observed frequencies"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Model Calibration Analysis', fontsize=16)
    
    best_model = max(PLOT_DATA, key=lambda x: x['test_r2'])
    y_true = best_model['y_true']
    y_pred = best_model['y_pred']
    
    # Calibration for payer prediction
    ax = axes[0]
    y_true_binary = (y_true > 0).astype(int)
    
    # Create bins based on predicted LTV
    pred_probs = y_pred / y_pred.max() if y_pred.max() > 0 else y_pred
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = []
    bin_true_probs = []
    bin_pred_probs = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (pred_probs >= bin_lower) & (pred_probs < bin_upper)
        if in_bin.sum() > 0:
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_true_probs.append(y_true_binary[in_bin].mean())
            bin_pred_probs.append(pred_probs[in_bin].mean())
    
    if bin_centers:
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.plot(bin_pred_probs, bin_true_probs, 'ro-', label='Model Calibration')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Observed Frequency')
        ax.set_title('Calibration Plot - Payer Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # LTV prediction accuracy by deciles
    ax = axes[1]
    deciles = np.percentile(y_pred, np.arange(0, 101, 10))
    decile_centers = []
    decile_actual = []
    decile_predicted = []
    
    for i in range(len(deciles) - 1):
        in_decile = (y_pred >= deciles[i]) & (y_pred < deciles[i + 1])
        if in_decile.sum() > 0:
            decile_centers.append(i + 1)
            decile_actual.append(y_true[in_decile].mean())
            decile_predicted.append(y_pred[in_decile].mean())
    
    ax.plot(decile_centers, decile_actual, 'bo-', label='Actual LTV', linewidth=2)
    ax.plot(decile_centers, decile_predicted, 'ro-', label='Predicted LTV', linewidth=2)
    ax.set_xlabel('Prediction Decile')
    ax.set_ylabel('Mean LTV ($)')
    ax.set_title('LTV Calibration by Deciles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/validation/calibration_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

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

def train_hybrid_model(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train a hybrid model: classification (payer/non-payer) + regression (LTV for payers)"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import Ridge
    
    print("   Training Classification + Regression hybrid model...")
    
    # Step 1: Classification - predict payers vs non-payers
    y_payer_train = (y_train > 0).astype(int)
    y_payer_val = (y_val > 0).astype(int)
    y_payer_test = (y_test > 0).astype(int)
    
    # Train classifier
    classifier = RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_split=10,
        random_state=42, n_jobs=-1
    )
    classifier.fit(X_train, y_payer_train)
    
    # Get payer probabilities
    train_payer_prob = classifier.predict_proba(X_train)[:, 1]
    val_payer_prob = classifier.predict_proba(X_val)[:, 1]
    test_payer_prob = classifier.predict_proba(X_test)[:, 1]
    
    # Step 2: Regression - predict LTV for payers only
    payer_mask_train = y_train > 0
    if payer_mask_train.sum() > 10:  # Need enough payer samples
        X_payers_train = X_train[payer_mask_train]
        y_ltv_payers_train = y_train[payer_mask_train]
        
        # Train regressor on payers only
        regressor = Ridge(alpha=10.0)
        regressor.fit(X_payers_train, y_ltv_payers_train)
        
        # Predict LTV for all users (will be multiplied by payer probability)
        train_ltv_raw = regressor.predict(X_train)
        val_ltv_raw = regressor.predict(X_val) 
        test_ltv_raw = regressor.predict(X_test)
    else:
        print("   Warning: Too few payers for regression, using mean LTV")
        mean_ltv = y_train[y_train > 0].mean() if (y_train > 0).sum() > 0 else 1.0
        train_ltv_raw = np.full(len(y_train), mean_ltv)
        val_ltv_raw = np.full(len(y_val), mean_ltv)
        test_ltv_raw = np.full(len(y_test), mean_ltv)
        regressor = None
    
    # Step 3: Combine predictions
    # Final LTV = P(payer) × predicted_LTV_for_payers
    train_pred = train_payer_prob * np.maximum(train_ltv_raw, 0)
    val_pred = val_payer_prob * np.maximum(val_ltv_raw, 0)
    test_pred = test_payer_prob * np.maximum(test_ltv_raw, 0)
    
    # Create hybrid model object
    hybrid_model = {
        'classifier': classifier,
        'regressor': regressor,
        'type': 'hybrid'
    }
    
    print(f"   Payer classification accuracy: {(classifier.predict(X_test) == y_payer_test).mean():.3f}")
    print(f"   Payer rate in test: {y_payer_test.mean():.3f}")
    
    return train_pred, val_pred, test_pred, hybrid_model

def train_ensemble_model(models, predictions, X_train, X_val, X_test, y_train, y_val, y_test):
    """Create an ensemble of the best performing models"""
    print("   Creating ensemble of best models...")
    
    if len(models) < 3:
        print("   Warning: Need at least 3 models for ensemble, using simple average")
        # Use simple average if not enough models
        if len(predictions) > 0:
            train_preds = np.array([pred['train'] for pred in predictions.values()])
            val_preds = np.array([pred['val'] for pred in predictions.values()])
            test_preds = np.array([pred['test'] for pred in predictions.values()])
            
            train_pred = np.mean(train_preds, axis=0)
            val_pred = np.mean(val_preds, axis=0) 
            test_pred = np.mean(test_preds, axis=0)
        else:
            train_pred = np.zeros(len(y_train))
            val_pred = np.zeros(len(y_val))
            test_pred = np.zeros(len(y_test))
    else:
        # Select top 3 models by validation R²
        model_scores = []
        for name, model in models.items():
            if name in predictions:
                val_r2 = r2_score(y_val, predictions[name]['val'])
                model_scores.append((name, val_r2))
        
        # Sort by validation R² and take top 3
        model_scores.sort(key=lambda x: x[1], reverse=True)
        top_models = [name for name, _ in model_scores[:3]]
        
        print(f"   Selected top 3 models: {top_models}")
        
        # Create weighted ensemble based on validation performance
        weights = []
        train_preds = []
        val_preds = []
        test_preds = []
        
        total_score = sum(score for _, score in model_scores[:3])
        
        for name, score in model_scores[:3]:
            weight = score / total_score if total_score > 0 else 1.0 / 3
            weights.append(weight)
            train_preds.append(predictions[name]['train'])
            val_preds.append(predictions[name]['val'])
            test_preds.append(predictions[name]['test'])
        
        # Weighted ensemble predictions
        train_pred = np.average(train_preds, axis=0, weights=weights)
        val_pred = np.average(val_preds, axis=0, weights=weights)
        test_pred = np.average(test_preds, axis=0, weights=weights)
        
        print(f"   Ensemble weights: {dict(zip(top_models, weights))}")
    
    # Ensure non-negative predictions
    train_pred = np.maximum(train_pred, 0)
    val_pred = np.maximum(val_pred, 0)
    test_pred = np.maximum(test_pred, 0)
    
    # Create ensemble model info
    ensemble_model = {
        'type': 'ensemble',
        'models': list(models.keys()) if len(models) >= 3 else ['simple_average'],
        'weights': weights if len(models) >= 3 else [1.0 / len(models)] * len(models)
    }
    
    return train_pred, val_pred, test_pred, ensemble_model

def train_ensemble_ltv_models(X_train, X_val, X_test, y_train, y_val, y_test, feature_cols):
    """Train multiple LTV prediction models and ensemble them"""
    
    print("\n" + "="*80)
    print("TRAINING ENSEMBLE LTV PREDICTION MODELS")
    print("="*80)
    
    # Clear any previous plot data
    global PLOT_DATA
    PLOT_DATA.clear()
    
    # Feature scaling for linear models
    from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
    print("\nPreparing scaled features for linear models...")
    
    # Try different scalers for linear models
    robust_scaler = RobustScaler()
    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    
    # Will scale selected features after feature selection
    
    # Use robust scaler as default
    scaler = robust_scaler
    
    models = {}
    predictions = {}
    results = []
    
    # Import necessary libraries
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, ElasticNet
    import xgboost as xgb
    from catboost import CatBoostRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import joblib
    import os
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Streamlined model configs - only the 7 best performing models
    model_configs = {
        # Best linear models
        'Ridge Optimal': ('ridge_optimal', Ridge(alpha=10.0)),
        'ElasticNet Balanced': ('elastic_balanced', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000)),
        
        # Best tree-based models  
        'XGBoost Optimized': ('xgboost_opt', xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )),
        'Random Forest Balanced': ('rf_balanced', RandomForestRegressor(
            n_estimators=150, max_depth=5, min_samples_split=15, min_samples_leaf=8,
            max_features=0.8, random_state=42, n_jobs=-1
        )),
        'Gradient Boosting': ('gbm', GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        )),
        
        # Advanced models
        'CatBoost Optimized': ('catboost_opt', CatBoostRegressor(
            iterations=200, depth=4, learning_rate=0.1,
            random_seed=42, silent=True
        )),
        
        # Classification + Regression approach
        'Hybrid (Classification + Regression)': ('hybrid', 'hybrid_model'),
        
        # Ensemble of best performers  
        'Best Models Ensemble': ('ensemble', 'ensemble_model')
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
    
    # Add feature selection to improve performance
    print("\nPerforming feature selection...")
    from sklearn.feature_selection import SelectKBest, f_regression
    
    # Select top K features based on F-test
    selector = SelectKBest(f_regression, k=min(50, X_train.shape[1]))  # Select top 50 features
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)  
    X_test_selected = selector.transform(X_test)
    
    selected_features = selector.get_support()
    selected_feature_names = [feature_cols[i] for i in range(len(feature_cols)) if selected_features[i]]
    print(f"Selected {len(selected_feature_names)} most important features")
    print(f"Top 10 selected features: {selected_feature_names[:10]}")
    
    # Scale the selected features
    X_train_robust = robust_scaler.fit_transform(X_train_selected)
    X_val_robust = robust_scaler.transform(X_val_selected)
    X_test_robust = robust_scaler.transform(X_test_selected)
    
    X_train_standard = standard_scaler.fit_transform(X_train_selected)
    X_val_standard = standard_scaler.transform(X_val_selected)
    X_test_standard = standard_scaler.transform(X_test_selected)
    
    X_train_minmax = minmax_scaler.fit_transform(X_train_selected)
    X_val_minmax = minmax_scaler.transform(X_val_selected)
    X_test_minmax = minmax_scaler.transform(X_test_selected)
    
    print(f"\nTraining {len(model_configs)} different models with feature selection...")
    
    for name, config in model_configs.items():
        print(f"\n--- Training {name} ---")
        try:
            # Handle different model configurations
            if isinstance(config, tuple):
                scaler_type, model = config
            else:
                scaler_type, model = 'tree', config
            
            # Handle hybrid classification + regression model
            if model == 'hybrid_model':
                train_pred, val_pred, test_pred, model = train_hybrid_model(
                    X_train, X_val, X_test, y_train, y_val, y_test
                )
            # Handle ensemble model
            elif model == 'ensemble_model':
                train_pred, val_pred, test_pred, model = train_ensemble_model(
                    models, predictions, X_train, X_val, X_test, y_train, y_val, y_test
                )
            else:
                # Select appropriate data based on scaler type
                if 'robust' in scaler_type:
                    X_tr, X_v, X_te = X_train_robust, X_val_robust, X_test_robust
                elif 'standard' in scaler_type:
                    X_tr, X_v, X_te = X_train_standard, X_val_standard, X_test_standard
                elif 'minmax' in scaler_type:
                    X_tr, X_v, X_te = X_train_minmax, X_val_minmax, X_test_minmax
                elif 'bayesian' in scaler_type or 'huber' in scaler_type:
                    X_tr, X_v, X_te = X_train_robust, X_val_robust, X_test_robust
                else:  # tree models - use selected features
                    X_tr, X_v, X_te = X_train_selected, X_val_selected, X_test_selected
                
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
            
            # Debug R² calculation
            if 'Ridge Weak' in name:  # Debug for first Ridge model
                print(f"   DEBUG R² calculation:")
                print(f"   y_test mean: {y_test.mean():.6f}")
                print(f"   test_pred mean: {test_pred.mean():.6f}")
                print(f"   test_pred range: [{test_pred.min():.6f}, {test_pred.max():.6f}]")
                
                # Manual R² calculation
                ss_res = np.sum((y_test - test_pred) ** 2)
                ss_tot = np.sum((y_test - y_test.mean()) ** 2)
                manual_r2 = 1 - (ss_res / ss_tot)
                print(f"   Manual R² calculation: {manual_r2:.6f}")
                print(f"   sklearn R² calculation: {test_r2:.6f}")
                print(f"   SS_res: {ss_res:.6f}, SS_tot: {ss_tot:.6f}")
            
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
        
        # Save scaler and feature columns
        joblib.dump(scaler, 'models/feature_scaler.pkl')
        with open('models/feature_columns.txt', 'w') as f:
            f.write('\n'.join(feature_cols))
        print(f"Saved feature scaler and columns to 'models/' directory")
        
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
    """Load temporal datasets using D0-D3 behavioral features → D4-D33 LTV targets"""
    
    # First try BigQuery temporal approach
    try:
        from google.cloud import bigquery
        return load_temporal_datasets()
        
    except Exception as e:
        print(f"BigQuery temporal loading failed: {e}")
        print("Falling back to CSV file with enhanced synthetic LTV targets...")
        return load_from_csv_with_ltv()

def load_temporal_datasets():
    """Load temporal datasets using only behavioral features (no revenue features)"""
    try:
        from google.cloud import bigquery
    except ImportError:
        print("Warning: google-cloud-bigquery not installed. Skipping BigQuery functionality.")
        return None, None, None, None, None, None, None, None, None, None
    
    print("="*80)
    print("LOADING DATASETS - BEHAVIORAL FEATURES ONLY")
    print("="*80)
    print("Train: 70% of users, D0-D3 behavioral features → D4-D33 LTV targets")
    print("Val: 15% of users, D0-D3 behavioral features → D4-D33 LTV targets")
    print("Test: 15% of users, D0-D3 behavioral features → D4-D33 LTV targets")
    print("Split Method: Random ratio-based (reproducible with seed=42)")
    print("="*80)
    
    PROJECT_ID = "gc-forecasting-dev"
    DATASET = "test_data"
    TABLE = "app_data"
    FEATURE_DAYS = 3  # D0-D3 (4 days)
    PREDICTION_DAYS = 30  # D4-D33 (30 days)
    
    client = bigquery.Client(project=PROJECT_ID)
    table_path = f"{PROJECT_ID}.{DATASET}.{TABLE}"
    
    # Your complete BigQuery query with D0-D3 features → D4-D33 LTV targets
    data_query = f"""
    WITH install_cohort AS (
      SELECT 
          COALESCE(gaid, idfa, android_id, custom_user_id) as user_id,
          DATE(MIN(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64))))) as install_date,
          MIN(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64)))) as install_timestamp
      FROM `{table_path}`
      WHERE DATE(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64)))) >= '2024-06-01'
          AND DATE(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64)))) <= '2024-12-01'
          AND COALESCE(gaid, idfa, android_id, custom_user_id) IS NOT NULL
          AND COALESCE(gaid, idfa, android_id, custom_user_id) != ''
      GROUP BY COALESCE(gaid, idfa, android_id, custom_user_id)
    ),
    
    -- D0-D2 behavioral features (training features)
    behavioral_features AS (
      SELECT 
          ic.user_id,
          ic.install_date,
          
          -- Tutorial completion
          MAX(CASE WHEN DATE(TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64))) = ic.install_date 
                   AND e.name = 'tutorial_completed' THEN 1 ELSE 0 END) as d0_tutorial_completed,
          MAX(CASE WHEN DATE(TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64))) = DATE_ADD(ic.install_date, INTERVAL 1 DAY)
                   AND e.name = 'tutorial_completed' THEN 1 ELSE 0 END) as d1_tutorial_completed,
          MAX(CASE WHEN DATE(TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64))) = DATE_ADD(ic.install_date, INTERVAL 2 DAY)
                   AND e.name = 'tutorial_completed' THEN 1 ELSE 0 END) as d2_tutorial_completed,
          
          -- Session counts by day
          COUNT(DISTINCT CASE WHEN DATE(TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64))) = ic.install_date 
                              AND e.name = 'session_start' THEN e.session_id END) as d0_sessions,
          COUNT(DISTINCT CASE WHEN DATE(TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64))) = DATE_ADD(ic.install_date, INTERVAL 1 DAY)
                              AND e.name = 'session_start' THEN e.session_id END) as d1_sessions,
          COUNT(DISTINCT CASE WHEN DATE(TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64))) = DATE_ADD(ic.install_date, INTERVAL 2 DAY)
                              AND e.name = 'session_start' THEN e.session_id END) as d2_sessions,
          
          -- Level progression
          MAX(CASE WHEN DATE(TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64))) = ic.install_date 
                   AND e.name = 'level_completed' THEN CAST(JSON_EXTRACT_SCALAR(e.arguments, '$.level') AS INT64) END) as d0_max_level,
          MAX(CASE WHEN DATE(TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64))) = DATE_ADD(ic.install_date, INTERVAL 1 DAY)
                   AND e.name = 'level_completed' THEN CAST(JSON_EXTRACT_SCALAR(e.arguments, '$.level') AS INT64) END) as d1_max_level,
          MAX(CASE WHEN DATE(TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64))) = DATE_ADD(ic.install_date, INTERVAL 2 DAY)
                   AND e.name = 'level_completed' THEN CAST(JSON_EXTRACT_SCALAR(e.arguments, '$.level') AS INT64) END) as d2_max_level,
          
          -- Total events per day
          COUNT(CASE WHEN DATE(TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64))) = ic.install_date THEN 1 END) as d0_total_events,
          COUNT(CASE WHEN DATE(TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64))) = DATE_ADD(ic.install_date, INTERVAL 1 DAY) THEN 1 END) as d1_total_events,
          COUNT(CASE WHEN DATE(TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64))) = DATE_ADD(ic.install_date, INTERVAL 2 DAY) THEN 1 END) as d2_total_events,
          
          -- Demographics
          ANY_VALUE(e.country) as country,
          ANY_VALUE(CASE 
              WHEN e.idfa IS NOT NULL OR e.idfa_md5 IS NOT NULL THEN 'ios'
              WHEN e.gaid IS NOT NULL OR e.android_id IS NOT NULL THEN 'android'
              ELSE 'unknown' END) as platform,
          ANY_VALUE(e.os_version) as os_version,
          ANY_VALUE(e.app_version) as app_version
          
      FROM install_cohort ic
      LEFT JOIN `{table_path}` e
          ON ic.user_id = COALESCE(e.gaid, e.idfa, e.android_id, e.custom_user_id)
          AND DATE(TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64))) 
              BETWEEN ic.install_date AND DATE_ADD(ic.install_date, INTERVAL 2 DAY)
      GROUP BY ic.user_id, ic.install_date
    ),
    
    -- D3-D32 revenue targets (what we're predicting)
    revenue_targets AS (
      SELECT 
          ic.user_id,
          SUM(COALESCE(e.converted_revenue, 0)) as ltv_target
      FROM install_cohort ic
      LEFT JOIN `{table_path}` e
          ON ic.user_id = COALESCE(e.gaid, e.idfa, e.android_id, e.custom_user_id)
          AND DATE(TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64))) 
              BETWEEN DATE_ADD(ic.install_date, INTERVAL 3 DAY) 
              AND DATE_ADD(ic.install_date, INTERVAL 32 DAY)
          AND COALESCE(e.converted_revenue, 0) > 0
      GROUP BY ic.user_id
    )
    
    SELECT 
        bf.*,
        COALESCE(rt.ltv_target, 0) as ltv_target
    FROM behavioral_features bf
    LEFT JOIN revenue_targets rt ON bf.user_id = rt.user_id
    WHERE bf.d0_total_events > 0  -- Ensure user was active on D0
    """
    
    print("Executing BigQuery query...")
    data = client.query(query).to_dataframe()
    
    print(f"Loaded BigQuery data with shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Handle missing values
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(0)
    
    # Fill missing categorical columns
    categorical_cols = ['country', 'platform', 'os_version', 'app_version']
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].fillna('unknown')
    
    print(f"LTV Target distribution from actual revenue:")
    print(f"  Mean: ${data['ltv_target'].mean():.2f}")
    print(f"  Median: ${data['ltv_target'].median():.2f}")
    print(f"  75th percentile: ${data['ltv_target'].quantile(0.75):.2f}")
    print(f"  95th percentile: ${data['ltv_target'].quantile(0.95):.2f}")
    print(f"  Max: ${data['ltv_target'].max():.2f}")
    print(f"  Paying users: {(data['ltv_target'] > 0).sum()}/{len(data)} ({(data['ltv_target'] > 0).mean()*100:.1f}%)")
    
    # Prepare features and target
    target_col = 'ltv_target'
    feature_cols = [col for col in data.columns if col not in ['user_id', 'install_date', target_col]]
    
    X = data[feature_cols].copy()
    y = data[target_col].values
    
    print(f"Features: {len(feature_cols)} columns")
    print(f"Target: {target_col}")
    
    # Train/validation/test split (70/15/15) - no stratification for regression
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42  # 0.176 ≈ 0.15/0.85
    )
    
    # Create raw data splits for reference
    train_indices = X_train.index
    val_indices = X_val.index  
    test_indices = X_test.index
    
    train_data = data.iloc[train_indices].copy()
    val_data = data.iloc[val_indices].copy()
    test_data = data.iloc[test_indices].copy()
    
    print(f"Data split completed:")
    print(f"  Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(data)*100:.1f}%)")
    print(f"  Val: {X_val.shape[0]} samples ({X_val.shape[0]/len(data)*100:.1f}%)")
    print(f"  Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(data)*100:.1f}%)")
    
    return train_data, val_data, test_data, X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

def load_from_csv_with_ltv():
    """Load from CSV file and create realistic LTV targets based on engagement"""
    import os
    
    # Load the main dataset
    data_file = 'data/d0_d2_data.csv'
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!")
        return None, None, None, None, None, None, None, None, None, None
    
    data = pd.read_csv(data_file)
    print(f"Loaded CSV data with shape: {data.shape}")
    
    # Map CSV columns to BigQuery-like columns for consistency
    column_mapping = {
        'd0_tutorial_completed_flag': 'd0_tutorial_completed',
        'd1_tutorial_completed_flag': 'd1_tutorial_completed', 
        'd2_tutorial_completed_flag': 'd2_tutorial_completed',
        'd0_total_sessions': 'd0_sessions',
        'd1_total_sessions': 'd1_sessions',
        'd2_total_sessions': 'd2_sessions',
        'd0_levels_count': 'd0_max_level',
        'd1_levels_count': 'd1_max_level', 
        'd2_levels_count': 'd2_max_level',
        # Calculate total events as proxy - this is approximate
        'd0_total_playtime_mins': 'd0_total_events',  # Use playtime as proxy for events
        'd1_total_playtime_mins': 'd1_total_events',
        'd2_total_playtime_mins': 'd2_total_events'
    }
    
    # Rename columns to match BigQuery format
    for old_col, new_col in column_mapping.items():
        if old_col in data.columns:
            data[new_col] = data[old_col]
    
    # Scale playtime to events (rough approximation: 1 min = 10 events)
    for col in ['d0_total_events', 'd1_total_events', 'd2_total_events']:
        if col in data.columns:
            data[col] = (data[col].fillna(0) * 10).astype(int)
    
    # Create realistic LTV targets based on engagement patterns
    print("Creating realistic LTV targets based on engagement patterns...")
    
    # Fill NaN values first
    numeric_cols = ['d0_tutorial_completed', 'd1_tutorial_completed', 'd2_tutorial_completed',
                   'd0_sessions', 'd1_sessions', 'd2_sessions', 
                   'd0_max_level', 'd1_max_level', 'd2_max_level',
                   'd0_total_events', 'd1_total_events', 'd2_total_events']
    
    for col in numeric_cols:
        if col in data.columns:
            data[col] = data[col].fillna(0)
    
    # Create sophisticated LTV model based on behavioral patterns
    # Tutorial completion contributes to conversion likelihood
    tutorial_score = (
        data.get('d0_tutorial_completed', 0) * 0.3 +
        data.get('d1_tutorial_completed', 0) * 0.4 +
        data.get('d2_tutorial_completed', 0) * 0.5
    )
    
    # Session engagement (consistency and volume)
    session_score = (
        data.get('d0_sessions', 0) * 1.0 +
        data.get('d1_sessions', 0) * 1.2 +
        data.get('d2_sessions', 0) * 1.5
    )
    
    # Level progression (skill and engagement)
    level_score = (
        data.get('d0_max_level', 0) * 0.5 +
        data.get('d1_max_level', 0) * 0.7 +
        data.get('d2_max_level', 0) * 1.0
    )
    
    # Total activity (overall engagement)
    activity_score = (
        data.get('d0_total_events', 0) * 0.01 +
        data.get('d1_total_events', 0) * 0.015 +
        data.get('d2_total_events', 0) * 0.02
    )
    
    # Combine scores with realistic monetization rates
    base_ltv = (
        tutorial_score * 2.0 +      # Tutorial completers more likely to spend
        session_score * 0.5 +       # More sessions = more opportunities
        level_score * 1.0 +         # Progression indicates engagement
        activity_score              # Overall activity
    )
    
    # Apply realistic conversion rates and spending patterns
    # Only ~3-5% of mobile game users typically spend money
    conversion_threshold = np.percentile(base_ltv, 95)  # Top 5% convert
    
    data['ltv_target'] = 0.0
    converters_mask = base_ltv >= conversion_threshold
    
    # For converters, create realistic spending distribution
    n_converters = converters_mask.sum()
    if n_converters > 0:
        # Most spenders spend small amounts, few spend large amounts (Pareto distribution)
        spending_amounts = np.random.pareto(1.2, n_converters) * 10  # Scale to reasonable amounts
        spending_amounts = np.clip(spending_amounts, 0.99, 500)  # Min $0.99, max $500
        
        # Apply spending based on engagement level (higher engagement = higher spending)
        engagement_multiplier = base_ltv[converters_mask] / base_ltv[converters_mask].max()
        spending_amounts = spending_amounts * (0.5 + 1.5 * engagement_multiplier)  # 0.5x to 2x multiplier
        
        data.loc[converters_mask, 'ltv_target'] = spending_amounts
    
    # Add some randomness for realism but preserve the behavioral correlation
    noise = np.random.normal(0, 0.5, len(data))
    data['ltv_target'] = np.maximum(0, data['ltv_target'] + noise)
    
    print(f"Created realistic LTV targets:")
    print(f"  Mean: ${data['ltv_target'].mean():.2f}")
    print(f"  Median: ${data['ltv_target'].median():.2f}")
    print(f"  75th percentile: ${data['ltv_target'].quantile(0.75):.2f}")
    print(f"  95th percentile: ${data['ltv_target'].quantile(0.95):.2f}")
    print(f"  Max: ${data['ltv_target'].max():.2f}")
    print(f"  Paying users: {(data['ltv_target'] > 0).sum()}/{len(data)} ({(data['ltv_target'] > 0).mean()*100:.1f}%)")
    
    # Prepare features and target (remove original columns that were mapped)
    target_col = 'ltv_target'
    exclude_cols = ['device_id', 'install_date', target_col, 'is_payer_30d'] + list(column_mapping.keys())
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    X = data[feature_cols].copy()
    y = data[target_col].values
    
    print(f"Features: {len(feature_cols)} columns")
    print(f"Target: {target_col}")
    
    # Train/validation/test split (70/15/15) - no stratification for regression
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42  # 0.176 ≈ 0.15/0.85
    )
    
    # Create raw data splits for reference
    train_indices = X_train.index
    val_indices = X_val.index  
    test_indices = X_test.index
    
    train_data = data.iloc[train_indices].copy()
    val_data = data.iloc[val_indices].copy()
    test_data = data.iloc[test_indices].copy()
    
    print(f"Data split completed:")
    print(f"  Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(data)*100:.1f}%)")
    print(f"  Val: {X_val.shape[0]} samples ({X_val.shape[0]/len(data)*100:.1f}%)")
    print(f"  Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(data)*100:.1f}%)")
    
    return train_data, val_data, test_data, X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

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
    
    # Load baseline metrics and perform accuracy loss analysis
    load_baseline_metrics()
    
    if BASELINE_METRICS and len(PLOT_DATA) > 0:
        print("\nPerforming accuracy loss analysis...")
        accuracy_loss = plot_accuracy_loss_analysis()
        
        if accuracy_loss and accuracy_loss['needs_retraining']:
            print("\n🔄 RECOMMENDATION: Consider updating baseline metrics if current model is performing well on new data patterns")
    elif len(PLOT_DATA) > 0:
        # Save current run as baseline if no baseline exists
        print("\nSaving current model performance as baseline...")
        model_results = []
        for data in PLOT_DATA:
            model_results.append({
                'model_name': data['model_name'],
                'test_r2': data['test_r2'],
                'test_rmse': data['test_rmse'],
                'val_r2': data['val_r2'],
                'val_rmse': data['val_rmse']
            })
        save_baseline_metrics(model_results)
    
    # Create comprehensive plots
    if len(PLOT_DATA) > 0:
        print("\nCreating visualizations...")
        create_comprehensive_plots()
        
        # Create enhanced validation plots
        create_enhanced_validation_plots()
        create_roc_pr_curves()
        create_individual_model_plots()
    
    print("Pipeline completed successfully!")
    return models, predictions, comparison_df, test_data

def update_baseline_from_current():
    """Update baseline metrics from current model results - useful when model improves"""
    import pickle
    import os
    
    if not os.path.exists('baseline_metrics.pkl'):
        print("No baseline metrics file found.")
        return
    
    # Load current results
    if not PLOT_DATA:
        print("No current model results available. Run the pipeline first.")
        return
    
    model_results = []
    for data in PLOT_DATA:
        model_results.append({
            'model_name': data['model_name'],
            'test_r2': data['test_r2'],
            'test_rmse': data['test_rmse'],
            'val_r2': data['val_r2'],
            'val_rmse': data['val_rmse']
        })
    
    print("Updating baseline metrics with current best model performance...")
    save_baseline_metrics(model_results)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "update_baseline":
        print("Updating baseline metrics...")
        # Load existing PLOT_DATA if available
        try:
            models, predictions, comparison_df, data = main()
            update_baseline_from_current()
        except Exception as e:
            print(f"Error updating baseline: {e}")
    else:
        models, predictions, comparison_df, data = main()
