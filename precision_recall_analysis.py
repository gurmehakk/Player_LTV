"""
Enhanced Classification Analysis with Precision-Recall Curves and Metrics at Specific Recall Levels
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import (
    precision_recall_curve, roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score,
    average_precision_score
)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_extractor import DataExtractor
from src.feature_engineer import FeatureEngineer
from src.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_metrics_at_recall(y_true, y_prob, target_recall=0.9):
    """Find precision and other metrics at a specific recall level"""
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Find the index where recall is closest to target_recall
    recall_diff = np.abs(recall - target_recall)
    best_idx = np.argmin(recall_diff)
    
    # Get the threshold and metrics at that point
    if best_idx < len(thresholds):
        threshold = thresholds[best_idx]
        actual_recall = recall[best_idx]
        precision_at_recall = precision[best_idx]
    else:
        # Edge case: use the last available point
        threshold = thresholds[-1] if len(thresholds) > 0 else 0.5
        actual_recall = recall[best_idx]
        precision_at_recall = precision[best_idx]
    
    # Calculate other metrics at this threshold
    y_pred = (y_prob >= threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        'threshold': threshold,
        'target_recall': target_recall,
        'actual_recall': actual_recall,
        'precision': precision_at_recall,
        'accuracy': accuracy,
        'f1_score': f1,
        'recall_diff': abs(actual_recall - target_recall)
    }


def analyze_precision_recall_curves():
    """Comprehensive precision-recall analysis for all classification targets"""
    
    logger.info("🎯 PRECISION-RECALL ANALYSIS FOR LTV CLASSIFICATION")
    logger.info("=" * 60)
    
    # 1. Load and prepare data
    logger.info("Step 1: Loading data...")
    config = Config()
    extractor = DataExtractor(project_id=config.project_id, credentials_path=config.credentials_path)
    raw_data = extractor.extract_player_data()
    
    feature_engineer = FeatureEngineer()
    engineered_data = feature_engineer.create_features(raw_data)
    
    # Get features and targets
    X = feature_engineer.get_feature_matrix(engineered_data, apply_smote=False)
    ltv_values = engineered_data['ltv_target'].values
    historical_revenue = engineered_data['total_revenue'].values
    
    logger.info(f"Dataset: {len(X):,} players, {len(feature_engineer.feature_columns)} features")
    
    # 2. Create classification targets
    logger.info("\\nStep 2: Creating classification targets...")
    
    payers = ltv_values[ltv_values > 0]
    
    if len(payers) > 0:
        thresholds = {
            'will_spend': 0.01,
            'low_spender': np.percentile(payers, 25),
            'medium_spender': np.percentile(payers, 75), 
            'high_spender': np.percentile(payers, 90)
        }
    else:
        thresholds = {
            'will_spend': 0.01,
            'low_spender': 0.10, 
            'medium_spender': 1.00,
            'high_spender': 5.00
        }
    
    targets = {
        'will_spend': (ltv_values >= thresholds['will_spend']).astype(int),
        'low_spender': (ltv_values >= thresholds['low_spender']).astype(int),
        'medium_spender': (ltv_values >= thresholds['medium_spender']).astype(int),
        'high_spender': (ltv_values >= thresholds['high_spender']).astype(int)
    }
    
    logger.info("Classification thresholds:")
    for name, threshold in thresholds.items():
        positive_rate = np.mean(targets[name])
        logger.info(f"  {name}: ${threshold:.3f} ({positive_rate*100:.1f}% positive)")
    
    # 3. Train models and analyze precision-recall
    logger.info("\\nStep 3: Training models and analyzing precision-recall...")
    
    results_summary = []
    pr_curves = {}
    
    # Create subplots for precision-recall curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    fig.suptitle('Precision-Recall Curves for LTV Classification Models', fontsize=16, fontweight='bold')
    
    for idx, (target_name, y) in enumerate(targets.items()):
        if np.sum(y) < 10:  # Skip if not enough positive examples
            logger.warning(f"Skipping {target_name} - insufficient positive examples ({np.sum(y)})")
            continue
        
        logger.info(f"\\nAnalyzing {target_name}...")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train LightGBM model
        model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=200,
            learning_rate=0.1,
            num_leaves=31,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=42,
            verbose=-1,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
        auc_pr = average_precision_score(y_test, y_prob)
        auc_roc = roc_auc_score(y_test, y_prob)
        
        # Find metrics at different recall levels
        recall_levels = [0.5, 0.7, 0.8, 0.9, 0.95]
        metrics_at_recalls = {}
        
        for target_recall in recall_levels:
            metrics = find_metrics_at_recall(y_test, y_prob, target_recall)
            metrics_at_recalls[target_recall] = metrics
        
        # Store results
        pr_curves[target_name] = {
            'precision': precision,
            'recall': recall,
            'thresholds': pr_thresholds,
            'auc_pr': auc_pr,
            'auc_roc': auc_roc,
            'metrics_at_recalls': metrics_at_recalls,
            'y_test': y_test,
            'y_prob': y_prob
        }
        
        # Plot precision-recall curve
        ax = axes[idx]
        ax.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {auc_pr:.3f})')
        
        # Add points for specific recall levels
        for target_recall in [0.8, 0.9]:
            if target_recall in metrics_at_recalls:
                metrics = metrics_at_recalls[target_recall]
                ax.plot(metrics['actual_recall'], metrics['precision'], 
                       'ro', markersize=8, 
                       label=f'Recall {target_recall:.1f}: Precision {metrics["precision"]:.3f}')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'{target_name.replace("_", " ").title()} Classification')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Log detailed results
        logger.info(f"  AUC-PR: {auc_pr:.4f}, AUC-ROC: {auc_roc:.4f}")
        logger.info(f"  Metrics at specific recall levels:")
        for target_recall in recall_levels:
            if target_recall in metrics_at_recalls:
                m = metrics_at_recalls[target_recall]
                logger.info(f"    Recall {target_recall:.1f}: Precision={m['precision']:.3f}, "
                          f"Threshold={m['threshold']:.3f}, F1={m['f1_score']:.3f}")
        
        # Add to summary
        results_summary.append({
            'target': target_name,
            'auc_pr': auc_pr,
            'auc_roc': auc_roc,
            'precision_at_90_recall': metrics_at_recalls.get(0.9, {}).get('precision', 0),
            'threshold_at_90_recall': metrics_at_recalls.get(0.9, {}).get('threshold', 0),
            'f1_at_90_recall': metrics_at_recalls.get(0.9, {}).get('f1_score', 0),
            'precision_at_80_recall': metrics_at_recalls.get(0.8, {}).get('precision', 0),
            'positive_rate': np.mean(y_test)
        })
    
    plt.tight_layout()
    plt.savefig('output/precision_recall_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Create metrics comparison matrix
    logger.info("\\nStep 4: Creating metrics comparison matrix...")
    
    if len(results_summary) > 0:
        results_df = pd.DataFrame(results_summary)
        
        # Create heatmap of metrics at different recall levels
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Metrics at 0.9 recall
        metrics_90 = results_df[['target', 'precision_at_90_recall', 'f1_at_90_recall', 'threshold_at_90_recall']].copy()
        metrics_90.columns = ['Target', 'Precision', 'F1 Score', 'Threshold']
        metrics_90 = metrics_90.set_index('Target')
        
        sns.heatmap(metrics_90, annot=True, fmt='.3f', cmap='RdYlGn', 
                   ax=axes[0], cbar_kws={'label': 'Score'})
        axes[0].set_title('Metrics at 90% Recall')
        
        # AUC comparison
        auc_comparison = results_df[['target', 'auc_pr', 'auc_roc']].copy()
        auc_comparison.columns = ['Target', 'AUC-PR', 'AUC-ROC']
        auc_comparison = auc_comparison.set_index('Target')
        
        sns.heatmap(auc_comparison, annot=True, fmt='.3f', cmap='Blues',
                   ax=axes[1], cbar_kws={'label': 'AUC Score'})
        axes[1].set_title('AUC Comparison')
        
        plt.tight_layout()
        plt.savefig('output/metrics_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 5. Create detailed metrics table
        logger.info("\\nStep 5: Creating detailed metrics table...")
        
        # Create comprehensive metrics table
        detailed_metrics = []
        
        for target_name, pr_data in pr_curves.items():
            metrics_at_recalls = pr_data['metrics_at_recalls']
            
            base_row = {
                'target': target_name,
                'auc_pr': pr_data['auc_pr'],
                'auc_roc': pr_data['auc_roc'],
                'threshold_dollar': thresholds[target_name]
            }
            
            # Add metrics for each recall level
            for recall_level in [0.5, 0.7, 0.8, 0.9, 0.95]:
                if recall_level in metrics_at_recalls:
                    m = metrics_at_recalls[recall_level]
                    base_row.update({
                        f'precision_at_{int(recall_level*100)}': m['precision'],
                        f'threshold_at_{int(recall_level*100)}': m['threshold'],
                        f'f1_at_{int(recall_level*100)}': m['f1_score'],
                        f'accuracy_at_{int(recall_level*100)}': m['accuracy']
                    })
            
            detailed_metrics.append(base_row)
        
        detailed_df = pd.DataFrame(detailed_metrics)
        detailed_df.to_csv('output/detailed_precision_recall_metrics.csv', index=False)
        
        # 6. Print summary results
        logger.info("\\n" + "=" * 60)
        logger.info("📊 PRECISION-RECALL ANALYSIS RESULTS")
        logger.info("=" * 60)
        
        logger.info("\\n🎯 METRICS AT 90% RECALL:")
        for _, row in results_df.iterrows():
            target = row['target'].replace('_', ' ').title()
            precision = row['precision_at_90_recall']
            threshold = row['threshold_at_90_recall']
            f1 = row['f1_at_90_recall']
            
            if precision > 0:
                logger.info(f"  {target:15} | Precision: {precision:.3f} | Threshold: {threshold:.3f} | F1: {f1:.3f}")
            else:
                logger.info(f"  {target:15} | Could not achieve 90% recall")
        
        logger.info("\\n📈 OVERALL AUC PERFORMANCE:")
        for _, row in results_df.iterrows():
            target = row['target'].replace('_', ' ').title()
            auc_pr = row['auc_pr']
            auc_roc = row['auc_roc']
            logger.info(f"  {target:15} | AUC-PR: {auc_pr:.3f} | AUC-ROC: {auc_roc:.3f}")
        
        # 7. Business recommendations
        logger.info("\\n" + "=" * 60)
        logger.info("💡 BUSINESS RECOMMENDATIONS")
        logger.info("=" * 60)
        
        # Find best model for high recall
        best_90_recall = results_df.loc[results_df['precision_at_90_recall'].idxmax()]
        
        if best_90_recall['precision_at_90_recall'] > 0.1:
            logger.info(f"✅ HIGH RECALL STRATEGY:")
            logger.info(f"   Best model: {best_90_recall['target'].replace('_', ' ').title()}")
            logger.info(f"   At 90% recall: {best_90_recall['precision_at_90_recall']:.1%} precision")
            logger.info(f"   Use case: Broad targeting campaign (catch most potential spenders)")
            logger.info(f"   Expected false positive rate: {(1-best_90_recall['precision_at_90_recall'])*100:.1f}%")
        
        # Find best model for balanced performance
        best_balanced = results_df.loc[results_df['auc_pr'].idxmax()]
        
        logger.info(f"\\n✅ BALANCED STRATEGY:")
        logger.info(f"   Best model: {best_balanced['target'].replace('_', ' ').title()}")
        logger.info(f"   AUC-PR: {best_balanced['auc_pr']:.3f}")
        logger.info(f"   At 80% recall: {best_balanced['precision_at_80_recall']:.1%} precision")
        logger.info(f"   Use case: Efficient targeting with good precision-recall balance")
        
        # Save all results
        logger.info("\\n📊 Results saved:")
        logger.info("  • output/precision_recall_curves.png")
        logger.info("  • output/metrics_heatmap.png") 
        logger.info("  • output/detailed_precision_recall_metrics.csv")
        
        return {
            'results_summary': results_df,
            'detailed_metrics': detailed_df,
            'pr_curves': pr_curves,
            'thresholds': thresholds
        }
    
    else:
        logger.warning("No valid classification targets found!")
        return None


if __name__ == "__main__":
    analyze_precision_recall_curves()