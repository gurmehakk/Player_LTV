"""
FINAL SOLUTION: Classification-Based LTV Prediction
Addresses the fundamental regression challenges by switching to classification
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_extractor import DataExtractor
from src.feature_engineer import FeatureEngineer
from src.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LTVClassificationModel:
    """Classification-based LTV prediction model"""
    
    def __init__(self, model_type='lgb'):
        self.model_type = model_type
        self.model = None
        self.thresholds = {}
        self.training_stats = {}
        
    def prepare_classification_targets(self, ltv_values):
        """Create meaningful classification targets from LTV values"""
        
        # Define meaningful spending thresholds
        payers = ltv_values[ltv_values > 0]
        
        if len(payers) == 0:
            # Fallback thresholds
            self.thresholds = {
                'will_spend': 0.01,
                'low_spender': 0.10, 
                'medium_spender': 1.00,
                'high_spender': 5.00
            }
        else:
            # Data-driven thresholds
            self.thresholds = {
                'will_spend': 0.01,
                'low_spender': np.percentile(payers, 25),
                'medium_spender': np.percentile(payers, 75), 
                'high_spender': np.percentile(payers, 90)
            }
        
        logger.info("Classification thresholds:")
        for name, threshold in self.thresholds.items():
            logger.info(f"  {name}: ${threshold:.3f}")
        
        # Create classification targets
        targets = {
            'will_spend': (ltv_values >= self.thresholds['will_spend']).astype(int),
            'low_spender': (ltv_values >= self.thresholds['low_spender']).astype(int),
            'medium_spender': (ltv_values >= self.thresholds['medium_spender']).astype(int),
            'high_spender': (ltv_values >= self.thresholds['high_spender']).astype(int)
        }
        
        # Log class distributions
        logger.info("Class distributions:")
        for name, target in targets.items():
            positive_rate = np.mean(target)
            logger.info(f"  {name}: {positive_rate*100:.1f}% positive ({np.sum(target):,}/{len(target):,})")
        
        return targets
    
    def train_classification_models(self, X, targets):
        """Train classification models for each target"""
        
        models = {}
        
        for target_name, y in targets.items():
            logger.info(f"\\nTraining {target_name} classifier...")
            
            # Skip if not enough positive examples
            if np.sum(y) < 10:
                logger.warning(f"Skipping {target_name} - insufficient positive examples ({np.sum(y)})")
                continue
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            if self.model_type == 'lgb':
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
            else:  # Random Forest
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=10,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            
            metrics = {
                'auc': roc_auc_score(y_test, y_pred_proba),
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'positive_rate': np.mean(y_train)
            }
            
            logger.info(f"  AUC: {metrics['auc']:.4f}")
            logger.info(f"  F1 Score: {metrics['f1']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            
            models[target_name] = {
                'model': model,
                'metrics': metrics,
                'test_data': (X_test, y_test)
            }
        
        return models
    
    def create_business_rules(self, ltv_values, targets):
        """Create simple business rules for comparison"""
        
        logger.info("\\nCreating business rules baseline...")
        
        # Simple rule: historical spenders are likely to spend again
        historical_revenue = self.historical_revenue if hasattr(self, 'historical_revenue') else np.zeros_like(ltv_values)
        historical_payers = historical_revenue > 0
        
        rules = {}
        
        for target_name, y_true in targets.items():
            if np.sum(y_true) < 10:
                continue
                
            # Rule: Historical payers are more likely to be future spenders
            y_pred_rule = historical_payers.astype(int)
            
            if len(np.unique(y_true)) > 1 and len(np.unique(y_pred_rule)) > 1:
                rule_auc = roc_auc_score(y_true, y_pred_rule)
                rule_precision = precision_score(y_true, y_pred_rule, zero_division=0)
                rule_recall = recall_score(y_true, y_pred_rule, zero_division=0)
                rule_f1 = f1_score(y_true, y_pred_rule, zero_division=0)
                
                rules[target_name] = {
                    'auc': rule_auc,
                    'precision': rule_precision,
                    'recall': rule_recall,
                    'f1': rule_f1
                }
                
                logger.info(f"  {target_name} rule - AUC: {rule_auc:.4f}, F1: {rule_f1:.4f}")
        
        return rules


def run_classification_solution():
    """Run the complete classification-based solution"""
    
    logger.info("🎯 CLASSIFICATION-BASED LTV PREDICTION SOLUTION")
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
    
    # 2. Create classification model
    logger.info("\\nStep 2: Setting up classification models...")
    classifier = LTVClassificationModel(model_type='lgb')
    classifier.historical_revenue = historical_revenue  # For business rules
    
    # 3. Prepare classification targets
    logger.info("\\nStep 3: Creating classification targets...")
    targets = classifier.prepare_classification_targets(ltv_values)
    
    # 4. Train models
    logger.info("\\nStep 4: Training classification models...")
    trained_models = classifier.train_classification_models(X, targets)
    
    # 5. Business rules baseline
    logger.info("\\nStep 5: Creating business rules baseline...")
    business_rules = classifier.create_business_rules(ltv_values, targets)
    
    # 6. Compare approaches
    logger.info("\\n" + "=" * 60)
    logger.info("📊 RESULTS COMPARISON")
    logger.info("=" * 60)
    
    comparison_results = []
    
    for target_name in targets.keys():
        if target_name in trained_models and target_name in business_rules:
            ml_metrics = trained_models[target_name]['metrics']
            rule_metrics = business_rules[target_name]
            
            logger.info(f"\\n{target_name.upper()} PREDICTION:")
            logger.info(f"  Machine Learning Model:")
            logger.info(f"    AUC: {ml_metrics['auc']:.4f}")
            logger.info(f"    F1:  {ml_metrics['f1']:.4f}")
            logger.info(f"    Precision: {ml_metrics['precision']:.4f}")
            logger.info(f"    Recall: {ml_metrics['recall']:.4f}")
            
            logger.info(f"  Business Rule Baseline:")
            logger.info(f"    AUC: {rule_metrics['auc']:.4f}")
            logger.info(f"    F1:  {rule_metrics['f1']:.4f}")
            logger.info(f"    Precision: {rule_metrics['precision']:.4f}")
            logger.info(f"    Recall: {rule_metrics['recall']:.4f}")
            
            improvement = ml_metrics['auc'] - rule_metrics['auc']
            logger.info(f"  ML Improvement: {improvement:+.4f} AUC points")
            
            comparison_results.append({
                'target': target_name,
                'ml_auc': ml_metrics['auc'],
                'rule_auc': rule_metrics['auc'],
                'improvement': improvement,
                'ml_f1': ml_metrics['f1'],
                'feasible': ml_metrics['auc'] > 0.6
            })
    
    # 7. Final recommendations
    logger.info("\\n" + "=" * 60)
    logger.info("🎯 FINAL RECOMMENDATIONS")
    logger.info("=" * 60)
    
    feasible_targets = [r for r in comparison_results if r['feasible']]
    
    if len(feasible_targets) > 0:
        logger.info("✅ SUCCESSFUL CLASSIFICATION MODELS:")
        for result in feasible_targets:
            logger.info(f"  • {result['target']}: AUC = {result['ml_auc']:.3f} (improvement: {result['improvement']:+.3f})")
        
        logger.info("\\n💡 IMPLEMENTATION STRATEGY:")
        logger.info("  1. Deploy classification models instead of regression")
        logger.info("  2. Focus on business-actionable predictions")
        logger.info("  3. Use ensemble of multiple thresholds")
        logger.info("  4. A/B test against business rules")
        
        # Best model recommendation
        best_result = max(feasible_targets, key=lambda x: x['ml_auc'])
        logger.info(f"\\n🏆 RECOMMENDED PRIMARY MODEL: {best_result['target']}")
        logger.info(f"   Expected AUC: {best_result['ml_auc']:.3f}")
        logger.info(f"   Business Impact: Can identify top spenders with {best_result['ml_auc']*100:.1f}% accuracy")
        
    else:
        logger.info("⚠️  CLASSIFICATION CHALLENGES IDENTIFIED:")
        logger.info("  • All models show limited predictive power")
        logger.info("  • 3-day observation period may be insufficient")
        logger.info("  • Consider collecting additional features")
        
        logger.info("\\n🎯 ALTERNATIVE APPROACHES:")
        logger.info("  1. Reduce prediction horizon to 7-14 days")
        logger.info("  2. Focus on immediate next-session predictions")
        logger.info("  3. Implement cohort-based analysis")
        logger.info("  4. Use engagement metrics instead of revenue")
    
    # 8. Save results
    logger.info("\\nStep 6: Saving results...")
    
    results_summary = pd.DataFrame(comparison_results)
    results_summary.to_csv('output/classification_results.csv', index=False)
    
    # Save detailed model info
    model_details = []
    for target_name, model_info in trained_models.items():
        metrics = model_info['metrics']
        model_details.append({
            'target': target_name,
            'model_type': 'LightGBM',
            'auc': metrics['auc'],
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'train_samples': metrics['train_size'],
            'test_samples': metrics['test_size'],
            'positive_rate': metrics['positive_rate']
        })
    
    model_details_df = pd.DataFrame(model_details)
    model_details_df.to_csv('output/classification_model_details.csv', index=False)
    
    logger.info("📊 Results saved:")
    logger.info("  • output/classification_results.csv")
    logger.info("  • output/classification_model_details.csv")
    
    return {
        'trained_models': trained_models,
        'comparison_results': comparison_results,
        'feasible_targets': feasible_targets
    }


if __name__ == "__main__":
    run_classification_solution()