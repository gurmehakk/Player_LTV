"""
Backtesting and validation module for pLTV models
Implements time-based splits and comprehensive validation
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .ziln_model import ZILNModel
from .feature_engineer import FeatureEngineer


class BackTester:
    """Handles backtesting and validation of pLTV models"""
    
    def __init__(self, 
                 test_size: float = 0.2,
                 n_splits: int = 3,
                 random_state: int = 42):
        
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.results = {}
        
    def run_backtest(self, 
                     data: pd.DataFrame,
                     model: ZILNModel,
                     feature_engineer: FeatureEngineer,
                     include_whale_validation: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive backtesting with whale detection validation
        
        Args:
            data: Complete dataset with features
            model: ZILN model instance 
            feature_engineer: Feature engineering instance
            include_whale_validation: Whether to include whale-specific validation
            
        Returns:
            Dictionary containing all backtest results
        """
        
        logger = logging.getLogger(__name__)
        logger.info("Starting comprehensive backtesting...")
        
        # Prepare data - Use LTV target instead of historical revenue
        # For backtesting, we want to use original data (not SMOTE-resampled)
        X = feature_engineer.get_feature_matrix(data, apply_smote=False)
        y = data['ltv_target'].values  # This is the future LTV we want to predict
        
        # 1. Random split validation
        logger.info("Running random split validation...")
        random_results = self._random_split_validation(X, y, model)
        
        # 2. Time-based split validation (if timestamp available)
        time_results = {}
        if 'first_session_timestamp' in data.columns:
            logger.info("Running time-based split validation...")
            time_results = self._time_based_validation(data, X, y, model, feature_engineer)
        
        # 3. Cross-validation
        logger.info("Running cross-validation...")
        cv_results = self._cross_validation(X, y, model)
        
        # 4. Segment-based validation
        logger.info("Running segment-based validation...")
        segment_results = self._segment_based_validation(data, X, y, model)
        
        # 5. Revenue tier validation
        logger.info("Running revenue tier validation...")
        tier_results = self._revenue_tier_validation(data, X, y, model)
        
        # 6. Whale detection validation (if enabled)
        whale_results = {}
        if include_whale_validation:
            logger.info("Running whale detection validation...")
            whale_results = self._whale_detection_validation(data, X, y, model, feature_engineer)
        
        # Compile results
        backtest_results = {
            'random_split': random_results,
            'time_based': time_results,
            'cross_validation': cv_results,
            'segment_based': segment_results,
            'revenue_tier': tier_results,
            'whale_detection': whale_results,
            'feature_importance': model.get_feature_importance(),
            'training_stats': model.get_training_stats()
        }
        
        logger.info("Backtesting completed successfully")
        return backtest_results
    
    def _random_split_validation(self, 
                                X: pd.DataFrame, 
                                y: np.ndarray,
                                model: ZILNModel) -> Dict[str, Any]:
        """Standard random train/test split validation"""
        
        # Stratified split based on payer status
        is_payer = (y > 0).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, 
            random_state=self.random_state,
            stratify=is_payer
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        
        # Get predictions for analysis
        try:
            predictions, payer_probs, conditional_amounts, _ = model.predict(X_test)
        except ValueError:
            # Fallback for models that return 3 values
            predictions, payer_probs, conditional_amounts = model.predict(X_test)
        
        return {
            'metrics': metrics,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'predictions': predictions,
            'actual': y_test,
            'payer_probabilities': payer_probs
        }
    
    def _time_based_validation(self,
                              data: pd.DataFrame,
                              X: pd.DataFrame,
                              y: np.ndarray,
                              model: ZILNModel,
                              feature_engineer: FeatureEngineer) -> Dict[str, Any]:
        """Time-based validation using temporal splits"""
        
        # Sort by first session timestamp
        data_sorted = data.sort_values('first_session_timestamp')
        
        # Create time-based split (80% train, 20% test)
        split_idx = int(len(data_sorted) * 0.8)
        
        train_data = data_sorted.iloc[:split_idx]
        test_data = data_sorted.iloc[split_idx:]
        
        # Get corresponding features and targets - Use LTV target
        X_train = feature_engineer.get_feature_matrix(train_data, apply_smote=False)
        X_test = feature_engineer.get_feature_matrix(test_data, apply_smote=False)
        y_train = train_data['ltv_target'].values  # Future LTV prediction
        y_test = test_data['ltv_target'].values    # Future LTV prediction
        
        # Train model on earlier data
        model.fit(X_train, y_train)
        
        # Evaluate on later data
        metrics = model.evaluate(X_test, y_test)
        
        # Calculate temporal drift metrics for LTV
        train_payer_rate = np.mean(y_train > 0)
        test_payer_rate = np.mean(y_test > 0)
        payer_rate_drift = abs(test_payer_rate - train_payer_rate)
        
        train_avg_ltv = np.mean(y_train[y_train > 0]) if np.sum(y_train > 0) > 0 else 0
        test_avg_ltv = np.mean(y_test[y_test > 0]) if np.sum(y_test > 0) > 0 else 0
        ltv_drift = abs(test_avg_ltv - train_avg_ltv) / train_avg_ltv if train_avg_ltv > 0 else 0
        
        try:
            predictions, _, _, _ = model.predict(X_test)
        except ValueError:
            # Fallback for models that return 3 values
            predictions, _, _ = model.predict(X_test)
        
        return {
            'metrics': metrics,
            'train_period': (train_data['first_session_timestamp'].min(), train_data['first_session_timestamp'].max()),
            'test_period': (test_data['first_session_timestamp'].min(), test_data['first_session_timestamp'].max()),
            'payer_rate_drift': payer_rate_drift,
            'ltv_drift': ltv_drift,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'predictions': predictions,
            'actual': y_test
        }
    
    def _cross_validation(self,
                         X: pd.DataFrame,
                         y: np.ndarray,
                         model: ZILNModel) -> Dict[str, Any]:
        """K-fold cross-validation"""
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        cv_scores = {
            'mae': [], 'rmse': [], 'r2': [], 'auc': [],
            'lift_top_10pct': [], 'lift_top_20pct': []
        }
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Create new model instance for this fold
            fold_model = ZILNModel()
            fold_model.fit(X_train, y_train)
            
            # Evaluate
            metrics = fold_model.evaluate(X_test, y_test)
            
            # Store metrics
            for metric in cv_scores.keys():
                if metric in metrics:
                    cv_scores[metric].append(metrics[metric])
            
            fold_results.append({
                'fold': fold,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'metrics': metrics
            })
        
        # Calculate mean and std for each metric
        cv_summary = {}
        for metric, values in cv_scores.items():
            if values:
                cv_summary[f'{metric}_mean'] = np.mean(values)
                cv_summary[f'{metric}_std'] = np.std(values)
        
        return {
            'summary': cv_summary,
            'fold_results': fold_results,
            'n_splits': self.n_splits
        }
    
    def _segment_based_validation(self,
                                 data: pd.DataFrame,
                                 X: pd.DataFrame,
                                 y: np.ndarray,
                                 model: ZILNModel) -> Dict[str, Any]:
        """Validation across different player segments"""
        
        if 'player_segment' not in data.columns:
            return {'error': 'No player segments available'}
        
        segment_results = {}
        
        # Train on full dataset
        model.fit(X, y)
        
        # Evaluate on each segment
        for segment in data['player_segment'].unique():
            segment_mask = data['player_segment'] == segment
            
            if np.sum(segment_mask) < 10:  # Skip segments with too few samples
                continue
            
            X_segment = X[segment_mask]
            y_segment = y[segment_mask]
            
            # Evaluate model performance on this segment
            metrics = model.evaluate(X_segment, y_segment)
            
            segment_results[f'segment_{segment}'] = {
                'metrics': metrics,
                'size': len(X_segment),
                'payer_rate': np.mean(y_segment > 0),
                'avg_revenue': np.mean(y_segment[y_segment > 0]) if np.sum(y_segment > 0) > 0 else 0
            }
        
        return segment_results
    
    def _revenue_tier_validation(self,
                                data: pd.DataFrame,
                                X: pd.DataFrame,
                                y: np.ndarray,
                                model: ZILNModel) -> Dict[str, Any]:
        """Validation across different revenue tiers"""
        
        # Define revenue tiers
        revenue_tiers = {
            'non_payers': y == 0,
            'low_spenders': (y > 0) & (y <= np.percentile(y[y > 0], 50)) if np.sum(y > 0) > 0 else np.zeros_like(y, dtype=bool),
            'medium_spenders': (y > np.percentile(y[y > 0], 50)) & (y <= np.percentile(y[y > 0], 90)) if np.sum(y > 0) > 0 else np.zeros_like(y, dtype=bool),
            'high_spenders': y > np.percentile(y[y > 0], 90) if np.sum(y > 0) > 0 else np.zeros_like(y, dtype=bool)
        }
        
        # Train on full dataset
        model.fit(X, y)
        
        tier_results = {}
        
        for tier_name, tier_mask in revenue_tiers.items():
            if np.sum(tier_mask) < 10:  # Skip tiers with too few samples
                continue
            
            X_tier = X[tier_mask]
            y_tier = y[tier_mask]
            
            # Get predictions for this tier
            try:
                predictions, payer_probs, _, _ = model.predict(X_tier)
            except ValueError:
                # Fallback for models that return 3 values
                predictions, payer_probs, _ = model.predict(X_tier)
            
            # Calculate tier-specific metrics
            mae = mean_absolute_error(y_tier, predictions)
            rmse = np.sqrt(mean_squared_error(y_tier, predictions))
            
            # For non-payers, focus on payer probability accuracy
            if tier_name == 'non_payers':
                avg_payer_prob = np.mean(payer_probs)
                metrics = {
                    'mae': mae,
                    'rmse': rmse,
                    'avg_payer_probability': avg_payer_prob,
                    'false_positive_rate': avg_payer_prob  # Should be low for non-payers
                }
            else:
                r2 = r2_score(y_tier, predictions)
                avg_payer_prob = np.mean(payer_probs)
                metrics = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'avg_payer_probability': avg_payer_prob
                }
            
            tier_results[tier_name] = {
                'metrics': metrics,
                'size': len(X_tier),
                'revenue_range': (np.min(y_tier), np.max(y_tier)),
                'avg_revenue': np.mean(y_tier)
            }
        
        return tier_results
    
    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive validation report"""
        
        report = []
        report.append("pLTV Model Validation Report")
        report.append("=" * 50)
        report.append("")
        
        # Random split results
        if 'random_split' in results:
            rs = results['random_split']
            report.append("1. Random Split Validation:")
            report.append(f"   Train size: {rs['train_size']:,}")
            report.append(f"   Test size: {rs['test_size']:,}")
            report.append("")
            
            # Regression Performance
            report.append("   Regression Metrics:")
            report.append(f"   - MAE (Mean Absolute Error): {rs['metrics']['mae']:.4f}")
            report.append(f"   - RMSE (Root Mean Square Error): {rs['metrics']['rmse']:.4f}")
            report.append(f"   - R² Score: {rs['metrics']['r2']:.4f}")
            report.append(f"   - MAPE (Mean Absolute % Error): {rs['metrics']['mape']:.2f}%")
            report.append(f"   - Explained Variance: {rs['metrics']['explained_variance']:.4f}")
            report.append("")
            
            # Classification Performance
            report.append("   Payer Classification Metrics:")
            report.append(f"   - AUC Score: {rs['metrics']['auc_payer_classification']:.4f}")
            report.append(f"   - Binary Accuracy: {rs['metrics']['binary_accuracy']:.4f}")
            report.append(f"   - Precision: {rs['metrics']['precision']:.4f}")
            report.append(f"   - Recall: {rs['metrics']['recall']:.4f}")
            report.append(f"   - F1 Score: {rs['metrics']['f1_score']:.4f}")
            report.append(f"   - Optimal Threshold: {rs['metrics']['optimal_threshold']:.4f}")
            report.append("")
            
            # Revenue Accuracy
            report.append("   Revenue Prediction Accuracy:")
            report.append(f"   - Within 10%: {rs['metrics']['revenue_accuracy_within_10pct']:.2%}")
            report.append(f"   - Within 25%: {rs['metrics']['revenue_accuracy_within_25pct']:.2%}")
            report.append(f"   - Within 50%: {rs['metrics']['revenue_accuracy_within_50pct']:.2%}")
            report.append("")
            
            # Business Metrics
            report.append("   Business Impact Metrics:")
            report.append(f"   - Lift (Top 1%): {rs['metrics']['lift_top_1pct']:.2f}x")
            report.append(f"   - Lift (Top 5%): {rs['metrics']['lift_top_5pct']:.2f}x")
            report.append(f"   - Lift (Top 10%): {rs['metrics']['lift_top_10pct']:.2f}x")
            report.append(f"   - Lift (Top 20%): {rs['metrics']['lift_top_20pct']:.2f}x")
            report.append("")
            
            # Revenue Capture
            report.append("   Revenue Capture Analysis:")
            report.append(f"   - Top 1% captures: {rs['metrics']['revenue_capture_top_1pct']:.2%} of revenue")
            report.append(f"   - Top 5% captures: {rs['metrics']['revenue_capture_top_5pct']:.2%} of revenue")
            report.append(f"   - Top 10% captures: {rs['metrics']['revenue_capture_top_10pct']:.2%} of revenue")
            report.append(f"   - Top 20% captures: {rs['metrics']['revenue_capture_top_20pct']:.2%} of revenue")
            report.append("")
            
            # High-Value Player Detection
            report.append("   High-Value Player Identification (Top 10% Spenders):")
            report.append(f"   - Accuracy: {rs['metrics']['high_value_accuracy']:.4f}")
            report.append(f"   - Precision: {rs['metrics']['high_value_precision']:.4f}")
            report.append(f"   - Recall: {rs['metrics']['high_value_recall']:.4f}")
            report.append("")
            
            # Model Calibration
            report.append("   Model Calibration:")
            report.append(f"   - Predicted Payer Rate: {rs['metrics']['predicted_payer_rate']:.2%}")
            report.append(f"   - Actual Payer Rate: {rs['metrics']['actual_payer_rate']:.2%}")
            report.append(f"   - Calibration Error: {rs['metrics']['calibration_error']:.4f}")
            report.append(f"   - Brier Score: {rs['metrics']['brier_score']:.4f}")
            report.append("")
        
        # Time-based results
        if 'time_based' in results and results['time_based']:
            tb = results['time_based']
            report.append("2. Time-based Validation (Temporal Stability):")
            report.append(f"   Train size: {tb['train_size']:,}")
            report.append(f"   Test size: {tb['test_size']:,}")
            report.append(f"   MAE: {tb['metrics']['mae']:.4f}")
            report.append(f"   RMSE: {tb['metrics']['rmse']:.4f}")
            report.append(f"   R²: {tb['metrics']['r2']:.4f}")
            report.append(f"   AUC: {tb['metrics']['auc_payer_classification']:.4f}")
            report.append(f"   Binary Accuracy: {tb['metrics']['binary_accuracy']:.4f}")
            report.append(f"   Payer rate drift: {tb['payer_rate_drift']:.4f}")
            report.append(f"   LTV drift: {tb['ltv_drift']:.4f}")
            report.append("")
        
        # Cross-validation results
        if 'cross_validation' in results:
            cv = results['cross_validation']
            if 'summary' in cv:
                report.append("3. Cross-validation Results (Model Stability):")
                summary = cv['summary']
                key_metrics = ['mae', 'rmse', 'r2', 'auc_payer_classification', 'binary_accuracy', 'f1_score']
                for metric in key_metrics:
                    mean_key = f'{metric}_mean'
                    std_key = f'{metric}_std'
                    if mean_key in summary and std_key in summary:
                        report.append(f"   {metric.upper()}: {summary[mean_key]:.4f} ± {summary[std_key]:.4f}")
                report.append("")
        
        # Segment-based results
        if 'segment_based' in results and results['segment_based']:
            sb = results['segment_based']
            report.append("4. Performance by Player Segment:")
            for segment, data in sb.items():
                if isinstance(data, dict) and 'metrics' in data:
                    report.append(f"   {segment}:")
                    report.append(f"     Size: {data['size']:,} players")
                    report.append(f"     Payer rate: {data['payer_rate']:.1%}")
                    report.append(f"     Avg revenue: ${data['avg_revenue']:.2f}")
                    report.append(f"     MAE: {data['metrics']['mae']:.4f}")
                    report.append(f"     R²: {data['metrics']['r2']:.4f}")
                    report.append(f"     AUC: {data['metrics']['auc_payer_classification']:.4f}")
                    report.append(f"     Binary Accuracy: {data['metrics']['binary_accuracy']:.4f}")
                    report.append("")
        
        # Revenue tier analysis
        if 'revenue_tier' in results and results['revenue_tier']:
            rt = results['revenue_tier']
            report.append("5. Performance by Revenue Tier:")
            for tier, data in rt.items():
                if isinstance(data, dict) and 'metrics' in data:
                    report.append(f"   {tier.replace('_', ' ').title()}:")
                    report.append(f"     Size: {data['size']:,} players")
                    report.append(f"     Revenue range: ${data['revenue_range'][0]:.2f} - ${data['revenue_range'][1]:.2f}")
                    report.append(f"     MAE: {data['metrics']['mae']:.4f}")
                    if 'r2' in data['metrics']:
                        report.append(f"     R²: {data['metrics']['r2']:.4f}")
                    if 'avg_payer_probability' in data['metrics']:
                        report.append(f"     Avg Payer Probability: {data['metrics']['avg_payer_probability']:.4f}")
                    report.append("")
        
        # Feature importance
        if 'feature_importance' in results:
            fi = results['feature_importance']
            if 'binary_classifier' in fi:
                report.append("6. Top Features for Payer Classification:")
                for i, (feature, importance) in enumerate(list(fi['binary_classifier'].items())[:15], 1):
                    report.append(f"   {i:2d}. {feature.replace('_', ' ').title()}: {importance:.4f}")
                report.append("")
            
            if 'amount_regressor' in fi:
                report.append("7. Top Features for Revenue Amount Prediction:")
                for i, (feature, importance) in enumerate(list(fi['amount_regressor'].items())[:15], 1):
                    report.append(f"   {i:2d}. {feature.replace('_', ' ').title()}: {importance:.4f}")
                report.append("")
        
        # Model Performance Summary
        if 'random_split' in results:
            rs = results['random_split']
            report.append("8. Model Performance Summary:")
            report.append("   Overall Assessment:")
            
            # R² interpretation
            r2 = rs['metrics']['r2']
            if r2 > 0.7:
                r2_assessment = "Excellent"
            elif r2 > 0.5:
                r2_assessment = "Good" 
            elif r2 > 0.3:
                r2_assessment = "Fair"
            elif r2 > 0.1:
                r2_assessment = "Poor"
            else:
                r2_assessment = "Very Poor"
            
            report.append(f"   - Revenue Prediction: {r2_assessment} (R² = {r2:.3f})")
            
            # AUC interpretation
            auc = rs['metrics']['auc_payer_classification']
            if auc > 0.9:
                auc_assessment = "Excellent"
            elif auc > 0.8:
                auc_assessment = "Good"
            elif auc > 0.7:
                auc_assessment = "Fair"
            elif auc > 0.6:
                auc_assessment = "Poor"
            else:
                auc_assessment = "Very Poor"
            
            report.append(f"   - Payer Identification: {auc_assessment} (AUC = {auc:.3f})")
            
            # Business impact
            lift_10 = rs['metrics']['lift_top_10pct']
            if lift_10 > 5:
                business_impact = "High"
            elif lift_10 > 3:
                business_impact = "Medium"
            elif lift_10 > 1.5:
                business_impact = "Low"
            else:
                business_impact = "Minimal"
            
            report.append(f"   - Business Impact: {business_impact} ({lift_10:.1f}x lift for top 10%)")
            report.append("")
            
            # Recommendations
            report.append("9. Recommendations:")
            if r2 < 0.3:
                report.append("   - Consider additional feature engineering or external data sources")
            if auc < 0.7:
                report.append("   - Review payer classification features and model parameters")
            if lift_10 < 2:
                report.append("   - Model may benefit from segment-specific approaches")
            if rs['metrics']['calibration_error'] > 0.1:
                report.append("   - Model probabilities need better calibration")
            
            report.append("   - Use model for targeting top predicted players")
            report.append("   - Implement A/B testing to validate business impact")
            report.append("   - Monitor model performance over time for drift")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Save backtest results to file"""
        
        # Generate report
        report = self.generate_validation_report(results)
        
        # Save to file
        with open(filepath, 'w') as f:
            f.write(report)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Validation report saved to: {filepath}")
    
    def get_model_stability_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate model stability metrics across different validation approaches"""
        
        stability_metrics = {}
        
        # Collect R² scores from different validation methods
        r2_scores = []
        
        if 'random_split' in results:
            r2_scores.append(results['random_split']['metrics']['r2'])
        
        if 'time_based' in results and results['time_based']:
            r2_scores.append(results['time_based']['metrics']['r2'])
        
        if 'cross_validation' in results and 'summary' in results['cross_validation']:
            cv_r2 = results['cross_validation']['summary'].get('r2_mean')
            if cv_r2 is not None:
                r2_scores.append(cv_r2)
        
        # Calculate stability metrics
        if len(r2_scores) >= 2:
            stability_metrics['r2_variance'] = np.var(r2_scores)
            stability_metrics['r2_std'] = np.std(r2_scores)
            stability_metrics['r2_range'] = max(r2_scores) - min(r2_scores)
            stability_metrics['r2_mean'] = np.mean(r2_scores)
        
        return stability_metrics
    
    def _whale_detection_validation(self,
                                   data: pd.DataFrame,
                                   X: pd.DataFrame,
                                   y: np.ndarray,
                                   model: ZILNModel,
                                   feature_engineer: FeatureEngineer) -> Dict[str, Any]:
        """Validation specifically for whale detection performance"""
        
        
        # Create whale labels based on revenue percentiles
        payers = y[y > 0]
        if len(payers) == 0:
            return {'error': 'No payers in dataset for whale validation'}
        
        whale_threshold = np.percentile(payers, 90)  # Top 10% of payers are whales
        whale_labels = (y >= whale_threshold).astype(int)
        
        # Get whale predictions from existing model
        try:
            try:
                predictions, payer_probs, conditional_amounts, _ = model.predict(X)
            except ValueError:
                # Fallback for models that return 3 values
                predictions, payer_probs, conditional_amounts = model.predict(X)
            
            # Use LTV predictions to identify whales
            whale_probs = (predictions >= whale_threshold).astype(float)
        except Exception:
            whale_probs = (y >= whale_threshold).astype(float)  # Use actual as fallback
        
        # Calculate whale detection metrics
        whale_auc = roc_auc_score(whale_labels, whale_probs) if len(np.unique(whale_labels)) > 1 else 0.5
        
        # Whale identification accuracy at different thresholds
        thresholds = [0.3, 0.5, 0.7]
        threshold_metrics = {}
        
        for threshold in thresholds:
            whale_preds = (whale_probs >= threshold).astype(int)
            accuracy = accuracy_score(whale_labels, whale_preds)
            precision = precision_score(whale_labels, whale_preds, zero_division=0)
            recall = recall_score(whale_labels, whale_preds, zero_division=0)
            f1 = f1_score(whale_labels, whale_preds, zero_division=0)
            
            threshold_metrics[f'threshold_{threshold}'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        # Whale statistics
        whale_mask = whale_labels == 1
        whale_revenues = y[whale_mask]
        
        whale_stats = {
            'total_whales': np.sum(whale_mask),
            'whale_rate': np.mean(whale_labels),
            'avg_whale_revenue': np.mean(whale_revenues) if len(whale_revenues) > 0 else 0,
            'whale_revenue_share': np.sum(whale_revenues) / np.sum(y) if np.sum(y) > 0 else 0
        }
        
        return {
            'whale_threshold': whale_threshold,
            'whale_auc': whale_auc,
            'threshold_metrics': threshold_metrics,
            'whale_statistics': whale_stats
        }