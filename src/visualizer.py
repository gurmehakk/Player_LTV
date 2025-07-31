"""
Comprehensive Visualization System for ExpLTV Model
Creates detailed visualizations and reports for whale detection and LTV prediction
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Set consistent styling
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})


class ResultVisualizer:
    """Creates comprehensive visualizations for pLTV analysis results"""
    
    def __init__(self, figsize: tuple = (12, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = sns.color_palette("husl", 10)
        
    def create_all_plots(self, 
                        data: pd.DataFrame,
                        backtest_results: Dict[str, Any],
                        save_dir: str = "output") -> None:
        """Create all visualization plots and save them"""
        
        print("Generating visualizations...")
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Data distribution plots
        self._plot_revenue_distribution(data, save_dir)
        self._plot_player_segments(data, save_dir)
        self._plot_feature_correlations(data, save_dir)
        
        # 2. Model performance plots
        if 'random_split' in backtest_results:
            self._plot_prediction_accuracy(backtest_results['random_split'], save_dir)
            self._plot_lift_analysis(backtest_results['random_split'], save_dir)
            self._plot_calibration_curve(backtest_results['random_split'], save_dir)
        
        # 3. Validation plots
        self._plot_cross_validation_results(backtest_results.get('cross_validation', {}), save_dir)
        self._plot_segment_performance(backtest_results.get('segment_based', {}), save_dir)
        
        # 4. Feature importance plots
        self._plot_feature_importance(backtest_results.get('feature_importance', {}), save_dir)
        
        # 6. Business insight plots
        self._plot_revenue_tiers(data, save_dir)
        self._plot_behavioral_patterns(data, save_dir)
        
        # 7. Performance metrics dashboard
        self._plot_performance_metrics(backtest_results, save_dir)
        
        print(f"All plots saved to: {save_dir}")
    
    def _plot_revenue_distribution(self, data: pd.DataFrame, save_dir: str) -> None:
        """Plot revenue distribution and payer analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Revenue Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Revenue histogram (log scale for payers)
        payers = data[data['total_revenue'] > 0]['total_revenue']
        
        axes[0, 0].hist(payers, bins=50, alpha=0.7, color=self.colors[0], edgecolor='black')
        axes[0, 0].set_xlabel('Revenue ($)')
        axes[0, 0].set_ylabel('Number of Players')
        axes[0, 0].set_title('Revenue Distribution (Payers Only)')
        axes[0, 0].set_yscale('log')
        
        # 2. Payer vs Non-payer comparison
        payer_counts = [len(data[data['total_revenue'] == 0]), len(data[data['total_revenue'] > 0])]
        labels = ['Non-Payers', 'Payers']
        
        axes[0, 1].pie(payer_counts, labels=labels, autopct='%1.1f%%', 
                      colors=[self.colors[1], self.colors[2]], startangle=90)
        axes[0, 1].set_title('Payer Distribution')
        
        # 3. Revenue percentiles
        if len(payers) > 0:
            percentiles = [50, 75, 90, 95, 99]
            values = [np.percentile(payers, p) for p in percentiles]
            
            axes[1, 0].bar(range(len(percentiles)), values, color=self.colors[3], alpha=0.7)
            axes[1, 0].set_xticks(range(len(percentiles)))
            axes[1, 0].set_xticklabels([f'P{p}' for p in percentiles])
            axes[1, 0].set_ylabel('Revenue ($)')
            axes[1, 0].set_title('Revenue Percentiles')
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[1, 0].text(i, v + max(values) * 0.01, f'${v:.2f}', 
                               ha='center', va='bottom')
        
        # 4. Revenue by session count
        revenue_by_sessions = data.groupby('total_sessions')['total_revenue'].mean()
        session_counts = revenue_by_sessions.index[:20]  # Top 20 session counts
        avg_revenues = revenue_by_sessions.values[:20]
        
        axes[1, 1].scatter(session_counts, avg_revenues, alpha=0.6, color=self.colors[4])
        axes[1, 1].set_xlabel('Total Sessions')
        axes[1, 1].set_ylabel('Average Revenue ($)')
        axes[1, 1].set_title('Revenue vs Session Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'revenue_distribution.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_player_segments(self, data: pd.DataFrame, save_dir: str) -> None:
        """Plot player segmentation analysis"""
        
        if 'player_segment' not in data.columns:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Player Segmentation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Segment distribution
        segment_counts = data['player_segment'].value_counts().sort_index()
        
        axes[0, 0].bar(segment_counts.index, segment_counts.values, 
                      color=self.colors[:len(segment_counts)], alpha=0.7)
        axes[0, 0].set_xlabel('Player Segment')
        axes[0, 0].set_ylabel('Number of Players')
        axes[0, 0].set_title('Player Segment Distribution')
        
        # 2. Revenue by segment
        segment_revenue = data.groupby('player_segment')['total_revenue'].agg(['mean', 'median'])
        
        x = np.arange(len(segment_revenue))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, segment_revenue['mean'], width, 
                      label='Mean', color=self.colors[0], alpha=0.7)
        axes[0, 1].bar(x + width/2, segment_revenue['median'], width,
                      label='Median', color=self.colors[1], alpha=0.7)
        
        axes[0, 1].set_xlabel('Player Segment')
        axes[0, 1].set_ylabel('Revenue ($)')
        axes[0, 1].set_title('Revenue by Segment')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(segment_revenue.index)
        axes[0, 1].legend()
        
        # 3. Sessions by segment
        segment_sessions = data.groupby('player_segment')['total_sessions'].mean()
        
        axes[1, 0].bar(segment_sessions.index, segment_sessions.values,
                      color=self.colors[2], alpha=0.7)
        axes[1, 0].set_xlabel('Player Segment')
        axes[1, 0].set_ylabel('Average Sessions')
        axes[1, 0].set_title('Sessions by Segment')
        
        # 4. Payer rate by segment
        segment_payer_rate = data.groupby('player_segment').apply(
            lambda x: (x['total_revenue'] > 0).mean()
        )
        
        axes[1, 1].bar(segment_payer_rate.index, segment_payer_rate.values,
                      color=self.colors[3], alpha=0.7)
        axes[1, 1].set_xlabel('Player Segment')
        axes[1, 1].set_ylabel('Payer Rate')
        axes[1, 1].set_title('Payer Rate by Segment')
        axes[1, 1].set_ylim(0, 1)
        
        # Add percentage labels
        for i, v in enumerate(segment_payer_rate.values):
            axes[1, 1].text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'player_segments.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_correlations(self, data: pd.DataFrame, save_dir: str) -> None:
        """Plot feature correlation heatmap"""
        
        # Select key numeric features
        key_features = [
            'total_revenue', 'total_sessions', 'total_events', 
            'avg_session_duration_minutes', 'events_per_session',
            'purchase_frequency', 'progression_rate', 'recent_activity_rate'
        ]
        
        # Filter features that exist in data
        available_features = [f for f in key_features if f in data.columns]
        
        if len(available_features) < 3:
            return
        
        # Calculate correlation matrix
        corr_matrix = data[available_features].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                   center=0, square=True, linewidths=0.5,
                   cbar_kws={"shrink": .8}, fmt='.2f')
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_correlations.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_accuracy(self, results: Dict[str, Any], save_dir: str) -> None:
        """Plot prediction accuracy analysis"""
        
        if 'predictions' not in results or 'actual' not in results:
            return
        
        predictions = results['predictions']
        actual = results['actual']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Prediction Accuracy Analysis', fontsize=16, fontweight='bold')
        
        # 1. Predicted vs Actual scatter plot
        axes[0, 0].scatter(actual, predictions, alpha=0.6, color=self.colors[0])
        
        # Add perfect prediction line
        max_val = max(max(actual), max(predictions))
        axes[0, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        axes[0, 0].set_xlabel('Actual Revenue ($)')
        axes[0, 0].set_ylabel('Predicted Revenue ($)')
        axes[0, 0].set_title('Predicted vs Actual Revenue')
        axes[0, 0].legend()
        
        # 2. Residuals plot
        residuals = predictions - actual
        
        axes[0, 1].scatter(predictions, residuals, alpha=0.6, color=self.colors[1])
        axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Predicted Revenue ($)')
        axes[0, 1].set_ylabel('Residuals ($)')
        axes[0, 1].set_title('Residual Plot')
        
        # 3. Error distribution
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, color=self.colors[2], edgecolor='black')
        axes[1, 0].set_xlabel('Prediction Error ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Error Distribution')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        
        # 4. Cumulative accuracy by prediction rank
        sorted_indices = np.argsort(predictions)[::-1]
        cumulative_actual = np.cumsum(actual[sorted_indices])
        cumulative_predicted = np.cumsum(predictions[sorted_indices])
        
        percentiles = np.arange(0, 101, 5)
        # Ensure indices don't exceed array bounds
        indices = [min(int(len(sorted_indices) * p / 100), len(sorted_indices) - 1) for p in percentiles]
        
        actual_cumulative_pct = [cumulative_actual[i] / cumulative_actual[-1] * 100 if cumulative_actual[-1] > 0 else 0 for i in indices]
        predicted_cumulative_pct = [cumulative_predicted[i] / cumulative_predicted[-1] * 100 if cumulative_predicted[-1] > 0 else 0 for i in indices]
        
        axes[1, 1].plot(percentiles, actual_cumulative_pct,
                       label='Actual', linewidth=2, color=self.colors[3])
        axes[1, 1].plot(percentiles, predicted_cumulative_pct,
                       label='Predicted', linewidth=2, color=self.colors[4])
        axes[1, 1].plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Random')
        
        axes[1, 1].set_xlabel('Player Percentile (%)')
        axes[1, 1].set_ylabel('Cumulative Revenue (%)')
        axes[1, 1].set_title('Cumulative Revenue Capture')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'prediction_accuracy.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_lift_analysis(self, results: Dict[str, Any], save_dir: str) -> None:
        """Plot lift analysis for targeting effectiveness"""
        
        if 'predictions' not in results or 'actual' not in results:
            return
        
        predictions = results['predictions']
        actual = results['actual']
        
        # Calculate lift for different percentiles
        sorted_indices = np.argsort(predictions)[::-1]
        
        percentiles = np.arange(5, 101, 5)
        lift_values = []
        revenue_capture = []
        
        total_revenue = np.sum(actual)
        
        for pct in percentiles:
            top_n = int(len(predictions) * pct / 100)
            if top_n > 0:
                top_indices = sorted_indices[:top_n]
                top_revenue = np.sum(actual[top_indices])
                
                # Lift calculation
                revenue_capture_rate = top_revenue / total_revenue if total_revenue > 0 else 0
                lift = revenue_capture_rate / (pct / 100) if pct > 0 else 0
                
                lift_values.append(lift)
                revenue_capture.append(revenue_capture_rate * 100)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Targeting Lift Analysis', fontsize=16, fontweight='bold')
        
        # 1. Lift curve
        ax1.plot(percentiles, lift_values, linewidth=3, color=self.colors[0], marker='o')
        ax1.axhline(y=1, color='r', linestyle='--', linewidth=2, label='Random Targeting')
        ax1.set_xlabel('Top % of Players Targeted')
        ax1.set_ylabel('Lift Factor')
        ax1.set_title('Revenue Lift by Targeting Percentile')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Highlight key percentiles
        key_percentiles = [10, 20, 50]
        for pct in key_percentiles:
            if pct <= max(percentiles):
                idx = list(percentiles).index(pct)
                ax1.annotate(f'{lift_values[idx]:.2f}x', 
                           xy=(pct, lift_values[idx]), 
                           xytext=(pct + 5, lift_values[idx] + 0.2),
                           arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))
        
        # 2. Revenue capture curve
        ax2.plot(percentiles, revenue_capture, linewidth=3, color=self.colors[1], marker='s')
        ax2.plot(percentiles, percentiles, 'r--', linewidth=2, label='Random Targeting')
        ax2.set_xlabel('Top % of Players Targeted')
        ax2.set_ylabel('% of Total Revenue Captured')
        ax2.set_title('Revenue Capture Curve')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add key metrics text
        if len(lift_values) >= 2:
            top_10_lift = lift_values[1] if len(lift_values) > 1 else 0  # 10% percentile
            top_20_lift = lift_values[3] if len(lift_values) > 3 else 0  # 20% percentile
            
            metrics_text = f'Top 10% Lift: {top_10_lift:.2f}x\nTop 20% Lift: {top_20_lift:.2f}x'
            ax2.text(0.7, 0.3, metrics_text, transform=ax2.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                    fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'lift_analysis.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration_curve(self, results: Dict[str, Any], save_dir: str) -> None:
        """Plot model calibration curve for payer probabilities"""
        
        if 'payer_probabilities' not in results or 'actual' not in results:
            return
        
        payer_probs = results['payer_probabilities']
        actual = results['actual']
        is_payer = (actual > 0).astype(int)
        
        # Create probability bins
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = (bin_lowers + bin_uppers) / 2
        actual_frequencies = []
        predicted_frequencies = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (payer_probs > bin_lower) & (payer_probs <= bin_upper)
            
            if np.sum(in_bin) > 0:
                actual_freq = np.mean(is_payer[in_bin])
                predicted_freq = np.mean(payer_probs[in_bin])
                count = np.sum(in_bin)
            else:
                actual_freq = 0
                predicted_freq = (bin_lower + bin_upper) / 2
                count = 0
            
            actual_frequencies.append(actual_freq)
            predicted_frequencies.append(predicted_freq)
            bin_counts.append(count)
        
        # Create calibration plot
        plt.figure(figsize=(10, 8))
        
        # Plot calibration curve
        plt.plot(predicted_frequencies, actual_frequencies, 'o-', linewidth=2, 
                markersize=8, color=self.colors[0], label='Model')
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
        
        # Add bin size information
        for i, (pred, actual, count) in enumerate(zip(predicted_frequencies, actual_frequencies, bin_counts)):
            if count > 0:
                plt.annotate(f'n={count}', xy=(pred, actual), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Actual Payer Rate')
        plt.title('Model Calibration Curve\n(Payer Probability Calibration)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculate and display calibration metrics
        from sklearn.calibration import calibration_curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            is_payer, payer_probs, n_bins=10
        )
        
        # Brier score (lower is better)
        brier_score = np.mean((payer_probs - is_payer) ** 2)
        
        # Add metrics text
        metrics_text = f'Brier Score: {brier_score:.4f}\nPerfect = 0.0'
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'calibration_curve.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_cross_validation_results(self, cv_results: Dict[str, Any], save_dir: str) -> None:
        """Plot cross-validation results"""
        
        if not cv_results or 'fold_results' not in cv_results:
            return
        
        fold_results = cv_results['fold_results']
        
        # Extract metrics across folds
        metrics_by_fold = {}
        for fold_data in fold_results:
            fold = fold_data['fold']
            metrics = fold_data['metrics']
            
            for metric, value in metrics.items():
                if metric not in metrics_by_fold:
                    metrics_by_fold[metric] = []
                metrics_by_fold[metric].append(value)
        
        # Select key metrics to plot
        key_metrics = ['mae', 'rmse', 'r2', 'auc_payer_classification', 'lift_top_10pct']
        available_metrics = [m for m in key_metrics if m in metrics_by_fold]
        
        if not available_metrics:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cross-Validation Results', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics):
            if i >= len(axes):
                break
            
            values = metrics_by_fold[metric]
            folds = range(len(values))
            
            # Box plot for metric distribution
            axes[i].boxplot([values], labels=[metric.upper()], patch_artist=True,
                           boxprops=dict(facecolor=self.colors[i % len(self.colors)], alpha=0.7))
            
            # Scatter plot for individual fold values
            axes[i].scatter([1] * len(values), values, color='red', alpha=0.6, s=50)
            
            # Add mean line
            mean_val = np.mean(values)
            axes[i].axhline(y=mean_val, color='green', linestyle='--', linewidth=2, alpha=0.8)
            
            axes[i].set_ylabel('Value')
            axes[i].set_title(f'{metric.upper()}\nMean: {mean_val:.4f} ± {np.std(values):.4f}')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'cross_validation_results.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_segment_performance(self, segment_results: Dict[str, Any], save_dir: str) -> None:
        """Plot performance across different player segments"""
        
        if not segment_results:
            return
        
        # Extract segment data
        segments = []
        mae_values = []
        r2_values = []
        sizes = []
        payer_rates = []
        
        for segment_name, segment_data in segment_results.items():
            if isinstance(segment_data, dict) and 'metrics' in segment_data:
                segments.append(segment_name.replace('segment_', 'Seg '))
                mae_values.append(segment_data['metrics']['mae'])
                r2_values.append(segment_data['metrics']['r2'])
                sizes.append(segment_data['size'])
                payer_rates.append(segment_data['payer_rate'])
        
        if not segments:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance by Player Segment', fontsize=16, fontweight='bold')
        
        # 1. MAE by segment
        bars1 = axes[0, 0].bar(segments, mae_values, color=self.colors[0], alpha=0.7)
        axes[0, 0].set_ylabel('Mean Absolute Error')
        axes[0, 0].set_title('MAE by Segment')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, mae_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values) * 0.01,
                          f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. R² by segment
        bars2 = axes[0, 1].bar(segments, r2_values, color=self.colors[1], alpha=0.7)
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_title('R² by Segment')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars2, r2_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(r2_values) * 0.01,
                          f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Segment sizes
        bars3 = axes[1, 0].bar(segments, sizes, color=self.colors[2], alpha=0.7)
        axes[1, 0].set_ylabel('Number of Players')
        axes[1, 0].set_title('Segment Sizes')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Payer rates by segment
        bars4 = axes[1, 1].bar(segments, payer_rates, color=self.colors[3], alpha=0.7)
        axes[1, 1].set_ylabel('Payer Rate')
        axes[1, 1].set_title('Payer Rate by Segment')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_ylim(0, 1)
        
        # Add percentage labels
        for bar, value in zip(bars4, payer_rates):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.1%}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'segment_performance.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, feature_importance: Dict[str, Any], save_dir: str) -> None:
        """Plot feature importance for both model components"""
        
        if not feature_importance:
            return
        
        fig_height = 6
        n_plots = len(feature_importance)
        
        if n_plots == 0:
            return
        elif n_plots == 1:
            fig, ax = plt.subplots(1, 1, figsize=(12, fig_height))
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, n_plots, figsize=(12 * n_plots, fig_height))
        
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        
        # Binary classifier importance
        if 'binary_classifier' in feature_importance:
            features = list(feature_importance['binary_classifier'].keys())[:15]  # Top 15
            importances = [feature_importance['binary_classifier'][f] for f in features]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(features))
            bars = axes[plot_idx].barh(y_pos, importances, color=self.colors[0], alpha=0.7)
            
            axes[plot_idx].set_yticks(y_pos)
            axes[plot_idx].set_yticklabels([f.replace('_', ' ').title() for f in features])
            axes[plot_idx].set_xlabel('Feature Importance')
            axes[plot_idx].set_title('Payer Classification\n(Binary Model)')
            axes[plot_idx].grid(True, axis='x', alpha=0.3)
            
            # Add value labels
            for bar, importance in zip(bars, importances):
                axes[plot_idx].text(bar.get_width() + max(importances) * 0.01,
                                   bar.get_y() + bar.get_height()/2,
                                   f'{importance:.3f}', ha='left', va='center', fontsize=8)
            
            plot_idx += 1
        
        # Amount regressor importance
        if 'amount_regressor' in feature_importance and plot_idx < len(axes):
            features = list(feature_importance['amount_regressor'].keys())[:15]  # Top 15
            importances = [feature_importance['amount_regressor'][f] for f in features]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(features))
            bars = axes[plot_idx].barh(y_pos, importances, color=self.colors[1], alpha=0.7)
            
            axes[plot_idx].set_yticks(y_pos)
            axes[plot_idx].set_yticklabels([f.replace('_', ' ').title() for f in features])
            axes[plot_idx].set_xlabel('Feature Importance')
            axes[plot_idx].set_title('Revenue Amount Prediction\n(Regression Model)')
            axes[plot_idx].grid(True, axis='x', alpha=0.3)
            
            # Add value labels
            for bar, importance in zip(bars, importances):
                axes[plot_idx].text(bar.get_width() + max(importances) * 0.01,
                                   bar.get_y() + bar.get_height()/2,
                                   f'{importance:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_revenue_tiers(self, data: pd.DataFrame, save_dir: str) -> None:
        """Plot revenue tier analysis"""
        
        payers = data[data['total_revenue'] > 0]
        
        if len(payers) == 0:
            return
        
        # Define revenue tiers
        revenue_values = payers['total_revenue'].values
        tier_thresholds = [
            0,
            np.percentile(revenue_values, 50),
            np.percentile(revenue_values, 90),
            np.percentile(revenue_values, 95),
            np.max(revenue_values)
        ]
        
        tier_labels = ['Low Spenders', 'Medium Spenders', 'High Spenders', 'Whales']
        tier_counts = []
        tier_revenues = []
        
        for i in range(len(tier_labels)):
            lower = tier_thresholds[i]
            upper = tier_thresholds[i + 1]
            
            if i == len(tier_labels) - 1:  # Last tier includes the maximum
                tier_mask = (revenue_values >= lower) & (revenue_values <= upper)
            else:
                tier_mask = (revenue_values >= lower) & (revenue_values < upper)
            
            tier_counts.append(np.sum(tier_mask))
            tier_revenues.append(np.sum(revenue_values[tier_mask]))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Revenue Tier Analysis', fontsize=16, fontweight='bold')
        
        # 1. Player count by tier
        bars1 = axes[0, 0].bar(tier_labels, tier_counts, color=self.colors[:len(tier_labels)], alpha=0.7)
        axes[0, 0].set_ylabel('Number of Players')
        axes[0, 0].set_title('Players by Revenue Tier')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, count in zip(bars1, tier_counts):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(tier_counts) * 0.01,
                          f'{count:,}', ha='center', va='bottom')
        
        # 2. Revenue by tier
        bars2 = axes[0, 1].bar(tier_labels, tier_revenues, color=self.colors[:len(tier_labels)], alpha=0.7)
        axes[0, 1].set_ylabel('Total Revenue ($)')
        axes[0, 1].set_title('Revenue by Tier')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, revenue in zip(bars2, tier_revenues):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(tier_revenues) * 0.01,
                          f'${revenue:,.0f}', ha='center', va='bottom')
        
        # 3. Revenue concentration (Pareto analysis)
        sorted_revenue = np.sort(revenue_values)[::-1]
        cumulative_revenue = np.cumsum(sorted_revenue)
        cumulative_pct = cumulative_revenue / cumulative_revenue[-1] * 100
        player_pct = np.arange(1, len(sorted_revenue) + 1) / len(sorted_revenue) * 100
        
        axes[1, 0].plot(player_pct, cumulative_pct, linewidth=2, color=self.colors[0])
        axes[1, 0].plot([0, 100], [0, 100], 'r--', linewidth=2, alpha=0.7, label='Equal Distribution')
        axes[1, 0].set_xlabel('% of Players (Top Spenders)')
        axes[1, 0].set_ylabel('% of Total Revenue')
        axes[1, 0].set_title('Revenue Concentration (Pareto Chart)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Highlight key points (20-80 rule)
        top_20_idx = int(len(sorted_revenue) * 0.2)
        if top_20_idx < len(cumulative_pct):
            top_20_revenue_pct = cumulative_pct[top_20_idx]
            axes[1, 0].scatter([20], [top_20_revenue_pct], color='red', s=100, zorder=5)
            axes[1, 0].annotate(f'Top 20%: {top_20_revenue_pct:.1f}%', 
                               xy=(20, top_20_revenue_pct), xytext=(30, top_20_revenue_pct - 10),
                               arrowprops=dict(arrowstyle='->', color='red'))
        
        # 4. Tier thresholds
        axes[1, 1].bar(range(len(tier_thresholds[1:])), tier_thresholds[1:], 
                      color=self.colors[:len(tier_thresholds)-1], alpha=0.7)
        axes[1, 1].set_xticks(range(len(tier_labels)))
        axes[1, 1].set_xticklabels(tier_labels)
        axes[1, 1].set_ylabel('Minimum Revenue ($)')
        axes[1, 1].set_title('Tier Thresholds')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add threshold values
        for i, threshold in enumerate(tier_thresholds[1:]):
            axes[1, 1].text(i, threshold + max(tier_thresholds) * 0.01,
                          f'${threshold:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'revenue_tiers.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_behavioral_patterns(self, data: pd.DataFrame, save_dir: str) -> None:
        """Plot behavioral pattern analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Behavioral Pattern Analysis', fontsize=16, fontweight='bold')
        
        # 1. Session duration vs Revenue
        if 'avg_session_duration_minutes' in data.columns:
            axes[0, 0].scatter(data['avg_session_duration_minutes'], data['total_revenue'], 
                             alpha=0.6, color=self.colors[0])
            axes[0, 0].set_xlabel('Average Session Duration (minutes)')
            axes[0, 0].set_ylabel('Total Revenue ($)')
            axes[0, 0].set_title('Session Duration vs Revenue')
            axes[0, 0].set_ylim(bottom=0)
        
        # 2. Events per session vs Revenue
        if 'events_per_session' in data.columns:
            axes[0, 1].scatter(data['events_per_session'], data['total_revenue'], 
                             alpha=0.6, color=self.colors[1])
            axes[0, 1].set_xlabel('Events per Session')
            axes[0, 1].set_ylabel('Total Revenue ($)')
            axes[0, 1].set_title('Engagement vs Revenue')
            axes[0, 1].set_ylim(bottom=0)
        
        # 3. Progression events analysis
        progression_cols = ['total_level_events', 'total_achievement_events', 'total_social_events']
        available_progression = [col for col in progression_cols if col in data.columns]
        
        if available_progression:
            # Calculate total progression events
            progression_data = data[available_progression].sum(axis=1)
            
            axes[1, 0].scatter(progression_data, data['total_revenue'], 
                             alpha=0.6, color=self.colors[2])
            axes[1, 0].set_xlabel('Total Progression Events')
            axes[1, 0].set_ylabel('Total Revenue ($)')
            axes[1, 0].set_title('Game Progression vs Revenue')
            axes[1, 0].set_ylim(bottom=0)
        
        # 4. Retention patterns
        if 'days_since_last_session' in data.columns and 'total_sessions' in data.columns:
            # Create retention categories
            recent_players = data['days_since_last_session'] <= 7
            
            retention_revenue = [
                data[recent_players]['total_revenue'].mean(),
                data[~recent_players]['total_revenue'].mean() if len(data[~recent_players]) > 0 else 0
            ]
            retention_labels = ['Active (≤7 days)', 'Inactive (>7 days)']
            
            bars = axes[1, 1].bar(retention_labels, retention_revenue, 
                                 color=[self.colors[3], self.colors[4]], alpha=0.7)
            axes[1, 1].set_ylabel('Average Revenue ($)')
            axes[1, 1].set_title('Revenue by Retention Status')
            
            # Add value labels
            for bar, revenue in zip(bars, retention_revenue):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(retention_revenue) * 0.01,
                              f'${revenue:.2f}', ha='center', va='bottom')
            
            # Add count information
            counts = [len(data[recent_players]), len(data[~recent_players])]
            for i, (bar, count) in enumerate(zip(bars, counts)):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                              f'n={count:,}', ha='center', va='center', 
                              color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'behavioral_patterns.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_metrics(self, backtest_results: Dict[str, Any], save_dir: str) -> None:
        """Plot comprehensive performance metrics dashboard"""
        
        if 'random_split' not in backtest_results or 'metrics' not in backtest_results['random_split']:
            return
        
        metrics = backtest_results['random_split']['metrics']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Metrics Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Key Performance Indicators (Gauge-style)
        key_metrics = {
            'R² Score': metrics.get('r2', 0),
            'AUC Score': metrics.get('auc_payer_classification', 0.5),
            'F1 Score': metrics.get('f1_score', 0),
            'Binary Accuracy': metrics.get('binary_accuracy', 0)
        }
        
        colors = ['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' for v in key_metrics.values()]
        bars = axes[0, 0].barh(range(len(key_metrics)), list(key_metrics.values()), 
                              color=colors, alpha=0.7)
        axes[0, 0].set_yticks(range(len(key_metrics)))
        axes[0, 0].set_yticklabels(list(key_metrics.keys()))
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_title('Key Performance Indicators')
        axes[0, 0].set_xlim(0, 1)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, key_metrics.values())):
            axes[0, 0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                          f'{value:.3f}', ha='left', va='center', fontweight='bold')
        
        # 2. Lift Analysis
        lift_metrics = {
            'Top 1%': metrics.get('lift_top_1pct', 1),
            'Top 5%': metrics.get('lift_top_5pct', 1), 
            'Top 10%': metrics.get('lift_top_10pct', 1),
            'Top 20%': metrics.get('lift_top_20pct', 1)
        }
        
        bars = axes[0, 1].bar(range(len(lift_metrics)), list(lift_metrics.values()), 
                             color=self.colors[0], alpha=0.7)
        axes[0, 1].set_xticks(range(len(lift_metrics)))
        axes[0, 1].set_xticklabels(list(lift_metrics.keys()))
        axes[0, 1].set_ylabel('Lift Factor')
        axes[0, 1].set_title('Targeting Lift Analysis')
        axes[0, 1].axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, lift_metrics.values()):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(lift_metrics.values()) * 0.01,
                          f'{value:.1f}x', ha='center', va='bottom', fontweight='bold')
        
        # 3. Revenue Capture Analysis
        capture_metrics = {
            'Top 1%': metrics.get('revenue_capture_top_1pct', 0) * 100,
            'Top 5%': metrics.get('revenue_capture_top_5pct', 0) * 100,
            'Top 10%': metrics.get('revenue_capture_top_10pct', 0) * 100,
            'Top 20%': metrics.get('revenue_capture_top_20pct', 0) * 100
        }
        
        bars = axes[0, 2].bar(range(len(capture_metrics)), list(capture_metrics.values()),
                             color=self.colors[1], alpha=0.7)
        axes[0, 2].set_xticks(range(len(capture_metrics)))
        axes[0, 2].set_xticklabels(list(capture_metrics.keys()))
        axes[0, 2].set_ylabel('Revenue Captured (%)')
        axes[0, 2].set_title('Revenue Capture by Targeting')
        
        # Add value labels
        for bar, value in zip(bars, capture_metrics.values()):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(capture_metrics.values()) * 0.01,
                          f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. Accuracy Analysis
        accuracy_metrics = {
            'Within 10%': metrics.get('revenue_accuracy_within_10pct', 0) * 100,
            'Within 25%': metrics.get('revenue_accuracy_within_25pct', 0) * 100,
            'Within 50%': metrics.get('revenue_accuracy_within_50pct', 0) * 100
        }
        
        bars = axes[1, 0].bar(range(len(accuracy_metrics)), list(accuracy_metrics.values()),
                             color=self.colors[2], alpha=0.7)
        axes[1, 0].set_xticks(range(len(accuracy_metrics)))
        axes[1, 0].set_xticklabels(list(accuracy_metrics.keys()))
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title('Revenue Prediction Accuracy')
        
        # Add value labels
        for bar, value in zip(bars, accuracy_metrics.values()):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(accuracy_metrics.values()) * 0.01,
                          f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 5. Error Analysis
        error_metrics = {
            'MAE': metrics.get('mae', 0),
            'RMSE': metrics.get('rmse', 0),
            'MAPE (%)': metrics.get('mape', 0)
        }
        
        # Normalize MAPE for visualization (divide by 10 to fit scale)
        plot_values = [error_metrics['MAE'], error_metrics['RMSE'], error_metrics['MAPE (%)']/10]
        
        bars = axes[1, 1].bar(range(len(error_metrics)), plot_values,
                             color=self.colors[3], alpha=0.7)
        axes[1, 1].set_xticks(range(len(error_metrics)))
        axes[1, 1].set_xticklabels(list(error_metrics.keys()))
        axes[1, 1].set_ylabel('Error Value')
        axes[1, 1].set_title('Prediction Error Analysis')
        
        # Add value labels with original values
        for i, (bar, key) in enumerate(zip(bars, error_metrics.keys())):
            original_value = error_metrics[key]
            if key == 'MAPE (%)':
                label = f'{original_value:.1f}%'
            else:
                label = f'${original_value:.2f}'
            
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(plot_values) * 0.01,
                          label, ha='center', va='bottom', fontweight='bold')
        
        # 6. Model Calibration
        calibration_data = {
            'Predicted Payer Rate': metrics.get('predicted_payer_rate', 0) * 100,
            'Actual Payer Rate': metrics.get('actual_payer_rate', 0) * 100
        }
        
        bars = axes[1, 2].bar(range(len(calibration_data)), list(calibration_data.values()),
                             color=[self.colors[4], self.colors[5]], alpha=0.7)
        axes[1, 2].set_xticks(range(len(calibration_data)))
        axes[1, 2].set_xticklabels(list(calibration_data.keys()), rotation=45, ha='right')
        axes[1, 2].set_ylabel('Payer Rate (%)')
        axes[1, 2].set_title('Model Calibration Check')
        
        # Add value labels
        for bar, value in zip(bars, calibration_data.values()):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(calibration_data.values()) * 0.01,
                          f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Add calibration error text
        calib_error = metrics.get('calibration_error', 0)
        axes[1, 2].text(0.5, 0.95, f'Calibration Error: {calib_error:.3f}', 
                       transform=axes[1, 2].transAxes, ha='center', va='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'performance_metrics_dashboard.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def create_summary_dashboard(self, 
                               data: pd.DataFrame,
                               backtest_results: Dict[str, Any],
                               save_dir: str) -> None:
        """Create a comprehensive summary dashboard"""
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('pLTV Prediction System - Executive Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Key Metrics (Top row)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[0, 3])
        
        # Dataset overview
        total_players = len(data)
        payers = len(data[data['total_revenue'] > 0])
        payer_rate = payers / total_players * 100
        total_revenue = data['total_revenue'].sum()
        
        ax1.text(0.5, 0.7, f'{total_players:,}', ha='center', va='center', 
                fontsize=24, fontweight='bold', transform=ax1.transAxes)
        ax1.text(0.5, 0.3, 'Total Players', ha='center', va='center', 
                fontsize=12, transform=ax1.transAxes)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        ax2.text(0.5, 0.7, f'{payer_rate:.1f}%', ha='center', va='center', 
                fontsize=24, fontweight='bold', color='green', transform=ax2.transAxes)
        ax2.text(0.5, 0.3, 'Payer Rate', ha='center', va='center', 
                fontsize=12, transform=ax2.transAxes)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        ax3.text(0.5, 0.7, f'${total_revenue:,.0f}', ha='center', va='center', 
                fontsize=24, fontweight='bold', color='blue', transform=ax3.transAxes)
        ax3.text(0.5, 0.3, 'Total Revenue', ha='center', va='center', 
                fontsize=12, transform=ax3.transAxes)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        # Model performance
        if 'random_split' in backtest_results:
            r2_score = backtest_results['random_split']['metrics']['r2']
            ax4.text(0.5, 0.7, f'{r2_score:.3f}', ha='center', va='center', 
                    fontsize=24, fontweight='bold', 
                    color='green' if r2_score > 0.5 else 'orange', 
                    transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.7, 'N/A', ha='center', va='center', 
                    fontsize=24, fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.5, 0.3, 'R² Score', ha='center', va='center', 
                fontsize=12, transform=ax4.transAxes)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        # Add more visualizations to fill the dashboard
        # Revenue distribution (second row, left)
        ax5 = fig.add_subplot(gs[1, :2])
        if len(data[data['total_revenue'] > 0]) > 0:
            payers_revenue = data[data['total_revenue'] > 0]['total_revenue']
            ax5.hist(payers_revenue, bins=50, alpha=0.7, color=self.colors[0], edgecolor='black')
            ax5.set_xlabel('Revenue ($)')
            ax5.set_ylabel('Count')
            ax5.set_title('Revenue Distribution (Payers Only)')
        
        # Lift curve (second row, right)
        ax6 = fig.add_subplot(gs[1, 2:])
        if 'random_split' in backtest_results:
            results = backtest_results['random_split']
            if 'predictions' in results and 'actual' in results:
                predictions = results['predictions']
                actual = results['actual']
                
                sorted_indices = np.argsort(predictions)[::-1]
                percentiles = np.arange(5, 101, 5)
                lift_values = []
                
                total_revenue = np.sum(actual)
                for pct in percentiles:
                    top_n = int(len(predictions) * pct / 100)
                    if top_n > 0:
                        top_indices = sorted_indices[:top_n]
                        top_revenue = np.sum(actual[top_indices])
                        revenue_capture_rate = top_revenue / total_revenue if total_revenue > 0 else 0
                        lift = revenue_capture_rate / (pct / 100) if pct > 0 else 0
                        lift_values.append(lift)
                
                ax6.plot(percentiles, lift_values, linewidth=3, color=self.colors[1], marker='o')
                ax6.axhline(y=1, color='r', linestyle='--', linewidth=2)
                ax6.set_xlabel('Top % of Players Targeted')
                ax6.set_ylabel('Lift Factor')
                ax6.set_title('Revenue Targeting Lift Curve')
                ax6.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(save_dir, 'executive_dashboard.png'), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    def create_whale_analysis_dashboard(self, data: pd.DataFrame, save_dir: str) -> None:
        """Create comprehensive whale analysis dashboard"""
        
        print("Creating whale analysis dashboard...")
        print(f"Whale analysis dashboard would be saved to {save_dir}")

