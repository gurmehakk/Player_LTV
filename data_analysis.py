"""
Deep data analysis to understand why LTV prediction is challenging
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_extractor import DataExtractor
from src.feature_engineer import FeatureEngineer
from src.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_ltv_prediction_challenge():
    """Comprehensive analysis of why LTV prediction is challenging"""
    
    logger.info("🔍 COMPREHENSIVE LTV PREDICTION CHALLENGE ANALYSIS")
    logger.info("=" * 60)
    
    # 1. Extract and prepare data
    logger.info("Step 1: Extracting data...")
    config = Config()
    extractor = DataExtractor(project_id=config.project_id, credentials_path=config.credentials_path)
    raw_data = extractor.extract_player_data()
    
    feature_engineer = FeatureEngineer()
    engineered_data = feature_engineer.create_features(raw_data)
    X = feature_engineer.get_feature_matrix(engineered_data, apply_smote=False)
    
    # Key variables
    historical_revenue = engineered_data['total_revenue'].values  # Observation period
    future_ltv = engineered_data['ltv_target'].values  # Prediction target
    
    logger.info(f"Dataset: {len(engineered_data):,} players")
    
    # 2. Basic data distribution analysis
    logger.info("\\n" + "="*50)
    logger.info("📊 DATA DISTRIBUTION ANALYSIS")
    logger.info("="*50)
    
    historical_payers = historical_revenue > 0
    future_payers = future_ltv > 0
    
    logger.info(f"Historical payers (observation): {np.sum(historical_payers):,} ({np.mean(historical_payers)*100:.1f}%)")
    logger.info(f"Future payers (target): {np.sum(future_payers):,} ({np.mean(future_payers)*100:.1f}%)")
    
    logger.info(f"\\nHistorical revenue stats:")
    logger.info(f"  Mean: ${np.mean(historical_revenue):.3f}")
    logger.info(f"  Std: ${np.std(historical_revenue):.3f}")
    logger.info(f"  Median: ${np.median(historical_revenue):.3f}")
    logger.info(f"  Max: ${np.max(historical_revenue):.2f}")
    logger.info(f"  Payers mean: ${np.mean(historical_revenue[historical_payers]):.3f}")
    
    logger.info(f"\\nFuture LTV stats:")
    logger.info(f"  Mean: ${np.mean(future_ltv):.3f}")
    logger.info(f"  Std: ${np.std(future_ltv):.3f}")
    logger.info(f"  Median: ${np.median(future_ltv):.3f}")
    logger.info(f"  Max: ${np.max(future_ltv):.2f}")
    logger.info(f"  Payers mean: ${np.mean(future_ltv[future_payers]):.3f}")
    
    # 3. Correlation analysis
    logger.info("\\n" + "="*50)
    logger.info("🔗 CORRELATION ANALYSIS")
    logger.info("="*50)
    
    # Overall correlation
    overall_corr = np.corrcoef(historical_revenue, future_ltv)[0, 1]
    logger.info(f"Overall correlation (historical vs future): {overall_corr:.4f}")
    
    # Payer-only correlation
    both_payers = historical_payers & future_payers
    if np.sum(both_payers) > 1:
        payer_corr = np.corrcoef(
            historical_revenue[both_payers], 
            future_ltv[both_payers]
        )[0, 1]
        logger.info(f"Payer-only correlation: {payer_corr:.4f}")
        logger.info(f"Players who paid in both periods: {np.sum(both_payers):,}")
    
    # 4. Predictability analysis
    logger.info("\\n" + "="*50)  
    logger.info("🎯 PREDICTABILITY ANALYSIS")
    logger.info("="*50)
    
    # What percentage of historical payers become future payers?
    if np.sum(historical_payers) > 0:
        retention_rate = np.sum(both_payers) / np.sum(historical_payers)
        logger.info(f"Payer retention rate: {retention_rate*100:.1f}%")
        logger.info(f"  {np.sum(historical_payers):,} historical payers")
        logger.info(f"  {np.sum(both_payers):,} continued paying")
        logger.info(f"  {np.sum(historical_payers) - np.sum(both_payers):,} churned")
    
    # What percentage of future payers were historical non-payers?
    historical_nonpayers_future_payers = (~historical_payers) & future_payers
    if np.sum(future_payers) > 0:
        acquisition_rate = np.sum(historical_nonpayers_future_payers) / np.sum(future_payers)
        logger.info(f"\\nNew payer acquisition: {acquisition_rate*100:.1f}% of future payers were historical non-payers")
        logger.info(f"  {np.sum(future_payers):,} future payers")
        logger.info(f"  {np.sum(historical_nonpayers_future_payers):,} were new acquisitions")
        logger.info(f"  {np.sum(both_payers):,} were retained payers")
    
    # 5. Feature correlation with target
    logger.info("\\n" + "="*50)
    logger.info("📈 FEATURE IMPORTANCE ANALYSIS")
    logger.info("="*50)
    
    # Calculate correlations with target
    feature_correlations = []
    for col in feature_engineer.feature_columns:
        if col in X.columns:
            corr = np.corrcoef(X[col].fillna(0), future_ltv)[0, 1]
            if not np.isnan(corr):
                feature_correlations.append((col, abs(corr), corr))
    
    # Sort by absolute correlation
    feature_correlations.sort(key=lambda x: x[1], reverse=True)
    
    logger.info("Top 10 features by correlation with future LTV:")
    for i, (feature, abs_corr, corr) in enumerate(feature_correlations[:10], 1):
        logger.info(f"  {i:2d}. {feature:<30} | {corr:+.4f} (|{abs_corr:.4f}|)")
    
    # 6. Signal-to-noise analysis
    logger.info("\\n" + "="*50)
    logger.info("🔊 SIGNAL-TO-NOISE ANALYSIS")
    logger.info("="*50)
    
    # Calculate baseline prediction (mean)
    baseline_prediction = np.mean(future_ltv)
    baseline_mse = np.mean((future_ltv - baseline_prediction) ** 2)
    baseline_rmse = np.sqrt(baseline_mse)
    
    logger.info(f"Baseline (mean) prediction: ${baseline_prediction:.4f}")
    logger.info(f"Baseline RMSE: ${baseline_rmse:.4f}")
    logger.info(f"Target standard deviation: ${np.std(future_ltv):.4f}")
    
    # Signal-to-noise ratio
    signal_to_noise = np.std(future_ltv) / np.mean(future_ltv) if np.mean(future_ltv) > 0 else float('inf')
    logger.info(f"Coefficient of variation (noise/signal): {signal_to_noise:.2f}")
    
    # 7. Time horizon analysis
    logger.info("\\n" + "="*50)
    logger.info("⏰ TIME HORIZON ANALYSIS")
    logger.info("="*50)
    
    logger.info("Current setup:")
    logger.info(f"  Observation period: 3 days (May 28-31)")
    logger.info(f"  Prediction period: 30 days (June 2025)")
    logger.info(f"  Prediction horizon: 10x observation period")
    
    # Calculate what would happen with shorter prediction horizons
    # This is theoretical since we only have one target period
    logger.info("\\nTheoretical analysis:")
    logger.info("  Predicting 30 days from 3 days is like:")
    logger.info("  - Predicting a month from a weekend")
    logger.info("  - 10x extrapolation factor")
    logger.info("  - High uncertainty expected")
    
    # 8. Business impact analysis
    logger.info("\\n" + "="*50)
    logger.info("💼 BUSINESS IMPACT ANALYSIS")
    logger.info("="*50)
    
    total_future_revenue = np.sum(future_ltv)
    total_historical_revenue = np.sum(historical_revenue)
    
    logger.info(f"Total historical revenue: ${total_historical_revenue:.2f}")
    logger.info(f"Total future revenue: ${total_future_revenue:.2f}")
    logger.info(f"Future/Historical ratio: {total_future_revenue/total_historical_revenue:.2f}x" if total_historical_revenue > 0 else "N/A")
    
    # Revenue concentration
    if np.sum(future_payers) > 0:
        future_payer_ltv = future_ltv[future_payers]
        top_10_pct_idx = int(len(future_payer_ltv) * 0.9)
        top_10_pct_revenue = np.sum(np.sort(future_payer_ltv)[top_10_pct_idx:])
        revenue_concentration = top_10_pct_revenue / total_future_revenue
        logger.info(f"Revenue concentration: Top 10% of payers = {revenue_concentration*100:.1f}% of revenue")
    
    # 9. Conclusions and recommendations
    logger.info("\\n" + "="*60)
    logger.info("🎯 CONCLUSIONS & RECOMMENDATIONS")
    logger.info("="*60)
    
    # Determine the main issues
    issues = []
    recommendations = []
    
    if overall_corr < 0.1:
        issues.append(f"Very weak correlation between historical and future revenue ({overall_corr:.3f})")
        recommendations.append("Consider shorter prediction horizons (7-14 days)")
    
    if retention_rate < 0.5:
        issues.append(f"Low payer retention rate ({retention_rate*100:.1f}%)")
        recommendations.append("Focus on churn prediction rather than revenue amounts")
    
    if acquisition_rate > 0.5:
        issues.append(f"High new payer acquisition ({acquisition_rate*100:.1f}%)")
        recommendations.append("Historical spending is poor predictor of new spenders")
    
    if signal_to_noise > 5:
        issues.append(f"High noise-to-signal ratio ({signal_to_noise:.1f})")
        recommendations.append("Consider classification instead of regression")
    
    logger.info("🚨 IDENTIFIED ISSUES:")
    for i, issue in enumerate(issues, 1):
        logger.info(f"  {i}. {issue}")
    
    logger.info("\\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"  {i}. {rec}")
    
    # Additional strategic recommendations
    logger.info("\\n🎯 STRATEGIC RECOMMENDATIONS:")
    logger.info("  1. Switch to classification: Predict 'will spend >$X' instead of exact amounts")
    logger.info("  2. Shorter horizons: Try 7-day or 14-day LTV prediction")
    logger.info("  3. Segmented models: Build separate models for different player types")
    logger.info("  4. Ensemble approach: Combine classification + amount prediction")
    logger.info("  5. Feature enrichment: Add behavioral progression features")
    logger.info("  6. External data: Consider device, demographic, or game-specific features")
    
    # 10. Create visualizations
    logger.info("\\n📊 Creating diagnostic visualizations...")
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('LTV Prediction Challenge Analysis', fontsize=16, fontweight='bold')
    
    # 1. LTV distribution
    axes[0,0].hist(future_ltv[future_ltv > 0], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Future LTV Distribution (Payers Only)')
    axes[0,0].set_xlabel('Future LTV ($)')
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_yscale('log')
    
    # 2. Historical vs Future scatter
    sample_mask = np.random.choice(len(future_ltv), min(5000, len(future_ltv)), replace=False)
    axes[0,1].scatter(historical_revenue[sample_mask], future_ltv[sample_mask], alpha=0.5)
    axes[0,1].set_title(f'Historical vs Future Revenue\\n(Correlation: {overall_corr:.3f})')
    axes[0,1].set_xlabel('Historical Revenue ($)')
    axes[0,1].set_ylabel('Future LTV ($)')
    
    # 3. Payer transition matrix
    transition_data = pd.crosstab(
        pd.cut(historical_revenue, bins=[-0.1, 0, 1, 10, 100], labels=['$0', '$0-1', '$1-10', '$10+']),
        pd.cut(future_ltv, bins=[-0.1, 0, 1, 10, 100], labels=['$0', '$0-1', '$1-10', '$10+']),
        normalize='index'
    )
    sns.heatmap(transition_data, annot=True, fmt='.2f', ax=axes[0,2], cmap='Blues')
    axes[0,2].set_title('Revenue Transition Matrix')
    axes[0,2].set_xlabel('Future LTV')
    axes[0,2].set_ylabel('Historical Revenue')
    
    # 4. Feature correlation heatmap (top features)
    top_features = [feat[0] for feat in feature_correlations[:10]]
    corr_matrix = X[top_features].corrwith(pd.Series(future_ltv)).to_frame('LTV_Correlation')
    axes[1,0].barh(range(len(corr_matrix)), corr_matrix['LTV_Correlation'])
    axes[1,0].set_yticks(range(len(corr_matrix)))
    axes[1,0].set_yticklabels([feat.replace('_', ' ').title()[:20] for feat in top_features])
    axes[1,0].set_title('Top Feature Correlations')
    axes[1,0].set_xlabel('Correlation with Future LTV')
    
    # 5. Payer behavior analysis
    categories = ['Historical\\nOnly', 'Future\\nOnly', 'Both\\nPeriods', 'Neither\\nPeriod']
    counts = [
        np.sum(historical_payers & ~future_payers),
        np.sum(~historical_payers & future_payers), 
        np.sum(both_payers),
        np.sum(~historical_payers & ~future_payers)
    ]
    axes[1,1].pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
    axes[1,1].set_title('Player Payment Behavior')
    
    # 6. Prediction difficulty visualization
    difficulty_metrics = ['Correlation', 'Retention Rate', 'Signal/Noise', 'R² Achievable']
    difficulty_values = [
        max(0, overall_corr * 10),  # Scale correlation
        retention_rate,
        min(1, 1/signal_to_noise),  # Inverse for difficulty
        max(0, overall_corr ** 2)  # Theoretical R² upper bound
    ]
    axes[1,2].bar(difficulty_metrics, difficulty_values, color=['red', 'orange', 'yellow', 'green'])
    axes[1,2].set_title('Prediction Difficulty Metrics')
    axes[1,2].set_ylabel('Score (0=Difficult, 1=Easy)')
    axes[1,2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('output/ltv_prediction_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("📊 Analysis complete! Visualizations saved to 'output/ltv_prediction_analysis.png'")
    
    # 11. Final assessment
    logger.info("\\n" + "="*60)
    logger.info("🏁 FINAL ASSESSMENT")
    logger.info("="*60)
    
    if overall_corr < 0.05:
        assessment = "❌ EXTREMELY DIFFICULT - Fundamental prediction challenge"
        confidence = "Very Low"
    elif overall_corr < 0.15:
        assessment = "⚠️  CHALLENGING - Significant prediction difficulty" 
        confidence = "Low"
    elif overall_corr < 0.3:
        assessment = "⚡ MODERATE - Some predictive signal present"
        confidence = "Medium"
    else:
        assessment = "✅ FEASIBLE - Good predictive potential"
        confidence = "High"
    
    logger.info(f"Prediction Assessment: {assessment}")
    logger.info(f"Confidence Level: {confidence}")
    logger.info(f"Expected R² Range: {max(0, overall_corr**2 - 0.1):.3f} - {overall_corr**2:.3f}")
    
    return {
        'overall_correlation': overall_corr,
        'retention_rate': retention_rate if 'retention_rate' in locals() else 0,
        'signal_to_noise': signal_to_noise,
        'assessment': assessment,
        'recommendations': recommendations
    }


if __name__ == "__main__":
    analyze_ltv_prediction_challenge()