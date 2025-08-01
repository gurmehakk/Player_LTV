"""
Test script to evaluate simplified LTV regression approach
"""

import sys
import os
import logging
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_extractor import DataExtractor
from src.feature_engineer import FeatureEngineer
from src.config import Config
from simple_ltv_model import compare_simple_models

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test simplified LTV regression approaches"""
    
    logger.info("Testing Simplified LTV Regression Models")
    logger.info("=" * 50)
    
    try:
        # 1. Extract data
        logger.info("Step 1: Extracting data...")
        config = Config()
        extractor = DataExtractor(
            project_id=config.project_id,
            credentials_path=config.credentials_path
        )
        raw_data = extractor.extract_player_data()
        logger.info(f"Extracted {len(raw_data):,} players")
        
        # 2. Engineer features (without SMOTE for fair comparison)
        logger.info("Step 2: Engineering features...")
        feature_engineer = FeatureEngineer()
        engineered_data = feature_engineer.create_features(raw_data)
        
        # Get feature matrix (no SMOTE)
        X = feature_engineer.get_feature_matrix(engineered_data, apply_smote=False)
        y = engineered_data['ltv_target'].values
        
        logger.info(f"Features: {len(feature_engineer.feature_columns)}")
        logger.info(f"Samples: {len(y):,}")
        logger.info(f"Payer rate: {np.mean(y > 0)*100:.1f}%")
        logger.info(f"LTV stats: mean=${np.mean(y):.3f}, std=${np.std(y):.3f}, max=${np.max(y):.2f}")
        
        # 3. Compare simple models
        logger.info("\\nStep 3: Comparing simplified models...")
        results = compare_simple_models(X, y)
        
        # 4. Print detailed results
        logger.info("\\n" + "=" * 50)
        logger.info("DETAILED RESULTS")
        logger.info("=" * 50)
        
        for model_type, result in results.items():
            if model_type == 'best_model' or model_type == 'test_data':
                continue
                
            if 'metrics' in result:
                metrics = result['metrics']
                logger.info(f"\\n{model_type.upper()} Model Performance:")
                logger.info(f"  R² Score: {metrics['r2']:.4f}")
                logger.info(f"  RMSE: ${metrics['rmse']:.4f}")
                logger.info(f"  MAE: ${metrics['mae']:.4f}")
                logger.info(f"  MAPE: {metrics['mape']:.1f}%")
                logger.info(f"  Payer R²: {metrics['payer_r2']:.4f}")
                logger.info(f"  Payer MAE: ${metrics['payer_mae']:.4f}")
                logger.info(f"  Total Actual: ${metrics['total_actual']:.2f}")
                logger.info(f"  Total Predicted: ${metrics['total_predicted']:.2f}")
                logger.info(f"  Prediction Error: {metrics['prediction_error_pct']:.1f}%")
                
                # Assessment
                r2 = metrics['r2']
                if r2 > 0.1:
                    assessment = "✅ GOOD - Positive R²"
                elif r2 > 0:
                    assessment = "⚠️  FAIR - Small positive R²"
                else:
                    assessment = "❌ POOR - Negative R²"
                logger.info(f"  Assessment: {assessment}")
            else:
                logger.info(f"\\n{model_type.upper()} Model: ERROR - {result.get('error', 'Unknown error')}")
        
        # 5. Best model summary
        if 'best_model' in results:
            best_type = results['best_model']
            if best_type and best_type in results:
                best_metrics = results[best_type]['metrics']
                logger.info(f"\\n🏆 BEST MODEL: {best_type.upper()}")
                logger.info(f"   R² Score: {best_metrics['r2']:.4f}")
                logger.info(f"   RMSE: ${best_metrics['rmse']:.4f}")
                
                if best_metrics['r2'] > 0:
                    logger.info("✅ SUCCESS: Achieved positive R² score!")
                    
                    # Save best model results
                    test_X = results['test_data']['X'] 
                    test_y = results['test_data']['y']
                    best_model = results[best_type]['model']
                    predictions = best_model.predict(test_X)
                    
                    # Create results dataframe
                    results_df = pd.DataFrame({
                        'actual_ltv': test_y,
                        'predicted_ltv': predictions,
                        'error': test_y - predictions,
                        'abs_error': np.abs(test_y - predictions)
                    })
                    
                    results_df.to_csv('output/simple_ltv_results.csv', index=False)
                    logger.info("📊 Results saved to output/simple_ltv_results.csv")
                    
                else:
                    logger.info("❌ Still negative R² - Need further investigation")
        
        # 6. Recommendations
        logger.info("\\n" + "=" * 50)
        logger.info("RECOMMENDATIONS")
        logger.info("=" * 50)
        
        best_r2 = max([r['metrics']['r2'] for r in results.values() if 'metrics' in r])
        
        if best_r2 > 0.1:
            logger.info("✅ Model performance is acceptable for business use")
            logger.info("   - Use best model for LTV predictions")
            logger.info("   - Consider ensemble approach for robustness")
        elif best_r2 > 0:
            logger.info("⚠️  Model shows promise but needs improvement")
            logger.info("   - Consider feature selection/engineering")
            logger.info("   - Try ensemble methods")
            logger.info("   - Collect more diverse features")
        else:
            logger.info("❌ Fundamental prediction challenge identified")
            logger.info("   - 3-day observation may be insufficient for 30-day LTV")
            logger.info("   - Consider shorter prediction horizons")
            logger.info("   - Focus on payer classification instead of exact amounts")
            logger.info("   - Investigate data quality issues")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in simplified LTV testing: {e}")
        raise


if __name__ == "__main__":
    main()