"""
Production-Ready Player LTV Prediction Pipeline
Orchestrates the complete ExpLTV system using modular src/ components
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modular components
from src.data_extractor import DataExtractor
from src.feature_engineer import FeatureEngineer
from src.ziln_model import ZILNModel
from src.backtester import BackTester
from src.visualizer import ResultVisualizer
from src.config import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pltv_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ExpLTVPipeline:
    """Complete ExpLTV prediction pipeline orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize pipeline with configuration"""
        
        logger.info("Initializing ExpLTV Pipeline...")
        
        # Load configuration
        self.config = Config(config_path) if config_path else Config()
        
        # Initialize components
        self.data_extractor = DataExtractor(
            project_id=self.config.project_id,
            credentials_path=self.config.credentials_path
        )
        
        self.feature_engineer = FeatureEngineer()
        self.model = ZILNModel(
            use_neural_network=True,  # Force neural network usage
            nn_params=None  # Use defaults
        )
        self.backtester = BackTester(
            test_size=self.config.test_size,
            n_splits=5,
            random_state=self.config.random_state
        )
        self.visualizer = ResultVisualizer()
        
        # Create output directory
        self.output_dir = self.config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Pipeline initialized. Output directory: {self.output_dir}")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete ExpLTV prediction pipeline"""
        
        logger.info("Starting complete ExpLTV pipeline execution...")
        
        pipeline_results = {
            'start_time': datetime.now(),
            'config': self.config.__dict__,
            'data_extraction': {},
            'feature_engineering': {},
            'user_categorization': {},
            'model_training': {},
            'backtesting': {},
            'visualization': {},
            'final_metrics': {}
        }
        
        try:
            # Step 1: Data Extraction
            logger.info("Step 1: Extracting data from BigQuery...")
            raw_data = self._extract_data()
            pipeline_results['data_extraction'] = {
                'status': 'success',
                'total_players': len(raw_data),
                'data_shape': raw_data.shape,
                'payer_rate': len(raw_data[raw_data['total_revenue'] > 0]) / len(raw_data),
                'total_revenue': raw_data['total_revenue'].sum()
            }
            
            # Step 2: Feature Engineering
            logger.info("Step 2: Engineering advanced behavioral features...")
            engineered_data = self._engineer_features(raw_data)
            pipeline_results['feature_engineering'] = {
                'status': 'success',
                'total_features': len(self.feature_engineer.feature_columns),
                'numeric_features': len(self.feature_engineer.numeric_features),
                'categorical_features': len(self.feature_engineer.categorical_features)
            }
            
            # Step 3: User Categorization
            logger.info("Step 3: Creating user categories (Whales, Low Spenders, Non-payers)...")
            categorized_data = self._categorize_users(engineered_data)
            pipeline_results['user_categorization'] = {
                'status': 'success',
                'category_distribution': categorized_data['future_user_category'].value_counts().to_dict()
            }
            
            # Step 4: Model Training
            logger.info("Step 4: Training ExpLTV model with whale detection...")
            trained_model = self._train_model(categorized_data)
            pipeline_results['model_training'] = {
                'status': 'success',
                'training_stats': self.model.get_training_stats()
            }
            
            # Step 5: Comprehensive Backtesting
            logger.info("Step 5: Running comprehensive backtesting and validation...")
            backtest_results = self._run_backtesting(categorized_data)
            pipeline_results['backtesting'] = {
                'status': 'success',
                'validation_methods': list(backtest_results.keys())
            }
            
            # Step 6: Visualization and Reporting
            logger.info("Step 6: Creating visualizations and reports...")
            self._create_visualizations(categorized_data, backtest_results)
            pipeline_results['visualization'] = {
                'status': 'success',
                'output_directory': self.output_dir
            }
            
            # Step 7: Save Results
            logger.info("Step 7: Saving engineered data and results...")
            self._save_results(categorized_data, backtest_results, pipeline_results)
            
            # Extract final metrics
            if 'random_split' in backtest_results:
                metrics = backtest_results['random_split']['metrics']
                pipeline_results['final_metrics'] = {
                    'r2_score': metrics.get('r2', 0),
                    'rmse': metrics.get('rmse', 0),
                    'auc_payer_classification': metrics.get('auc_payer_classification', 0),
                    'f1_score': metrics.get('f1_score', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'accuracy': metrics.get('binary_accuracy', 0),  # Fixed key name
                    'lift_top_10pct': metrics.get('lift_top_10pct', 1),
                    'revenue_capture_top_10pct': metrics.get('revenue_capture_top_10pct', 0)
                }
            
            pipeline_results['end_time'] = datetime.now()
            pipeline_results['duration_minutes'] = (
                pipeline_results['end_time'] - pipeline_results['start_time']
            ).total_seconds() / 60
            
            logger.info(f"Pipeline completed successfully in {pipeline_results['duration_minutes']:.1f} minutes")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            pipeline_results['error'] = str(e)
            pipeline_results['status'] = 'failed'
            raise
    
    def _extract_data(self) -> pd.DataFrame:
        """Extract data from BigQuery"""
        
        data = self.data_extractor.extract_player_data()
        
        if len(data) == 0:
            raise ValueError("No data extracted from BigQuery")
        
        logger.info(f"Extracted {len(data):,} players from BigQuery")
        return data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced behavioral features"""
        
        engineered_data = self.feature_engineer.create_features(data)
        
        logger.info(f"Created {len(self.feature_engineer.feature_columns)} features")
        return engineered_data
    
    def _categorize_users(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create user categories"""
        
        categorized_data = self.feature_engineer.create_user_categories(data)
        
        category_counts = categorized_data['future_user_category'].value_counts()
        logger.info(f"User categorization complete: {dict(category_counts)}")
        
        return categorized_data
    
    def _train_model(self, data: pd.DataFrame) -> ZILNModel:
        """Train the ExpLTV model"""
        
        # Prepare training data - Apply SMOTE if enabled
        X_data = self.feature_engineer.get_feature_matrix(data)
        
        # If SMOTE is enabled and was applied, X_data will include ltv_target
        if hasattr(self.feature_engineer, 'use_smote') and self.feature_engineer.use_smote and 'ltv_target' in X_data.columns:
            y = X_data['ltv_target'].values
            X = X_data.drop('ltv_target', axis=1)
            logger.info(f"Using SMOTE-resampled data: {len(X)} samples")
        else:
            X = X_data
            y = data['ltv_target'].values  # Use LTV target, not historical revenue
        
        # Create whale labels based on future LTV
        if hasattr(self.feature_engineer, 'use_smote') and self.feature_engineer.use_smote and 'ltv_target' in X_data.columns:
            # For SMOTE data, create whale labels from resampled LTV values
            whale_threshold = np.percentile(y[y > 0], 95) if len(y[y > 0]) > 0 else 0
            whale_labels = (y >= whale_threshold).astype(int)
            logger.info(f"Created whale labels for SMOTE data: {len(whale_labels)} labels, {whale_labels.sum()} whales")
        elif 'future_user_category' in data.columns:
            whale_labels = (data['future_user_category'] == 'Whale').astype(int).values
            # If SMOTE was applied but we don't have resampled categories, extend whale labels
            if len(X) != len(whale_labels):
                logger.warning(f"Mismatch: X has {len(X)} samples, whale_labels has {len(whale_labels)}. Creating new whale labels from LTV.")
                whale_threshold = np.percentile(y[y > 0], 95) if len(y[y > 0]) > 0 else 0
                whale_labels = (y >= whale_threshold).astype(int)
        else:
            # Fallback: create whale labels from LTV target
            whale_threshold = np.percentile(y[y > 0], 95) if len(y[y > 0]) > 0 else 0
            whale_labels = (y >= whale_threshold).astype(int)
        
        # Final safety check for sample size consistency
        if len(X) != len(y) or len(X) != len(whale_labels):
            logger.error(f"Sample size mismatch: X={len(X)}, y={len(y)}, whale_labels={len(whale_labels)}")
            raise ValueError(f"Inconsistent sample sizes: X={len(X)}, y={len(y)}, whale_labels={len(whale_labels)}")
        
        # Train model
        self.model.fit(X, y, whale_labels=whale_labels)
        
        logger.info("Model training completed")
        return self.model
    
    def _run_backtesting(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive backtesting"""
        
        backtest_results = self.backtester.run_backtest(
            data=data,
            model=self.model,
            feature_engineer=self.feature_engineer,
            include_whale_validation=True
        )
        
        logger.info("Backtesting completed")
        return backtest_results
    
    def _create_visualizations(self, data: pd.DataFrame, backtest_results: Dict[str, Any]) -> None:
        """Create comprehensive visualizations"""
        
        # Create all plots
        self.visualizer.create_all_plots(
            data=data,
            backtest_results=backtest_results,
            save_dir=self.output_dir
        )
        
        # Create whale analysis dashboard
        if hasattr(self.visualizer, 'create_whale_analysis_dashboard'):
            self.visualizer.create_whale_analysis_dashboard(data, self.output_dir)
        
        logger.info("Visualizations created")
    
    def _save_results(self, data: pd.DataFrame, backtest_results: Dict[str, Any], 
                     pipeline_results: Dict[str, Any]) -> None:
        """Save all results and engineered data"""
        
        # Save engineered data
        if self.config.save_enhanced_data:
            engineered_data_path = os.path.join(self.output_dir, 'engineered_data.csv')
            data.to_csv(engineered_data_path, index=False)
            logger.info(f"Engineered data saved to {engineered_data_path}")
        
        # Save backtest results
        backtest_results_path = os.path.join(self.output_dir, 'backtest_results.json')
        with open(backtest_results_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = self._make_json_serializable(backtest_results)
            json.dump(serializable_results, f, indent=2, default=str)
        logger.info(f"Backtest results saved to {backtest_results_path}")
        
        # Save pipeline results
        pipeline_results_path = os.path.join(self.output_dir, 'pipeline_results.json')
        with open(pipeline_results_path, 'w') as f:
            serializable_pipeline = self._make_json_serializable(pipeline_results)
            json.dump(serializable_pipeline, f, indent=2, default=str)
        logger.info(f"Pipeline results saved to {pipeline_results_path}")
        
        # Generate validation report
        if hasattr(self.backtester, 'generate_validation_report'):
            report = self.backtester.generate_validation_report(backtest_results)
            report_path = os.path.join(self.output_dir, 'validation_report.txt')
            with open(report_path, 'w') as f:
                f.write(report)
            logger.info(f"Validation report saved to {report_path}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types"""
        
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        else:
            return obj


def main():
    """Main execution function"""
    
    logger.info("Starting ExpLTV Production Pipeline")
    logger.info("Player Lifetime Value Prediction with Whale Detection")
    
    try:
        # Initialize and run pipeline
        pipeline = ExpLTVPipeline()
        results = pipeline.run_complete_pipeline()
        
        # Log execution summary
        logger.info("Pipeline execution completed successfully")
        logger.info(f"Duration: {results['duration_minutes']:.1f} minutes")
        logger.info(f"Total Players: {results['data_extraction']['total_players']:,}")
        logger.info(f"Payer Rate: {results['data_extraction']['payer_rate']*100:.1f}%")
        logger.info(f"Total Features: {results['feature_engineering']['total_features']}")
        
        if 'final_metrics' in results:
            metrics = results['final_metrics']
            logger.info("Model Performance Metrics:")
            logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
            logger.info(f"  AUC (Payer Classification): {metrics['auc_payer_classification']:.4f}")
            logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
            logger.info(f"  Top 10% Lift: {metrics['lift_top_10pct']:.2f}x")
            logger.info(f"  Revenue Capture (Top 10%): {metrics['revenue_capture_top_10pct']*100:.1f}%")
        
        logger.info(f"All results saved to: {pipeline.output_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        logger.error("Check the log file for detailed error information.")
        raise


if __name__ == "__main__":
    main()