"""
Simple script to run the ExpLTV pipeline
"""

import sys
import os
import numpy as np

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_extractor import DataExtractor
from src.feature_engineer import FeatureEngineer
from src.ziln_model import ZILNModel
from src.backtester import BackTester
from src.visualizer import ResultVisualizer

def run_pltv_pipeline():
    """Run the complete pLTV pipeline"""
    
    print("="*60)
    print("ExpLTV Production Pipeline")
    print("Player Lifetime Value Prediction with Whale Detection")
    print("="*60)
    
    try:
        # Initialize components
        print("\\nInitializing components...")
        
        data_extractor = DataExtractor(project_id='gc-forecasting-dev')
        feature_engineer = FeatureEngineer()
        model = ZILNModel(use_neural_network=True)  # Use LightGBM by default
        backtester = BackTester()
        visualizer = ResultVisualizer()
        
        # Step 1: Extract data with proper LTV targets
        print("\\n1. Extracting data from BigQuery...")
        print("   Using Jan-May observation period and May-June prediction period for LTV")
        # Use fixed parameters since we're using static dates now
        raw_data = data_extractor.extract_player_data()
        print(f"   Extracted {len(raw_data):,} players with LTV targets")
        print("   Columns include historical features + ltv_target (future revenue)")
        print(f"   Sample columns: {list(raw_data.columns[:10])}...")
        
        os.makedirs('output', exist_ok=True)
        raw_data.to_csv('output/raw_data.csv', index=False)
        
        # Step 2: Engineer features
        print("\\n2. Engineering features...")
        engineered_data = feature_engineer.create_features(raw_data)
        print(f"   Created {len(feature_engineer.feature_columns)} features")
        
        engineered_data.to_csv('output/engineered_data.csv', index=False)
        
        # Step 3: Create user categories (future-focused for whale detection)
        print("\\n3. Creating user categories...")
        categorized_data = feature_engineer.create_user_categories(engineered_data)
        
        # Step 4: Train LTV prediction model
        print("\\n4. Training LTV prediction model...")
        print("   Target: ltv_target (30-day future revenue)")
        print("   Features: Based on 90-day observation period only")
        X = feature_engineer.get_feature_matrix(categorized_data)
        y = categorized_data['ltv_target'].values  # Use LTV target, not historical revenue
        whale_labels = categorized_data['will_be_whale'].values  # Future whale prediction
        
        model.fit(X, y, whale_labels=whale_labels)
        print("   Model training completed")
        
        # Step 5: Run backtesting
        print("\\n5. Running backtesting...")
        backtest_results = backtester.run_backtest(
            data=categorized_data,
            model=model,
            feature_engineer=feature_engineer,
            include_whale_validation=True
        )
        
        # Step 6: Create visualizations
        print("\\n6. Creating visualizations...")
        os.makedirs('output', exist_ok=True)
        visualizer.create_all_plots(
            data=categorized_data,
            backtest_results=backtest_results,
            save_dir='output'
        )
        
        # Save engineered data
        print("\\n7. Saving results...")
        categorized_data.to_csv('output/engineered_data.csv', index=False)
        
        # Print summary
        if 'random_split' in backtest_results:
            metrics = backtest_results['random_split']['metrics']
            print("\\n" + "="*60)
            print("LTV PREDICTION PIPELINE RESULTS")
            print("="*60)
            print(f"Total Players: {len(categorized_data):,}")
            print(f"Observation Period: Jan-May 2025 | Prediction Period: May-Jun 2025")
            print(f"Historical Payers: {np.sum(categorized_data['total_revenue'] > 0):,}")
            print(f"Future Payers (LTV > 0): {np.sum(categorized_data['ltv_target'] > 0):,}")
            print("")
            print("MODEL PERFORMANCE:")
            print(f"R² Score (LTV Prediction): {metrics.get('r2', 0):.4f}")
            print(f"AUC (Future Payer Classification): {metrics.get('auc_payer_classification', 0):.4f}")
            print(f"F1 Score: {metrics.get('f1_score', 0):.4f}")
            print(f"Top 10% Lift: {metrics.get('lift_top_10pct', 1):.2f}x")
            print(f"LTV Capture (Top 10%): {metrics.get('revenue_capture_top_10pct', 0)*100:.1f}%")
            print(f"\\nResults saved to: output/")
            print("="*60)
        
        return backtest_results
        
    except Exception as e:
        print(f"\\nERROR: Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_pltv_pipeline()