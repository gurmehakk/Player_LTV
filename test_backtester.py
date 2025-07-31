"""
Quick test to verify backtester fixes
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from src.backtester import BackTester
from src.ziln_model import ZILNModel
from src.feature_engineer import FeatureEngineer

def test_backtester():
    """Test the backtester with synthetic data"""
    
    print("Testing BackTester fixes...")
    
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic features
    data = pd.DataFrame({
        'total_revenue': np.random.exponential(10, n_samples) * np.random.binomial(1, 0.3, n_samples),
        'total_sessions': np.random.randint(1, 50, n_samples),
        'total_events': np.random.randint(10, 1000, n_samples),
        'player_lifetime_days': np.random.randint(1, 365, n_samples),
        'avg_session_duration_minutes': np.random.uniform(1, 60, n_samples),
        'avg_events_per_session': np.random.uniform(5, 50, n_samples),
        'first_session_timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='H')
    })
    
    # Initialize components
    feature_engineer = FeatureEngineer()
    
    # Create basic features (simplified)
    data['events_per_day'] = data['total_events'] / np.maximum(data['player_lifetime_days'], 1)
    data['sessions_per_day'] = data['total_sessions'] / np.maximum(data['player_lifetime_days'], 1)
    data['events_per_session'] = data['total_events'] / np.maximum(data['total_sessions'], 1)
    data['is_payer'] = (data['total_revenue'] > 0).astype(int)
    data['player_segment'] = np.random.randint(0, 5, n_samples)
    
    # Set feature columns manually for testing
    feature_engineer.feature_columns = [
        'total_sessions', 'total_events', 'player_lifetime_days',
        'events_per_day', 'sessions_per_day', 'events_per_session',
        'avg_session_duration_minutes', 'avg_events_per_session', 'is_payer'
    ]
    
    # Initialize model and backtester
    model = ZILNModel(use_neural_network=False)  # Use LightGBM for testing
    backtester = BackTester(test_size=0.3, n_splits=2)
    
    try:
        # Test the main backtesting function
        print("Running backtest...")
        results = backtester.run_backtest(
            data=data,
            model=model,
            feature_engineer=feature_engineer,
            include_whale_validation=True
        )
        
        print("✓ Backtesting completed successfully!")
        print(f"Results keys: {list(results.keys())}")
        
        # Test validation report generation
        print("Generating validation report...")
        report = backtester.generate_validation_report(results)
        print("✓ Validation report generated successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_backtester()
    if success:
        print("\\n✅ All backtester fixes verified successfully!")
    else:
        print("\\n❌ Some issues remain in the backtester.")