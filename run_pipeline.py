#!/usr/bin/env python3
"""
Production LTV Prediction Pipeline Runner
Clean, production-ready entry point for the LTV prediction system
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main_pipeline import ExpLTVPipeline


def main(use_smote=False):
    """Run the LTV prediction pipeline
    
    Args:
        use_smote: Whether to use SMOTE for handling class imbalance
    """
    
    smote_text = "with SMOTE" if use_smote else "without SMOTE" 
    print(f"Starting LTV Prediction Pipeline ({smote_text})...")
    
    try:
        # Initialize and run pipeline
        pipeline = ExpLTVPipeline()
        
        # Set SMOTE usage
        pipeline.feature_engineer.use_smote = use_smote
        
        # Optimize for faster training
        pipeline.model.nn_params.update({
            'epochs': 500,
            'batch_size': 1024,
            'hidden_dim': 256,
            'embedding_dim': 64,
            'early_stopping_patience': 100
        })
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline()
        
        # Print summary
        print(f"\nPipeline Status: {'SUCCESS' if results.get('status') != 'failed' else 'FAILED'}")
        print(f"Duration: {results['duration_minutes']:.1f} minutes")
        print(f"Observation Period: 3 days (May 28-31)")
        print(f"Prediction Period: 30 days (June 2025)")
        print(f"Total Players: {results['data_extraction']['total_players']:,}")
        print(f"Payer Rate: {results['data_extraction']['payer_rate']*100:.1f}%")
        print(f"SMOTE Applied: {'Yes' if use_smote else 'No'}")
        
        if 'final_metrics' in results:
            metrics = results['final_metrics']
            print(f"\nLTV Prediction Performance:")
            print(f"R² Score: {metrics['r2_score']:.4f}")
            print(f"RMSE: ${metrics.get('rmse', 0):.2f}")
            
            print(f"\nPayer Classification Performance:")
            print(f"AUC Score: {metrics['auc_payer_classification']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"Precision: {metrics.get('precision', 0):.4f}")
            print(f"Recall: {metrics.get('recall', 0):.4f}")
            print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
        
        print(f"\nResults saved to: {pipeline.output_dir}")
        
        return results
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        raise


def run_comparison():
    """Run pipeline with and without SMOTE for comparison"""
    
    print("Running LTV Pipeline Comparison: With and Without SMOTE")
    print("=" * 60)
    
    # Run without SMOTE
    print("\n1. Running WITHOUT SMOTE...")
    results_no_smote = main(use_smote=False)
    
    print("\n" + "=" * 60)
    
    # Run with SMOTE
    print("\n2. Running WITH SMOTE...")
    results_with_smote = main(use_smote=True)
    
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    # Compare results
    if 'final_metrics' in results_no_smote and 'final_metrics' in results_with_smote:
        no_smote = results_no_smote['final_metrics']
        with_smote = results_with_smote['final_metrics']
        
        print(f"{'Metric':<20} {'No SMOTE':<12} {'With SMOTE':<12} {'Improvement':<12}")
        print("-" * 60)
        print(f"{'R² Score':<20} {no_smote['r2_score']:<12.4f} {with_smote['r2_score']:<12.4f} {with_smote['r2_score']-no_smote['r2_score']:+.4f}")
        print(f"{'AUC Score':<20} {no_smote['auc_payer_classification']:<12.4f} {with_smote['auc_payer_classification']:<12.4f} {with_smote['auc_payer_classification']-no_smote['auc_payer_classification']:+.4f}")
        print(f"{'F1 Score':<20} {no_smote['f1_score']:<12.4f} {with_smote['f1_score']:<12.4f} {with_smote['f1_score']-no_smote['f1_score']:+.4f}")
        print(f"{'Precision':<20} {no_smote.get('precision',0):<12.4f} {with_smote.get('precision',0):<12.4f} {with_smote.get('precision',0)-no_smote.get('precision',0):+.4f}")
        print(f"{'Recall':<20} {no_smote.get('recall',0):<12.4f} {with_smote.get('recall',0):<12.4f} {with_smote.get('recall',0)-no_smote.get('recall',0):+.4f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        run_comparison()
    elif len(sys.argv) > 1 and sys.argv[1] == "smote":
        main(use_smote=True)
    else:
        main(use_smote=False)