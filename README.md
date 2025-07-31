# ExpLTV Production System

A comprehensive Player Lifetime Value (pLTV) prediction system that combines BigQuery data extraction with ExpLTV methodology, whale detection, and comprehensive backtesting.

## Features

- **BigQuery Data Extraction**: Comprehensive feature extraction from app_data table
- **Advanced Feature Engineering**: 50+ behavioral, engagement, and progression features  
- **User Categorization**: Automatic classification into Whales, Medium Spenders, Low Spenders, and Non-payers
- **ExpLTV Methodology**: Implements research-based approach with expert routing
- **Whale Detection**: Specialized detection and analysis of high-value players
- **ZILN Model**: Zero-Inflated Log-Normal model with LightGBM implementation
- **Comprehensive Backtesting**: Multiple validation approaches including time-based splits
- **Rich Visualizations**: Comprehensive dashboards and analysis plots
- **Production Ready**: Modular architecture with proper configuration management

## Directory Structure

```
Player_LTV/
├── src/                          # Modular components
│   ├── data_extractor.py        # BigQuery data extraction
│   ├── feature_engineer.py      # Advanced feature engineering
│   ├── ziln_model.py            # ExpLTV model with whale detection
│   ├── backtester.py            # Comprehensive validation
│   ├── visualizer.py            # Visualization system
│   └── config.py                # Configuration management
├── run_pipeline.py              # Simple pipeline runner
├── main_pipeline.py             # Comprehensive pipeline orchestrator
├── main.py                      # Original ExpLTV implementation
├── requirements.txt             # Python dependencies
└── output/                      # Generated results
    ├── engineered_data.csv      # Feature-engineered dataset
    ├── visualizations/          # Analysis plots
    ├── backtest_results.json    # Validation results
    └── validation_report.txt    # Performance report
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up BigQuery Authentication

```bash
# Set up Google Cloud credentials
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
```

### 3. Run the Pipeline

```bash
# Simple run
python run_pipeline.py

# Or comprehensive pipeline
python main_pipeline.py
```

## Key Components

### Data Extraction (`src/data_extractor.py`)
- Extracts comprehensive player behavioral data from BigQuery
- Handles session-level aggregation and feature computation
- Supports configurable lookback periods

### Feature Engineering (`src/feature_engineer.py`)
- Creates 50+ advanced behavioral features
- Implements user categorization (Whales, Low/Medium Spenders, Non-payers)
- Handles feature scaling and encoding

### Model (`src/ziln_model.py`)
- Zero-Inflated Log-Normal (ZILN) model implementation
- Supports both neural network and LightGBM backends
- Includes whale detection capabilities

### Backtesting (`src/backtester.py`)
- Multiple validation approaches (random, time-based, cross-validation)
- Whale detection performance analysis
- Comprehensive performance metrics

### Visualization (`src/visualizer.py`)
- Revenue distribution analysis
- Model performance dashboards
- Whale detection visualizations
- Feature importance plots

## Model Performance Metrics

The system tracks comprehensive metrics including:

- **Revenue Prediction**: R², MAE, RMSE, MAPE
- **Payer Classification**: AUC, Precision, Recall, F1-Score
- **Business Impact**: Lift analysis, Revenue capture rates
- **Whale Detection**: Whale-specific AUC, Precision, Recall
- **Model Calibration**: Brier score, Calibration error

## Expected Results

With default configuration, you should expect:

- **Data**: 10K-100K+ players depending on lookback period
- **Features**: 50+ engineered behavioral features
- **Categories**: ~10% Whales, ~15% Medium Spenders, ~25% Low Spenders, ~50% Non-payers
- **Performance**: R² > 0.3, AUC > 0.8, Top 10% Lift > 3x

## Advanced Usage

### Custom Feature Engineering

```python
from src.feature_engineer import FeatureEngineer

fe = FeatureEngineer()
data = fe.create_features(raw_data)
categorized_data = fe.create_user_categories(data)
```

### Model Training

```python
from src.ziln_model import ZILNModel

model = ZILNModel(use_neural_network=False)  # Use LightGBM
model.fit(X, y, whale_labels=whale_labels)
predictions, payer_probs, amounts, whale_probs = model.predict(X_test)
```

### Comprehensive Backtesting

```python
from src.backtester import BackTester

backtester = BackTester()
results = backtester.run_backtest(data, model, feature_engineer)
```

## Output Files

After running the pipeline, check the `output/` directory for:

1. **engineered_data.csv**: Complete dataset with all engineered features
2. **visualizations/**: Comprehensive analysis plots and dashboards
3. **backtest_results.json**: Detailed validation results
4. **validation_report.txt**: Human-readable performance summary

## Troubleshooting

### Common Issues

1. **BigQuery Authentication**: Ensure GOOGLE_APPLICATION_CREDENTIALS is set
2. **Memory Issues**: Reduce lookback period or batch size
3. **Feature Errors**: Check for missing columns in source data

### Performance Optimization

- Use LightGBM backend for faster training (`use_neural_network=False`)
- Reduce lookback period for smaller datasets
- Adjust batch_size for memory constraints

## Research Background

This implementation is inspired by the ExpLTV methodology for mobile game player lifetime value prediction.