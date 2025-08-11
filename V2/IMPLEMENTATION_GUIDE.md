# LTV Prediction Pipeline - Complete Implementation Guide

## 📋 Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Module Structure](#module-structure)
4. [Data Flow](#data-flow)
5. [Model Training](#model-training)
6. [Visualization System](#visualization-system)
7. [Monitoring & Accuracy Loss](#monitoring--accuracy-loss)
8. [Usage Guide](#usage-guide)
9. [Configuration](#configuration)
10. [Deployment](#deployment)

## 🎯 Overview

This is a comprehensive **Lifetime Value (LTV) Prediction Pipeline** designed for mobile game monetization. The system predicts user spending behavior using early behavioral features (D0-D3) to forecast long-term revenue potential (D4-D33).

### Key Features
- **Modular Architecture**: Clean separation of concerns across 7 modules
- **8 ML Models**: Ensemble of linear, tree-based, and hybrid models
- **Comprehensive Monitoring**: Accuracy loss tracking and retraining alerts
- **Advanced Visualization**: 15+ different plots and analysis charts
- **Production Ready**: Scalable, maintainable, and well-documented

### Business Impact
- **Early Revenue Prediction**: Identify high-value users within 3 days
- **Targeted Marketing**: Optimize ad spend on promising users
- **Retention Strategies**: Personalize content based on LTV predictions
- **Portfolio Management**: Understand game monetization potential

## 🏗️ Architecture

### System Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Feature Eng   │    │   ML Models     │
│                 │    │                 │    │                 │
│ • BigQuery      │ -> │ • Behavioral    │ -> │ • Ensemble      │
│ • CSV Files     │    │ • Engagement    │    │ • Tree Models   │
│ • App Events    │    │ • Temporal      │    │ • Linear Models │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         v                       v                       v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Preprocessing  │    │   Validation    │    │  Visualization  │
│                 │    │                 │    │                 │
│ • Scaling       │    │ • Cross-Val     │ -> │ • Performance   │
│ • Encoding      │    │ • Overfitting   │    │ • Monitoring    │
│ • Missing Data  │    │ • Drift         │    │ • Comparisons   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Technology Stack
- **Core ML**: scikit-learn, XGBoost, CatBoost
- **Data**: pandas, numpy, BigQuery
- **Visualization**: matplotlib, seaborn
- **Storage**: pickle, joblib, CSV
- **Monitoring**: Custom accuracy loss tracking

## 📁 Module Structure

### 1. **config.py** - Configuration & Imports
```python
# Key Responsibilities:
- Global imports and constants
- Feature availability flags (XGBoost, CatBoost, TensorFlow)
- Random seeds and configuration parameters
- PLOT_DATA global variable management
```

### 2. **data_loader.py** - Data Loading & Preparation
```python
# Key Functions:
- load_existing_datasets()      # Main data loading orchestrator
- load_temporal_datasets()      # BigQuery temporal data loading
- load_from_csv_with_ltv()      # CSV fallback with synthetic LTV
- create_synthetic_ltv_targets() # Realistic LTV generation
- save_datasets()               # Data persistence
```

**Data Loading Strategy:**
1. **Primary**: BigQuery temporal approach (D0-D3 → D4-D33)
2. **Fallback**: CSV with sophisticated synthetic LTV generation
3. **Validation**: Comprehensive data quality checks
4. **Persistence**: Automatic saving for reproducibility

### 3. **preprocessing.py** - Feature Engineering
```python
# Key Functions:
- prepare_features()     # Main preprocessing orchestrator
- engineer_features()    # Create derived features
- handle_categorical()   # Label encoding for categorical data
- handle_numerical()     # Missing value imputation
```

**Feature Engineering Pipeline:**
- **Behavioral Features**: Session patterns, tutorial completion
- **Engagement Metrics**: Level progression, ad interactions
- **Temporal Features**: Day-over-day changes, retention scores
- **Composite Features**: Events per session, activity consistency

### 4. **models.py** - Model Training & Ensembling
```python
# Key Functions:
- train_ensemble_ltv_models()  # Main model training orchestrator
- train_hybrid_model()         # Classification + Regression approach
- train_ensemble_model()       # Weighted ensemble creation
```

**Model Architecture:**
```
8 Core Models:
├── Linear Models
│   ├── Ridge Regression (α=10.0)
│   └── ElasticNet (α=0.1, l1_ratio=0.5)
├── Tree-Based Models
│   ├── Random Forest (100 trees, depth=8)
│   ├── XGBoost (200 trees, depth=4)
│   ├── CatBoost (200 iterations, depth=6)
│   └── Gradient Boosting (200 trees)
├── Hybrid Model
│   └── Classification → Regression (2-stage)
└── Ensemble Model
    └── Weighted average of top 3 models
```

### 5. **evaluation.py** - Performance Monitoring
```python
# Key Functions:
- collect_evaluation_data()     # Comprehensive metrics collection
- calculate_accuracy_loss()     # Degradation analysis
- save_baseline_metrics()       # Performance baselines
- load_baseline_metrics()       # Baseline restoration
```

**Monitoring Metrics:**
- **Performance**: RMSE, R², MAE, MAPE
- **Business**: Revenue capture rate, spender detection
- **Validation**: Top percentile analysis, correlation metrics
- **Degradation**: Accuracy loss tracking, retraining alerts

### 6. **visualization.py** - Comprehensive Plotting
```python
# Key Functions:
- create_train_test_accuracy_plots()  # Training vs testing analysis
- create_comprehensive_plots()        # Model comparison charts
- create_roc_pr_curves()             # Binary classification curves
- create_individual_model_plots()     # Detailed model analysis
```

**Visualization Suite:**
1. **Training Analysis**: Train/test accuracy, overfitting detection
2. **Performance Comparison**: R², RMSE, revenue capture
3. **Binary Classification**: ROC/PR curves for LTV>0 detection
4. **Individual Analysis**: Residuals, distributions, metrics
5. **Monitoring Dashboards**: Accuracy loss, health status

### 7. **main.py** - Pipeline Orchestration
```python
# Key Functions:
- main()                        # Full pipeline execution
- update_baseline_from_current() # Baseline management
```

## 🔄 Data Flow

### End-to-End Process
```
1. Data Loading
   ├── BigQuery Query Execution (D0-D3 → D4-D33)
   ├── Fallback to CSV with Synthetic LTV
   └── Data Quality Validation

2. Preprocessing
   ├── Feature Selection (72 → 50 features)
   ├── Categorical Encoding (14 categorical features)
   ├── Feature Engineering (8 derived features)
   └── Train/Val/Test Split (70%/15%/15%)

3. Model Training
   ├── Feature Scaling (RobustScaler for linear models)
   ├── Individual Model Training (8 models)
   ├── Ensemble Creation (top 3 weighted average)
   └── Model Persistence (joblib pickle)

4. Evaluation & Monitoring
   ├── Comprehensive Metrics Collection
   ├── Baseline Comparison & Accuracy Loss
   ├── Performance Visualization (15+ plots)
   └── Retraining Alert System

5. Output Generation
   ├── Model Comparison CSV
   ├── LTV Detection Performance CSV
   ├── Visualization PNG Files
   └── Model Artifacts (PKL files)
```

## 🤖 Model Training

### Training Strategy
1. **Feature Selection**: SelectKBest with F-regression (top 50)
2. **Scaling Strategy**: RobustScaler for linear, raw features for trees
3. **Validation**: 15% holdout validation set
4. **Ensemble Method**: Weighted averaging based on validation R²

### Model Configurations
```python
Ridge Optimal:
- Alpha: 10.0
- Features: Scaled (RobustScaler)
- Use Case: Baseline linear model

ElasticNet Balanced:
- Alpha: 0.1, L1 Ratio: 0.5
- Features: Scaled
- Use Case: Feature selection + regularization

XGBoost Optimized:
- Trees: 200, Depth: 4, LR: 0.1
- Features: Raw (handles scaling internally)
- Use Case: Non-linear pattern detection

Random Forest Balanced:
- Trees: 100, Depth: 8
- Min Samples: Split=10, Leaf=5
- Use Case: Robust ensemble baseline

CatBoost Optimized:
- Iterations: 200, Depth: 6
- Learning Rate: 0.1
- Use Case: Categorical feature handling

Gradient Boosting:
- Trees: 200, Depth: 4, LR: 0.1
- Subsample: 0.8
- Use Case: Sequential error correction

Hybrid Model:
- Stage 1: RandomForest Classifier (payer detection)
- Stage 2: Ridge Regressor (LTV prediction)
- Combination: P(payer) × Predicted_LTV

Ensemble Model:
- Method: Weighted average of top 3 models
- Weights: Normalized validation R² scores
- Selection: Dynamic based on performance
```

### Performance Expectations
- **R² Range**: 0.08 - 0.15 (typical for early behavioral prediction)
- **RMSE Range**: $13 - $15 (depending on LTV distribution)
- **Spender Detection AUC**: 0.54 - 0.56 (better than random)
- **Revenue Capture**: 85-95% (top 10% users capture 50%+ revenue)

## 📊 Visualization System

### 1. Training/Testing Analysis (NEW)
**File**: `plots/train_test_accuracy_analysis.png`
- **Purpose**: Detect overfitting and training issues
- **Charts**: Train vs Test R², RMSE comparison, overfitting gaps
- **Features**: Color-coded health status, performance rankings

### 2. Model Comparison Dashboard
**File**: `plots/model_comparison_comprehensive.png`
- **Purpose**: Compare all models across key metrics
- **Charts**: R² bars, RMSE bars, revenue capture, performance scatter

### 3. ROC/PR Curves
**File**: `plots/roc_pr_curves_all_models.png`
- **Purpose**: Evaluate binary classification performance (LTV > $0)
- **Metrics**: ROC-AUC, PR-AUC, Precision@90%Recall

### 4. Individual Model Analysis
**Directory**: `plots/individual_models/`
- **Purpose**: Detailed analysis for each model
- **Charts**: Actual vs Predicted, Residuals, LTV distributions, Metrics summary

### 5. Accuracy Loss Monitoring
**File**: `plots/monitoring/accuracy_loss_analysis.png`
- **Purpose**: Production monitoring and retraining alerts
- **Charts**: Baseline comparison, degradation timeline, health dashboard

## 📈 Monitoring & Accuracy Loss

### Baseline Management
```python
# Automatic baseline creation on first run
save_baseline_metrics(model_results)

# Load existing baseline for comparison
load_baseline_metrics()

# Manual baseline update when needed
python run_pipeline.py update_baseline
```

### Degradation Thresholds
- **R² Loss > 15%**: 🚨 Critical - Retraining recommended
- **RMSE Increase > 20%**: 🚨 Critical - Retraining recommended
- **R² Loss 5-15%**: 🟡 Warning - Monitor closely
- **R² Loss < 5%**: 🟢 Healthy - Normal operation

### Monitoring Dashboard Features
- **Visual Comparison**: Baseline vs current performance
- **Timeline Analysis**: Performance degradation over time
- **Alert System**: Automated retraining recommendations
- **Health Status**: Color-coded model health indicators

## 🚀 Usage Guide

### Basic Pipeline Execution
```bash
# Run full pipeline
python run_pipeline.py

# Update baseline metrics
python run_pipeline.py update_baseline
```

### Module-Level Usage
```python
# Import specific modules
from modules.data_loader import load_existing_datasets
from modules.models import train_ensemble_ltv_models
from modules.visualization import create_train_test_accuracy_plots

# Custom pipeline
data = load_existing_datasets()
models, predictions, results = train_ensemble_ltv_models(*data)
create_train_test_accuracy_plots()
```

### Configuration Customization
```python
# Modify model parameters
from modules.models import train_ensemble_ltv_models

# Custom Ridge parameters
model_configs = {
    'Custom Ridge': Ridge(alpha=5.0, fit_intercept=True)
}
```

## ⚙️ Configuration

### Environment Variables
```bash
# BigQuery Configuration
export GOOGLE_CLOUD_PROJECT="gc-forecasting-dev"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"

# Pipeline Configuration
export LTV_RANDOM_STATE=42
export LTV_TEST_SIZE=0.15
export LTV_VAL_SIZE=0.15
```

### File Structure
```
V2/
├── modules/                    # Modular components
│   ├── __init__.py
│   ├── config.py              # Configuration & imports
│   ├── data_loader.py         # Data loading & CSV fallback
│   ├── preprocessing.py       # Feature engineering
│   ├── models.py              # Model training & ensembling
│   ├── evaluation.py          # Performance monitoring
│   ├── visualization.py       # Comprehensive plotting
│   └── main.py               # Pipeline orchestration
├── run_pipeline.py            # Main entry point
├── data/                      # Generated datasets
│   ├── X_train.csv, X_val.csv, X_test.csv
│   ├── y_train.csv, y_val.csv, y_test.csv
│   └── dataset_summary.json
├── models/                    # Trained model artifacts
│   ├── *.pkl                  # Individual model files
│   ├── feature_scaler.pkl     # Preprocessing artifacts
│   └── feature_columns.txt    # Feature metadata
├── plots/                     # Generated visualizations
│   ├── train_test_accuracy_analysis.png
│   ├── model_comparison_comprehensive.png
│   ├── roc_pr_curves_all_models.png
│   ├── individual_models/     # Individual model plots
│   └── monitoring/            # Accuracy loss plots
└── baseline_metrics.pkl       # Performance baseline
```

## 🚢 Deployment

### Production Deployment
1. **Container Setup**: Docker containerization with requirements.txt
2. **Scheduling**: Cron jobs or Airflow for regular retraining
3. **Monitoring**: Integration with logging and alerting systems
4. **Scaling**: BigQuery for large-scale data processing

### Model Serving
```python
# Load trained model
import joblib
model = joblib.load('models/best_models_ensemble_model.pkl')

# Load preprocessor
scaler = joblib.load('models/feature_scaler.pkl')

# Prediction pipeline
def predict_ltv(user_features):
    processed_features = scaler.transform(user_features)
    ltv_prediction = model.predict(processed_features)
    return max(0, ltv_prediction[0])
```

### Performance Benchmarks
- **Training Time**: ~2-5 minutes (522K users, 72 features)
- **Memory Usage**: ~2-4GB peak during training
- **Prediction Latency**: <10ms per user (batch prediction)
- **Model Size**: ~50MB total (all 8 models + artifacts)

## 🎯 Key Improvements in Modular Version

### 1. **Maintainability**
- Clear separation of concerns
- Easy to modify individual components
- Reduced code duplication

### 2. **Testability**
- Each module can be unit tested
- Mock dependencies for integration tests
- Isolated failure debugging

### 3. **Scalability**
- Easy to add new models
- Pluggable data sources
- Configurable preprocessing steps

### 4. **NEW: Enhanced Visualizations**
- Training vs Testing accuracy analysis
- Overfitting detection dashboard
- Model health monitoring
- Color-coded performance indicators

### 5. **Production Readiness**
- Comprehensive error handling
- Detailed logging and monitoring
- Baseline management system
- Automated retraining alerts

## 📚 Next Steps

### Immediate Enhancements
1. **Hyperparameter Tuning**: GridSearchCV integration
2. **Feature Selection**: Automated feature importance analysis
3. **Model Interpretability**: SHAP values integration
4. **A/B Testing**: Model comparison framework

### Advanced Features
1. **Real-time Prediction**: API endpoint development
2. **Drift Detection**: Statistical drift monitoring
3. **AutoML Integration**: Automated model selection
4. **Multi-game Support**: Configurable game-specific models

This modular architecture provides a solid foundation for enterprise-scale LTV prediction systems while maintaining simplicity and ease of use.