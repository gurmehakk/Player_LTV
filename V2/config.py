# config.py

PROJECT_ID = "gc-forecasting-dev"
DATASET = "test_data"
TABLE = "app_data"

# Date range for analysis
MIN_DATE = "2025-04-01"

# Output paths
TRAIN_CSV = "train_data.csv"
VAL_CSV = "val_data.csv"
TEST_CSV = "test_data.csv"

# Sampling configuration  
TRAIN_SAMPLES = 100000    # 50% of total users
VAL_SAMPLES = 50000     # 25% of total users  
TEST_SAMPLES = 50000     # 25% of total users

# LTV prediction configuration
FEATURE_DAYS = 3  # Use D0-D3 for features
PREDICTION_DAYS = 30  # Predict next 30 days LTV
WHALE_THRESHOLD = 1.0  # Revenue threshold for whale classification (lowered for better balance)
