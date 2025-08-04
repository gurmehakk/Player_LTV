# LTV Prediction Pipeline

A comprehensive pipeline for predicting player lifetime value (LTV) and whale classification using gaming analytics data.

## Features

This pipeline creates a comprehensive feature set from gaming event data including:

### Session Features
- Total sessions, session lengths, playtime
- Session timing patterns (hours, gaps between sessions)
- Time from install to first session

### Activity Features  
- Levels played, retries, progression speed
- Time per level, levels per session

### Monetization Features
- Ad viewing behavior (interstitial, rewarded, banner)
- Currency earning and spending patterns
- Store and purchase interactions

### Meta Features
- Attribution data (channel, campaign, publisher)
- Geographic and platform information

### Behavioral Features
- Navigation patterns, social interactions
- Achievement unlocks, tutorial completion
- Frustration signals (failures, exits)


## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your BigQuery settings in `config.py`

3. Run the pipeline:
```bash
python main.py
```

## Output

The pipeline generates three CSV files:
- `train_data.csv` - Training set (100k samples by default)
- `val_data.csv` - Validation set (20k samples by default)  
- `test_data.csv` - Test set (20k samples by default)

Each dataset contains:
- **Features**: D0-D3 behavioral features
- **Targets**: 
  - `ltv_30_days`: Regression target (30-day revenue)
  - `is_whale`: Classification target (whale vs non-whale)

## Configuration

Key parameters in `config.py`:
- `FEATURE_DAYS`: Days to use for feature extraction (default: 3)
- `PREDICTION_DAYS`: Days to predict LTV over (default: 30)
- `WHALE_THRESHOLD`: Revenue threshold for whale classification (default: $50)
- Sample sizes for train/val/test splits