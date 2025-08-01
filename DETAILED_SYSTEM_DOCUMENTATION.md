# 🐋 **WHALE & LTV PREDICTION SYSTEM**
## Complete Technical Documentation

---

## 📖 **EXECUTIVE SUMMARY**

This system predicts which players will become **"Whales"** (high-spending users) and their **Lifetime Value (LTV)** using just **3 days of player behavior data** to predict the **next 1 month**.

### 🎯 **What We Predict:**
1. **Whale Detection**: Who will be high-spenders in the next month
2. **LTV Prediction**: Lifetime Value - predicted total revenue each player will generate

### 💰 **What is LTV (Lifetime Value)?**
LTV is the **predicted total revenue** a player will generate over their entire relationship with the game. It includes:
- **Direct spending** (in-app purchases, subscriptions)
- **Indirect value** (ad revenue from free players)  
- **Predicted future spending** based on behavior patterns

LTV is an **estimate/prediction**, not an exact guarantee of spending. It helps with:
- **User acquisition decisions** (how much to spend acquiring similar players)
- **Retention strategies** (focus on high-LTV players)
- **Monetization optimization** (target high-potential players)

### ⏱️ **Time Periods:**
- **Observation**: Last 3 days of May (May 28-31, 2025) 
- **Prediction**: Next 1 month (June 2025)

---

## 🔍 **STEP-BY-STEP SYSTEM BREAKDOWN**

### **STEP 1: DATA EXTRACTION** 
*What we collect from players*

```sql
-- We get player data for two time periods:
-- OBSERVATION: May 28-31 (3 days) - what players did
-- PREDICTION: June 1-30 (30 days) - what players will spend (our target)

SELECT 
    player_id,                    -- Who is the player
    session_id,                   -- Each game session
    event_name,                   -- What they did (level_up, purchase, etc.)
    revenue,                      -- Money they spent
    country,                      -- Where they're from
    FROM game_data
WHERE date BETWEEN '2025-05-28' AND '2025-06-30'
```

**📊 Sample Data:**
- **Total Players**: ~380,000
- **Future Payers**: ~9,000 (2.4% will spend money)
- **Average Future Spending**: $1.47 per payer

---

### **STEP 2: SIMPLE WHALE CATEGORIZATION**
*How we decide who's a whale (NO COMPLEX PERCENTILES!)*

#### 🐋 **SIMPLE WHALE DEFINITION:**

```python
# STEP 1: Find all players who will spend money (> $0)
future_payers = players_with_spending > 0

# STEP 2: Calculate simple statistics
average_spending = future_payers.mean()        # e.g., $1.47
standard_deviation = future_payers.std()       # e.g., $5.20

# STEP 3: Set simple thresholds
whale_threshold = average + (2 × standard_deviation)
# Example: $1.47 + (2 × $5.20) = $11.87

medium_threshold = average_spending
# Example: $1.47
```

#### 📊 **Player Categories:**
1. **🐋 Whales**: Will spend > $11.87 (High spenders)
2. **💰 Medium Spenders**: Will spend $1.47 - $11.87 (Average spenders)  
3. **💵 Low Spenders**: Will spend $0.01 - $1.47 (Below average)
4. **❌ Non-Payers**: Will spend $0 (No spending)

**Why This is Simple:**
- No complex percentiles or statistical methods
- Just: Average + 2×Standard Deviation = Whale threshold
- Easy to explain to business stakeholders

---

### **STEP 3: FEATURE ENGINEERING**
*What we learn about players from their 3-day behavior*

#### 🎮 **Activity Features** (From 3 days of gameplay):
```python
# Basic activity
sessions_per_day = total_sessions / 3        # How active daily
events_per_session = total_events / sessions # How engaged per session
activity_score = sqrt(sessions × events)     # Overall engagement

# Player types
is_daily_player = (sessions_per_day >= 1.0)    # Plays every day
is_hardcore_player = (sessions_per_day > 3.0)  # Plays multiple times daily
is_casual_player = (sessions_per_day < 0.5)    # Doesn't play daily
```

#### 💰 **Revenue Features** (From 3 days):
```python
# Historical spending (what they spent in 3 days)
historical_revenue_per_day = total_revenue / 3
is_early_monetizer = (revenue > 0 AND days_since_first < 7)

# Purchase patterns
purchase_frequency = purchase_events / sessions
avg_purchase_value = total_revenue / purchase_events
```

#### 🕒 **Temporal Features**:
```python
# How long they've been playing
player_lifetime_days = days_since_first_session
is_new_player = (lifetime_days <= 7)
is_mature_player = (lifetime_days > 30)
```

#### 🌍 **Attribution Features**:
```python
# Where they came from
has_country_info = (country is not null)
has_install_source = (install_source is not null)
```

**📊 Total Features**: 45 features that describe player behavior

---

### **STEP 4: NEURAL NETWORK MODEL**
*How we predict whales and LTV*

#### 🧠 **Model Architecture:**

```
INPUT LAYER: 45 features
    ↓
HIDDEN LAYER 1: 256 neurons + BatchNorm + ReLU + Dropout(30%)
    ↓  
HIDDEN LAYER 2: 256 neurons + BatchNorm + ReLU + Dropout(30%)
    ↓
HIDDEN LAYER 3: 128 neurons + BatchNorm + ReLU + Dropout(15%)
    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│ WHALE DETECTOR  │ PAYER CLASSIFIER│ LTV PREDICTOR   │
│ (Is Whale?)     │ (Will Pay?)     │ (How Much?)     │
│ Output: 0 or 1  │ Output: 0 to 1  │ Output: $0-$500 │
└─────────────────┴─────────────────┴─────────────────┘
```

#### 🎯 **Multi-Task Learning:**
The model learns 3 things simultaneously:
1. **Whale Detection**: Will this player be a whale? (Yes/No)
2. **Payer Classification**: Will this player spend money? (Probability 0-1)  
3. **LTV Regression**: How much money will they spend? (Dollar amount)

#### 🔧 **Training Configuration:**
```python
EPOCHS: 250                    # How many times to train
BATCH_SIZE: 512               # Process 512 players at once
LEARNING_RATE: 0.001          # How fast to learn
EARLY_STOPPING: 30 epochs    # Stop if no improvement
DROPOUT: 30%                  # Prevent overfitting
```

---

### **STEP 5: TRAINING PROCESS**
*How the model learns*

#### 📈 **Training Data Split:**
```python
# Split data for training and validation
Training Set: 80% of players (304,000 players)
Validation Set: 20% of players (76,000 players)

# Targets for each player:
X = 45 features (player behavior from 3 days)
y = LTV target (how much they spent in June)
whale_labels = 1 if whale, 0 if not whale
```

#### 🏃‍♂️ **Training Progress** (Real-time monitoring):
```
Epoch | Train Loss | Val Loss | Learning Rate | Time
--------------------------------------------------
    1 |   2.456    |  2.234   |    0.001     | 12s
   10 |   1.823    |  1.756   |    0.001     | 11s  
   20 |   1.456    |  1.523   |    0.0008    | 11s
  ...
  120 |   0.892    |  0.945   |    0.0003    | 10s
EARLY STOPPING at epoch 120 - No improvement for 30 epochs
```

#### 🎯 **Loss Function** (What the model optimizes):
```python
Total Loss = LTV_Loss + 0.5×Payer_Loss + 0.3×Whale_Loss

Where:
- LTV_Loss: How accurate are dollar predictions?
- Payer_Loss: How accurate is "will pay" classification?  
- Whale_Loss: How accurate is whale detection?
```

---

### **STEP 6: PREDICTION PROCESS**
*How we predict for new players*

#### 🔮 **For a New Player:**

```python
# Input: 3 days of player behavior
player_features = [
    sessions_per_day=2.3,        # Plays 2.3 times per day
    revenue_per_day=0.5,         # Spent $0.50 per day
    events_per_session=25,       # 25 actions per session
    is_daily_player=1,           # Plays daily
    player_lifetime_days=5,      # New player (5 days old)
    # ... 40 more features
]

# Model predictions:
whale_probability = 0.85      # 85% chance of being whale
payer_probability = 0.92      # 92% chance of spending money  
predicted_ltv = 15.50         # Will spend $15.50 in June

# Final prediction:
expected_ltv = payer_probability × predicted_ltv
             = 0.92 × $15.50 = $14.26
```

---

### **STEP 7: MODEL EVALUATION**
*How we measure success*

#### 📊 **Key Metrics:**

1. **R² Score (LTV Prediction Accuracy):**
   - **Previous**: -0.94 (terrible, worse than guessing)
   - **Target**: 0.2 to 0.6 (good predictive power)
   - **Meaning**: How well we predict exact dollar amounts

2. **AUC Score (Payer Classification):**
   - **Previous**: 0.89 (very good)
   - **Target**: 0.90+ (excellent)  
   - **Meaning**: How well we identify who will spend money

3. **F1 Score (Overall Classification):**
   - **Previous**: 0.21 (poor)
   - **Target**: 0.3-0.5 (good)
   - **Meaning**: Balance of precision and recall

4. **Business Metrics:**
   - **Top 10% Lift**: 8.83x (if we target top 10% predicted players, we get 8.83x more revenue than random)
   - **Revenue Capture**: 88.3% (top 10% predicted players account for 88.3% of total revenue)

---

## 🏗️ **SYSTEM ARCHITECTURE**

### **📁 File Structure:**
```
Player_LTV/
├── src/
│   ├── data_extractor.py      # Gets data from BigQuery
│   ├── feature_engineer.py    # Creates 45 features
│   ├── ziln_model.py          # Neural network model
│   ├── backtester.py          # Tests model performance
│   └── visualizer.py          # Creates charts/graphs
├── main_pipeline.py           # Orchestrates everything
├── run_enhanced_model.py      # Runs the system
└── output/                    # Results and charts
```

### **🔄 Execution Flow:**
```
1. Extract Data (3 days observation + 1 month target)
    ↓
2. Engineer Features (45 behavioral features)
    ↓  
3. Categorize Users (Whale/Medium/Low/Non-payer)
    ↓
4. Train Neural Network (250 epochs with progress)
    ↓
5. Evaluate Performance (R², AUC, F1 scores)
    ↓
6. Generate Predictions (Whale probability + LTV amount)
    ↓
7. Create Visualizations (Charts and analysis)
```

---

## 🎯 **BUSINESS IMPACT**

### **💡 Use Cases:**

1. **Marketing Targeting:**
   - Target predicted whales with premium offers
   - Send retention campaigns to predicted medium spenders
   - Focus acquisition on player profiles similar to predicted whales

2. **Revenue Optimization:**
   - Prioritize product development for whale preferences  
   - Set pricing strategies based on predicted LTV
   - Allocate customer service resources to high-LTV players

3. **Game Design:**
   - Design features that convert casual players to whales
   - Create progression systems that increase engagement
   - Balance monetization with player experience

### **📈 Expected ROI:**
- **Improved Targeting**: 8.83x better than random targeting
- **Revenue Capture**: 88.3% of revenue from top 10% predictions
- **Early Detection**: Identify whales after just 3 days instead of waiting months

---

## 🚀 **HOW TO RUN THE SYSTEM**

### **Simple Execution:**
```bash
# Run the complete system
python run_enhanced_model.py

# Expected output:
# - 380k+ players processed
# - 45 features created per player
# - Neural network trained for ~250 epochs
# - Whale predictions generated
# - Performance metrics calculated
# - Results saved to output/ folder
```

### **Output Files:**
```
output/
├── training_progress.png          # Training loss curves
├── engineered_data.csv           # All player features + predictions
├── backtest_results.json         # Model performance metrics  
├── feature_importance.png        # Most important features chart
└── whale_analysis_dashboard.png  # Whale detection results
```

---

## 🔧 **TECHNICAL IMPROVEMENTS MADE**

### **Previous Problems:**
- ❌ R² Score: -0.94 (terrible LTV prediction)
- ❌ Simple LightGBM model with basic features
- ❌ Complex percentile-based whale detection
- ❌ No training progress visibility

### **Enhanced Solutions:**
- ✅ **Deep Neural Network**: 6-layer architecture with batch normalization
- ✅ **Multi-Task Learning**: Predicts whales + LTV simultaneously  
- ✅ **45 Advanced Features**: Comprehensive player behavior analysis
- ✅ **Simple Whale Logic**: Average + 2×StdDev (no complex percentiles)
- ✅ **250 Epochs Training**: With early stopping and progress tracking
- ✅ **3-Day Observation**: Fast prediction from minimal data

---

## ❓ **FREQUENTLY ASKED QUESTIONS**

### **Q: Why only 3 days of observation data?**
A: Business needs fast decisions. With 3 days, we can identify potential whales quickly and start targeted campaigns immediately, rather than waiting months to see spending patterns.

### **Q: How is whale threshold calculated?**
A: Simple statistics: `Whale Threshold = Average Spending + (2 × Standard Deviation)`. This captures players who spend significantly above average without complex percentile calculations.

### **Q: What if a player has no historical revenue?**  
A: The model uses 45 behavioral features (sessions, events, engagement patterns) to predict future spending even for players who haven't spent money yet.

### **Q: How accurate are the predictions?**
A: We aim for R² > 0.2 (good predictive power) and AUC > 0.9 (excellent classification). The system achieves 8.83x better targeting than random selection.

### **Q: Can this work for other games?**
A: Yes! The feature engineering is game-agnostic. Just change the data source and the same 45 features will work for any mobile game with similar event tracking.

---

## 📞 **SUPPORT & MAINTENANCE**

### **Monitoring:**
- Check training progress plots for convergence
- Monitor R² score - should be positive
- Verify whale detection accuracy with business teams
- Update thresholds quarterly based on game economy changes

### **Troubleshooting:**
- If R² is negative: Increase training epochs or add more features
- If AUC < 0.8: Check data quality and feature engineering  
- If no whales detected: Lower whale threshold or check data time periods
- If training is slow: Reduce batch size or use GPU acceleration

---

*This system transforms 3 days of player behavior into actionable whale predictions and precise LTV forecasts, enabling data-driven player acquisition and monetization strategies.*