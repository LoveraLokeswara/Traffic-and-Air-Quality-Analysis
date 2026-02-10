# Traffic and Air Quality Analysis

A comprehensive machine learning pipeline for analyzing the relationship between traffic patterns and air quality (NO₂ concentrations) across multiple Canadian monitoring stations. The project employs advanced feature engineering, multiple regression models, and time-series cross-validation to understand how traffic impacts air pollution.

## Project Overview

This analysis investigates whether traffic data is a significant predictor of nitrogen dioxide (NO₂) levels by:
- Training models **with traffic data** (Scenario A)
- Training models **without traffic data** (Scenario B)
- Comparing performance metrics to quantify traffic's impact on air quality

### Key Features
- **Four ML Models:** XGBoost, Random Forest, Generalized Additive Models (GAM), and Stacking Ensemble
- **Time-Series Cross-Validation:** Expanding window approach with 5 folds to prevent look-ahead bias
- **Comprehensive Feature Engineering:** Temporal, lag, rolling averages, interactions, and wind components
- **Multi-Station Analysis:** Simultaneous modeling across 5+ Canadian air quality stations
- **Detailed Visualizations:** 10+ composite figures comparing model performance and predictions

---

## Project Structure

```
Traffic-and-Air-Quality-Analysis/
├── README.md                          # Project documentation
├── analysis.py                        # Main analysis pipeline (moved to src/)
├── main.py                            # Entry point
├── pyproject.toml                     # Project configuration
│
├── src/
│   ├── analysis.py                    # Core ML pipeline
│   ├── collect_air_quality.py         # Air quality data collection
│   ├── collect_weather.py             # Weather data collection
│   ├── data-clean.py                  # Data cleaning & preprocessing
│   └── process_traffic.py             # Traffic data processing
│
├── data/
│   ├── air_quality/
│   │   ├── station_info.csv           # Station metadata
│   │   └── aq_data/                   # Raw station CSV files
│   ├── data_clean/
│   │   ├── cleaned_data_*.csv         # Processed station data (5 stations)
│   │   └── stations.txt               # Station list
│   ├── traffic/
│   │   ├── tf-ft-eng.csv              # Raw traffic counts
│   │   └── traffic_filter_summary.ipynb
│   └── weather/
│       ├── meteorological_metadata.yml
│       └── weather_data/              # Weather CSV by station
│
└── results/
    ├── eda_*.png                      # EDA figures (timeseries, distribution, etc.)
    ├── model_performance_*.png        # RMSE & R² comparisons
    ├── feature_importance_*.png       # XGBoost feature rankings
    ├── predictions_*.png              # Actual vs predicted plots
    ├── scatter_actual_vs_pred_*.png   # Residual scatters
    ├── residuals_*.png                # Residual distributions
    ├── cv_vs_test_*.png               # CV vs test RMSE
    ├── traffic_impact_comparison.png  # With/without traffic comparison
    ├── all_model_results.csv          # Full results table
    ├── rmse_comparison_pivot.csv      # Pivot table for quick comparison
    └── eda_summary_statistics.csv     # Descriptive statistics by station
```

---

## Datasets

### Air Quality Data
- **Source:** Canadian National Air Quality Monitoring Network
- **Stations:** 5 Canadian cities (Ottawa Downtown, Sarnia, Sault Ste. Marie, Toronto Downtown, Toronto West)
- **Target Variable:** NO₂_Mean (ppb - parts per billion)
- **Coverage:** Multiple years of daily observations

### Traffic Data
- **Source:** Local traffic monitoring infrastructure
- **Variable:** Daily traffic counts
- **Features:** Raw counts, lags (1-day), rolling averages (7-day)

### Weather Data
- **Variables:** Temperature, Precipitation, Wind Gust, Wind Direction
- **Processing:** Linear interpolation for missing values, wind component decomposition

---

## Feature Engineering

The pipeline creates a comprehensive feature set for each station:

### Raw Features
- `Traffic_Count` / `Temp` / `Precip` / `Wind_Gust`

### Temporal Features
- `Day_of_Week`, `Month`, `Is_Weekend`, `Season`

### Lag Features
- `NO2_Lag_1`, `NO2_Lag_7` (target history)
- `Traffic_Lag_1` (traffic history, Scenario A only)

### Rolling Averages (7-day)
- `NO2_Rolling_7`, `Temp_Rolling_7`, `Traffic_Rolling_7`

### Interaction Features
- `Traffic_x_Wind`, `Temp_x_Traffic` (Scenario A)
- `Temp_x_Wind`, `Precip_x_Wind` (Scenario B)

### Wind Components
- `Wind_NS`, `Wind_EW` (decomposed from direction/magnitude)

---

## Model Architecture

### Models Trained
1. **XGBoost** – Gradient boosting with 200 estimators
2. **Random Forest** – 200 trees, sqrt feature selection
3. **GAM** – Generalized Additive Model with spline/factor terms
4. **Stacking Ensemble** – Meta-learner (Ridge) combining base models

### Cross-Validation Strategy

**Temporal Data Split (80% Train / 20% Test):**
```
[─────── 80% Training Data ───────][── 20% Test Data ──]
         (Used for CV only)         (Final evaluation)
```

**Time-Series Cross-Validation** (expanding window with 5 folds within training data):
```
Fold 1: [Train: 0-16%  ][Validate: 16-32%]
Fold 2: [Train: 0-32%  ][Validate: 32-48%]
Fold 3: [Train: 0-48%  ][Validate: 48-64%]
Fold 4: [Train: 0-64%  ][Validate: 64-80%]
(Fold 5 training on full 80%)
```

**Key Properties:**
- **Expanding Window:** Training set grows with each fold; validation data always comes after training (no look-ahead bias)
- **Temporal Order Preserved:** All folds respect chronological sequence → realistic for time-series
- **No Data Leakage:** Test set (20%) is completely isolated and only used for final evaluation
- **Implementation:** `sklearn.model_selection.TimeSeriesSplit(n_splits=5)`

### Stacking Ensemble Details
- **Base Learners:** XGBoost, Random Forest, Linear Regression
- **Meta-Learner:** Ridge (α=1.0)
- **OOF Generation:** Out-of-fold predictions from CV folds

---

## Evaluation Metrics

All models are evaluated using:
- **RMSE** (Root Mean Squared Error) – primary metric
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error)

Results are stored in:
- `all_model_results.csv` – Full results table
- `rmse_comparison_pivot.csv` – Quick comparison across stations/scenarios

---

## Running the Analysis

### Prerequisites

Using `uv` (recommended):
```bash
uv sync
```

Or install dependencies directly:
```bash
uv pip install -r requirements.txt
```

### Execute the Pipeline

```bash
uv run python src/analysis.py
```

Or from main entry point:
```bash
uv run python main.py
```

### Output
The script generates:
1. **10+ PNG figures** in `results/` (EDA, model performance, predictions)
2. **3 CSV files** with detailed metrics and statistics
3. **Console summary** showing best model per station/scenario

---

## Key Findings

### Methodology
- Comparing Scenario A (with traffic) vs Scenario B (no traffic) reveals traffic's predictive power
- If Scenario A RMSE << Scenario B RMSE, traffic is a strong predictor
- If similar, other meteorological factors dominate NO₂ variation

### Outputs Explained
- **model_performance_with_traffic.png** – RMSE/R² for each model with traffic
- **feature_importance_with_traffic.png** – Top 10 XGBoost features
- **traffic_impact_comparison.png** – Direct RMSE comparison (with vs without traffic)
- **cv_vs_test_with_traffic.png** – Generalization: CV RMSE vs test RMSE
- **scatter_actual_vs_pred_with_traffic.png** – Prediction accuracy across all models

---

## Data Sources & Preprocessing

### Stage 1: Data Collection (`src/collect_*.py`)
- Downloads air quality, weather, and traffic data
- Stores raw data in `data/air_quality/`, `data/weather/`, `data/traffic/`

### Stage 2: Cleaning (`src/data-clean.py`)
- Merges datasets by date
- Handles missing values (interpolation, forward-fill)
- Removes outliers and invalid records
- Outputs cleaned files to `data/data_clean/`

### Stage 3: Analysis (`src/analysis.py`)
- Loads cleaned data
- Performs EDA and feature engineering
- Trains models under both scenarios
- Generates results and visualizations

---

## Dependencies

Key libraries (see `pyproject.toml`):
- **ML & Statistics:** scikit-learn, xgboost, pygam
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Time-Series:** sklearn.model_selection.TimeSeriesSplit
