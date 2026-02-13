# Traffic and Air Quality Analysis

This repository contains the code and data for analyzing the relationship between traffic patterns and Nitrogen Dioxide (NO₂) concentrations across multiple Canadian monitoring stations. The project investigates whether incorporating traffic flow data significantly improves air quality forecasting compared to meteorological data alone.

## Project Overview

This analysis investigates whether traffic data is a significant predictor of nitrogen dioxide (NO₂) levels by:

- Training models **with traffic data** (Scenario A)
- Training models **without traffic data** (Scenario B)
- Comparing performance metrics to quantify traffic's impact on air quality

---

## Project Structure

```
Traffic-and-Air-Quality-Analysis/
├── README.md                          # Project documentation
├── pyproject.toml                     # Project configuration and dependencies
├── uv.lock                            # Lock file for reproducible environments
│
├── src/
│   ├── analysis.py                    # Main analysis script: trains models and generates results
│   ├── collect_air_quality.py         # Script to fetch raw air quality data
│   ├── collect_weather.py             # Script to fetch raw weather data
│   └── data_clean.py                  # Main processing script: cleans and merges all datasets
│
├── data/
│   ├── air_quality/                   # Raw air quality data and station metadata
│   ├── traffic/                       # Raw traffic volume data
│   ├── weather/                       # Raw meteorological data
│   └── data_clean/                    # Final processed CSVs ready for modeling
│
├── models/                            # Serialized trained models (.pkl) for all stations
├── results/                           # Generated figures (EDA, RMSE plots) and CSV metrics
└── report/                            # Source files for the Executive Report
```

## How to Run the Analysis

This project uses **[uv](https://github.com/astral-sh/uv)** for efficient Python package management.

### 1. Setup & Installation
Clone the repository and set up the environment:
```bash
git clone https://github.com/LoveraLokeswara/Traffic-and-Air-Quality-Analysis
cd Traffic-and-Air-Quality-Analysis
uv sync
```

### 2. Data Processing
To regenerate the cleaned datasets from raw sources, run:
```bash
uv run python src/data_clean.py
```

### 3. Model Training and Analysis
To train the models and generate all plots in the `results/` folder, execute:
```bash
uv run python src/analysis.py
```
## Datasets Overview

See the [Data Dictionary](data/data_description.md) for detailed information about the datasets and their columns.

## Pipeline Stages
### Stage 1: Data Collection (`src/collect_*.py`)
- Downloads air quality, weather, and traffic data
- Stores raw data in `data/air_quality/`, `data/weather/`, `data/traffic/`

### Stage 2: Cleaning (`src/data-clean.py`)
- Merges datasets by date
- Handles missing values 
- Removes outliers and invalid records
- Outputs cleaned files to `data/data_clean/`

### Stage 3: Analysis (`src/analysis.py`)
- Loads cleaned data
- Performs EDA and feature engineering
- Trains models under both scenarios
- Generates results and visualizations

---
## Models
The following models are trained for every station:

### Models Trained
1. **XGBoost** 
2. **Random Forest** 
3. **Generalized Additive Model** 
4. **Stacking Ensemble** – Meta-learner (Ridge) combining base models
## Evaluation Metrics

All models are evaluated using:
- **RMSE** (Root Mean Squared Error) – primary metric
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error)

Full results are stored in:

- `all_model_results.csv` – Full results table

## Key Outputs
The analysis generates the following artifacts in the `results/` directory:

* **`traffic_impact_comparison.png`:** Direct RMSE comparison between traffic and no-traffic models

* **`feature_importance_*.png`:** Top predictors identified by XGBoost

* **`model_performance_*.png`:** Comprehensive RMSE and R²