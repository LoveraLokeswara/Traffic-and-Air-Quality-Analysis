#!/usr/bin/env python3
"""
analysis.py - Automated Air Quality Modeling Pipeline
=====================================================
Iterates over every CSV in data/data_clean/, performs EDA, engineers features,
and trains four models (XGBoost, Random Forest, GAM, Stacking Ensemble) under
two scenarios (with traffic / without traffic).

All composite figures (5 subplots per image, one per station) and summary
tables are saved to the results/ folder.

All trained models are saved to the models/ folder with naming convention:
    {station}_{scenario}_{model_type}.pkl

To load a saved model:
    import joblib
    
    # For XGBoost or Random Forest:
    model = joblib.load('models/station_name_with_traffic_xgboost.pkl')
    predictions = model.predict(X_new)
    
    # For GAM (includes scaler):
    gam_pkg = joblib.load('models/station_name_with_traffic_gam.pkl')
    model = gam_pkg['model']
    scaler = gam_pkg['scaler']
    if scaler:
        X_scaled = scaler.transform(X_new)
        predictions = model.predict(X_scaled)
    else:
        predictions = model.predict(X_new)
    
    # For Stacking Ensemble:
    stack_pkg = joblib.load('models/station_name_with_traffic_stacking.pkl')
    base_models = stack_pkg['base_models']
    meta_learner = stack_pkg['meta_learner']
    # Generate base predictions
    base_preds = np.column_stack([model.predict(X_new) for _, model in base_models])
    # Final prediction
    predictions = meta_learner.predict(base_preds)
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import stats as sp_stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.base import clone
from xgboost import XGBRegressor
from pygam import LinearGAM, s, f

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# ── Global font sizes for publication-quality figures ─────────────────────
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 24,
})

# ── Global spacing parameters for multi-subplot figures ────────────────────
GLOBAL_HSPACE = 0.50  # Vertical spacing between rows (increased for wider spacing)
GLOBAL_WSPACE = 1.0   # Horizontal spacing between columns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "data_clean")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET = "NO2_Mean"
TRAFFIC_COL = "Traffic_Count"
N_SPLITS = 5
RANDOM_STATE = 42

MODEL_NAMES = ["XGBoost", "Random Forest", "GAM", "Stacking Ensemble"]
MODEL_COLORS = {
    "XGBoost": "#e67e22",
    "Random Forest": "#27ae60",
    "GAM": "#2980b9",
    "Stacking Ensemble": "#c0392b",
}

# Columns that are related to traffic (to be dropped in scenario B)
TRAFFIC_RELATED = [
    "Traffic_Count",
    "Traffic_Lag_1",
    "Traffic_Rolling_7",
    "Traffic_x_Wind",
    "Temp_x_Traffic",
]


# ═══════════════════════════════════════════════════════════════════════════
#  1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
def load_datasets(data_dir: str) -> dict[str, pd.DataFrame]:
    """Read every CSV in *data_dir* and return {station_name: DataFrame}."""
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    datasets: dict[str, pd.DataFrame] = {}
    for fpath in csv_files:
        df = pd.read_csv(fpath, parse_dates=["Date"])
        station = df["Station"].iloc[0]
        datasets[station] = df
        print(f"  Loaded {station:25s}  ({len(df):>5} rows)  from {os.path.basename(fpath)}")
    return datasets


# ═══════════════════════════════════════════════════════════════════════════
#  2. EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
def run_eda(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Print summary statistics and save composite EDA figures.

    Returns a summary DataFrame with per-station statistics.
    """
    stations = list(datasets.keys())
    n = len(stations)

    # ── 2a. Summary statistics table ──────────────────────────────────────
    summary_rows = []
    for name, df in datasets.items():
        summary_rows.append(
            {
                "Station": name,
                "Rows": len(df),
                "Date_Min": df["Date"].min().date(),
                "Date_Max": df["Date"].max().date(),
                "NO2_Mean": df[TARGET].mean(),
                "NO2_Std": df[TARGET].std(),
                "NO2_Min": df[TARGET].min(),
                "NO2_Max": df[TARGET].max(),
                "Traffic_Mean": df[TRAFFIC_COL].mean(),
                "Temp_Mean": df["Temp"].mean(),
                "Missing": df.isnull().sum().sum(),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "eda_summary_statistics.csv"), index=False)
    print("\n  Summary statistics saved to results/eda_summary_statistics.csv")

    # ── 2b. NO2 time-series ──────────────────────────────────────────────
    toronto, others = _split_stations(stations)
    fig, axes_dict = _create_station_figure(stations, sharey=True)
    for name in stations:
        ax = axes_dict[name]
        df = datasets[name]
        ax.plot(df["Date"], df[TARGET], linewidth=0.6, alpha=0.8)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Date")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y")
        ax.grid(True, alpha=0.3)
    if toronto:
        axes_dict[toronto[0]].set_ylabel("NO$_2$ (ppb)")
    if others:
        axes_dict[others[0]].set_ylabel("NO$_2$ (ppb)")
    fig.suptitle("NO$_2$ Time Series by Station", fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "eda_no2_timeseries.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── 2c. NO2 distribution ─────────────────────────────────────────────
    fig, axes_dict = _create_station_figure(stations, sharey=True)
    for name in stations:
        ax = axes_dict[name]
        df = datasets[name]
        ax.hist(df[TARGET].dropna(), bins=40, edgecolor="black", alpha=0.7)
        ax.axvline(df[TARGET].mean(), color="red", linestyle="--", label=f"mean={df[TARGET].mean():.1f}")
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("NO$_2$ (ppb)")
        ax.legend()
        ax.tick_params(axis="x")
        ax.tick_params(axis="y")
        ax.grid(True, alpha=0.3, axis="y")
    if toronto:
        axes_dict[toronto[0]].set_ylabel("Frequency")
    if others:
        axes_dict[others[0]].set_ylabel("Frequency")
    fig.suptitle("NO$_2$ Distribution by Station", fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "eda_no2_distribution.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── 2d. Seasonal (monthly) NO2 ───────────────────────────────────────
    fig, axes_dict = _create_station_figure(stations, sharey=True)
    for name in stations:
        ax = axes_dict[name]
        df = datasets[name].copy()
        df["Month"] = df["Date"].dt.month
        monthly = df.groupby("Month")[TARGET].mean()
        ax.plot(monthly.index, monthly.values, marker="o", linewidth=2, markersize=6)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_xticks(range(1, 13))
        ax.tick_params(axis="x")
        ax.tick_params(axis="y")
        ax.grid(True, alpha=0.3)
    if toronto:
        axes_dict[toronto[0]].set_ylabel("Mean NO$_2$ (ppb)")
    if others:
        axes_dict[others[0]].set_ylabel("Mean NO$_2$ (ppb)")
    fig.suptitle("Seasonal NO$_2$ Pattern by Station", fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "eda_seasonal_pattern.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── 2e. Correlation heatmaps ─────────────────────────────────────────
    corr_cols = [TARGET, TRAFFIC_COL, "Temp", "Precip", "Wind_Gust", "Wind_Dir"]
    fig, axes_dict = _create_station_figure(stations, cell_h=5.5, hspace=0.45, wspace=0.75)
    for name in stations:
        ax = axes_dict[name]
        cm = datasets[name][corr_cols].corr()
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    square=True, linewidths=0.5, cbar=False, ax=ax,
                    annot_kws={"size": 11})
        ax.set_title(name, fontweight="bold")
    fig.suptitle("Feature Correlation by Station", fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "eda_correlation_matrices.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("  EDA composite figures saved to results/")
    return summary_df


# ═══════════════════════════════════════════════════════════════════════════
#  3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
def engineer_features(df_raw: pd.DataFrame, with_traffic: bool) -> tuple[pd.DataFrame, list[str]]:
    """Create temporal, lag, rolling, interaction, and wind-component features.

    Parameters
    ----------
    df_raw : raw station DataFrame
    with_traffic : if False, traffic-related columns are excluded.

    Returns
    -------
    df_clean : DataFrame with engineered features (NaN rows dropped)
    feature_cols : ordered list of feature column names
    """
    df = df_raw.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # Handle missing values in raw columns
    df["Temp"] = df["Temp"].interpolate(method="linear")
    df["Precip"] = df["Precip"].fillna(0)
    df["Wind_Gust"] = df["Wind_Gust"].interpolate(method="linear")
    df["Wind_Dir"] = df["Wind_Dir"].interpolate(method="linear")

    # Temporal features
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["Day_of_Year"] = df["Date"].dt.dayofyear
    df["Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(int)
    df["Season"] = df["Month"] % 12 // 3 + 1  # 1=Winter … 4=Fall

    # Lag features
    df["NO2_Lag_1"] = df[TARGET].shift(1)
    df["NO2_Lag_7"] = df[TARGET].shift(7)

    # Rolling averages
    df["NO2_Rolling_7"] = df[TARGET].rolling(window=7, min_periods=1).mean()
    df["Temp_Rolling_7"] = df["Temp"].rolling(window=7, min_periods=1).mean()

    # Wind components
    df["Wind_NS"] = df["Wind_Gust"] * np.cos(np.radians(df["Wind_Dir"]))
    df["Wind_EW"] = df["Wind_Gust"] * np.sin(np.radians(df["Wind_Dir"]))

    if with_traffic:
        df["Traffic_Lag_1"] = df[TRAFFIC_COL].shift(1)
        df["Traffic_Rolling_7"] = df[TRAFFIC_COL].rolling(window=7, min_periods=1).mean()
        df["Traffic_x_Wind"] = df[TRAFFIC_COL] * df["Wind_Gust"]
        df["Temp_x_Traffic"] = df["Temp"] * df[TRAFFIC_COL]

        feature_cols = [
            # Raw features
            "Traffic_Count", "Temp", "Precip", "Wind_Gust",
            # Temporal
            "Day_of_Week", "Month", "Is_Weekend", "Season",
            # Lag
            "NO2_Lag_1", "NO2_Lag_7", "Traffic_Lag_1",
            # Rolling
            "NO2_Rolling_7", "Traffic_Rolling_7", "Temp_Rolling_7",
            # Interaction
            "Traffic_x_Wind", "Temp_x_Traffic",
            # Wind components
            "Wind_NS", "Wind_EW",
        ]
    else:
        feature_cols = [
            # Raw features (no traffic)
            "Temp", "Precip", "Wind_Gust",
            # Temporal
            "Day_of_Week", "Month", "Is_Weekend", "Season",
            # Lag
            "NO2_Lag_1", "NO2_Lag_7",
            # Rolling
            "NO2_Rolling_7", "Temp_Rolling_7", 
            # Wind components
            "Wind_NS", "Wind_EW",
        ]

    df_clean = df.dropna(subset=feature_cols + [TARGET]).reset_index(drop=True)
    return df_clean, feature_cols


# ═══════════════════════════════════════════════════════════════════════════
#  4. MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════
def _get_xgb():
    return XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )


def _get_rf():
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )


def _build_gam_formula(feature_cols: list[str]):
    """Return a LinearGAM whose terms match *feature_cols* ordering.

    Categorical features → factor terms (f);  continuous → spline terms (s).
    """
    categorical = {"Day_of_Week", "Month", "Is_Weekend", "Season"}
    terms = None
    for i, col in enumerate(feature_cols):
        term = f(i) if col in categorical else s(i)
        terms = term if terms is None else terms + term
    return LinearGAM(terms, max_iter=200)


# ═══════════════════════════════════════════════════════════════════════════
#  5. EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def evaluate(y_true, y_pred) -> dict:
    """Return dict of regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # Guard against zero division in MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


# ═══════════════════════════════════════════════════════════════════════════
#  6. TRAINING + EVALUATION PIPELINE (single station, single scenario)
# ═══════════════════════════════════════════════════════════════════════════
def train_and_evaluate(
    df_clean: pd.DataFrame,
    feature_cols: list[str],
    station: str,
    scenario: str,
) -> dict:
    """Train all four models, evaluate, save models, and return a results dict.

    Returns
    -------
    dict with keys: results (metrics dict per model), predictions (arrays),
    feature_importance (DataFrame), y_test, test_dates, meta_coefficients.
    """
    X = df_clean[feature_cols]
    y = df_clean[TARGET]

    # Temporal train / test split (80 / 20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_dates = df_clean["Date"].iloc[split_idx:].reset_index(drop=True)

    # Scaled copies for GAM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}
    cv_results: dict[str, dict] = {}

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    # Create clean filename prefix
    scenario_safe = scenario.lower().replace(" ", "_")
    station_safe = station.replace(" ", "_").replace("/", "_")

    # ── XGBoost ──────────────────────────────────────────────────────────
    xgb_model = _get_xgb()
    cv_scores = _cv_loop(xgb_model, X_train, y_train, tscv)
    cv_results["XGBoost"] = cv_scores
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    results["XGBoost"] = evaluate(y_test.values, xgb_pred)
    predictions["XGBoost"] = xgb_pred
    
    # Save XGBoost model
    xgb_path = os.path.join(MODELS_DIR, f"{station_safe}_{scenario_safe}_xgboost.pkl")
    joblib.dump(xgb_model, xgb_path)

    # Feature importance from XGBoost
    fi = pd.DataFrame({"Feature": feature_cols, "Importance": xgb_model.feature_importances_})
    fi = fi.sort_values("Importance", ascending=False).reset_index(drop=True)

    # ── Random Forest ────────────────────────────────────────────────────
    rf_model = _get_rf()
    cv_scores = _cv_loop(rf_model, X_train, y_train, tscv)
    cv_results["Random Forest"] = cv_scores
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    results["Random Forest"] = evaluate(y_test.values, rf_pred)
    predictions["Random Forest"] = rf_pred
    
    # Save Random Forest model
    rf_path = os.path.join(MODELS_DIR, f"{station_safe}_{scenario_safe}_random_forest.pkl")
    joblib.dump(rf_model, rf_path)

    # ── GAM ──────────────────────────────────────────────────────────────
    gam_model = _build_gam_formula(feature_cols)
    cv_scores = _cv_loop_gam(feature_cols, X_train, y_train, tscv)
    cv_results["GAM"] = cv_scores
    gam_uses_scaler = True
    try:
        gam_model.fit(X_train_scaled, y_train)
        gam_pred = gam_model.predict(X_test_scaled)
    except Exception:
        # Fallback: use unscaled data
        gam_model = _build_gam_formula(feature_cols)
        gam_model.fit(X_train.values, y_train)
        gam_pred = gam_model.predict(X_test.values)
        gam_uses_scaler = False
    results["GAM"] = evaluate(y_test.values, gam_pred)
    predictions["GAM"] = gam_pred
    
    # Save GAM model (with scaler if used)
    gam_path = os.path.join(MODELS_DIR, f"{station_safe}_{scenario_safe}_gam.pkl")
    gam_package = {"model": gam_model, "scaler": scaler if gam_uses_scaler else None, "feature_cols": feature_cols}
    joblib.dump(gam_package, gam_path)

    # ── Stacking Ensemble (manual, TimeSeriesSplit-aware) ────────────────
    base_models = [
        ("XGBoost", _get_xgb()),
        ("Random Forest", _get_rf()),
        ("Linear Regression", LinearRegression(n_jobs=-1)),
    ]
    n_train = len(X_train)
    n_base = len(base_models)
    oof_preds = np.zeros((n_train, n_base))
    test_preds = np.zeros((len(X_test), n_base))

    # Store trained base models for stacking
    trained_base_models = []
    for idx, (name, model) in enumerate(base_models):
        for train_idx, val_idx in tscv.split(X_train):
            m = clone(model)
            m.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            oof_preds[val_idx, idx] = m.predict(X_train.iloc[val_idx])
        # Retrain on full training set
        model.fit(X_train, y_train)
        trained_base_models.append((name, model))
        test_preds[:, idx] = model.predict(X_test)

    meta_learner = Ridge(alpha=1.0)
    meta_learner.fit(oof_preds, y_train)
    stacking_pred = meta_learner.predict(test_preds)

    results["Stacking Ensemble"] = evaluate(y_test.values, stacking_pred)
    predictions["Stacking Ensemble"] = stacking_pred
    cv_results["Stacking Ensemble"] = {
        "mean_rmse": np.sqrt(mean_squared_error(y_train, meta_learner.predict(oof_preds))),
        "std_rmse": 0.0,
        "mean_r2": r2_score(y_train, meta_learner.predict(oof_preds)),
    }

    meta_coefficients = {
        name: coef for (name, _), coef in zip(base_models, meta_learner.coef_)
    }
    meta_coefficients["Intercept"] = meta_learner.intercept_
    
    # Save Stacking Ensemble (base models + meta learner)
    stacking_path = os.path.join(MODELS_DIR, f"{station_safe}_{scenario_safe}_stacking.pkl")
    stacking_package = {
        "base_models": trained_base_models,
        "meta_learner": meta_learner,
        "feature_cols": feature_cols,
    }
    joblib.dump(stacking_package, stacking_path)

    tag = f"[{station} | {scenario}]"
    print(f"    {tag}  RMSE → XGB {results['XGBoost']['RMSE']:.3f}  "
          f"RF {results['Random Forest']['RMSE']:.3f}  "
          f"GAM {results['GAM']['RMSE']:.3f}  "
          f"Stack {results['Stacking Ensemble']['RMSE']:.3f}")

    return {
        "results": results,
        "predictions": predictions,
        "cv_results": cv_results,
        "feature_importance": fi,
        "y_test": y_test.values,
        "test_dates": test_dates,
        "meta_coefficients": meta_coefficients,
    }


def _cv_loop(model, X_train, y_train, tscv):
    """Run TimeSeriesSplit CV for a tree-based model."""
    rmses, r2s = [], []
    for train_idx, val_idx in tscv.split(X_train):
        m = clone(model)
        m.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        pred = m.predict(X_train.iloc[val_idx])
        rmses.append(np.sqrt(mean_squared_error(y_train.iloc[val_idx], pred)))
        r2s.append(r2_score(y_train.iloc[val_idx], pred))
    return {"mean_rmse": np.mean(rmses), "std_rmse": np.std(rmses), "mean_r2": np.mean(r2s)}


def _cv_loop_gam(feature_cols, X_train, y_train, tscv):
    """Run TimeSeriesSplit CV for GAM (with per-fold scaling)."""
    rmses, r2s = [], []
    for train_idx, val_idx in tscv.split(X_train):
        try:
            fold_scaler = StandardScaler()
            gam = _build_gam_formula(feature_cols)
            Xt = fold_scaler.fit_transform(X_train.iloc[train_idx])
            Xv = fold_scaler.transform(X_train.iloc[val_idx])
            gam.fit(Xt, y_train.iloc[train_idx])
            pred = gam.predict(Xv)
            rmses.append(np.sqrt(mean_squared_error(y_train.iloc[val_idx], pred)))
            r2s.append(r2_score(y_train.iloc[val_idx], pred))
        except Exception:
            continue
    if rmses:
        return {"mean_rmse": np.mean(rmses), "std_rmse": np.std(rmses), "mean_r2": np.mean(r2s)}
    return {"mean_rmse": np.nan, "std_rmse": np.nan, "mean_r2": np.nan}


# ═══════════════════════════════════════════════════════════════════════════
#  7. COMPOSITE VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════
def _scenario_label(with_traffic: bool) -> str:
    return "With Traffic" if with_traffic else "No Traffic"


def _fname(name: str) -> str:
    return os.path.join(RESULTS_DIR, name)


def _split_stations(stations):
    """Split stations into Toronto (top row) and non-Toronto (bottom row)."""
    toronto = [s for s in stations if "Toronto" in s]
    others = [s for s in stations if "Toronto" not in s]
    return toronto, others


def _create_station_figure(stations, cell_w=7.5, cell_h=4.5, sharey=False,
                           hspace=GLOBAL_HSPACE, wspace=GLOBAL_WSPACE):
    """Create figure with centered 2-row layout (Toronto top, others bottom).

    Returns (fig, axes_dict) where axes_dict maps station_name -> Axes.
    """
    toronto, others = _split_stations(stations)
    n_top, n_bot = len(toronto), len(others)
    ncols_gs = 6  # LCM(2, 3) for even column spanning
    col_span = 2

    fig = plt.figure(figsize=(cell_w * 3, cell_h * 2))
    gs = fig.add_gridspec(2, ncols_gs, hspace=hspace, wspace=wspace)

    axes_dict = {}
    all_axes = []

    # Top row - Toronto stations, centered
    top_offset = (ncols_gs - n_top * col_span) // 2
    for i, station in enumerate(toronto):
        c = top_offset + i * col_span
        ax = fig.add_subplot(gs[0, c:c + col_span])
        axes_dict[station] = ax
        all_axes.append(ax)

    # Bottom row - other stations, centered
    bot_offset = (ncols_gs - n_bot * col_span) // 2
    for i, station in enumerate(others):
        c = bot_offset + i * col_span
        ax = fig.add_subplot(gs[1, c:c + col_span])
        axes_dict[station] = ax
        all_axes.append(ax)

    if sharey and len(all_axes) > 1:
        for ax in all_axes[1:]:
            ax.sharey(all_axes[0])
        for ax in all_axes:
            ax.tick_params(labelleft=True)

    return fig, axes_dict


def plot_model_performance(all_results: dict, stations: list[str], scenario_tag: str):
    """Bar chart of RMSE for every model, centered 2-row layout."""
    fig, axes_dict = _create_station_figure(stations, cell_w=6.0, cell_h=5.0, wspace=0.75)
    toronto, others = _split_stations(stations)

    for station in stations:
        ax = axes_dict[station]
        res = all_results[station]["results"]
        names = list(res.keys())
        rmses = [res[m]["RMSE"] for m in names]
        colors = [MODEL_COLORS.get(m, "grey") for m in names]

        bars = ax.barh(names, rmses, color=colors)
        for bar, v in zip(bars, rmses):
            ax.text(v + 0.05, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                    va="center", fontweight="bold")
        ax.set_title(station, fontweight="bold")
        ax.set_xlabel("RMSE")
        ax.invert_yaxis()
        ax.tick_params(axis="x")
        ax.tick_params(axis="y")
        # Only show y-axis labels on leftmost subplots
        if station != (toronto[0] if toronto else None) and station != (others[0] if others else None):
            ax.set_yticklabels([])
        ax.grid(True, alpha=0.3, axis="x")

    if toronto:
        axes_dict[toronto[0]].set_ylabel("RMSE")
    if others:
        axes_dict[others[0]].set_ylabel("RMSE")

    fig.suptitle(f"Model Performance – RMSE ({scenario_tag})",
                 fontweight="bold", y=0.98)
    fig.tight_layout()
    safe = scenario_tag.lower().replace(" ", "_")
    fig.savefig(_fname(f"model_performance_{safe}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(all_results: dict, stations: list[str], scenario_tag: str):
    """Top-10 XGBoost feature importances, centered 2-row layout."""
    from matplotlib.patches import Patch

    fig, axes_dict = _create_station_figure(stations, cell_w=8.0, cell_h=5.5, wspace=1.2)
    for station in stations:
        ax = axes_dict[station]
        fi = all_results[station]["feature_importance"].head(10)
        # Color traffic-related features red, others blue
        colors = ["#e74c3c" if "Traffic" in feat else "#3498db"
                  for feat in fi["Feature"]]
        ax.barh(fi["Feature"], fi["Importance"], color=colors)
        ax.set_title(station, fontweight="bold")
        ax.set_xlabel("Importance")
        ax.tick_params(axis="x")
        ax.tick_params(axis="y")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")

    # Add legend when traffic features may be present
    if "Traffic" in scenario_tag:
        legend_elements = [Patch(facecolor="#e74c3c", label="Traffic-related"),
                           Patch(facecolor="#3498db", label="Non-traffic")]
        fig.legend(handles=legend_elements, loc="upper right",
                   framealpha=0.9, bbox_to_anchor=(0.98, 0.98))

    fig.suptitle(f"XGBoost Feature Importance – Top 10 ({scenario_tag})",
                 fontweight="bold", y=0.98)
    fig.tight_layout()
    safe = scenario_tag.lower().replace(" ", "_")
    fig.savefig(_fname(f"feature_importance_{safe}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(all_results: dict, stations: list[str], scenario_tag: str):
    """Actual vs Stacking Ensemble predictions, centered 2-row layout."""
    fig, axes_dict = _create_station_figure(stations, wspace=0.75)
    toronto, others = _split_stations(stations)
    for station in stations:
        ax = axes_dict[station]
        r = all_results[station]
        dates = r["test_dates"]
        actual = r["y_test"]
        pred = r["predictions"]["Stacking Ensemble"]
        ax.plot(dates, actual, label="Actual", linewidth=1, alpha=0.8, color="blue")
        ax.plot(dates, pred, label="Stacking", linewidth=1, alpha=0.8,
                linestyle="--", color=MODEL_COLORS["Stacking Ensemble"])
        rmse = r["results"]["Stacking Ensemble"]["RMSE"]
        ax.set_title(f"{station}\nRMSE={rmse:.3f}", fontweight="bold")
        ax.set_xlabel("Date")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    if toronto:
        axes_dict[toronto[0]].set_ylabel("NO$_2$ (ppb)")
    if others:
        axes_dict[others[0]].set_ylabel("NO$_2$ (ppb)")
    fig.suptitle(f"Stacking Ensemble Predictions ({scenario_tag})",
                 fontweight="bold", y=0.98)
    fig.tight_layout()
    safe = scenario_tag.lower().replace(" ", "_")
    fig.savefig(_fname(f"predictions_{safe}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_residuals(all_results: dict, stations: list[str], scenario_tag: str):
    """Residual distribution for Stacking Ensemble, centered 2-row layout."""
    fig, axes_dict = _create_station_figure(stations, wspace=0.75)
    toronto, others = _split_stations(stations)
    for station in stations:
        ax = axes_dict[station]
        r = all_results[station]
        residuals = r["y_test"] - r["predictions"]["Stacking Ensemble"]
        ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7, color="#8e44ad")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(f"{station}\nmean={residuals.mean():.2f}  std={residuals.std():.2f}",
                     fontweight="bold")
        ax.set_xlabel("Residual")
        ax.tick_params(axis="x")
        ax.tick_params(axis="y")
        ax.grid(True, alpha=0.3, axis="y")
    if toronto:
        axes_dict[toronto[0]].set_ylabel("Frequency")
    if others:
        axes_dict[others[0]].set_ylabel("Frequency")
    fig.suptitle(f"Stacking Ensemble Residuals ({scenario_tag})",
                 fontweight="bold", y=0.98)
    fig.tight_layout()
    safe = scenario_tag.lower().replace(" ", "_")
    fig.savefig(_fname(f"residuals_{safe}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_traffic_impact(results_with: dict, results_without: dict, stations: list[str]):
    """Compare RMSE with/without traffic, centered 2-row layout."""
    fig, axes_dict = _create_station_figure(stations, cell_w=6.5, cell_h=5.0, wspace=0.75)
    toronto, others = _split_stations(stations)
    bar_width = 0.35

    for station in stations:
        ax = axes_dict[station]
        rw = results_with[station]["results"]
        rn = results_without[station]["results"]
        names = MODEL_NAMES
        rmse_w = [rw[m]["RMSE"] for m in names]
        rmse_n = [rn[m]["RMSE"] for m in names]
        x = np.arange(len(names))

        ax.barh(x - bar_width / 2, rmse_w, bar_width, label="With Traffic",
                color="#2ecc71", alpha=0.85)
        ax.barh(x + bar_width / 2, rmse_n, bar_width, label="No Traffic",
                color="#e74c3c", alpha=0.85)
        ax.set_yticks(x)
        ax.tick_params(axis="x")
        # Only show y-axis labels on leftmost subplots
        if station == (toronto[0] if toronto else None) or station == (others[0] if others else None):
            ax.set_yticklabels(names)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("RMSE")
        ax.invert_yaxis()
        ax.set_title(station, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
    if toronto:
        axes_dict[toronto[0]].set_ylabel("Model")
    if others:
        axes_dict[others[0]].set_ylabel("Model")
    fig.suptitle("Traffic Impact on Model RMSE", fontweight="bold", y=0.98)
    # Create a single legend for the entire figure
    handles, labels = axes_dict[stations[0]].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98), framealpha=0.9)
    fig.tight_layout()
    fig.savefig(_fname("traffic_impact_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cv_vs_test(all_results: dict, stations: list[str], scenario_tag: str):
    """CV RMSE vs Test RMSE, centered 2-row layout."""
    fig, axes_dict = _create_station_figure(stations, cell_w=6.5, cell_h=5.0, wspace=0.75)
    toronto, others = _split_stations(stations)
    bar_width = 0.35

    for station in stations:
        ax = axes_dict[station]
        r = all_results[station]
        res = r["results"]
        cv = r["cv_results"]
        names = MODEL_NAMES
        cv_rmses = [cv[m]["mean_rmse"] if m in cv and not np.isnan(cv[m]["mean_rmse"]) else 0 for m in names]
        test_rmses = [res[m]["RMSE"] for m in names]
        x = np.arange(len(names))

        ax.barh(x - bar_width / 2, cv_rmses, bar_width, label="CV RMSE", color="#3498db", alpha=0.85)
        ax.barh(x + bar_width / 2, test_rmses, bar_width, label="Test RMSE", color="#e67e22", alpha=0.85)
        ax.set_yticks(x)
        ax.tick_params(axis="x")
        # Only show y-axis labels on leftmost subplots
        if station == (toronto[0] if toronto else None) or station == (others[0] if others else None):
            ax.set_yticklabels(names)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("RMSE")
        ax.invert_yaxis()
        ax.set_title(station, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
    if toronto:
        axes_dict[toronto[0]].set_ylabel("Model")
    if others:
        axes_dict[others[0]].set_ylabel("Model")
    fig.suptitle(f"Cross-Validation vs Test RMSE ({scenario_tag})",
                 fontweight="bold", y=0.98)
    # Create a single legend for the entire figure
    handles, labels = axes_dict[stations[0]].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98), framealpha=0.9)
    fig.tight_layout()
    safe = scenario_tag.lower().replace(" ", "_")
    fig.savefig(_fname(f"cv_vs_test_{safe}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  8. RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════
def build_results_table(all_results_with, all_results_without, stations):
    """Build and save a comprehensive results CSV."""
    rows = []
    for station in stations:
        for scenario, res_dict in [("With Traffic", all_results_with),
                                    ("No Traffic", all_results_without)]:
            for model_name in MODEL_NAMES:
                m = res_dict[station]["results"][model_name]
                rows.append({
                    "Station": station,
                    "Scenario": scenario,
                    "Model": model_name,
                    "RMSE": round(m["RMSE"], 4),
                    "MAE": round(m["MAE"], 4),
                    "R2": round(m["R2"], 4),
                    "MAPE": round(m["MAPE"], 2) if not np.isnan(m["MAPE"]) else np.nan,
                })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "all_model_results.csv"), index=False)

    # Pivot for quick comparison
    pivot = df.pivot_table(index=["Station", "Scenario"], columns="Model", values="RMSE")
    pivot.to_csv(os.path.join(RESULTS_DIR, "rmse_comparison_pivot.csv"))

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  Air Quality Modeling Pipeline")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading datasets …")
    datasets = load_datasets(DATA_DIR)
    stations = list(datasets.keys())
    print(f"\n  Stations found: {len(stations)}")

    # ── EDA ───────────────────────────────────────────────────────────────
    print("\n[2/5] Exploratory Data Analysis …")
    eda_summary = run_eda(datasets)
    print(eda_summary.to_string(index=False))

    # ── Scenario A: With Traffic ──────────────────────────────────────────
    print("\n[3/5] Scenario A – With Traffic …")
    all_results_with: dict[str, dict] = {}
    for station, df_raw in datasets.items():
        df_clean, feat_cols = engineer_features(df_raw, with_traffic=True)
        all_results_with[station] = train_and_evaluate(
            df_clean, feat_cols, station, "With Traffic"
        )

    # ── Scenario B: No Traffic ────────────────────────────────────────────
    print("\n[4/5] Scenario B – No Traffic …")
    all_results_without: dict[str, dict] = {}
    for station, df_raw in datasets.items():
        df_clean, feat_cols = engineer_features(df_raw, with_traffic=False)
        all_results_without[station] = train_and_evaluate(
            df_clean, feat_cols, station, "No Traffic"
        )

    # ── Visualisation & results ───────────────────────────────────────────
    print("\n[5/5] Generating composite figures & saving results …")

    for scenario_flag, res, tag in [
        (True, all_results_with, "With Traffic"),
        (False, all_results_without, "No Traffic"),
    ]:
        plot_model_performance(res, stations, tag)
        plot_feature_importance(res, stations, tag)
        plot_predictions(res, stations, tag)
        plot_residuals(res, stations, tag)
        plot_cv_vs_test(res, stations, tag)

    plot_traffic_impact(all_results_with, all_results_without, stations)

    results_df = build_results_table(all_results_with, all_results_without, stations)

    # ── Print final summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 70)
    for scenario in ["With Traffic", "No Traffic"]:
        print(f"\n  ── {scenario} ──")
        sub = results_df[results_df["Scenario"] == scenario]
        for station in stations:
            best = sub[sub["Station"] == station].sort_values("RMSE").iloc[0]
            print(f"    {station:25s}  Best={best['Model']:20s}  RMSE={best['RMSE']:.4f}  R²={best['R2']:.4f}")

    print("\n" + "=" * 70)
    print(f"  All outputs saved to: {RESULTS_DIR}/")
    print("  Files:")
    for f_name in sorted(os.listdir(RESULTS_DIR)):
        print(f"    - {f_name}")
    
    print(f"\n  All trained models saved to: {MODELS_DIR}/")
    print("  Models:")
    model_files = sorted(os.listdir(MODELS_DIR))
    for f_name in model_files:
        print(f"    - {f_name}")
    print(f"\n  Total models saved: {len(model_files)}")
    
    print("=" * 70)
    print("  Pipeline complete.")


if __name__ == "__main__":
    main()
