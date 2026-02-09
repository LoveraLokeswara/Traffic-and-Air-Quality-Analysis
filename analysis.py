#!/usr/bin/env python3
"""
analysis.py - Automated Air Quality Modeling Pipeline
=====================================================
Iterates over every CSV in data/data_clean/, performs EDA, engineers features,
and trains four models (XGBoost, Random Forest, GAM, Stacking Ensemble) under
two scenarios (with traffic / without traffic).

All composite figures (5 subplots per image, one per station) and summary
tables are saved to the results/ folder.
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "data_clean")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

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
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    for ax, name in zip(axes, stations):
        df = datasets[name]
        ax.plot(df["Date"], df[TARGET], linewidth=0.6, alpha=0.8)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Date")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("NO$_2$ (ppb)")
    fig.suptitle("NO$_2$ Time Series by Station", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "eda_no2_timeseries.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── 2c. NO2 distribution ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    for ax, name in zip(axes, stations):
        df = datasets[name]
        ax.hist(df[TARGET].dropna(), bins=40, edgecolor="black", alpha=0.7)
        ax.axvline(df[TARGET].mean(), color="red", linestyle="--", label=f"mean={df[TARGET].mean():.1f}")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("NO$_2$ (ppb)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    axes[0].set_ylabel("Frequency")
    fig.suptitle("NO$_2$ Distribution by Station", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "eda_no2_distribution.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── 2d. Seasonal (monthly) NO2 ───────────────────────────────────────
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    for ax, name in zip(axes, stations):
        df = datasets[name].copy()
        df["Month"] = df["Date"].dt.month
        monthly = df.groupby("Month")[TARGET].mean()
        ax.plot(monthly.index, monthly.values, marker="o", linewidth=2, markersize=6)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_xticks(range(1, 13))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Mean NO$_2$ (ppb)")
    fig.suptitle("Seasonal NO$_2$ Pattern by Station", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "eda_seasonal_pattern.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── 2e. Correlation heatmaps ─────────────────────────────────────────
    corr_cols = [TARGET, TRAFFIC_COL, "Temp", "Precip", "Wind_Gust", "Wind_Dir"]
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    for ax, name in zip(axes, stations):
        cm = datasets[name][corr_cols].corr()
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    square=True, linewidths=0.5, cbar=False, ax=ax,
                    annot_kws={"size": 8})
        ax.set_title(name, fontsize=11, fontweight="bold")
    fig.suptitle("Feature Correlation by Station", fontsize=14, fontweight="bold", y=1.02)
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
        df["Wind_Rolling_7"] = df["Wind_Gust"].rolling(window=7, min_periods=1).mean()
        df["Temp_x_Wind"] = df["Temp"] * df["Wind_Gust"]
        df["Precip_x_Wind"] = df["Precip"] * df["Wind_Gust"]

        feature_cols = [
            # Raw features (no traffic)
            "Temp", "Precip", "Wind_Gust",
            # Temporal
            "Day_of_Week", "Month", "Is_Weekend", "Season",
            # Lag
            "NO2_Lag_1", "NO2_Lag_7",
            # Rolling
            "NO2_Rolling_7", "Temp_Rolling_7", "Wind_Rolling_7",
            # Interaction
            "Temp_x_Wind", "Precip_x_Wind",
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
    """Train all four models, evaluate, and return a results dict.

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

    # ── XGBoost ──────────────────────────────────────────────────────────
    xgb_model = _get_xgb()
    cv_scores = _cv_loop(xgb_model, X_train, y_train, tscv)
    cv_results["XGBoost"] = cv_scores
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    results["XGBoost"] = evaluate(y_test.values, xgb_pred)
    predictions["XGBoost"] = xgb_pred

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

    # ── GAM ──────────────────────────────────────────────────────────────
    gam_model = _build_gam_formula(feature_cols)
    cv_scores = _cv_loop_gam(feature_cols, X_train, y_train, tscv)
    cv_results["GAM"] = cv_scores
    try:
        gam_model.fit(X_train_scaled, y_train)
        gam_pred = gam_model.predict(X_test_scaled)
    except Exception:
        # Fallback: use unscaled data
        gam_model = _build_gam_formula(feature_cols)
        gam_model.fit(X_train.values, y_train)
        gam_pred = gam_model.predict(X_test.values)
    results["GAM"] = evaluate(y_test.values, gam_pred)
    predictions["GAM"] = gam_pred

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

    for idx, (name, model) in enumerate(base_models):
        for train_idx, val_idx in tscv.split(X_train):
            m = clone(model)
            m.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            oof_preds[val_idx, idx] = m.predict(X_train.iloc[val_idx])
        # Retrain on full training set
        model.fit(X_train, y_train)
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


def plot_model_performance(all_results: dict, stations: list[str], scenario_tag: str):
    """Bar chart of RMSE & R² for every model, 5 subplots (one per station)."""
    n = len(stations)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), sharey="row")

    for col, station in enumerate(stations):
        res = all_results[station]["results"]
        names = list(res.keys())
        rmses = [res[m]["RMSE"] for m in names]
        r2s = [res[m]["R2"] for m in names]
        colors = [MODEL_COLORS.get(m, "grey") for m in names]

        # RMSE
        ax = axes[0, col]
        bars = ax.barh(names, rmses, color=colors)
        for bar, v in zip(bars, rmses):
            ax.text(v + 0.05, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                    va="center", fontsize=8, fontweight="bold")
        ax.set_title(station, fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")
        if col == 0:
            ax.set_xlabel("RMSE (lower is better)")

        # R²
        ax = axes[1, col]
        bars = ax.barh(names, r2s, color=colors)
        for bar, v in zip(bars, r2s):
            ax.text(max(v + 0.01, 0.01), bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                    va="center", fontsize=8, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")
        if col == 0:
            ax.set_xlabel("R² (higher is better)")

    fig.suptitle(f"Model Performance Comparison ({scenario_tag})",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    safe = scenario_tag.lower().replace(" ", "_")
    fig.savefig(_fname(f"model_performance_{safe}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(all_results: dict, stations: list[str], scenario_tag: str):
    """Top-10 XGBoost feature importances, one subplot per station."""
    n = len(stations)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    for ax, station in zip(axes, stations):
        fi = all_results[station]["feature_importance"].head(10)
        ax.barh(fi["Feature"], fi["Importance"], color="#3498db")
        ax.set_title(station, fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")
    fig.suptitle(f"XGBoost Feature Importance – Top 10 ({scenario_tag})",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    safe = scenario_tag.lower().replace(" ", "_")
    fig.savefig(_fname(f"feature_importance_{safe}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(all_results: dict, stations: list[str], scenario_tag: str):
    """Actual vs Stacking Ensemble predictions, one subplot per station."""
    n = len(stations)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    for ax, station in zip(axes, stations):
        r = all_results[station]
        dates = r["test_dates"]
        actual = r["y_test"]
        pred = r["predictions"]["Stacking Ensemble"]
        ax.plot(dates, actual, label="Actual", linewidth=1, alpha=0.8, color="blue")
        ax.plot(dates, pred, label="Stacking", linewidth=1, alpha=0.8,
                linestyle="--", color=MODEL_COLORS["Stacking Ensemble"])
        rmse = r["results"]["Stacking Ensemble"]["RMSE"]
        r2 = r["results"]["Stacking Ensemble"]["R2"]
        ax.set_title(f"{station}\nRMSE={rmse:.3f}  R²={r2:.3f}", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("NO$_2$ (ppb)")
    fig.suptitle(f"Stacking Ensemble Predictions ({scenario_tag})",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    safe = scenario_tag.lower().replace(" ", "_")
    fig.savefig(_fname(f"predictions_{safe}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_actual_vs_pred(all_results: dict, stations: list[str], scenario_tag: str):
    """Scatter: actual vs predicted for every model, one subplot per station."""
    n = len(stations)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    for ax, station in zip(axes, stations):
        r = all_results[station]
        actual = r["y_test"]
        for model_name in MODEL_NAMES:
            pred = r["predictions"][model_name]
            ax.scatter(actual, pred, alpha=0.35, s=15, label=model_name,
                       color=MODEL_COLORS[model_name])
        lo = min(actual.min(), min(r["predictions"][m].min() for m in MODEL_NAMES))
        hi = max(actual.max(), max(r["predictions"][m].max() for m in MODEL_NAMES))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="Perfect")
        ax.set_title(station, fontsize=10, fontweight="bold")
        ax.set_xlabel("Actual NO$_2$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc="upper left")
    axes[0].set_ylabel("Predicted NO$_2$")
    fig.suptitle(f"Actual vs Predicted ({scenario_tag})",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    safe = scenario_tag.lower().replace(" ", "_")
    fig.savefig(_fname(f"scatter_actual_vs_pred_{safe}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(all_results: dict, stations: list[str], scenario_tag: str):
    """Residual distribution for Stacking Ensemble, one subplot per station."""
    n = len(stations)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    for ax, station in zip(axes, stations):
        r = all_results[station]
        residuals = r["y_test"] - r["predictions"]["Stacking Ensemble"]
        ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7, color="#8e44ad")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(f"{station}\nmean={residuals.mean():.2f}  std={residuals.std():.2f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Residual")
        ax.grid(True, alpha=0.3, axis="y")
    axes[0].set_ylabel("Frequency")
    fig.suptitle(f"Stacking Ensemble Residuals ({scenario_tag})",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    safe = scenario_tag.lower().replace(" ", "_")
    fig.savefig(_fname(f"residuals_{safe}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_traffic_impact(results_with: dict, results_without: dict, stations: list[str]):
    """Compare RMSE with/without traffic for every model, one subplot per station."""
    n = len(stations)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    bar_width = 0.35

    for ax, station in zip(axes, stations):
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
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(station, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        ax.legend(fontsize=7)
    axes[0].set_xlabel("RMSE")
    fig.suptitle("Traffic Impact on Model RMSE", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(_fname("traffic_impact_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cv_vs_test(all_results: dict, stations: list[str], scenario_tag: str):
    """CV RMSE vs Test RMSE for each model, one subplot per station."""
    n = len(stations)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    bar_width = 0.35

    for ax, station in zip(axes, stations):
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
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(station, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        ax.legend(fontsize=7)
    axes[0].set_xlabel("RMSE")
    fig.suptitle(f"Cross-Validation vs Test RMSE ({scenario_tag})",
                 fontsize=14, fontweight="bold", y=1.02)
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
        plot_scatter_actual_vs_pred(res, stations, tag)
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
    print("=" * 70)
    print("  Pipeline complete.")


if __name__ == "__main__":
    main()
