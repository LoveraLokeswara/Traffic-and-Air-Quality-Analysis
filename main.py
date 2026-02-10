#!/usr/bin/env python3
"""
main.py - Load Saved Models and Generate Evaluation Plots
==========================================================
Loads pre-trained models from models/ folder, evaluates them on test data,
and generates comprehensive evaluation plots without retraining.

This script is designed for CI/CD workflows where you want to evaluate
existing models without the computational cost of retraining.
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "data_clean")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "evaluation")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET = "NO2_Mean"
TRAFFIC_COL = "Traffic_Count"
RANDOM_STATE = 42

MODEL_NAMES = ["XGBoost", "Random Forest", "GAM", "Stacking Ensemble"]
MODEL_COLORS = {
    "XGBoost": "#e67e22",
    "Random Forest": "#27ae60",
    "GAM": "#2980b9",
    "Stacking Ensemble": "#c0392b",
}


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
def load_datasets(data_dir: str) -> dict[str, pd.DataFrame]:
    """Read every CSV in *data_dir* and return {station_name: DataFrame}."""
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    datasets: dict[str, pd.DataFrame] = {}
    for fpath in csv_files:
        df = pd.read_csv(fpath, parse_dates=["Date"])
        station = df["Station"].iloc[0]
        datasets[station] = df
        print(f"  Loaded {station:25s}  ({len(df):>5} rows)")
    return datasets


def engineer_features(df_raw: pd.DataFrame, with_traffic: bool) -> tuple[pd.DataFrame, list[str]]:
    """Create temporal, lag, rolling, interaction, and wind-component features."""
    df = df_raw.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # Handle missing values
    df["Temp"] = df["Temp"].interpolate(method="linear")
    df["Precip"] = df["Precip"].fillna(0)
    df["Wind_Gust"] = df["Wind_Gust"].interpolate(method="linear")
    df["Wind_Dir"] = df["Wind_Dir"].interpolate(method="linear")

    # Temporal features
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["Day_of_Year"] = df["Date"].dt.dayofyear
    df["Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(int)
    df["Season"] = df["Month"] % 12 // 3 + 1

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
            "Traffic_Count", "Temp", "Precip", "Wind_Gust",
            "Day_of_Week", "Month", "Is_Weekend", "Season",
            "NO2_Lag_1", "NO2_Lag_7", "Traffic_Lag_1",
            "NO2_Rolling_7", "Traffic_Rolling_7", "Temp_Rolling_7",
            "Traffic_x_Wind", "Temp_x_Traffic",
            "Wind_NS", "Wind_EW",
        ]
    else:
        df["Wind_Rolling_7"] = df["Wind_Gust"].rolling(window=7, min_periods=1).mean()
        df["Temp_x_Wind"] = df["Temp"] * df["Wind_Gust"]
        df["Precip_x_Wind"] = df["Precip"] * df["Wind_Gust"]

        feature_cols = [
            "Temp", "Precip", "Wind_Gust",
            "Day_of_Week", "Month", "Is_Weekend", "Season",
            "NO2_Lag_1", "NO2_Lag_7",
            "NO2_Rolling_7", "Temp_Rolling_7", "Wind_Rolling_7",
            "Temp_x_Wind", "Precip_x_Wind",
            "Wind_NS", "Wind_EW",
        ]

    df_clean = df.dropna(subset=feature_cols + [TARGET]).reset_index(drop=True)
    return df_clean, feature_cols


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL LOADING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
def evaluate(y_true, y_pred) -> dict:
    """Return dict of regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def load_and_evaluate_models(station: str, df_clean: pd.DataFrame, feature_cols: list[str], 
                             scenario: str) -> dict:
    """Load saved models and evaluate on test set."""
    X = df_clean[feature_cols]
    y = df_clean[TARGET]
    
    # Same 80/20 split as training
    split_idx = int(len(X) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    test_dates = df_clean["Date"].iloc[split_idx:].reset_index(drop=True)
    
    # Create safe filenames
    scenario_safe = scenario.lower().replace(" ", "_")
    station_safe = station.replace(" ", "_").replace("/", "_")
    
    results = {}
    predictions = {}
    
    # ── Load XGBoost ──
    xgb_path = os.path.join(MODELS_DIR, f"{station_safe}_{scenario_safe}_xgboost.pkl")
    if os.path.exists(xgb_path):
        xgb_model = joblib.load(xgb_path)
        xgb_pred = xgb_model.predict(X_test)
        results["XGBoost"] = evaluate(y_test.values, xgb_pred)
        predictions["XGBoost"] = xgb_pred
    else:
        print(f"  ⚠ Warning: {xgb_path} not found")
    
    # ── Load Random Forest ──
    rf_path = os.path.join(MODELS_DIR, f"{station_safe}_{scenario_safe}_random_forest.pkl")
    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
        rf_pred = rf_model.predict(X_test)
        results["Random Forest"] = evaluate(y_test.values, rf_pred)
        predictions["Random Forest"] = rf_pred
    else:
        print(f"  ⚠ Warning: {rf_path} not found")
    
    # ── Load GAM ──
    gam_path = os.path.join(MODELS_DIR, f"{station_safe}_{scenario_safe}_gam.pkl")
    if os.path.exists(gam_path):
        gam_pkg = joblib.load(gam_path)
        gam_model = gam_pkg["model"]
        scaler = gam_pkg["scaler"]
        
        if scaler:
            X_test_scaled = scaler.transform(X_test)
            gam_pred = gam_model.predict(X_test_scaled)
        else:
            gam_pred = gam_model.predict(X_test.values)
        
        results["GAM"] = evaluate(y_test.values, gam_pred)
        predictions["GAM"] = gam_pred
    else:
        print(f"  ⚠ Warning: {gam_path} not found")
    
    # ── Load Stacking Ensemble ──
    stack_path = os.path.join(MODELS_DIR, f"{station_safe}_{scenario_safe}_stacking.pkl")
    if os.path.exists(stack_path):
        stack_pkg = joblib.load(stack_path)
        base_models = stack_pkg["base_models"]
        meta_learner = stack_pkg["meta_learner"]
        
        # Generate base predictions
        base_preds = np.column_stack([model.predict(X_test) for _, model in base_models])
        stacking_pred = meta_learner.predict(base_preds)
        
        results["Stacking Ensemble"] = evaluate(y_test.values, stacking_pred)
        predictions["Stacking Ensemble"] = stacking_pred
    else:
        print(f"  ⚠ Warning: {stack_path} not found")
    
    tag = f"[{station} | {scenario}]"
    if results:
        print(f"  {tag}  RMSE → ", end="")
        for model_name in ["XGBoost", "Random Forest", "GAM", "Stacking Ensemble"]:
            if model_name in results:
                print(f"{model_name[:5]} {results[model_name]['RMSE']:.3f}  ", end="")
        print()
    
    return {
        "results": results,
        "predictions": predictions,
        "y_test": y_test.values,
        "test_dates": test_dates,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
def plot_model_performance(all_results: dict, stations: list[str], scenario_tag: str):
    """Bar chart of RMSE & R² for every model."""
    n = len(stations)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), sharey="row")
    
    for col, station in enumerate(stations):
        if station not in all_results or not all_results[station]["results"]:
            continue
            
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
    fig.savefig(os.path.join(RESULTS_DIR, f"eval_performance_{safe}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_predictions_comparison(all_results: dict, stations: list[str], scenario_tag: str):
    """Actual vs all model predictions."""
    n = len(stations)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    
    for ax, station in zip(axes, stations):
        if station not in all_results or not all_results[station]["results"]:
            continue
            
        r = all_results[station]
        dates = r["test_dates"]
        actual = r["y_test"]
        
        ax.plot(dates, actual, label="Actual", linewidth=1.5, alpha=0.9, color="black")
        
        for model_name in MODEL_NAMES:
            if model_name in r["predictions"]:
                pred = r["predictions"][model_name]
                ax.plot(dates, pred, label=model_name, linewidth=1, alpha=0.7,
                       linestyle="--", color=MODEL_COLORS[model_name])
        
        ax.set_title(station, fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
    
    axes[0].set_ylabel("NO$_2$ (ppb)")
    fig.suptitle(f"Model Predictions Comparison ({scenario_tag})",
                fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    safe = scenario_tag.lower().replace(" ", "_")
    fig.savefig(os.path.join(RESULTS_DIR, f"eval_predictions_{safe}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_actual_vs_pred(all_results: dict, stations: list[str], scenario_tag: str):
    """Scatter: actual vs predicted for every model."""
    n = len(stations)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    
    for ax, station in zip(axes, stations):
        if station not in all_results or not all_results[station]["results"]:
            continue
            
        r = all_results[station]
        actual = r["y_test"]
        
        for model_name in MODEL_NAMES:
            if model_name in r["predictions"]:
                pred = r["predictions"][model_name]
                ax.scatter(actual, pred, alpha=0.35, s=15, label=model_name,
                          color=MODEL_COLORS[model_name])
        
        lo = min(actual.min(), min(r["predictions"][m].min() for m in r["predictions"]))
        hi = max(actual.max(), max(r["predictions"][m].max() for m in r["predictions"]))
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
    fig.savefig(os.path.join(RESULTS_DIR, f"eval_scatter_{safe}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_residuals_comparison(all_results: dict, stations: list[str], scenario_tag: str):
    """Residual distribution for all models."""
    n_stations = len(stations)
    n_models = len(MODEL_NAMES)
    
    fig, axes = plt.subplots(n_models, n_stations, figsize=(5 * n_stations, 4 * n_models))
    
    for row, model_name in enumerate(MODEL_NAMES):
        for col, station in enumerate(stations):
            if station not in all_results or model_name not in all_results[station]["predictions"]:
                continue
            
            r = all_results[station]
            residuals = r["y_test"] - r["predictions"][model_name]
            
            ax = axes[row, col] if n_models > 1 else axes[col]
            ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7, 
                   color=MODEL_COLORS[model_name])
            ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
            ax.set_title(f"{station}\n{model_name}\nmean={residuals.mean():.2f}, std={residuals.std():.2f}",
                        fontsize=9, fontweight="bold")
            ax.set_xlabel("Residual")
            ax.grid(True, alpha=0.3, axis="y")
            
            if col == 0:
                ax.set_ylabel("Frequency")
    
    fig.suptitle(f"Residual Distributions ({scenario_tag})",
                fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()
    safe = scenario_tag.lower().replace(" ", "_")
    fig.savefig(os.path.join(RESULTS_DIR, f"eval_residuals_{safe}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_traffic_impact(results_with: dict, results_without: dict, stations: list[str]):
    """Compare RMSE with/without traffic for every model."""
    n = len(stations)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    bar_width = 0.35
    
    for ax, station in zip(axes, stations):
        if station not in results_with or station not in results_without:
            continue
            
        rw = results_with[station]["results"]
        rn = results_without[station]["results"]
        
        # Only use models that exist in both scenarios
        common_models = [m for m in MODEL_NAMES if m in rw and m in rn]
        
        if not common_models:
            continue
        
        rmse_w = [rw[m]["RMSE"] for m in common_models]
        rmse_n = [rn[m]["RMSE"] for m in common_models]
        x = np.arange(len(common_models))
        
        ax.barh(x - bar_width / 2, rmse_w, bar_width, label="With Traffic",
               color="#2ecc71", alpha=0.85)
        ax.barh(x + bar_width / 2, rmse_n, bar_width, label="No Traffic",
               color="#e74c3c", alpha=0.85)
        ax.set_yticks(x)
        ax.set_yticklabels(common_models, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(station, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        ax.legend(fontsize=7)
    
    axes[0].set_xlabel("RMSE")
    fig.suptitle("Traffic Impact on Model RMSE", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "eval_traffic_impact.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_results_table(all_results_with, all_results_without, stations):
    """Build and save a comprehensive results CSV."""
    rows = []
    for station in stations:
        for scenario, res_dict in [("With Traffic", all_results_with),
                                   ("No Traffic", all_results_without)]:
            if station not in res_dict:
                continue
                
            for model_name in MODEL_NAMES:
                if model_name not in res_dict[station]["results"]:
                    continue
                    
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
    df.to_csv(os.path.join(RESULTS_DIR, "evaluation_results.csv"), index=False)
    
    # Pivot for quick comparison
    pivot = df.pivot_table(index=["Station", "Scenario"], columns="Model", values="RMSE")
    pivot.to_csv(os.path.join(RESULTS_DIR, "evaluation_rmse_pivot.csv"))
    
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  Model Evaluation Pipeline (No Training)")
    print("=" * 70)
    
    # Check if models directory exists
    if not os.path.exists(MODELS_DIR):
        print(f"\n❌ ERROR: Models directory not found at {MODELS_DIR}")
        print("Please run src/analysis.py first to train and save models.")
        return
    
    model_files = os.listdir(MODELS_DIR)
    if not model_files:
        print(f"\n❌ ERROR: No model files found in {MODELS_DIR}")
        print("Please run src/analysis.py first to train and save models.")
        return
    
    print(f"\n✓ Found {len(model_files)} model files in {MODELS_DIR}")
    
    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading datasets …")
    datasets = load_datasets(DATA_DIR)
    stations = list(datasets.keys())
    print(f"\n  Stations found: {len(stations)}")
    
    # ── Scenario A: With Traffic ──────────────────────────────────────────
    print("\n[2/4] Evaluating models – With Traffic …")
    all_results_with = {}
    for station, df_raw in datasets.items():
        df_clean, feat_cols = engineer_features(df_raw, with_traffic=True)
        all_results_with[station] = load_and_evaluate_models(
            station, df_clean, feat_cols, "With Traffic"
        )
    
    # ── Scenario B: No Traffic ────────────────────────────────────────────
    print("\n[3/4] Evaluating models – No Traffic …")
    all_results_without = {}
    for station, df_raw in datasets.items():
        df_clean, feat_cols = engineer_features(df_raw, with_traffic=False)
        all_results_without[station] = load_and_evaluate_models(
            station, df_clean, feat_cols, "No Traffic"
        )
    
    # ── Visualization & results ───────────────────────────────────────────
    print("\n[4/4] Generating evaluation plots …")
    
    for res, tag in [(all_results_with, "With Traffic"),
                     (all_results_without, "No Traffic")]:
        plot_model_performance(res, stations, tag)
        plot_predictions_comparison(res, stations, tag)
        plot_scatter_actual_vs_pred(res, stations, tag)
        plot_residuals_comparison(res, stations, tag)
    
    plot_traffic_impact(all_results_with, all_results_without, stations)
    results_df = build_results_table(all_results_with, all_results_without, stations)
    
    # ── Print final summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    
    for scenario in ["With Traffic", "No Traffic"]:
        print(f"\n  ── {scenario} ──")
        sub = results_df[results_df["Scenario"] == scenario]
        for station in stations:
            station_data = sub[sub["Station"] == station]
            if not station_data.empty:
                best = station_data.sort_values("RMSE").iloc[0]
                print(f"    {station:25s}  Best={best['Model']:20s}  "
                     f"RMSE={best['RMSE']:.4f}  R²={best['R2']:.4f}")
    
    print("\n" + "=" * 70)
    print(f"  All evaluation outputs saved to: {RESULTS_DIR}/")
    print("  Files:")
    for f_name in sorted(os.listdir(RESULTS_DIR)):
        print(f"    - {f_name}")
    print("=" * 70)
    print("  ✓ Evaluation complete (no training performed).")
    print("=" * 70)


if __name__ == "__main__":
    main()
