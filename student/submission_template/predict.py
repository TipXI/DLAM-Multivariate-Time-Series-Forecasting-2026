"""Inference entrypoint for final evaluation.

Executes offline inference without internet dependency.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from src.model import ForecastModel


def load_input_frames(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load inputs and forecast index dynamically."""
    idx_candidates = [
        input_dir / "forecast_index_test.csv",
        input_dir / "forecast_index_validation.csv",
    ]
    forecast_index_file = None
    for c in idx_candidates:
        if c.exists():
            forecast_index_file = c
            break

    if forecast_index_file is None:
        raise FileNotFoundError(f"Missing forecast index file in {input_dir}")

    input_candidates = [
        input_dir / "test_input.csv",
        input_dir / "validation_input.csv",
    ]
    input_file = None
    for c in input_candidates:
        if c.exists():
            input_file = c
            break

    if input_file is None:
        raise FileNotFoundError(f"Missing input csv in {input_dir}")

    forecast_index = pd.read_csv(forecast_index_file)
    input_df = pd.read_csv(input_file)
    return forecast_index, input_df


def build_full_features(df: pd.DataFrame, series2idx: dict[str, int]) -> pd.DataFrame:
    """Extract multi-scale rolling temporal features, physics ratios, and missingness masks."""
    d = df.copy()
    d["dt"] = pd.to_datetime(d["timestamp"], format="%Y-%m-%d %H:%M:%S")
    d["hour"] = d["dt"].dt.hour
    d["dow"] = d["dt"].dt.dayofweek
    d["hour_sin_calc"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos_calc"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["dow_sin_calc"] = np.sin(2 * np.pi * d["dow"] / 7)
    d["dow_cos_calc"] = np.cos(2 * np.pi * d["dow"] / 7)

    # 1. Missingness indicator masks
    missing_cols = [
        "demand_forecast",
        "queue_pressure_forecast",
        "network_pressure_forecast",
        "shock_risk",
        "unit_reliability_forecast",
        "event_load_forecast",
        "service_irregularity_risk_forecast",
        "throughput_disruption_risk_forecast",
        "staffing_forecast",
        "upstream_quality_forecast",
    ]
    for c in missing_cols:
        if c in d.columns:
            d[f"{c}_isna"] = d[c].isna().astype(float)

    # 2. Series-level forward/backward fill + median imputation
    cov_cols = [
        "workload_intensity",
        "demand_forecast",
        "staffing_forecast",
        "upstream_quality_forecast",
        "promotion_intensity",
        "shock_risk",
        "maintenance_known",
        "unit_reliability_forecast",
        "queue_pressure_forecast",
        "network_pressure_forecast",
        "event_load_forecast",
        "service_irregularity_risk_forecast",
        "throughput_disruption_risk_forecast",
        "nominal_capacity",
    ]
    for c in cov_cols:
        if c in d.columns:
            d[c] = d.groupby("series_id")[c].transform(lambda x: x.ffill().bfill())
            d[c] = d[c].fillna(d[c].median())

    # 3. Multi-scale forward rolling cumulative windows (3h, 6h, 12h, 24h)
    for win in [3, 6, 12, 24]:
        d[f"queue_pressure_roll_{win}"] = d.groupby("series_id")["queue_pressure_forecast"].transform(
            lambda x: x.rolling(win, min_periods=1, center=True).mean()
        )
        d[f"network_pressure_roll_{win}"] = d.groupby("series_id")["network_pressure_forecast"].transform(
            lambda x: x.rolling(win, min_periods=1, center=True).mean()
        )
        d[f"workload_roll_{win}"] = d.groupby("series_id")["workload_intensity"].transform(
            lambda x: x.rolling(win, min_periods=1, center=True).mean()
        )
        d[f"demand_roll_{win}"] = d.groupby("series_id")["demand_forecast"].transform(
            lambda x: x.rolling(win, min_periods=1, center=True).mean()
        )

    # 4. Domain physics and interaction features
    d["pressure_sum"] = d["queue_pressure_forecast"] + d["network_pressure_forecast"]
    d["pressure_mult"] = d["queue_pressure_forecast"] * d["network_pressure_forecast"]
    d["pressure_ratio"] = (d["queue_pressure_forecast"] + 0.1) / (d["network_pressure_forecast"] + 0.1)
    d["workload_pressure"] = d["workload_intensity"] * (d["pressure_sum"] + 1.0)
    d["workload_sq"] = d["workload_intensity"] ** 2
    d["capacity_utilization"] = d["workload_intensity"] / (d["nominal_capacity"] + 1e-5)
    d["capacity_pressure"] = d["pressure_sum"] / (d["nominal_capacity"] + 1e-5)
    d["risk_impact"] = d["shock_risk"] * (1.0 - d["unit_reliability_forecast"])
    d["shock_x_pressure"] = d["shock_risk"] * d["pressure_sum"]
    d["event_risk"] = d["event_load_forecast"] * (d["service_irregularity_risk_forecast"] + 1.0)
    d["staffing_ratio"] = d["demand_forecast"] / (d["staffing_forecast"].abs() + 1e-5)

    # 5. Map series_id to integer index
    d["series_idx"] = d["series_id"].map(series2idx).fillna(0).astype(int)
    return d


def main() -> None:
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Generate private predictions.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    forecast_index, input_df = load_input_frames(args.input_dir)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    feature_names = checkpoint["feature_names"]
    norm_stats = checkpoint["norm_stats"]
    series2idx = checkpoint["series2idx"]
    weights = checkpoint.get(
        "blend_weights",
        {"dl_weight": 0.55, "gbm_weight": 0.45, "gbm_a_ratio": 0.6, "gbm_b_ratio": 0.4},
    )

    # Prepend boundary context if available and contiguous
    df_to_feature = input_df.copy()
    min_input_ts = pd.to_datetime(input_df["timestamp"].min())
    train_tail = checkpoint.get("train_tail", None)
    val_tail = checkpoint.get("val_tail", None)

    if train_tail is not None and not train_tail.empty:
        max_train_ts = pd.to_datetime(train_tail["timestamp"].max())
        if abs((min_input_ts - max_train_ts).total_seconds()) <= 7200:
            df_to_feature = pd.concat([train_tail, input_df], ignore_index=True)
    elif val_tail is not None and not val_tail.empty:
        max_val_ts = pd.to_datetime(val_tail["timestamp"].max())
        if abs((min_input_ts - max_val_ts).total_seconds()) <= 7200:
            df_to_feature = pd.concat([val_tail, input_df], ignore_index=True)

    df_to_feature = df_to_feature.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    featured_df = build_full_features(df_to_feature, series2idx)

    # Align back to input rows
    eval_df = featured_df.merge(
        input_df[["series_id", "timestamp"]],
        on=["series_id", "timestamp"],
        how="inner",
    ).reset_index(drop=True)

    X_num = eval_df[feature_names].to_numpy(dtype=np.float32)
    s_idx = eval_df["series_idx"].to_numpy(dtype=np.int64)
    X_norm = (X_num - norm_stats["mean"]) / norm_stats["std"]
    X_full = np.column_stack([X_num, s_idx])

    # 1. PyTorch Deep Neural Network Ensemble Inference
    dl_preds_list = []
    for state_dict in checkpoint["dl_state_dicts"]:
        model = ForecastModel(num_continuous_features=len(feature_names), num_series=len(series2idx))
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            preds = model(
                torch.tensor(X_norm, dtype=torch.float32),
                torch.tensor(s_idx, dtype=torch.long),
            ).numpy().flatten()
            dl_preds_list.append(preds)
    dl_ensemble_preds = np.mean(dl_preds_list, axis=0)

    # 2. LightGBM Multi-Tree Gradient Boosting Booster Inference
    gbm_a = lgb.Booster(model_str=checkpoint["gbm_a_str"])
    gbm_b = lgb.Booster(model_str=checkpoint["gbm_b_str"])
    gbm_a_preds = gbm_a.predict(X_full)
    gbm_b_preds = gbm_b.predict(X_full)
    gbm_ensemble_preds = (
        weights["gbm_a_ratio"] * gbm_a_preds + weights["gbm_b_ratio"] * gbm_b_preds
    )

    # 3. Grand Ensemble Blending
    final_preds = weights["dl_weight"] * dl_ensemble_preds + weights["gbm_weight"] * gbm_ensemble_preds
    final_preds = np.clip(final_preds, a_min=0.0, a_max=None)

    eval_df["prediction"] = final_preds
    final_predictions = forecast_index.merge(
        eval_df[["series_id", "timestamp", "prediction"]],
        on=["series_id", "timestamp"],
        how="left",
    )

    if final_predictions["prediction"].isna().any():
        raise ValueError("Generated predictions contain NaN values.")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    final_predictions.to_csv(args.output_file, index=False)
    elapsed = time.time() - t0
    print(f"Predictions written to: {args.output_file} ({len(final_predictions)} rows in {elapsed:.2f}s)")


if __name__ == "__main__":
    main()
