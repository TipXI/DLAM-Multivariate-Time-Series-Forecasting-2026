"""Inference entrypoint for final evaluation.

Executes offline inference without internet dependency.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.model import ForecastModel

COVARIATE_COLS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "trend",
    "zone_sin",
    "zone_cos",
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

MISSING_CANDIDATE_COLS = [
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


def extract_features(
    df: pd.DataFrame,
    medians: dict[str, float],
    series2idx: dict[str, int],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features matching training pipeline."""
    d = df.copy()

    for c in MISSING_CANDIDATE_COLS:
        if c in d.columns:
            d[f"{c}_isna"] = d[c].isna().astype(float)

    for c in COVARIATE_COLS:
        if c in d.columns:
            d[c] = d.groupby("series_id")[c].transform(lambda x: x.ffill().bfill())
            d[c] = d[c].fillna(medians.get(c, 0.0))

    d["pressure_sum"] = d["queue_pressure_forecast"] + d["network_pressure_forecast"]
    d["pressure_mult"] = d["queue_pressure_forecast"] * d["network_pressure_forecast"]
    d["effective_workload"] = d["workload_intensity"] * (d["queue_pressure_forecast"] + 1.0)
    d["capacity_utilization"] = d["workload_intensity"] / (d["nominal_capacity"] + 1e-5)
    d["risk_impact"] = d["shock_risk"] * (1.0 - d["unit_reliability_forecast"])
    d["event_risk"] = d["event_load_forecast"] * (d["service_irregularity_risk_forecast"] + 1.0)
    d["demand_staff_ratio"] = d["demand_forecast"] / (d["staffing_forecast"].abs() + 1e-5)
    d["workload_x_demand"] = d["workload_intensity"] * d["demand_forecast"]

    d["series_idx"] = d["series_id"].map(series2idx).fillna(0).astype(int)

    num_cols = [c for c in feature_names if c != "series_idx"]
    X_num = d[num_cols].to_numpy(dtype=np.float32)
    s_idx = d["series_idx"].to_numpy(dtype=np.int64)
    return X_num, s_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate private predictions.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    forecast_index, input_df = load_input_frames(args.input_dir)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    norm_stats = checkpoint["norm_stats"]
    feature_names = checkpoint["feature_names"]
    medians = checkpoint["medians"]
    series2idx = checkpoint["series2idx"]

    num_cols = [c for c in feature_names if c != "series_idx"]
    model = ForecastModel(num_continuous_features=len(num_cols), num_series=len(series2idx))
    model.load_state_dict(state_dict)
    model.eval()

    X_num, s_idx = extract_features(input_df, medians, series2idx, feature_names)
    X_norm = (X_num - norm_stats["mean"]) / norm_stats["std"]

    with torch.no_grad():
        preds = model(
            torch.tensor(X_norm, dtype=torch.float32),
            torch.tensor(s_idx, dtype=torch.long),
        ).numpy().flatten()

    predictions = input_df[["series_id", "timestamp"]].copy()
    predictions["prediction"] = np.clip(preds, a_min=0.0, a_max=None)

    final_predictions = forecast_index.merge(predictions, on=["series_id", "timestamp"], how="left")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    final_predictions.to_csv(args.output_file, index=False)
    print(f"Predictions written to: {args.output_file} ({len(final_predictions)} rows)")


if __name__ == "__main__":
    main()
