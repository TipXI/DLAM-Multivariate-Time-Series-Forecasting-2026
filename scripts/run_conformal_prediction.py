"""Run Conformal Prediction Uncertainty Quantification on the validation set."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.conformal import ConformalPredictor


def main() -> None:
    results_dir = PROJECT_ROOT / "results" / "conformal"
    results_dir.mkdir(parents=True, exist_ok=True)

    data_dir = PROJECT_ROOT / "data"
    train_raw = pd.read_csv(data_dir / "train.csv").sort_values(["series_id", "timestamp"]).reset_index(drop=True)

    val_horizon = 336
    local_val_parts = []
    for _, g in train_raw.groupby("series_id", sort=False):
        local_val_parts.append(g.iloc[-val_horizon:])
    val_truth_df = pd.concat(local_val_parts).reset_index(drop=True)

    # Load top ensemble predictions
    pred_path = PROJECT_ROOT / "submissions" / "grand_master_v3.csv"
    if not pred_path.exists():
        pred_path = PROJECT_ROOT / "submissions" / "ensemble_v1.csv"
    preds_df = pd.read_csv(pred_path)

    # Evaluate conformal prediction at multiple confidence levels (80%, 90%, 95%)
    print("=" * 75)
    print("SPLIT CONFORMAL PREDICTION (Angelopoulos & Bates, 2021)")
    print("=" * 75)

    y_true = val_truth_df["target"].to_numpy()
    y_pred = preds_df["prediction"].to_numpy()

    # Calibration split: first half of validation as calibration, second half as test
    n_half = len(y_true) // 2
    y_calib_t, y_calib_p = y_true[:n_half], y_pred[:n_half]
    y_test_t, y_test_p = y_true[n_half:], y_pred[n_half:]

    results = []
    for alpha in [0.20, 0.10, 0.05]:
        cp = ConformalPredictor(alpha=alpha).calibrate(y_calib_t, y_calib_p)
        cov_metrics = cp.evaluate_coverage(y_test_t, y_test_p)
        results.append(cov_metrics)
        print(f"Confidence Level: {cov_metrics['Nominal Confidence (%)']:.0f}% -> "
              f"Empirical Coverage: {cov_metrics['Empirical Coverage (%)']:.2f}% | "
              f"Quantile: {cov_metrics['Calibrated Quantile (q_val)']:.3f} | "
              f"Interval Width: {cov_metrics['Average Interval Width']:.3f}")

    df_res = pd.DataFrame(results)
    df_res.to_csv(results_dir / "conformal_coverage_metrics.csv", index=False)
    print(f"\n[SUCCESS] Saved Conformal Prediction metrics to: {results_dir / 'conformal_coverage_metrics.csv'}")


if __name__ == "__main__":
    main()
