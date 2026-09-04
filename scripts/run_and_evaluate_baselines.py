"""Run baseline forecasting models, evaluate locally, and produce leaderboard submissions."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.evaluation import compute_all_metrics, create_local_validation_split
from student.baseline.baselines import make_all_baselines


def verify_submission(submission_df: pd.DataFrame, forecast_index_df: pd.DataFrame, name: str) -> None:
    """Verify that the generated submission strictly matches the required format."""
    expected_cols = ["series_id", "timestamp", "prediction"]
    assert list(submission_df.columns) == expected_cols, (
        f"[{name}] Invalid columns: {submission_df.columns}, expected {expected_cols}"
    )
    assert len(submission_df) == len(forecast_index_df), (
        f"[{name}] Row count mismatch: {len(submission_df)} vs expected {len(forecast_index_df)}"
    )
    assert not submission_df["prediction"].isna().any(), f"[{name}] Contains NaN values!"
    assert (submission_df["series_id"] == forecast_index_df["series_id"]).all(), (
        f"[{name}] Series order does not match forecast index!"
    )
    assert (submission_df["timestamp"] == forecast_index_df["timestamp"]).all(), (
        f"[{name}] Timestamp order does not match forecast index!"
    )
    print(f"  [OK] {name}: Verified ({len(submission_df)} rows, no NaNs, valid schema)")


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    train_path = data_dir / "train.csv"
    val_index_path = data_dir / "forecast_index_validation.csv"
    submissions_dir = PROJECT_ROOT / "submissions" / "baselines"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists() or not val_index_path.exists():
        raise FileNotFoundError(f"Missing data in {data_dir}. Run `python scripts/download_data.py` first.")

    print("=" * 70)
    print("1. Loading Training Data...")
    train_df = pd.read_csv(train_path)
    val_index_df = pd.read_csv(val_index_path)
    print(f"Loaded train.csv with {len(train_df)} rows and {train_df['series_id'].nunique()} series.")

    print("\n" + "=" * 70)
    print("2. Local Validation Benchmarking (Backtest on last 336 hours of train.csv)")
    print("=" * 70)
    local_train, local_val_truth, local_index = create_local_validation_split(train_df, val_horizon=336)
    print(f"Local train size: {len(local_train)}, Local val size: {len(local_val_truth)}")

    local_baselines = make_all_baselines(local_train, local_index)
    results = []

    for name, pred_df in local_baselines.items():
        # Merge predictions with truth
        merged = local_val_truth[["series_id", "timestamp", "target"]].merge(
            pred_df[["series_id", "timestamp", "prediction"]],
            on=["series_id", "timestamp"],
            how="left",
        )
        metrics = compute_all_metrics(merged["target"], merged["prediction"])
        metrics["Model"] = name
        results.append(metrics)

    res_df = pd.DataFrame(results)[["Model", "WAPE", "MAE", "MSE", "RMSE", "MAPE(%)", "sMAPE(%)"]]
    # Sort by primary metric WAPE (lower is better)
    res_df = res_df.sort_values("WAPE").reset_index(drop=True)
    print("\n--- Local Validation Benchmark Results ---")
    print(res_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("3. Generating Public Validation Leaderboard Submissions...")
    print("=" * 70)
    official_baselines = make_all_baselines(train_df, val_index_df)

    for name, pred_df in official_baselines.items():
        out_csv = submissions_dir / f"{name}.csv"
        pred_df.to_csv(out_csv, index=False)
        verify_submission(pred_df, val_index_df, name)
        print(f"  -> Saved submission: {out_csv.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 80)
    print("Baseline Evaluation Complete: All baselines evaluated and validation submissions generated!")
    print("=" * 80)


if __name__ == "__main__":
    main()
