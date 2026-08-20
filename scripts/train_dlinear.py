"""Train DLinear model and export leaderboard predictions."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import TimeSeriesPreprocessor, WindowDataset
from src.dlinear import DLinear
from src.evaluation import create_local_validation_split
from src.trainer import Trainer


def generate_submission(
    model: DLinear,
    preprocessor: TimeSeriesPreprocessor,
    train_df: pd.DataFrame,
    val_index_df: pd.DataFrame,
    lookback: int = 168,
    horizon: int = 336,
    device: str = "cpu",
    output_file: Path | str = "submissions/dlinear.csv",
) -> None:
    """Generate official public validation submission CSV."""
    model.eval()
    model.to(device)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = []
    with torch.no_grad():
        for series_id, group in train_df.groupby("series_id", sort=False):
            group = group.sort_values("timestamp").reset_index(drop=True)
            past_targets = group["target"].iloc[-lookback:].to_numpy(dtype=float)

            past_t_tensor = torch.tensor(past_targets, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            pred = model(past_t_tensor)
            pred_arr = pred.cpu().numpy().flatten()

            ser_idx_part = val_index_df[val_index_df["series_id"] == series_id].sort_values("timestamp")
            if len(ser_idx_part) != horizon:
                raise ValueError(f"Expected {horizon} rows for {series_id}, got {len(ser_idx_part)}")

            for (_, row), p_val in zip(ser_idx_part.iterrows(), pred_arr):
                predictions.append({
                    "series_id": series_id,
                    "timestamp": row["timestamp"],
                    "prediction": float(p_val),
                })

    sub_df = pd.DataFrame(predictions)
    sub_df.to_csv(output_path, index=False)
    print(f"Generated submission: {output_path} ({len(sub_df)} rows)")


def main() -> None:
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device.upper()}")

    data_dir = PROJECT_ROOT / "data"
    train_path = data_dir / "train.csv"
    val_index_path = data_dir / "forecast_index_validation.csv"

    train_df = pd.read_csv(train_path)
    val_index_df = pd.read_csv(val_index_path)

    # 1. Local Validation Split
    local_train, local_val_truth, _ = create_local_validation_split(train_df, val_horizon=336)

    # Fit preprocessor
    preprocessor = TimeSeriesPreprocessor().fit(local_train)

    # Create datasets
    train_dataset = WindowDataset(local_train, preprocessor, lookback=168, horizon=336, step=24, is_train=True)
    val_dataset = WindowDataset(train_df, preprocessor, lookback=168, horizon=336, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    print(f"DLinear training samples: {len(train_dataset)}, val samples: {len(val_dataset)}")

    # 2. Model, Optimizer, Loss
    model = DLinear(lookback=168, horizon=336, kernel_size=25, use_revin=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        loss_type="l1",
        device=device,
        checkpoint_dir=PROJECT_ROOT / "checkpoints",
        model_name="dlinear",
    )

    best_metrics = trainer.fit(train_loader, val_loader, epochs=25, early_stopping_patience=8)
    print("\nBest DLinear Validation Metrics:", best_metrics)

    # 3. Generate Submission with best model
    best_checkpoint = torch.load(trainer.best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["state_dict"])

    # Fit preprocessor on full data for submission
    full_preprocessor = TimeSeriesPreprocessor().fit(train_df)
    generate_submission(
        model=model,
        preprocessor=full_preprocessor,
        train_df=train_df,
        val_index_df=val_index_df,
        lookback=168,
        horizon=336,
        device=device,
        output_file=PROJECT_ROOT / "submissions" / "dlinear.csv",
    )


if __name__ == "__main__":
    main()
