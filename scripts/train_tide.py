"""Train TiDE (Time-series Dense Encoder) model and export leaderboard predictions."""

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
from src.evaluation import create_local_validation_split
from src.tide import TiDE
from src.trainer import Trainer


def generate_tide_submission(
    model: TiDE,
    preprocessor: TimeSeriesPreprocessor,
    train_df: pd.DataFrame,
    val_input_df: pd.DataFrame,
    val_index_df: pd.DataFrame,
    lookback: int = 168,
    horizon: int = 336,
    device: str = "cpu",
    output_file: Path | str = "submissions/tide.csv",
) -> None:
    """Generate official public validation submission CSV with TiDE."""
    model.eval()
    model.to(device)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = []

    with torch.no_grad():
        for series_id, train_group in train_df.groupby("series_id", sort=False):
            train_group = train_group.sort_values("timestamp").reset_index(drop=True)
            val_group = val_input_df[val_input_df["series_id"] == series_id].sort_values("timestamp").reset_index(drop=True)

            past_target = train_group["target"].iloc[-lookback:].to_numpy(dtype=float)
            past_covs = preprocessor.transform_covariates(train_group.iloc[-lookback:])
            future_covs = preprocessor.transform_covariates(val_group.iloc[:horizon])

            past_seasonal = preprocessor.get_seasonal_values(train_group.iloc[-lookback:])
            future_seasonal = preprocessor.get_seasonal_values(val_group.iloc[:horizon])

            series_idx = preprocessor.series2idx[series_id]

            # Convert to tensors
            past_t_t = torch.tensor(past_target, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            past_c_t = torch.tensor(past_covs, dtype=torch.float32).unsqueeze(0).to(device)
            fut_c_t = torch.tensor(future_covs, dtype=torch.float32).unsqueeze(0).to(device)
            s_idx_t = torch.tensor([series_idx], dtype=torch.long).to(device)
            past_s_t = torch.tensor(past_seasonal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            fut_s_t = torch.tensor(future_seasonal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

            pred = model(
                past_target=past_t_t,
                past_covariates=past_c_t,
                future_covariates=fut_c_t,
                series_idx=s_idx_t,
                future_seasonal=fut_s_t,
                past_seasonal=past_s_t,
            )
            pred_arr = pred.cpu().numpy().flatten()

            ser_idx_part = val_index_df[val_index_df["series_id"] == series_id].sort_values("timestamp")
            for (_, row), p_val in zip(ser_idx_part.iterrows(), pred_arr):
                predictions.append({
                    "series_id": series_id,
                    "timestamp": row["timestamp"],
                    "prediction": float(p_val),
                })

    sub_df = pd.DataFrame(predictions)
    sub_df.to_csv(output_path, index=False)
    print(f"Generated TiDE submission: {output_path} ({len(sub_df)} rows)")


def main() -> None:
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device.upper()}")

    data_dir = PROJECT_ROOT / "data"
    train_path = data_dir / "train.csv"
    val_input_path = data_dir / "validation_input.csv"
    val_index_path = data_dir / "forecast_index_validation.csv"

    train_df = pd.read_csv(train_path)
    val_input_df = pd.read_csv(val_input_path)
    val_index_df = pd.read_csv(val_index_path)

    # 1. Local Validation Split
    local_train, local_val_truth, _ = create_local_validation_split(train_df, val_horizon=336)

    # Fit preprocessor
    preprocessor = TimeSeriesPreprocessor().fit(local_train)

    # Sliding window datasets
    train_dataset = WindowDataset(
        local_train,
        preprocessor,
        lookback=168,
        horizon=336,
        step=24,
        is_train=True,
        use_covariate_dropout=True,
        covariate_dropout_prob=0.1,
    )
    val_dataset = WindowDataset(
        train_df,
        preprocessor,
        lookback=168,
        horizon=336,
        is_train=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    print(f"TiDE training samples: {len(train_dataset)}, val samples: {len(val_dataset)}")

    # 2. TiDE Architecture
    model = TiDE(
        lookback=168,
        horizon=336,
        num_dynamic_covariates=len(preprocessor.dynamic_cols),
        num_series=len(preprocessor.series2idx),
        static_embedding_dim=16,
        dynamic_projection_dim=16,
        hidden_dim=256,
        decoder_output_dim=16,
        temporal_hidden_dim=32,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1,
        use_seasonal_residual=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-5)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        loss_type="l1",
        device=device,
        checkpoint_dir=PROJECT_ROOT / "checkpoints",
        model_name="tide",
    )

    best_metrics = trainer.fit(train_loader, val_loader, epochs=35, early_stopping_patience=10)
    print("\nBest TiDE Validation Metrics:", best_metrics)

    # 3. Generate Submission on full training set
    best_checkpoint = torch.load(trainer.best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["state_dict"])

    full_preprocessor = TimeSeriesPreprocessor().fit(train_df)
    generate_tide_submission(
        model=model,
        preprocessor=full_preprocessor,
        train_df=train_df,
        val_input_df=val_input_df,
        val_index_df=val_index_df,
        lookback=168,
        horizon=336,
        device=device,
        output_file=PROJECT_ROOT / "submissions" / "tide.csv",
    )


if __name__ == "__main__":
    main()
