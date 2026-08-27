"""Train Deep Neural Network + GBDT and build the top-tier ensemble."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.deep_net import DeepOperationsNet
from src.evaluation import compute_all_metrics
from src.features import FeaturePipeline


def train_pytorch_deep_net(
    X_tr: np.ndarray,
    s_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    s_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple[DeepOperationsNet, np.ndarray, dict]:
    """Train DeepOperationsNet with L1 loss and cosine annealing."""
    # Standardize continuous features
    mean = np.mean(X_tr, axis=0, keepdims=True)
    std = np.std(X_tr, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    X_tr_norm = (X_tr - mean) / std
    X_val_norm = (X_val - mean) / std

    train_ds = TensorDataset(
        torch.tensor(X_tr_norm, dtype=torch.float32),
        torch.tensor(s_tr, dtype=torch.long),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(-1),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_norm, dtype=torch.float32),
        torch.tensor(s_val, dtype=torch.long),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DeepOperationsNet(
        num_continuous_features=X_tr.shape[1],
        num_series=96,
        embedding_dim=32,
        hidden_dim=256,
        num_blocks=4,
        dropout=0.1,
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_wape = float("inf")
    best_val_preds = None
    best_state_dict = None

    print(f"\n--- Training PyTorch DeepOperationsNet ({epochs} epochs) ---")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x_b, s_b, y_b in train_loader:
            x_b, s_b, y_b = x_b.to(device), s_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(x_b, s_b)
            loss = criterion(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for x_b, s_b, _ in val_loader:
                x_b, s_b = x_b.to(device), s_b.to(device)
                pred = model(x_b, s_b)
                val_preds_list.append(pred.cpu().numpy().flatten())

        val_preds = np.concatenate(val_preds_list)
        metrics = compute_all_metrics(y_val, val_preds)
        val_wape = metrics["WAPE"]
        elapsed = time.time() - t0

        if epoch % 5 == 0 or val_wape < best_val_wape:
            print(
                f"  Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
                f"Train Loss: {total_loss/n_batches:.4f} | "
                f"Val WAPE: {val_wape:.4f} | "
                f"Val MAE: {metrics['MAE']:.3f} | "
                f"Val RMSE: {metrics['RMSE']:.3f}",
                flush=True,
            )

        if val_wape < best_val_wape:
            best_val_wape = val_wape
            best_val_preds = val_preds
            best_state_dict = model.state_dict()

    model.load_state_dict(best_state_dict)
    return model, best_val_preds, {"mean": mean, "std": std}


def train_lightgbm(
    X_tr: np.ndarray,
    s_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    s_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[lgb.LGBMRegressor, np.ndarray]:
    """Train LightGBM with MAE/L1 objective."""
    print("\n--- Training LightGBM GBDT Regressor ---")
    # Combine continuous and series id
    X_tr_full = np.column_stack([X_tr, s_tr])
    X_val_full = np.column_stack([X_val, s_val])

    params = {
        "objective": "regression_l1",
        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbose": -1,
        "n_estimators": 1000,
        "random_state": 42,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr_full, y_tr)

    val_preds = model.predict(X_val_full)
    metrics = compute_all_metrics(y_val, val_preds)
    print(
        f"  LightGBM Val WAPE: {metrics['WAPE']:.4f} | "
        f"Val MAE: {metrics['MAE']:.3f} | "
        f"Val RMSE: {metrics['RMSE']:.3f}"
    )
    return model, val_preds


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device.upper()}")

    data_dir = PROJECT_ROOT / "data"
    train_df = pd.read_csv(data_dir / "train.csv").sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    val_input_df = pd.read_csv(data_dir / "validation_input.csv").sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    val_index_df = pd.read_csv(data_dir / "forecast_index_validation.csv")

    # 1. Local Validation Split (last 336 hours)
    val_horizon = 336
    local_train_parts = []
    local_val_parts = []
    for _, g in train_df.groupby("series_id", sort=False):
        local_train_parts.append(g.iloc[:-val_horizon])
        local_val_parts.append(g.iloc[-val_horizon:])
    local_train_df = pd.concat(local_train_parts).reset_index(drop=True)
    local_val_df = pd.concat(local_val_parts).reset_index(drop=True)

    # 2. Fit Feature Pipeline
    pipeline = FeaturePipeline().fit(local_train_df)

    X_tr, s_tr, y_tr = pipeline.get_feature_matrix(local_train_df)
    X_val, s_val, y_val = pipeline.get_feature_matrix(local_val_df)

    print(f"Feature matrix shape: {X_tr.shape} ({X_tr.shape[1]} continuous features + series embedding)")

    # 3. Train PyTorch Deep Network
    dl_model, dl_val_preds, norm_stats = train_pytorch_deep_net(
        X_tr, s_tr, y_tr, X_val, s_val, y_val, epochs=30, batch_size=256, lr=1e-3, device=device
    )

    # 4. Train LightGBM
    gbm_model, gbm_val_preds = train_lightgbm(X_tr, s_tr, y_tr, X_val, s_val, y_val)

    # 5. Evaluate Individual & Ensemble on Validation Set
    print("\n" + "=" * 70)
    print("VALIDATION BENCHMARK COMPARISON")
    print("=" * 70)

    dl_metrics = compute_all_metrics(y_val, dl_val_preds)
    gbm_metrics = compute_all_metrics(y_val, gbm_val_preds)

    # Search optimal blend weight
    best_w = 0.5
    best_ens_wape = float("inf")
    best_ens_preds = None

    for w in np.linspace(0.0, 1.0, 21):
        blend = w * dl_val_preds + (1.0 - w) * gbm_val_preds
        m = compute_all_metrics(y_val, blend)
        if m["WAPE"] < best_ens_wape:
            best_ens_wape = m["WAPE"]
            best_w = w
            best_ens_preds = blend

    ens_metrics = compute_all_metrics(y_val, best_ens_preds)

    summary_df = pd.DataFrame([
        {"Model": "PyTorch DeepOperationsNet", **dl_metrics},
        {"Model": "LightGBM GBDT", **gbm_metrics},
        {"Model": f"Ensemble (w_dl={best_w:.2f}, w_gbm={1-best_w:.2f})", **ens_metrics},
    ])[["Model", "WAPE", "MAE", "MSE", "RMSE", "MAPE(%)", "sMAPE(%)"]]

    print(summary_df.to_string(index=False))

    # 6. Save PyTorch Checkpoint for submission
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / "deep_net_best.pt"
    torch.save({
        "state_dict": dl_model.state_dict(),
        "norm_stats": norm_stats,
        "feature_names": pipeline.feature_names_,
        "medians": pipeline.medians_,
        "series2idx": pipeline.series2idx_,
        "ensemble_weights": {"w_dl": best_w, "w_gbm": 1.0 - best_w},
    }, checkpoint_path)
    print(f"\nSaved best PyTorch checkpoint to: {checkpoint_path}")

    # 7. Generate Public Validation Submissions
    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    # Transform public validation_input.csv
    X_pub, s_pub, _ = pipeline.get_feature_matrix(val_input_df)
    X_pub_norm = (X_pub - norm_stats["mean"]) / norm_stats["std"]

    # DL Predictions
    dl_model.eval()
    with torch.no_grad():
        dl_pub_preds = dl_model(
            torch.tensor(X_pub_norm, dtype=torch.float32).to(device),
            torch.tensor(s_pub, dtype=torch.long).to(device),
        ).cpu().numpy().flatten()

    # GBDT Predictions
    X_pub_full = np.column_stack([X_pub, s_pub])
    gbm_pub_preds = gbm_model.predict(X_pub_full)

    # Ensemble Predictions
    ens_pub_preds = best_w * dl_pub_preds + (1.0 - best_w) * gbm_pub_preds

    for name, p_arr in [
        ("deep_net_v2", dl_pub_preds),
        ("gbm_v1", gbm_pub_preds),
        ("ensemble_v1", ens_pub_preds),
    ]:
        sub = val_input_df[["series_id", "timestamp"]].copy()
        sub["prediction"] = np.clip(p_arr, a_min=0.0, a_max=None)

        # Merge strictly on forecast_index_validation
        merged_sub = val_index_df.merge(sub, on=["series_id", "timestamp"], how="left")
        assert len(merged_sub) == len(val_index_df)
        assert not merged_sub["prediction"].isna().any()

        out_path = submissions_dir / f"{name}.csv"
        merged_sub.to_csv(out_path, index=False)
        print(f"Generated verified submission: {out_path} ({len(merged_sub)} rows)")

    print("\n" + "=" * 70)
    print("Top-Tier Training & Ensemble Generation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
