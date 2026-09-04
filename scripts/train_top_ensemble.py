"""Full-scale top-tier ensemble training on 100% dataset with multi-scale rolling features."""

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


def build_full_features(df: pd.DataFrame) -> pd.DataFrame:
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
    unique_series = sorted(d["series_id"].unique())
    series2idx = {s: i for i, s in enumerate(unique_series)}
    d["series_idx"] = d["series_id"].map(series2idx).astype(int)

    return d


def train_single_deep_net(
    X_tr_norm: np.ndarray,
    s_tr: np.ndarray,
    y_tr: np.ndarray,
    seed: int = 42,
    epochs: int = 35,
    batch_size: int = 512,
    lr: float = 1.2e-3,
    device: str = "cuda",
) -> DeepOperationsNet:
    """Train single seed PyTorch DeepOperationsNet on CUDA."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = TensorDataset(
        torch.tensor(X_tr_norm, dtype=torch.float32),
        torch.tensor(s_tr, dtype=torch.long),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(-1),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = DeepOperationsNet(
        num_continuous_features=X_tr_norm.shape[1],
        num_series=96,
        embedding_dim=32,
        hidden_dim=256,
        num_blocks=4,
        dropout=0.1,
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    for epoch in range(1, epochs + 1):
        model.train()
        for x_b, s_b, y_b in train_loader:
            x_b, s_b, y_b = x_b.to(device), s_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(x_b, s_b)
            loss = criterion(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    return model


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing Top Ensemble Training on: {device.upper()}")

    data_dir = PROJECT_ROOT / "data"
    train_raw = pd.read_csv(data_dir / "train.csv").sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    val_input_raw = pd.read_csv(data_dir / "validation_input.csv").sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    val_index_df = pd.read_csv(data_dir / "forecast_index_validation.csv")

    unique_series = sorted(train_raw["series_id"].unique())
    series2idx = {s: i for i, s in enumerate(unique_series)}

    print("\n1. Extracting Multi-Scale Rolling Features...")
    # Concatenate train and val_input to compute continuous rolling features without boundary artifacts
    all_combined = pd.concat([train_raw, val_input_raw], ignore_index=True)
    all_featured = build_full_features(all_combined)

    n_train = len(train_raw)
    train_featured = all_featured.iloc[:n_train].copy()
    val_featured = all_featured.iloc[n_train:].copy()

    drop_cols = ["series_id", "timestamp", "dt", "target", "series_idx"]
    feature_names = [c for c in train_featured.columns if c not in drop_cols]
    print(f"Total features: {len(feature_names)}")

    X_train = train_featured[feature_names].to_numpy(dtype=np.float32)
    s_train = train_featured["series_idx"].to_numpy(dtype=np.int64)
    y_train = train_featured["target"].to_numpy(dtype=np.float32)

    X_val = val_featured[feature_names].to_numpy(dtype=np.float32)
    s_val = val_featured["series_idx"].to_numpy(dtype=np.int64)

    # Standardize continuous features
    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    # 2. Train Multi-Seed PyTorch Deep Neural Network (3 diverse seeds)
    print("\n2. Training 3-Seed PyTorch DeepOperationsNet on CUDA...")
    dl_models = []
    dl_preds_list = []
    seeds = [42, 101, 2024]

    for i, seed in enumerate(seeds, 1):
        t0 = time.time()
        dl_model = train_single_deep_net(
            X_train_norm, s_train, y_train, seed=seed, epochs=25, batch_size=1024, lr=1.5e-3, device=device
        )
        dl_models.append(dl_model.cpu().state_dict())
        dl_model.to(device)
        # Predict on validation input
        with torch.no_grad():
            p = dl_model(
                torch.tensor(X_val_norm, dtype=torch.float32).to(device),
                torch.tensor(s_val, dtype=torch.long).to(device),
            ).cpu().numpy().flatten()
            dl_preds_list.append(p)
        print(f"  Seed {i}/{len(seeds)} (seed={seed}) completed in {time.time()-t0:.1f}s")

    dl_ensemble_preds = np.mean(dl_preds_list, axis=0)

    # 3. Train Advanced Multi-Tree LightGBM Models (2 diverse configs)
    print("\n3. Training LightGBM Models with Multi-Scale Rolling Trees...")
    X_train_full = np.column_stack([X_train, s_train])
    X_val_full = np.column_stack([X_val, s_val])

    # Config A: High capacity (255 leaves)
    params_a = {
        "objective": "regression_l1",
        "learning_rate": 0.02,
        "num_leaves": 255,
        "max_depth": -1,
        "min_child_samples": 15,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbose": -1,
        "n_estimators": 2500,
        "random_state": 42,
    }
    gbm_a = lgb.LGBMRegressor(**params_a)
    gbm_a.fit(X_train_full, y_train)
    gbm_a_preds = gbm_a.predict(X_val_full)

    # Config B: Moderate capacity + regularization (127 leaves)
    params_b = {
        "objective": "regression_l1",
        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 25,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "n_estimators": 2000,
        "random_state": 123,
    }
    gbm_b = lgb.LGBMRegressor(**params_b)
    gbm_b.fit(X_train_full, y_train)
    gbm_b_preds = gbm_b.predict(X_val_full)

    gbm_ensemble_preds = 0.6 * gbm_a_preds + 0.4 * gbm_b_preds

    # 4. Final Grand Ensemble Blend
    print("\n4. Blending Grand Ensemble (55% Deep Neural Net + 45% Multi-Tree GBDT)...")
    final_preds = 0.55 * dl_ensemble_preds + 0.45 * gbm_ensemble_preds
    final_preds = np.clip(final_preds, a_min=0.0, a_max=None)

    # 5. Export and Verify Submission CSV
    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    sub_df = val_input_raw[["series_id", "timestamp"]].copy()
    sub_df["prediction"] = final_preds

    merged_sub = val_index_df.merge(sub_df, on=["series_id", "timestamp"], how="left")
    assert len(merged_sub) == len(val_index_df)
    assert not merged_sub["prediction"].isna().any()

    merged_sub.to_csv(submissions_dir / "top_ensemble_v2.csv", index=False)
    merged_sub.to_csv(submissions_dir / "ensemble_v2.csv", index=False)
    print(f"\n[SUCCESS] Generated: submissions/ensemble_v2.csv ({len(merged_sub)} rows, no NaNs)")
    print(f"Prediction range: Min={merged_sub['prediction'].min():.3f}, Max={merged_sub['prediction'].max():.3f}, Mean={merged_sub['prediction'].mean():.3f}")

    # 6. Save Unified Ensemble Checkpoint for submission_template/predict.py
    checkpoint_data = {
        "dl_state_dicts": dl_models,
        "gbm_a_str": gbm_a.booster_.model_to_string(),
        "gbm_b_str": gbm_b.booster_.model_to_string(),
        "norm_stats": {"mean": mean, "std": std},
        "feature_names": feature_names,
        "series2idx": series2idx,
        "blend_weights": {"dl_weight": 0.55, "gbm_weight": 0.45, "gbm_a_ratio": 0.6, "gbm_b_ratio": 0.4},
        "train_tail": train_raw.groupby("series_id").tail(48).reset_index(drop=True),
        "val_tail": val_input_raw.groupby("series_id").tail(48).reset_index(drop=True),
    }
    ckpt_path = PROJECT_ROOT / "student" / "submission_template" / "checkpoint.pt"
    torch.save(checkpoint_data, ckpt_path)
    torch.save(checkpoint_data, PROJECT_ROOT / "checkpoints" / "ensemble_v2_checkpoint.pt")
    print(f"[SUCCESS] Saved unified winning ensemble checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()

