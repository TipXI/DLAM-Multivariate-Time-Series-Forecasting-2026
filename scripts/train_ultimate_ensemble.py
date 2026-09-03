"""Ultimate Grand Ensemble: 5-Seed PyTorch CUDA DeepNet + LightGBM + XGBoost + CatBoost + Trend Anchor.

Designed for maximum performance and WAPE minimization on the full 100% dataset.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

from src.deep_net import DeepOperationsNet


def build_ultimate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract comprehensive multi-scale rolling temporal features, physics ratios, and horizon features."""
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
            d[c] = d.groupby("series_id")[c].transform(lambda x: x.ffill().bfill().fillna(x.median()))

    # 3. Comprehensive multi-scale rolling windows (3h to 168h weekly)
    windows = [3, 6, 12, 24, 48, 72, 168]
    for win in windows:
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

    # 4. Domain interaction and physics terms
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
    d["workload_per_staff"] = d["workload_intensity"] / (d["staffing_forecast"].abs() + 1e-5)
    d["demand_diff_workload"] = d["demand_forecast"] - d["workload_intensity"]

    # 5. Integer series index mapping
    unique_series = sorted(d["series_id"].unique())
    s2i = {s: i for i, s in enumerate(unique_series)}
    d["series_idx"] = d["series_id"].map(s2i).astype(int)

    return d


def train_single_deep_net(
    X_tr_norm: np.ndarray,
    s_tr: np.ndarray,
    y_tr: np.ndarray,
    seed: int = 42,
    epochs: int = 45,
    batch_size: int = 512,
    lr: float = 1.2e-3,
    device: str = "cuda",
) -> DeepOperationsNet:
    """Train high-capacity DeepOperationsNet on CUDA."""
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
        embedding_dim=48,
        hidden_dim=384,
        num_blocks=5,
        dropout=0.1,
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-5
    )

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
    print("=" * 80)
    print(f"ULTIMATE GRAND ENSEMBLE TRAINING (Device: {device.upper()})")
    print("=" * 80)

    data_dir = PROJECT_ROOT / "data"
    train_raw = pd.read_csv(data_dir / "train.csv").sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    val_input_raw = pd.read_csv(data_dir / "validation_input.csv").sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    val_index_df = pd.read_csv(data_dir / "forecast_index_validation.csv")

    print("\n[Step 1/6] Extracting Multi-Scale Rolling Features (78 features)...")
    all_combined = pd.concat([train_raw, val_input_raw], ignore_index=True)
    all_featured = build_ultimate_features(all_combined)

    n_train = len(train_raw)
    train_f = all_featured.iloc[:n_train].copy()
    val_f = all_featured.iloc[n_train:].copy()

    drop_cols = ["series_id", "timestamp", "dt", "target", "series_idx"]
    feature_names = [c for c in train_f.columns if c not in drop_cols]
    print(f"Total features: {len(feature_names)}")

    X_train = train_f[feature_names].to_numpy(dtype=np.float32)
    s_train = train_f["series_idx"].to_numpy(dtype=np.int64)
    y_train = train_f["target"].to_numpy(dtype=np.float32)

    X_val = val_f[feature_names].to_numpy(dtype=np.float32)
    s_val = val_f["series_idx"].to_numpy(dtype=np.int64)

    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    # [Step 2/6] Train 5-Seed PyTorch DeepOperationsNet on CUDA
    print("\n[Step 2/6] Training 5-Seed PyTorch DeepOperationsNet on CUDA (45 epochs each)...")
    dl_preds_list = []
    seeds = [42, 101, 777, 1337, 2026]

    for i, seed in enumerate(seeds, 1):
        t0 = time.time()
        dl_model = train_single_deep_net(
            X_train_norm, s_train, y_train, seed=seed, epochs=45, batch_size=512, lr=1.2e-3, device=device
        )
        with torch.no_grad():
            p = dl_model(
                torch.tensor(X_val_norm, dtype=torch.float32).to(device),
                torch.tensor(s_val, dtype=torch.long).to(device),
            ).cpu().numpy().flatten()
            dl_preds_list.append(p)
        print(f"  PyTorch Seed {i}/{len(seeds)} (seed={seed}) completed in {time.time()-t0:.1f}s")

    dl_preds = np.mean(dl_preds_list, axis=0)

    # [Step 3/6] Train LightGBM Multi-Configuration Regressors
    print("\n[Step 3/6] Training LightGBM Multi-Scale Regressors (2 configurations)...")
    X_train_full = np.column_stack([X_train, s_train])
    X_val_full = np.column_stack([X_val, s_val])

    # Model A: Deep trees (383 leaves, 3000 trees)
    params_lgb1 = {
        "objective": "regression_l1",
        "learning_rate": 0.018,
        "num_leaves": 383,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbose": -1,
        "n_estimators": 3000,
        "random_state": 42,
    }
    lgb1 = lgb.LGBMRegressor(**params_lgb1)
    lgb1.fit(X_train_full, y_train)
    lgb1_preds = lgb1.predict(X_val_full)

    # Model B: Regularized trees (191 leaves, 2500 trees)
    params_lgb2 = {
        "objective": "regression_l1",
        "learning_rate": 0.025,
        "num_leaves": 191,
        "max_depth": -1,
        "min_child_samples": 35,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "n_estimators": 2500,
        "random_state": 123,
    }
    lgb2 = lgb.LGBMRegressor(**params_lgb2)
    lgb2.fit(X_train_full, y_train)
    lgb2_preds = lgb2.predict(X_val_full)

    lgb_preds = 0.6 * lgb1_preds + 0.4 * lgb2_preds

    # [Step 4/6] Train XGBoost Regressor with L1 objective
    print("\n[Step 4/6] Training XGBoost Regressor (reg:absoluteerror, hist)...")
    t0 = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=9,
        subsample=0.85,
        colsample_bytree=0.8,
        objective="reg:absoluteerror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train_full, y_train)
    xgb_preds = xgb_model.predict(X_val_full)
    print(f"  XGBoost completed in {time.time()-t0:.1f}s")

    # [Step 5/6] Train CatBoost Regressor with categorical series ID
    print("\n[Step 5/6] Training CatBoost Regressor with Categorical Embeddings...")
    t0 = time.time()
    cb_cols = feature_names + ["series_id"]
    cb = CatBoostRegressor(
        loss_function="MAE",
        iterations=2500,
        learning_rate=0.035,
        depth=8,
        cat_features=["series_id"],
        verbose=0,
        random_seed=42,
    )
    cb.fit(train_f[cb_cols], y_train)
    cb_preds = cb.predict(val_f[cb_cols])
    print(f"  CatBoost completed in {time.time()-t0:.1f}s")

    # Per-Series Ridge Trend Anchor
    print("\nFitting Per-Series Trend Extrapolation Prior...")
    ridge_preds = []
    trend_cols = ["trend", "workload_intensity", "queue_pressure_forecast", "network_pressure_forecast"]
    for s, g_tr in train_f.groupby("series_id"):
        g_val = val_f[val_f["series_id"] == s]
        r = Ridge(alpha=30.0)
        r.fit(g_tr[trend_cols], g_tr["target"])
        ridge_preds.extend(r.predict(g_val[trend_cols]))
    ridge_preds = np.array(ridge_preds)

    # [Step 6/6] Grand Consensus Blending
    print("\n[Step 6/6] Blending Grand Consensus (50% PyTorch + 25% LightGBM + 12% XGBoost + 8% CatBoost + 5% Trend Anchor)...")
    final_preds = (
        0.50 * dl_preds
        + 0.25 * lgb_preds
        + 0.12 * xgb_preds
        + 0.08 * cb_preds
        + 0.05 * ridge_preds
    )
    final_preds = np.clip(final_preds, a_min=0.0, a_max=None)

    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    sub_df = val_input_raw[["series_id", "timestamp"]].copy()
    sub_df["prediction"] = final_preds

    merged_sub = val_index_df.merge(sub_df, on=["series_id", "timestamp"], how="left")
    assert len(merged_sub) == len(val_index_df)
    assert not merged_sub["prediction"].isna().any()

    output_path = submissions_dir / "ultimate_v4.csv"
    merged_sub.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print(f"[SUCCESS] Generated Ultimate Submission: {output_path}")
    print(f"Total Rows: {len(merged_sub)}")
    print(f"Summary: Min={merged_sub['prediction'].min():.3f}, Max={merged_sub['prediction'].max():.3f}, Mean={merged_sub['prediction'].mean():.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
