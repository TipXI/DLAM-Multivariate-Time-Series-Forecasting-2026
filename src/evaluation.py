"""Evaluation metrics and validation split helpers for DLAM Time Series Forecasting 2026."""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def wape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Weighted Absolute Percentage Error (WAPE). Primary course metric."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    denominator = np.sum(np.abs(y_t))
    if denominator == 0:
        return 0.0 if np.sum(np.abs(y_t - y_p)) == 0 else float("inf")
    return float(np.sum(np.abs(y_t - y_p)) / denominator)


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean Absolute Error."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_t - y_p)))


def mse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean Squared Error."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_t - y_p) ** 2))


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (in %)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs((y_t - y_p) / (np.abs(y_t) + eps))) * 100.0)


def smape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series, eps: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error (in %)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    numerator = 2.0 * np.abs(y_p - y_t)
    denominator = np.abs(y_t) + np.abs(y_p) + eps
    return float(np.mean(numerator / denominator) * 100.0)


def compute_all_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> Dict[str, float]:
    """Compute dictionary of all course leaderboard metrics."""
    return {
        "WAPE": wape(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE(%)": mape(y_true, y_pred),
        "sMAPE(%)": smape(y_true, y_pred),
    }


def create_local_validation_split(
    train_df: pd.DataFrame,
    val_horizon: int = 336,
    series_col: str = "series_id",
    time_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split train.csv into a local train history and local validation ground truth.

    Returns:
        (local_train_df, local_val_truth_df, local_forecast_index_df)
    """
    train_df = train_df.sort_values([series_col, time_col]).reset_index(drop=True)
    local_train_parts = []
    local_val_parts = []

    for series_id, group in train_df.groupby(series_col, sort=False):
        if len(group) <= val_horizon:
            raise ValueError(f"Series {series_id} has {len(group)} rows, <= val_horizon {val_horizon}")
        local_train_parts.append(group.iloc[:-val_horizon])
        local_val_parts.append(group.iloc[-val_horizon:])

    local_train_df = pd.concat(local_train_parts, ignore_index=True)
    local_val_truth_df = pd.concat(local_val_parts, ignore_index=True)
    local_forecast_index_df = local_val_truth_df[[series_col, time_col]].copy()

    return local_train_df, local_val_truth_df, local_forecast_index_df
