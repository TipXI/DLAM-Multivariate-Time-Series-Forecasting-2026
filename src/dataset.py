"""Dataset and preprocessing module for DLAM Time Series Forecasting 2026."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DYNAMIC_COVARIATES = [
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


class TimeSeriesPreprocessor:
    """Preprocesses series data, builds series mappings, imputes missing values, and calculates seasonal profiles."""

    def __init__(self, dynamic_cols: Optional[List[str]] = None) -> None:
        self.dynamic_cols = dynamic_cols or DYNAMIC_COVARIATES
        self.series2idx: Dict[str, int] = {}
        self.idx2series: Dict[int, str] = {}
        self.covariate_means: Optional[np.ndarray] = None
        self.covariate_stds: Optional[np.ndarray] = None
        self.seasonal_table: Optional[pd.DataFrame] = None
        self.series_means: Optional[pd.DataFrame] = None
        self.global_target_mean: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> TimeSeriesPreprocessor:
        unique_series = sorted(train_df["series_id"].unique())
        self.series2idx = {s: i for i, s in enumerate(unique_series)}
        self.idx2series = {i: s for s, i in self.series2idx.items()}

        df = train_df.copy()
        ts = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
        df["_hour"] = ts.dt.hour
        df["_dayofweek"] = ts.dt.dayofweek

        self.global_target_mean = float(df["target"].mean())
        self.seasonal_table = (
            df.groupby(["series_id", "_dayofweek", "_hour"], as_index=False)["target"]
            .mean()
            .rename(columns={"target": "seasonal_target"})
        )
        self.series_means = (
            df.groupby("series_id", as_index=False)["target"]
            .mean()
            .rename(columns={"target": "series_mean"})
        )

        cov_matrix = df[self.dynamic_cols].fillna(0.0).to_numpy(dtype=np.float32)
        self.covariate_means = np.nanmean(cov_matrix, axis=0, keepdims=True)
        self.covariate_stds = np.nanstd(cov_matrix, axis=0, keepdims=True)
        self.covariate_stds[self.covariate_stds < 1e-6] = 1.0

        return self

    def get_seasonal_values(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorized seasonal profile calculation for the entire dataframe."""
        if self.seasonal_table is None or self.series_means is None:
            raise RuntimeError("Preprocessor must be fit before computing seasonal values.")
        temp = df.copy()
        ts = pd.to_datetime(temp["timestamp"], format="%Y-%m-%d %H:%M:%S")
        temp["_hour"] = ts.dt.hour
        temp["_dayofweek"] = ts.dt.dayofweek
        temp["_orig_idx"] = np.arange(len(temp))

        merged = temp.merge(
            self.seasonal_table,
            on=["series_id", "_dayofweek", "_hour"],
            how="left",
        )
        merged = merged.merge(self.series_means, on="series_id", how="left")
        merged["seasonal_target"] = (
            merged["seasonal_target"]
            .fillna(merged["series_mean"])
            .fillna(self.global_target_mean)
        )
        merged = merged.sort_values("_orig_idx")
        return merged["seasonal_target"].to_numpy(dtype=np.float32)

    def transform_covariates(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorized covariate normalization and missingness imputation."""
        covs = df[self.dynamic_cols].copy()
        cov_arr = covs.fillna(0.0).to_numpy(dtype=np.float32)
        norm_covs = (cov_arr - self.covariate_means) / self.covariate_stds
        return norm_covs


class WindowDataset(Dataset):
    """
    High-performance sliding window dataset.
    Precomputes vectorized features across all series in sub-second time.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        preprocessor: TimeSeriesPreprocessor,
        lookback: int = 168,
        horizon: int = 336,
        step: int = 24,
        is_train: bool = True,
        use_covariate_dropout: bool = False,
        covariate_dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.is_train = is_train
        self.use_covariate_dropout = use_covariate_dropout
        self.covariate_dropout_prob = covariate_dropout_prob

        # Vectorize transformations for the whole DataFrame at once
        df_sorted = df.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
        all_covs = preprocessor.transform_covariates(df_sorted)
        all_seasonals = preprocessor.get_seasonal_values(df_sorted)
        has_target = "target" in df_sorted.columns
        all_targets = df_sorted["target"].to_numpy(dtype=np.float32) if has_target else None

        self.samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, Optional[np.ndarray], np.ndarray, np.ndarray]] = []

        # Split vectorized arrays by series
        start_idx = 0
        for series_id, count in df_sorted.groupby("series_id", sort=False).size().items():
            end_idx = start_idx + count
            series_idx = preprocessor.series2idx[series_id]

            covs = all_covs[start_idx:end_idx]
            seasonals = all_seasonals[start_idx:end_idx]
            targets = all_targets[start_idx:end_idx] if all_targets is not None else None

            n_rows = count
            if is_train:
                for i in range(0, n_rows - lookback - horizon + 1, step):
                    past_t = targets[i : i + lookback]
                    fut_t = targets[i + lookback : i + lookback + horizon]
                    past_c = covs[i : i + lookback]
                    fut_c = covs[i + lookback : i + lookback + horizon]
                    past_s = seasonals[i : i + lookback]
                    fut_s = seasonals[i + lookback : i + lookback + horizon]
                    self.samples.append((past_t, past_c, fut_c, series_idx, fut_t, past_s, fut_s))
            else:
                if n_rows >= lookback + horizon:
                    i = n_rows - lookback - horizon
                    past_t = targets[i : i + lookback] if targets is not None else np.zeros(lookback, dtype=np.float32)
                    fut_t = targets[i + lookback : i + lookback + horizon] if targets is not None else None
                    past_c = covs[i : i + lookback]
                    fut_c = covs[i + lookback : i + lookback + horizon]
                    past_s = seasonals[i : i + lookback]
                    fut_s = seasonals[i + lookback : i + lookback + horizon]
                    self.samples.append((past_t, past_c, fut_c, series_idx, fut_t, past_s, fut_s))

            start_idx = end_idx

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        past_t, past_c, fut_c, series_idx, fut_t, past_s, fut_s = self.samples[idx]

        past_covs = np.copy(past_c)
        fut_covs = np.copy(fut_c)

        if self.is_train and self.use_covariate_dropout and np.random.rand() < self.covariate_dropout_prob:
            mask = np.random.rand(*fut_covs.shape) > 0.15
            fut_covs = fut_covs * mask

        item = {
            "past_target": torch.tensor(past_t, dtype=torch.float32).unsqueeze(-1),
            "past_covariates": torch.tensor(past_covs, dtype=torch.float32),
            "future_covariates": torch.tensor(fut_covs, dtype=torch.float32),
            "series_idx": torch.tensor(series_idx, dtype=torch.long),
            "past_seasonal": torch.tensor(past_s, dtype=torch.float32).unsqueeze(-1),
            "future_seasonal": torch.tensor(fut_s, dtype=torch.float32).unsqueeze(-1),
        }

        if fut_t is not None:
            item["future_target"] = torch.tensor(fut_t, dtype=torch.float32).unsqueeze(-1)

        return item
