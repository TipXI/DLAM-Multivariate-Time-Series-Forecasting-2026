"""Feature engineering and preprocessing pipeline for high-precision operations forecasting."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd

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


class FeaturePipeline:
    """End-to-end feature extraction with robust missingness handling and domain interaction terms."""

    def __init__(self) -> None:
        self.medians_: dict[str, float] = {}
        self.series2idx_: dict[str, int] = {}
        self.feature_names_: List[str] = []

    def fit(self, train_df: pd.DataFrame) -> FeaturePipeline:
        unique_series = sorted(train_df["series_id"].unique())
        self.series2idx_ = {s: i for i, s in enumerate(unique_series)}

        # Compute medians for imputation
        for c in COVARIATE_COLS:
            if c in train_df.columns:
                self.medians_[c] = float(train_df[c].dropna().median())
            else:
                self.medians_[c] = 0.0

        # Run transform once to capture feature names
        sample_feat = self.transform(train_df.iloc[:500])
        self.feature_names_ = [
            c for c in sample_feat.columns if c not in ["series_id", "timestamp", "target"]
        ]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()

        # 1. Missingness indicator masks
        for c in MISSING_CANDIDATE_COLS:
            if c in d.columns:
                d[f"{c}_isna"] = d[c].isna().astype(float)

        # 2. Robust series-level forward/backward fill + median fallback
        for c in COVARIATE_COLS:
            if c in d.columns:
                d[c] = d.groupby("series_id")[c].transform(lambda x: x.ffill().bfill())
                d[c] = d[c].fillna(self.medians_.get(c, 0.0))

        # 3. Domain physics and interaction features
        d["pressure_sum"] = d["queue_pressure_forecast"] + d["network_pressure_forecast"]
        d["pressure_mult"] = d["queue_pressure_forecast"] * d["network_pressure_forecast"]
        d["effective_workload"] = d["workload_intensity"] * (d["queue_pressure_forecast"] + 1.0)
        d["capacity_utilization"] = d["workload_intensity"] / (d["nominal_capacity"] + 1e-5)
        d["risk_impact"] = d["shock_risk"] * (1.0 - d["unit_reliability_forecast"])
        d["event_risk"] = d["event_load_forecast"] * (d["service_irregularity_risk_forecast"] + 1.0)
        d["demand_staff_ratio"] = d["demand_forecast"] / (d["staffing_forecast"].abs() + 1e-5)
        d["workload_x_demand"] = d["workload_intensity"] * d["demand_forecast"]

        # 4. Integer series index
        d["series_idx"] = d["series_id"].map(self.series2idx_).fillna(0).astype(int)

        return d

    def get_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            (X_continuous, series_indices, y_targets_if_present)
        """
        feat_df = self.transform(df)
        num_cols = [c for c in self.feature_names_ if c != "series_idx"]

        X_num = feat_df[num_cols].to_numpy(dtype=np.float32)
        s_idx = feat_df["series_idx"].to_numpy(dtype=np.int64)

        if "target" in feat_df.columns:
            y = feat_df["target"].to_numpy(dtype=np.float32)
        else:
            y = None

        return X_num, s_idx, y
