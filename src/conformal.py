"""Split Conformal Prediction module for distribution-free uncertainty quantification.

Reference:
    Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction and
    Distribution-Free Uncertainty Quantification", arXiv:2107.07511, 2021.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ConformalPredictor:
    """Calibrates distribution-free prediction intervals on a held-out validation set."""

    def __init__(self, alpha: float = 0.10) -> None:
        """
        Args:
            alpha: Desired miscoverage rate (alpha=0.10 -> 90% confidence interval).
        """
        self.alpha = alpha
        self.q_val: float | None = None
        self.scores_: np.ndarray | None = None

    def calibrate(self, y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> ConformalPredictor:
        """Calibrate non-conformity quantile on held-out validation set."""
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)

        # Absolute residual non-conformity score
        self.scores_ = np.abs(y_t - y_p)
        n = len(self.scores_)

        # Finite-sample calibrated quantile: ceil((n + 1) * (1 - alpha)) / n
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(1.0, level)
        self.q_val = float(np.quantile(self.scores_, level, method="higher"))

        return self

    def predict_intervals(
        self, y_pred: np.ndarray | pd.Series
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Generate (lower_bound, upper_bound, interval_width) for predictions.
        """
        if self.q_val is None:
            raise RuntimeError("ConformalPredictor must be calibrated before predicting intervals.")
        y_p = np.asarray(y_pred, dtype=np.float64)

        lower = np.clip(y_p - self.q_val, a_min=0.0, a_max=None)
        upper = y_p + self.q_val
        width = 2.0 * self.q_val

        return lower, upper, width

    def evaluate_coverage(
        self, y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
    ) -> dict[str, float]:
        """Evaluate empirical coverage and average interval width on a test set."""
        lower, upper, width = self.predict_intervals(y_pred)
        y_t = np.asarray(y_true, dtype=np.float64)

        covered = (y_t >= lower) & (y_t <= upper)
        empirical_coverage = float(np.mean(covered) * 100.0)

        return {
            "Nominal Confidence (%)": (1.0 - self.alpha) * 100.0,
            "Empirical Coverage (%)": empirical_coverage,
            "Calibrated Quantile (q_val)": self.q_val,
            "Average Interval Width": width,
        }
