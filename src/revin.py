"""Reversible Instance Normalization (RevIN) module for time series.

Reference:
    Kim et al., "Reversible Instance Normalization for Accurate Time-Series
    Forecasting against Distribution Shift", ICLR 2022.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization layer."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        self.mean: torch.Tensor | None = None
        self.stdev: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'norm' or 'denorm'.")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, num_features)
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

        x_norm = (x - self.mean) / self.stdev
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
        return x_norm

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, horizon, num_features)
        if self.mean is None or self.stdev is None:
            raise RuntimeError("RevIN must normalize before denormalizing.")
        x_denorm = x
        if self.affine:
            x_denorm = (x_denorm - self.affine_bias) / (self.affine_weight + self.eps)
        x_denorm = x_denorm * self.stdev + self.mean
        return x_denorm
