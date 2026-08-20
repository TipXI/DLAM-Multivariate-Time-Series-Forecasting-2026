"""DLinear forecasting model.

Reference:
    Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from src.revin import RevIN


class MovingAvgDecomposition(nn.Module):
    """Series decomposition block using moving average pooling."""

    def __init__(self, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: (B, L, C)
        # Padding front and back to maintain length
        front_pad = (self.kernel_size - 1) // 2
        back_pad = (self.kernel_size - 1) - front_pad

        front = x[:, 0:1, :].repeat(1, front_pad, 1)
        back = x[:, -1:, :].repeat(1, back_pad, 1)
        x_padded = torch.cat([front, x, back], dim=1)

        # Average pool over time
        x_perm = x_padded.permute(0, 2, 1)  # (B, C, L_pad)
        trend = self.avg(x_perm).permute(0, 2, 1)  # (B, L, C)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """
    DLinear: A simple yet effective decomposition linear model for time series forecasting.
    """

    def __init__(
        self,
        lookback: int = 168,
        horizon: int = 336,
        kernel_size: int = 25,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.use_revin = use_revin

        self.decomposition = MovingAvgDecomposition(kernel_size=kernel_size)
        self.linear_seasonal = nn.Linear(lookback, horizon)
        self.linear_trend = nn.Linear(lookback, horizon)

        # Initialize weights
        self.linear_seasonal.weight.data.normal_(mean=0.0, std=0.01)
        self.linear_seasonal.bias.data.zero_()
        self.linear_trend.weight.data.normal_(mean=0.0, std=0.01)
        self.linear_trend.bias.data.zero_()

        if self.use_revin:
            self.revin = RevIN(num_features=1, affine=True)

    def forward(
        self,
        past_target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            past_target: (B, L, 1) past target sequence
        Returns:
            (B, H, 1) forecasted future target
        """
        x = past_target
        if self.use_revin:
            x = self.revin(x, mode="norm")

        seasonal, trend = self.decomposition(x)  # (B, L, 1)

        seasonal = seasonal.squeeze(-1)  # (B, L)
        trend = trend.squeeze(-1)        # (B, L)

        seasonal_pred = self.linear_seasonal(seasonal)  # (B, H)
        trend_pred = self.linear_trend(trend)            # (B, H)

        pred = (seasonal_pred + trend_pred).unsqueeze(-1)  # (B, H, 1)

        if self.use_revin:
            pred = self.revin(pred, mode="denorm")

        return pred
