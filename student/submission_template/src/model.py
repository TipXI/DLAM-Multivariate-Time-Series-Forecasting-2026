"""DeepOperationsNet architecture for the final submission package."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResMLPBlock(nn.Module):
    """Residual MLP Block with LayerNorm, GELU, and Dropout."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ForecastModel(nn.Module):
    """
    Final PyTorch deep learning model for operations load forecasting.
    Combines entity embeddings with non-linear feature representations.
    """

    def __init__(
        self,
        num_continuous_features: int = 40,
        num_series: int = 96,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.series_embedding = nn.Embedding(num_series, embedding_dim)

        self.input_layer = nn.Sequential(
            nn.Linear(num_continuous_features + embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList([ResMLPBlock(hidden_dim, dropout) for _ in range(num_blocks)])

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU(),
        )

    def forward(self, x_num: torch.Tensor, s_idx: torch.Tensor) -> torch.Tensor:
        emb = self.series_embedding(s_idx)
        x_cat = torch.cat([x_num, emb], dim=-1)

        h = self.input_layer(x_cat)
        for block in self.blocks:
            h = block(h)

        out = self.head(h)
        return out
