"""TiDE: Time-series Dense Encoder model for long-term multivariate forecasting with covariates.

Reference:
    Das et al., "Long-term Forecasting with TiDE: Time-series Dense Encoder",
    Transactions on Machine Learning Research (TMLR), 2023.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from src.revin import RevIN


class ResBlock(nn.Module):
    """Residual MLP Block with LayerNorm and Dropout."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

        if input_dim != output_dim:
            self.skip = nn.Linear(input_dim, output_dim)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.skip(x)
        out = self.dropout(self.fc2(self.relu(self.fc1(x))))
        return self.layer_norm(out + res)


class TiDE(nn.Module):
    """
    TiDE (Time-series Dense Encoder) architecture.
    """

    def __init__(
        self,
        lookback: int = 168,
        horizon: int = 336,
        num_dynamic_covariates: int = 22,
        num_series: int = 96,
        static_embedding_dim: int = 16,
        dynamic_projection_dim: int = 8,
        hidden_dim: int = 256,
        decoder_output_dim: int = 16,
        temporal_hidden_dim: int = 32,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
        use_revin: bool = True,
        use_seasonal_residual: bool = True,
    ) -> None:
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.use_revin = use_revin
        self.use_seasonal_residual = use_seasonal_residual
        self.decoder_output_dim = decoder_output_dim

        # 1. Static Series ID Embedding
        self.static_embedding = nn.Embedding(num_series, static_embedding_dim)

        # 2. Dynamic Covariate Projection
        self.covariate_projection = nn.Sequential(
            nn.Linear(num_dynamic_covariates, dynamic_projection_dim),
            nn.ReLU(),
            nn.Linear(dynamic_projection_dim, dynamic_projection_dim),
        )

        # Input dimension for Encoder:
        # past_target (L) + past_covariates (L * d_proj) + future_covariates (H * d_proj) + static_emb (d_static)
        encoder_input_dim = (
            lookback
            + (lookback * dynamic_projection_dim)
            + (horizon * dynamic_projection_dim)
            + static_embedding_dim
        )

        # 3. Dense Encoder
        encoder_layers = [ResBlock(encoder_input_dim, hidden_dim, hidden_dim, dropout)]
        for _ in range(num_encoder_layers - 1):
            encoder_layers.append(ResBlock(hidden_dim, hidden_dim, hidden_dim, dropout))
        self.encoder = nn.Sequential(*encoder_layers)

        # 4. Dense Decoder
        decoder_output_flat_dim = horizon * decoder_output_dim
        decoder_layers = [ResBlock(hidden_dim, hidden_dim, hidden_dim, dropout)]
        for _ in range(num_decoder_layers - 1):
            decoder_layers.append(ResBlock(hidden_dim, hidden_dim, hidden_dim, dropout))
        decoder_layers.append(nn.Linear(hidden_dim, decoder_output_flat_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # 5. Temporal Decoder (per future step t)
        # Combines decoded_feature (decoder_output_dim) + projected_future_covariate (d_proj) + static_emb (d_static)
        temporal_input_dim = decoder_output_dim + dynamic_projection_dim + static_embedding_dim
        self.temporal_decoder = ResBlock(temporal_input_dim, temporal_hidden_dim, 1, dropout)

        # 6. Global Residual Skip Connection (maps past target L directly to future H)
        self.residual_skip = nn.Linear(lookback, horizon)

        # 7. RevIN for scale stabilization
        if self.use_revin and not self.use_seasonal_residual:
            self.revin = RevIN(num_features=1, affine=True)

    def forward(
        self,
        past_target: torch.Tensor,
        past_covariates: torch.Tensor,
        future_covariates: torch.Tensor,
        series_idx: torch.Tensor,
        future_seasonal: torch.Tensor | None = None,
        past_seasonal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            past_target: (B, L, 1)
            past_covariates: (B, L, C)
            future_covariates: (B, H, C)
            series_idx: (B,)
            future_seasonal: (B, H, 1) (optional)
            past_seasonal: (B, L, 1) (optional)
        Returns:
            (B, H, 1) prediction
        """
        B = past_target.size(0)

        # Target preprocessing: Residual mode or RevIN
        if self.use_seasonal_residual and past_seasonal is not None:
            # Model predicts residual deviations from seasonal profile
            y_in = past_target - past_seasonal
        elif self.use_revin:
            y_in = self.revin(past_target, mode="norm")
        else:
            y_in = past_target

        # Flatten past target
        y_flat = y_in.squeeze(-1)  # (B, L)

        # Embed static series
        s_emb = self.static_embedding(series_idx)  # (B, d_static)

        # Project dynamic covariates
        past_cov_proj = self.covariate_projection(past_covariates)     # (B, L, d_proj)
        fut_cov_proj = self.covariate_projection(future_covariates)    # (B, H, d_proj)

        past_cov_flat = past_cov_proj.reshape(B, -1)                  # (B, L * d_proj)
        fut_cov_flat = fut_cov_proj.reshape(B, -1)                    # (B, H * d_proj)

        # Concatenate all inputs to Encoder
        enc_in = torch.cat([y_flat, past_cov_flat, fut_cov_flat, s_emb], dim=-1)  # (B, enc_in_dim)
        enc_out = self.encoder(enc_in)                                              # (B, hidden_dim)

        # Decode
        dec_out = self.decoder(enc_out)                                            # (B, H * decoder_output_dim)
        dec_matrix = dec_out.reshape(B, self.horizon, self.decoder_output_dim)      # (B, H, decoder_output_dim)

        # Expand static embedding across horizon
        s_emb_expanded = s_emb.unsqueeze(1).repeat(1, self.horizon, 1)             # (B, H, d_static)

        # Temporal decode per step
        temp_in = torch.cat([dec_matrix, fut_cov_proj, s_emb_expanded], dim=-1)   # (B, H, temp_in_dim)
        temp_out = self.temporal_decoder(temp_in)                                  # (B, H, 1)

        # Add global linear residual skip
        skip_out = self.residual_skip(y_flat).unsqueeze(-1)                        # (B, H, 1)
        pred_norm = temp_out + skip_out

        # Postprocessing: invert RevIN or add seasonal profile
        if self.use_seasonal_residual and future_seasonal is not None:
            pred = pred_norm + future_seasonal
        elif self.use_revin and not self.use_seasonal_residual:
            pred = self.revin(pred_norm, mode="denorm")
        else:
            pred = pred_norm

        return pred
