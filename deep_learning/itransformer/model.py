"""
QUANT_VVF — iTransformer: Inverted Variate Tokens for Multivariate Time Series.
Reference:
- Liu et al. (ICLR 2024): iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


class iTransformer(nn.Module):
    """
    iTransformer (Liu et al. 2024):
    Inverts the standard tokenization: individual variate channels across all time steps
    are embedded into token representations (N tokens of length d_model), and self-attention
    models cross-variate correlations while MLP decodes the temporal trajectory.
    """

    def __init__(
        self,
        num_features: int,
        seq_len: int = 48,
        forecast_horizon: int = 5,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model

        # Project entire temporal sequence of length seq_len into d_model per variate
        self.variate_embedding = nn.Linear(seq_len, d_model)

        # Standard Multi-head self attention across variates
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Temporal forecasting head per variate
        self.projection_head = nn.Linear(d_model, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, num_features)
        Returns: (batch, forecast_horizon, num_features)
        """
        B, L, N = x.shape
        # Invert: transpose to (B, N, L) -> each token is an entire time-series variate
        x_inv = x.permute(0, 2, 1)

        # Variate embedding: (B, N, d_model)
        tokens = self.variate_embedding(x_inv)

        # Self-attention captures cross-variate interaction
        encoded = self.transformer_encoder(tokens)  # (B, N, d_model)

        # Project tokens back to forecast horizon: (B, N, forecast_horizon)
        preds = self.projection_head(encoded)

        # Transpose back to standard shape: (B, forecast_horizon, N)
        return preds.permute(0, 2, 1)
