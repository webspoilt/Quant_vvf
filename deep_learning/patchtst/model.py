"""
QUANT_VVF — PatchTST Architecture (Challenger to TFT).
Reference:
- Nie, Nguyen, Sinthong & Kalagnanam (ICLR 2023):
  A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional


class PatchEmbedding(nn.Module):
    """
    Sub-series patching layer: splits a 1D sequence into overlapping patches
    and projects them into embedding space.
    """

    def __init__(self, patch_len: int = 16, stride: int = 8, d_model: int = 32) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.projection = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        x: (batch * num_channels, seq_len)
        returns: (batch * num_channels, num_patches, d_model), num_patches
        """
        B_N, L = x.shape
        # Unfold into overlapping patches
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # (B_N, num_patches, patch_len)
        num_patches = patches.shape[1]
        embeddings = self.projection(patches)  # (B_N, num_patches, d_model)
        return embeddings, num_patches


class PatchTST(nn.Module):
    """
    PatchTST: Channel-independent Transformer with sub-series patching.
    Challenger architecture to TFT for multi-horizon volatility and feature forecasting.
    """

    def __init__(
        self,
        num_features: int,
        seq_len: int = 64,
        forecast_horizon: int = 5,
        patch_len: int = 16,
        stride: int = 8,
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

        # Channel-independent patching
        self.patch_embed = PatchEmbedding(patch_len=patch_len, stride=stride, d_model=d_model)

        # Positional encoding
        num_patches = (seq_len - patch_len) // stride + 1
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)

        # Standard Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Linear forecasting head
        self.head = nn.Linear(num_patches * d_model, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, num_features)
        Returns: (batch, forecast_horizon, num_features)
        """
        B, L, N = x.shape
        # Channel independence: reshape to (B * N, L)
        x_channels = x.permute(0, 2, 1).reshape(B * N, L)

        # Instance normalization (RevIN style)
        mean = torch.mean(x_channels, dim=-1, keepdim=True)
        std = torch.std(x_channels, dim=-1, keepdim=True) + 1e-6
        x_norm = (x_channels - mean) / std

        # Patching and embedding
        enc_in, num_patches = self.patch_embed(x_norm)  # (B*N, num_patches, d_model)
        enc_in = enc_in + self.pos_embed

        # Transformer encoder
        enc_out = self.transformer_encoder(enc_in)  # (B*N, num_patches, d_model)

        # Flatten patches and project
        flat = enc_out.reshape(B * N, -1)
        preds = self.head(flat)  # (B*N, forecast_horizon)

        # Denormalize
        preds = preds * std + mean

        # Reshape back to (B, forecast_horizon, N)
        out = preds.reshape(B, N, self.forecast_horizon).permute(0, 2, 1)
        return out
