"""
QUANT_VVF — Temporal Fusion Transformer (TFT) Architecture.
Reference:
- Lim, Arık, Loeff & Pfister (2021): Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.
  International Journal of Forecasting, 37(4), 1748-1764.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional


class GatedLinearUnit(nn.Module):
    """GLU(x) = x_1 * sigmoid(x_2)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, d_model * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val, gate = self.fc(x).chunk(2, dim=-1)
        return val * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    """GRN: Non-linear processing with gating and residual skip connection."""

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None, dropout: float = 0.1) -> None:
        super().__init__()
        h_dim = hidden_dim or d_model
        self.fc1 = nn.Linear(d_model, h_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(h_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.glu(out)
        return self.layer_norm(out + residual)


class VariableSelectionNetwork(nn.Module):
    """VSN: Learns variable importance weights across feature inputs."""

    def __init__(self, num_features: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        # Independent GRN per feature
        self.feature_grns = nn.ModuleList([GatedResidualNetwork(d_model, dropout=dropout) for _ in range(num_features)])
        # Gating network for variable selection weights
        self.flattened_grn = GatedResidualNetwork(num_features * d_model, hidden_dim=d_model, dropout=dropout)
        self.softmax_linear = nn.Linear(num_features * d_model, num_features)

    def forward(self, feature_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        feature_embeddings: (batch, seq_len, num_features, d_model)
        Returns: (aggregated_representation, variable_weights)
        """
        B, T, N, D = feature_embeddings.shape
        flat = feature_embeddings.reshape(B, T, N * D)

        # Compute selection weights
        v_weights = F.softmax(self.softmax_linear(flat), dim=-1)  # (B, T, N)

        # Process each feature through its own GRN
        processed_features = []
        for i in range(N):
            feat_i = feature_embeddings[:, :, i, :]
            proc_i = self.feature_grns[i](feat_i)
            processed_features.append(proc_i)

        stacked = torch.stack(processed_features, dim=2)  # (B, T, N, D)
        weights_expanded = v_weights.unsqueeze(-1)  # (B, T, N, 1)

        weighted_sum = torch.sum(stacked * weights_expanded, dim=2)  # (B, T, D)
        return weighted_sum, v_weights


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable multi-head self-attention layer providing attention weight interpretability."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        Q = self.q_linear(x).reshape(B, T, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        K = self.k_linear(x).reshape(B, T, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        V = self.v_linear(x).reshape(B, T, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.d_head)
        # Causal mask
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.permute(0, 2, 1, 3).reshape(B, T, D)
        out = self.out_linear(context)

        # Average attention weights across heads for interpretability
        avg_weights = torch.mean(attn_weights, dim=1)  # (B, T, T)
        return out, avg_weights


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for Multi-Horizon Quantile Volatility Forecasting.
    Predicts quantiles: 10%, 50% (median), 90%.
    """

    def __init__(
        self,
        num_features: int,
        forecast_horizon: int = 5,
        d_model: int = 32,
        num_heads: int = 4,
        quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        self.quantiles = quantiles

        # Linear projection for raw inputs
        self.input_embed = nn.Linear(1, d_model)

        # Variable selection network
        self.vsn = VariableSelectionNetwork(num_features, d_model, dropout=dropout)

        # Local LSTM processing for sequential context
        self.lstm = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)

        # Attention layer
        self.attention = InterpretableMultiHeadAttention(d_model, num_heads=num_heads, dropout=dropout)
        self.post_attn_grn = GatedResidualNetwork(d_model, dropout=dropout)

        # Output projection to multi-horizon quantiles: shape (forecast_horizon, len(quantiles))
        self.output_heads = nn.Linear(d_model, forecast_horizon * len(quantiles))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, num_features)
        Returns:
            quantile_forecasts: (batch, forecast_horizon, len(quantiles))
            variable_importance: (batch, seq_len, num_features)
            temporal_attention: (batch, seq_len, seq_len)
        """
        B, T, N = x.shape
        x_exp = x.unsqueeze(-1)  # (B, T, N, 1)
        feat_embed = self.input_embed(x_exp)  # (B, T, N, d_model)

        # Variable Selection
        vsn_out, var_weights = self.vsn(feat_embed)  # (B, T, d_model)

        # Local sequential processing
        lstm_out, _ = self.lstm(vsn_out)  # (B, T, d_model)

        # Interpretable Attention
        attn_out, attn_weights = self.attention(lstm_out)
        enriched = self.post_attn_grn(attn_out + lstm_out)

        # Take last time step for multi-horizon forecast
        last_step = enriched[:, -1, :]  # (B, d_model)
        raw_preds = self.output_heads(last_step)  # (B, forecast_horizon * len(quantiles))
        quantile_forecasts = raw_preds.reshape(B, self.forecast_horizon, len(self.quantiles))

        return quantile_forecasts, var_weights, attn_weights


def quantile_loss(preds: torch.Tensor, target: torch.Tensor, quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90)) -> torch.Tensor:
    """
    Quantile / Pinball loss:
    L_q(y, y_hat) = max(q * (y - y_hat), (1 - q) * (y_hat - y))
    """
    losses = []
    for idx, q in enumerate(quantiles):
        pred_q = preds[:, :, idx]
        err = target - pred_q
        loss_q = torch.maximum(q * err, (q - 1.0) * err)
        losses.append(torch.mean(loss_q))
    return sum(losses)
