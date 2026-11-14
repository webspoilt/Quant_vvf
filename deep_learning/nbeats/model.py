"""
QUANT_VVF — N-BEATS: Neural Basis Expansion Analysis for Time Series.
Reference:
- Oreshkin, Carpov, Chapados & Bengio (ICLR 2020):
  N-BEATS: Neural basis expansion analysis for interpretable time series forecasting.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List


class NBeatsBlock(nn.Module):
    """Basic residual block with backward (backcast) and forward (forecast) basis expansion."""

    def __init__(self, backcast_len: int, forecast_len: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(backcast_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.theta_b = nn.Linear(hidden_dim, backcast_len)
        self.theta_f = nn.Linear(hidden_dim, forecast_len)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        backcast = self.theta_b(h)
        forecast = self.theta_f(h)
        return backcast, forecast


class NBeats(nn.Module):
    """
    N-BEATS architecture composed of stacked residual expansion blocks.
    x_l = x_{l-1} - \\hat{x}_{l-1} (residual subtraction)
    \\hat{y} = sum_l \\hat{y}_l (forecast aggregation)
    """

    def __init__(
        self,
        backcast_len: int = 48,
        forecast_len: int = 5,
        num_blocks: int = 3,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.backcast_len = backcast_len
        self.forecast_len = forecast_len
        self.blocks = nn.ModuleList([NBeatsBlock(backcast_len, forecast_len, hidden_dim) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, backcast_len) or (batch, backcast_len, 1)
        returns: (batch, forecast_len)
        """
        if x.dim() == 3:
            x = x.squeeze(-1)

        residuals = x
        total_forecast = torch.zeros(x.shape[0], self.forecast_len, device=x.device)

        for block in self.blocks:
            backcast, forecast = block(residuals)
            residuals = residuals - backcast
            total_forecast = total_forecast + forecast

        return total_forecast
