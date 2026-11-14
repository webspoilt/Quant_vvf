"""
QUANT_VVF — Deep Hedging Architecture for Non-Linear Crypto Derivatives.
Reference:
- Buehler, Gonon, Teichmann & Wood (2019): Deep Hedging.
  Quantitative Finance, 19(8), 1271-1291.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List, Optional


class DeepHedgingPolicy(nn.Module):
    """
    Recurrent Deep Hedging Policy Network (Buehler et al. 2019):
    At each rebalancing step t, takes current state:
    [Spot / Strike, Time_to_expiry, Current_IV, Current_Hedge_Delta, Log_Moneyness]
    and outputs the optimal updated hedge delta delta_t.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # Hedge position bounded in [-1.0, 1.0]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (batch, input_dim) -> delta_t: (batch, 1)."""
        return self.net(state)


class DeepHedgingEngine:
    """
    Simulates pathwise hedging trajectories, computes terminal hedging errors,
    penalizes proportional transaction costs, and trains via convex risk measure minimization.
    """

    def __init__(
        self,
        cost_bps: float = 3.0,
        risk_aversion: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.cost_multiplier = cost_bps * 1e-4
        self.risk_aversion = risk_aversion
        self.device = device
        self.policy = DeepHedgingPolicy().to(device)

    def simulate_and_evaluate(
        self,
        spot_paths: torch.Tensor,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        is_call: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        spot_paths: (num_paths, num_steps + 1)
        Returns:
            terminal_hedging_error: (num_paths,)
            total_transaction_costs: (num_paths,)
            hedge_trajectories: (num_paths, num_steps)
        """
        num_paths, total_steps = spot_paths.shape
        num_steps = total_steps - 1
        dt = time_to_expiry / num_steps

        current_delta = torch.zeros(num_paths, 1, device=self.device)
        portfolio_cash = torch.zeros(num_paths, 1, device=self.device)
        total_costs = torch.zeros(num_paths, 1, device=self.device)
        delta_history = []

        for step in range(num_steps):
            t_rem = max(1e-4, time_to_expiry - step * dt)
            S_t = spot_paths[:, step : step + 1]

            # Construct state vector
            state = torch.cat(
                [
                    S_t / strike,
                    torch.full_like(S_t, t_rem),
                    torch.full_like(S_t, volatility),
                    current_delta,
                    torch.log(S_t / strike),
                ],
                dim=-1,
            )

            # Policy emits new target hedge delta
            target_delta = self.policy(state)
            delta_change = target_delta - current_delta

            # Transaction costs: cost_mult * S_t * |delta_change|
            cost = self.cost_multiplier * S_t * torch.abs(delta_change)
            total_costs = total_costs + cost

            # Cash flow from trade
            portfolio_cash = portfolio_cash - delta_change * S_t - cost
            current_delta = target_delta
            delta_history.append(current_delta)

        # Terminal step payoff
        S_T = spot_paths[:, -1:]
        if is_call:
            terminal_payoff = torch.maximum(torch.zeros_like(S_T), S_T - strike)
        else:
            terminal_payoff = torch.maximum(torch.zeros_like(S_T), strike - S_T)

        # Final portfolio value = Cash + Delta_final * S_T - Liability
        final_portfolio = portfolio_cash + current_delta * S_T - terminal_payoff
        hedge_error = -final_portfolio  # Loss from short derivative perspective

        return hedge_error.squeeze(-1), total_costs.squeeze(-1), torch.cat(delta_history, dim=-1)

    def compute_loss(self, hedge_error: torch.Tensor, alpha_cvar: float = 0.95) -> torch.Tensor:
        """
        Convex risk measure loss: Mean + Risk Aversion * CVaR_alpha.
        """
        mean_loss = torch.mean(hedge_error)
        # CVaR approximation
        k = int((1.0 - alpha_cvar) * len(hedge_error))
        k = max(1, k)
        top_losses, _ = torch.topk(hedge_error, k=k)
        cvar = torch.mean(top_losses)

        return mean_loss + self.risk_aversion * cvar
