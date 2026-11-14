"""
QUANT_VVF — Avellaneda-Stoikov & Obizhaeva-Wang Execution Mathematics.
References:
- Avellaneda & Stoikov (2008): High-frequency trading in a limit order book. Quantitative Finance, 8(3), 217-224.
- Obizhaeva & Wang (2013): Optimal consumption and portfolio selection with transaction costs.
  Journal of Financial Markets, 16(1), 1-32.
"""

from __future__ import annotations
import math
from typing import Dict, Any, Tuple
import numpy as np


class AvellanedaStoikovMarketMaker:
    """
    Avellaneda & Stoikov (2008) Optimal High-Frequency Market Making:
    Reservation price:
        r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
    Optimal spread distances:
        delta_a + delta_b = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / kappa)
        delta_a = (r - s) + (1 / gamma) * ln(1 + gamma / kappa)
        delta_b = (s - r) + (1 / gamma) * ln(1 + gamma / kappa)
    where:
        s: current mid price
        q: current inventory (positive = long, negative = short)
        gamma: risk-aversion parameter
        sigma: volatility
        T - t: remaining trading session fraction
        kappa: order-book liquidity parameter (fill probability decay with distance)
    """

    def __init__(self, gamma: float = 0.1, kappa: float = 1.5) -> None:
        self.gamma = max(1e-4, float(gamma))
        self.kappa = max(1e-4, float(kappa))

    def compute_quotes(
        self,
        mid_price: float,
        inventory: float,
        volatility: float,
        time_remaining: float,
    ) -> Dict[str, float]:
        s = mid_price
        q = inventory
        gamma = self.gamma
        kappa = self.kappa
        sigma = volatility
        tau = max(1e-6, time_remaining)

        # 1. Reservation price
        r = s - q * gamma * (sigma ** 2) * tau

        # 2. Optimal half-spread adjustment
        spread_term = (1.0 / gamma) * math.log(1.0 + gamma / kappa)
        delta_a = (s - r) + spread_term
        delta_b = (r - s) + spread_term

        ask_quote = s + delta_a
        bid_quote = s - delta_b

        return {
            "reservation_price": float(r),
            "bid_quote": float(bid_quote),
            "ask_quote": float(ask_quote),
            "bid_spread": float(delta_b),
            "ask_spread": float(delta_a),
            "total_spread": float(delta_a + delta_b),
        }


class ObizhaevaWangExecution:
    """
    Obizhaeva & Wang (2013) Optimal Execution with Transient Market Impact & Order-Book Resilience:
    Impact decays exponentially: G(t - s) = lambda * exp(-rho * (t - s))
    where rho is order book recovery / resilience rate, lambda is instantaneous impact.
    """

    def __init__(self, impact_lambda: float = 1e-4, resilience_rho: float = 2.0) -> None:
        self.impact_lambda = float(impact_lambda)
        self.resilience_rho = float(resilience_rho)

    def transient_impact(self, trade_size: float, elapsed_time: float) -> float:
        """Impact remaining after elapsed_time: lambda * trade_size * exp(-rho * dt)."""
        return self.impact_lambda * trade_size * math.exp(-self.resilience_rho * max(0.0, elapsed_time))
