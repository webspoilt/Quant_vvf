"""
QUANT_VVF — Market Microstructure, Hawkes Processes, Microprice & Execution Costs.
References:
- Hawkes (1971): Spectra of Some Self-Exciting Point Processes
- Bacry et al. (2013): Some Properties of Mutually Exciting Hawkes Processes
- Stoikov (2018): The Micro-Price: a High-Frequency Estimator of Future Prices
- Almgren & Chriss (2000): Optimal Execution of Portfolio Transactions
"""

from __future__ import annotations
import math
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from scipy.optimize import minimize


class BivariateHawkesProcess:
    """
    Bivariate mutually exciting Hawkes process for buy vs sell aggressive order arrivals:
    lambda_1(t) = mu_1 + integral_0^t (alpha_11 e^{-beta_11(t-s)} dN_1(s) + alpha_12 e^{-beta_12(t-s)} dN_2(s))
    lambda_2(t) = mu_2 + integral_0^t (alpha_21 e^{-beta_21(t-s)} dN_1(s) + alpha_22 e^{-beta_22(t-s)} dN_2(s))
    Stability: spectral radius of matrix [alpha_ij / beta_ij] < 1.0.
    """

    def __init__(
        self,
        mu: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> None:
        self.mu = np.asarray(mu, dtype=np.float64)  # shape (2,)
        self.alpha = np.asarray(alpha, dtype=np.float64)  # shape (2, 2)
        self.beta = np.asarray(beta, dtype=np.float64)  # shape (2, 2)

    @property
    def branching_matrix(self) -> np.ndarray:
        """Matrix Gamma_ij = alpha_ij / beta_ij."""
        return self.alpha / np.maximum(1e-6, self.beta)

    @property
    def spectral_radius(self) -> float:
        """Spectral radius must be < 1.0 for stationary process stability."""
        eigvals = np.linalg.eigvals(self.branching_matrix)
        return float(np.max(np.abs(eigvals)))

    @property
    def is_stable(self) -> bool:
        return self.spectral_radius < 1.0

    def compute_intensities(self, event_times_1: np.ndarray, event_times_2: np.ndarray, eval_time: float) -> Tuple[float, float]:
        """Compute conditional intensity vector [lambda_1(t), lambda_2(t)] at eval_time."""
        t = eval_time
        lam1 = self.mu[0]
        lam2 = self.mu[1]

        # Contributions from process 1 (buy)
        past_1 = event_times_1[event_times_1 < t]
        if len(past_1) > 0:
            dt1 = t - past_1
            lam1 += np.sum(self.alpha[0, 0] * np.exp(-self.beta[0, 0] * dt1))
            lam2 += np.sum(self.alpha[1, 0] * np.exp(-self.beta[1, 0] * dt1))

        # Contributions from process 2 (sell)
        past_2 = event_times_2[event_times_2 < t]
        if len(past_2) > 0:
            dt2 = t - past_2
            lam1 += np.sum(self.alpha[0, 1] * np.exp(-self.beta[0, 1] * dt2))
            lam2 += np.sum(self.alpha[1, 1] * np.exp(-self.beta[1, 1] * dt2))

        return float(max(1e-6, lam1)), float(max(1e-6, lam2))

    @classmethod
    def fit_mle(
        cls, event_times_1: np.ndarray, event_times_2: np.ndarray, total_time: float
    ) -> BivariateHawkesProcess:
        """Maximum Likelihood Estimation of Hawkes parameters under exponential kernel."""
        t1 = np.sort(np.asarray(event_times_1, dtype=np.float64))
        t2 = np.sort(np.asarray(event_times_2, dtype=np.float64))
        T = float(total_time)

        # Vector parameter: [mu1, mu2, a11, a12, a21, a22, b11, b12, b21, b22]
        x0 = np.array([len(t1) / T * 0.5, len(t2) / T * 0.5, 0.2, 0.1, 0.1, 0.2, 1.0, 1.0, 1.0, 1.0])
        bounds = [(1e-4, 50.0)] * 2 + [(0.0, 5.0)] * 4 + [(0.1, 20.0)] * 4

        def neg_log_lik(p: np.ndarray) -> float:
            mu1, mu2 = p[0], p[1]
            a = p[2:6].reshape((2, 2))
            b = p[6:10].reshape((2, 2))

            # Spectral radius stability check
            gamma = a / b
            if np.max(np.abs(np.linalg.eigvals(gamma))) >= 0.999:
                return 1e9

            # Log-likelihood approximation over events
            ll = 0.0
            # Integrated intensity term: - sum_i (mu_i * T + sum_j (a_ij / b_ij) * sum_{t_k} (1 - exp(-b_ij*(T - t_k))))
            int_lam1 = mu1 * T + (a[0, 0] / b[0, 0]) * np.sum(1.0 - np.exp(-b[0, 0] * (T - t1))) + (
                a[0, 1] / b[0, 1]
            ) * np.sum(1.0 - np.exp(-b[0, 1] * (T - t2)))
            int_lam2 = mu2 * T + (a[1, 0] / b[1, 0]) * np.sum(1.0 - np.exp(-b[1, 0] * (T - t1))) + (
                a[1, 1] / b[1, 1]
            ) * np.sum(1.0 - np.exp(-b[1, 1] * (T - t2)))
            ll -= (int_lam1 + int_lam2)

            # Sum log lambda at arrival times
            for t in t1:
                p1 = t1[t1 < t]
                p2 = t2[t2 < t]
                lam = mu1 + np.sum(a[0, 0] * np.exp(-b[0, 0] * (t - p1))) + np.sum(a[0, 1] * np.exp(-b[0, 1] * (t - p2)))
                ll += math.log(max(1e-8, lam))

            for t in t2:
                p1 = t1[t1 < t]
                p2 = t2[t2 < t]
                lam = mu2 + np.sum(a[1, 0] * np.exp(-b[1, 0] * (t - p1))) + np.sum(a[1, 1] * np.exp(-b[1, 1] * (t - p2)))
                ll += math.log(max(1e-8, lam))

            return -float(ll) if np.isfinite(ll) else 1e9

        res = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 80})
        p = res.x
        return cls(
            mu=p[0:2],
            alpha=p[2:6].reshape((2, 2)),
            beta=p[6:10].reshape((2, 2)),
        )


def calculate_queue_imbalance(bid_qty: float, ask_qty: float) -> float:
    """Standard order book level-1 imbalance: (Q_b - Q_a) / (Q_b + Q_a)."""
    total = bid_qty + ask_qty
    if total <= 1e-9:
        return 0.0
    return float((bid_qty - ask_qty) / total)


def calculate_microprice(bid_price: float, ask_price: float, bid_qty: float, ask_qty: float) -> float:
    """
    Stoikov (2018) Micro-Price estimator:
    P_micro = P_ask * (Q_bid / (Q_bid + Q_ask)) + P_bid * (Q_ask / (Q_bid + Q_ask))
    """
    total_qty = bid_qty + ask_qty
    if total_qty <= 1e-9:
        return 0.5 * (bid_price + ask_price)
    w_bid = bid_qty / total_qty
    w_ask = ask_qty / total_qty
    return float(ask_price * w_bid + bid_price * w_ask)


def almgren_chriss_execution_cost(
    trade_size: float,
    daily_volume: float,
    volatility: float,
    temp_impact_factor: float = 0.1,
    perm_impact_factor: float = 0.05,
    half_spread_bps: float = 2.5,
) -> Dict[str, float]:
    """
    Almgren-Chriss (2000) execution cost model:
    - Spread cost: half-spread * trade_size
    - Permanent price impact: gamma * (trade_size / daily_volume) * spot
    - Temporary price impact: eta * (execution_rate / daily_volume) * spot
    """
    participation_rate = abs(trade_size) / max(1.0, daily_volume)
    spread_cost_fraction = half_spread_bps * 1e-4

    perm_impact = perm_impact_factor * participation_rate * volatility
    temp_impact = temp_impact_factor * math.sqrt(participation_rate) * volatility
    total_slippage_bps = (spread_cost_fraction + perm_impact + temp_impact) * 1e4

    return {
        "spread_cost_bps": float(spread_cost_fraction * 1e4),
        "perm_impact_bps": float(perm_impact * 1e4),
        "temp_impact_bps": float(temp_impact * 1e4),
        "total_slippage_bps": float(total_slippage_bps),
    }
