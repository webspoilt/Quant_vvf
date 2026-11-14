"""
Advanced Stochastic Calculus, Stochastic Control, Malliavin Greeks, Rough Volatility & HJB.

Modules:
1. Itô Lemma symbolic/numerical expansion and verification.
2. Change of Measure (Girsanov Theorem):
   - Measure P vs Measure Q transformation with Girsanov kernel lambda_t.
   - Novikov condition verification: E[exp(0.5 * int lambda_s^2 ds)] < infinity.
3. Malliavin Greeks:
   - Malliavin integration by parts for discontinuous payoffs (e.g. digital options, barrier options).
   - Malliavin weights: Delta = E[Phi(S_T) * pi_Delta], Vega = E[Phi(S_T) * pi_Vega].
4. Rough Volatility:
   - Fractional Brownian motion simulation via Cholesky / Riemann-Liouville Volterra representation.
   - Hurst exponent estimation via semivariogram / q-th order structure functions.
5. Hamilton-Jacobi-Bellman (HJB) Solver:
   - Optimal execution / liquidation with inventory penalty (Cartea & Jaimungal / Almgren-Chriss extension).
"""

from dataclasses import dataclass, field
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
import scipy.optimize as opt


@dataclass
class GirsanovMeasureEngine:
    """
    Computes Radon-Nikodym derivative dQ/dP and drift adjustments:
    dW_t^Q = dW_t^P + lambda_t dt
    Novikov check: int_0^T lambda_s^2 ds bounded.
    """
    lambda_market_price_of_risk: float = 0.20 # Excess return / volatility

    def check_novikov_condition(self, t_horizon: float) -> bool:
        """Verifies Novikov condition E[exp(0.5 * int_0^T lambda^2 dt)] < inf."""
        integral_val = 0.5 * (self.lambda_market_price_of_risk ** 2) * t_horizon
        return integral_val < 50.0 # Bounded exponent

    def radon_nikodym_density(self, w_p_path: np.ndarray, dt: float) -> np.ndarray:
        """
        Computes Z_t = dQ/dP = exp(-int lambda dW_P - 0.5 int lambda^2 dt)
        under convention dW^Q = dW^P + lambda dt.
        """
        dw = np.diff(w_p_path, prepend=0.0)
        t_steps = len(w_p_path)
        log_z = -self.lambda_market_price_of_risk * np.cumsum(dw) - 0.5 * (self.lambda_market_price_of_risk**2) * dt * np.arange(t_steps)
        return np.exp(log_z)


@dataclass
class MalliavinGreeksCalculator:
    """
    Computes option Greeks using Malliavin Integration by Parts weights.
    Particularly effective for discontinuous payoffs (e.g., Digital / Binary options)
    where pathwise derivatives fail (Dirac delta divergence).
    """
    s0: float
    k: float
    t_expiry: float
    r: float
    sigma: float

    def digital_call_malliavin_delta(self, n_paths: int = 100000, seed: int = 42) -> Tuple[float, float]:
        """
        Digital option payoff: 1_{S_T > K}.
        Malliavin weight for Delta:
        pi_Delta = W_T / (S0 * sigma * T)
        Delta = exp(-r*T) * E[ 1_{S_T > K} * pi_Delta ]
        Returns: (estimated_delta, standard_error)
        """
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(n_paths)
        w_t = math.sqrt(self.t_expiry) * z

        st = self.s0 * np.exp((self.r - 0.5 * self.sigma**2) * self.t_expiry + self.sigma * w_t)
        in_the_money = (st > self.k).astype(np.float64)

        # Malliavin weight for European claim under geometric Brownian motion
        weight_delta = w_t / (self.s0 * self.sigma * self.t_expiry)
        df = math.exp(-self.r * self.t_expiry)

        sample_deltas = df * in_the_money * weight_delta
        delta = float(np.mean(sample_deltas))
        se = float(np.std(sample_deltas, ddof=1) / math.sqrt(n_paths))
        return delta, se

    def analytical_digital_call_delta(self) -> float:
        """Analytical Black-Scholes Greek for digital cash-or-nothing call."""
        d1 = (math.log(self.s0 / self.k) + (self.r + 0.5 * self.sigma**2) * self.t_expiry) / (self.sigma * math.sqrt(self.t_expiry))
        d2 = d1 - self.sigma * math.sqrt(self.t_expiry)
        df = math.exp(-self.r * self.t_expiry)
        # Digital call price = df * N(d2). dPrice/dS0 = df * n(d2) / (S0 * sigma * sqrt(T))
        return float(df * stats.norm.pdf(d2) / (self.s0 * self.sigma * math.sqrt(self.t_expiry)))


@dataclass
class RoughVolatilityEngine:
    """
    Rough Fractional Volatility simulation and Hurst exponent estimator.
    Volterra representation:
    V_t = xi_0(t) * exp(eta * int_0^t (t - s)^{H - 1/2} dW_s - 0.5 * eta^2 * t^{2H})
    """
    hurst_h: float = 0.14 # Typical financial rough volatility H in (0.05, 0.20)
    eta: float = 1.2 # Vol of vol
    v0: float = 0.04

    def simulate_fractional_kernel_paths(
        self,
        n_paths: int = 1000,
        n_steps: int = 100,
        t_expiry: float = 1.0,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Simulates Volterra rough variance process using Riemann-Liouville fractional kernel:
        K(t - s) = (t - s)^{H - 1/2}
        """
        rng = np.random.default_rng(seed)
        dt = t_expiry / n_steps
        alpha = self.hurst_h - 0.5
        t_grid = np.linspace(0.0, t_expiry, n_steps + 1)

        # Generate independent Brownian increments
        dw = rng.normal(0.0, math.sqrt(dt), size=(n_paths, n_steps))
        v_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        v_paths[:, 0] = self.v0

        for i in range(1, n_steps + 1):
            t_curr = t_grid[i]
            # Time lags for steps j = 0...i-1
            s_grid = t_grid[:i]
            lags = t_curr - s_grid
            kernel = np.maximum(lags, 1e-8) ** alpha

            # Convolution sum
            conv = np.sum(dw[:, :i] * kernel[::-1][:i], axis=1)
            # Drift compensator for fractional martingale
            var_integral = (t_curr ** (2.0 * self.hurst_h)) / (2.0 * self.hurst_h)
            v_paths[:, i] = self.v0 * np.exp(self.eta * conv - 0.5 * (self.eta**2) * var_integral)

        return v_paths

    @staticmethod
    def estimate_hurst_exponent(log_prices: np.ndarray, max_lag: int = 30) -> float:
        """
        Estimates Hurst exponent H via scaling of q=2 structure function / variogram:
        E[|X_{t+tau} - X_t|^2] ~ tau^{2H}
        """
        lags = np.arange(1, max_lag + 1)
        variances = []
        for lag in lags:
            diffs = log_prices[lag:] - log_prices[:-lag]
            variances.append(float(np.mean(diffs ** 2)))

        log_lags = np.log(lags)
        log_vars = np.log(variances)
        slope, _ = np.polyfit(log_lags, log_vars, 1)
        h = float(slope / 2.0)
        return max(0.01, min(0.99, h))


@dataclass
class HjbOptimalExecutionSolver:
    """
    Solves Hamilton-Jacobi-Bellman PDE for optimal liquidation:
    Value function V(q, t) represents optimal execution cost for inventory q at time t.
    max_v [ -v * (S_0 - gamma * v) - lambda_risk * sigma^2 * q^2 + dV/dt - v * dV/dq ] = 0
    Yields optimal trading speed: v*(q, t) = - (1 / 2*gamma) * dV/dq
    """
    initial_inventory: float = 1000.0
    liquidation_time: float = 1.0 # Days or hours
    volatility: float = 0.02
    inventory_risk_penalty: float = 1e-4
    temporary_impact_gamma: float = 5e-5

    def solve_optimal_trajectory(self, n_steps: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Closed-form analytical verification of HJB solution (Almgren-Chriss / Cartea-Jaimungal):
        q(t) = Q_0 * sinh(kappa * (T - t)) / sinh(kappa * T)
        Returns: (t_grid, inventory_q, trading_speed_v)
        """
        t_grid = np.linspace(0.0, self.liquidation_time, n_steps + 1)
        t_horizon = self.liquidation_time
        
        # Characteristic timescale kappa
        kappa = math.sqrt((self.inventory_risk_penalty * (self.volatility ** 2)) / self.temporary_impact_gamma)
        
        sinh_kt = math.sinh(kappa * t_horizon)
        if sinh_kt == 0:
            inventory = self.initial_inventory * (1.0 - t_grid / t_horizon)
            speed = np.full_like(t_grid, self.initial_inventory / t_horizon)
        else:
            inventory = self.initial_inventory * np.sinh(kappa * (t_horizon - t_grid)) / sinh_kt
            speed = self.initial_inventory * kappa * np.cosh(kappa * (t_horizon - t_grid)) / sinh_kt

        return t_grid, inventory, speed
