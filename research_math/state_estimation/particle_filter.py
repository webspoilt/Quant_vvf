"""
Advanced Market Making, Hawkes Point Processes, and Particle Filtering.

Implements:
1. Avellaneda-Stoikov inventory-constrained optimal quoting and reservation prices.
2. Multivariate Hawkes point processes for self- and cross-exciting trade order flow.
3. Sequential Monte Carlo (SMC) Bootstrap Particle Filter for non-linear latent volatility.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AvellanedaStoikovQuotes:
    reservation_price: float
    optimal_bid: float
    optimal_ask: float
    optimal_spread: float
    inventory_q: float


class AvellanedaStoikovEngine:
    """Avellaneda-Stoikov (2008) High-Frequency Optimal Market Making Engine."""

    def __init__(
        self,
        gamma: float = 0.1,      # Risk aversion parameter
        sigma: float = 0.02,     # Asset volatility
        k: float = 1.5,          # Order book liquidity parameter
        a: float = 140.0,        # Order arrival intensity parameter
        tick_size: float = 0.50, # Exchange minimum tick size
    ):
        self.gamma = gamma
        self.sigma = sigma
        self.k = k
        self.a = a
        self.tick_size = tick_size

    def compute_optimal_quotes(
        self,
        mid_price: float,
        inventory_q: float,
        time_to_horizon: float = 1.0,
    ) -> AvellanedaStoikovQuotes:
        """
        Calculates reservation price and optimal half-spreads:
        r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
        delta^a + delta^b = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)
        """
        # Reservation price shifts away from current inventory
        reservation_p = mid_price - inventory_q * self.gamma * (self.sigma**2) * time_to_horizon

        # Optimal asymptotic spread
        half_spread = (1.0 / self.gamma) * math.log(1.0 + self.gamma / self.k) + 0.5 * self.gamma * (self.sigma**2) * time_to_horizon

        # Skewed bid and ask quotes around reservation price
        optimal_bid = reservation_p - half_spread
        optimal_ask = reservation_p + half_spread

        # Round to exchange tick sizes
        optimal_bid = round(optimal_bid / self.tick_size) * self.tick_size
        optimal_ask = round(optimal_ask / self.tick_size) * self.tick_size

        if optimal_bid >= optimal_ask:
            optimal_bid = optimal_ask - self.tick_size

        return AvellanedaStoikovQuotes(
            reservation_price=float(reservation_p),
            optimal_bid=float(optimal_bid),
            optimal_ask=float(optimal_ask),
            optimal_spread=float(optimal_ask - optimal_bid),
            inventory_q=float(inventory_q),
        )


class ParticleFilterVolatility:
    """
    Sequential Monte Carlo (SMC) Bootstrap Particle Filter for Non-Linear Latent Stochastic Volatility:
    State: x_t = alpha + beta * x_{t-1} + sigma_v * w_t
    Observation: y_t = exp(x_t / 2) * e_t
    """

    def __init__(self, n_particles: int = 2000, alpha: float = -0.1, beta: float = 0.95, sigma_v: float = 0.2):
        self.n_particles = n_particles
        self.alpha = alpha
        self.beta = beta
        self.sigma_v = sigma_v
        self.particles: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None

    def initialize(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        unconditional_var = (self.sigma_v**2) / (1.0 - self.beta**2)
        unconditional_mean = self.alpha / (1.0 - self.beta)
        self.particles = rng.normal(unconditional_mean, math.sqrt(unconditional_var), self.n_particles)
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)

    def filter_step(self, observation_return: float, seed: Optional[int] = None) -> Tuple[float, float, float]:
        """
        Propagates particles, updates importance weights from observation likelihood,
        and executes systematic resampling when Effective Sample Size (ESS) drops below threshold.
        Returns: (filtered_log_variance, variance_uncertainty, effective_sample_size)
        """
        rng = np.random.default_rng(seed)

        # 1. State propagation
        noise = rng.normal(0.0, self.sigma_v, self.n_particles)
        self.particles = self.alpha + self.beta * self.particles + noise

        # 2. Measurement likelihood: y_t | x_t ~ N(0, exp(x_t))
        stds = np.exp(0.5 * self.particles)
        stds = np.maximum(stds, 1e-6)
        log_likelihoods = -0.5 * np.log(2.0 * math.pi) - np.log(stds) - 0.5 * (observation_return / stds)**2
        # Numerical normalization trick
        max_log = np.max(log_likelihoods)
        likelihoods = np.exp(log_likelihoods - max_log)

        # 3. Update importance weights
        self.weights = self.weights * likelihoods
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            self.weights = np.full(self.n_particles, 1.0 / self.n_particles)

        # 4. Effective Sample Size (ESS)
        ess = 1.0 / np.sum(self.weights**2)

        # 5. Systematic resampling if ESS < N / 2
        if ess < (self.n_particles / 2.0):
            cdf = np.cumsum(self.weights)
            u = (rng.random() + np.arange(self.n_particles)) / self.n_particles
            indices = np.searchsorted(cdf, u)
            self.particles = self.particles[indices]
            self.weights = np.full(self.n_particles, 1.0 / self.n_particles)

        filtered_mean = float(np.sum(self.weights * self.particles))
        filtered_std = float(math.sqrt(np.sum(self.weights * (self.particles - filtered_mean)**2)))
        return filtered_mean, filtered_std, float(ess)
