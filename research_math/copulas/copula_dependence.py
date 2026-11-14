"""
QUANT_VVF — Copulas & Non-Linear Dependence Modeling.
References:
- Nelsen (2006): An Introduction to Copulas. Springer.
- Sklar (1959): Fonctions de répartition à n dimensions et leurs marges.
"""

from __future__ import annotations
import math
from typing import Dict, Any, Tuple
import numpy as np
from scipy.stats import norm, t


class ClaytonCopula:
    """
    Clayton Archimedean Copula:
    C(u, v; theta) = max(0, u^{-theta} + v^{-theta} - 1)^{-1 / theta}, theta > 0
    Features strong lower tail dependence: lambda_L = 2^{-1 / theta}, lambda_U = 0.
    Ideal for crash co-movement (e.g. BTC spot and derivative liquidity evaporation).
    """

    def __init__(self, theta: float = 2.0) -> None:
        self.theta = max(1e-4, float(theta))

    @property
    def lower_tail_dependence(self) -> float:
        return float(2.0 ** (-1.0 / self.theta))

    def cdf(self, u: float | np.ndarray, v: float | np.ndarray) -> float | np.ndarray:
        u_cl = np.clip(u, 1e-6, 1.0)
        v_cl = np.clip(v, 1e-6, 1.0)
        val = np.maximum(0.0, (u_cl ** (-self.theta)) + (v_cl ** (-self.theta)) - 1.0)
        return val ** (-1.0 / self.theta)

    def sample(self, n_samples: int) -> np.ndarray:
        """Sample bivariate uniform pairs (u, v) using conditional inversion."""
        u = np.random.uniform(1e-6, 1.0, size=n_samples)
        w = np.random.uniform(1e-6, 1.0, size=n_samples)
        # v = u * (w^{-theta / (1 + theta)} - 1 + u^theta)^{-1 / theta}
        term = (w ** (-self.theta / (1.0 + self.theta)) - 1.0) * (u ** self.theta) + 1.0
        v = np.clip(term ** (-1.0 / self.theta), 1e-6, 1.0)
        return np.column_stack([u, v])


class GumbelCopula:
    """
    Gumbel Archimedean Copula:
    C(u, v; theta) = exp( - ( (-ln u)^theta + (-ln v)^theta )^{1 / theta} ), theta >= 1
    Features strong upper tail dependence: lambda_U = 2 - 2^{1 / theta}, lambda_L = 0.
    """

    def __init__(self, theta: float = 2.0) -> None:
        self.theta = max(1.0, float(theta))

    @property
    def upper_tail_dependence(self) -> float:
        return float(2.0 - (2.0 ** (1.0 / self.theta)))

    def cdf(self, u: float | np.ndarray, v: float | np.ndarray) -> float | np.ndarray:
        u_cl = np.clip(u, 1e-6, 1.0)
        v_cl = np.clip(v, 1e-6, 1.0)
        term = ((-np.log(u_cl)) ** self.theta) + ((-np.log(v_cl)) ** self.theta)
        return np.exp(-(term ** (1.0 / self.theta)))
