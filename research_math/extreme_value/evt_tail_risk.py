"""
QUANT_VVF — Extreme Value Theory (EVT) for Tail Risk & Crash Estimation.
References:
- Embrechts, Klüppelberg & Mikosch (1997): Modelling Extremal Events for Insurance and Finance.
- McNeil & Frey (2000): Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach.
  Journal of Empirical Finance, 7(3-4), 271-300.
"""

from __future__ import annotations
import math
from typing import Dict, Any, Tuple
import numpy as np
from scipy.optimize import minimize


class GeneralizedParetoTailEstimator:
    """
    Peaks Over Threshold (POT) approach via Generalized Pareto Distribution (GPD):
    G_{xi, sigma}(y) = 1 - (1 + xi * y / sigma)^{-1 / xi}  for xi != 0
    where y = x - u > 0 is excess over high threshold u.
    xi > 0 indicates heavy Fréchet-type tails (typical of Bitcoin returns).
    """

    def __init__(self, threshold_percentile: float = 95.0) -> None:
        self.threshold_percentile = threshold_percentile
        self.threshold: float = 0.0
        self.xi: float = 0.0  # tail shape parameter
        self.sigma: float = 0.0  # scale parameter
        self.n_total: int = 0
        self.n_excess: int = 0
        self.is_fitted: bool = False

    def fit(self, losses: np.ndarray) -> Dict[str, Any]:
        """losses: positive values representing portfolio or asset losses."""
        L = np.asarray(losses, dtype=np.float64)
        L = L[np.isfinite(L)]
        self.n_total = len(L)
        if self.n_total < 50:
            raise ValueError(f"Need at least 50 loss observations for EVT, got {self.n_total}")

        self.threshold = float(np.percentile(L, self.threshold_percentile))
        excesses = L[L > self.threshold] - self.threshold
        self.n_excess = len(excesses)

        if self.n_excess < 10:
            raise ValueError(f"Too few excesses ({self.n_excess}) over threshold {self.threshold:.4f}")

        # MLE for GPD: params [xi, sigma]
        # Initial guess via moments
        mean_ex = float(np.mean(excesses))
        var_ex = float(np.var(excesses))
        xi_0 = 0.5 * (mean_ex ** 2 / max(1e-6, var_ex) - 1.0)
        sigma_0 = 0.5 * mean_ex * (mean_ex ** 2 / max(1e-6, var_ex) + 1.0)

        x0 = np.array([max(0.01, min(0.9, xi_0)), max(1e-4, sigma_0)])
        bounds = [(1e-4, 2.0), (1e-6, None)]

        def neg_log_lik(p: np.ndarray) -> float:
            xi, sig = p[0], p[1]
            if sig <= 0 or xi <= 0:
                return 1e10

            cond = 1.0 + (xi * excesses) / sig
            if np.any(cond <= 0):
                return 1e10

            ll = -self.n_excess * math.log(sig) - (1.0 + 1.0 / xi) * np.sum(np.log(cond))
            return -float(ll) if np.isfinite(ll) else 1e10

        res = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds)
        self.xi = float(res.x[0])
        self.sigma = float(res.x[1])
        self.is_fitted = True

        return {
            "threshold": self.threshold,
            "xi_shape": self.xi,
            "sigma_scale": self.sigma,
            "num_excesses": self.n_excess,
            "tail_index": 1.0 / self.xi if self.xi > 0 else np.inf,
        }

    def estimate_var_and_es(self, confidence_level: float = 0.99) -> Tuple[float, float]:
        """
        Analytical Value-at-Risk (VaR) and Expected Shortfall (ES) under GPD:
        VaR_p = u + (sigma / xi) * [ ((n / N_u) * (1 - p))^{-xi} - 1 ]
        ES_p  = (VaR_p + sigma - xi * u) / (1 - xi)
        """
        if not self.is_fitted:
            raise RuntimeError("Estimator must be fitted before computing VaR/ES.")

        p = confidence_level
        u = self.threshold
        xi = self.xi
        sig = self.sigma
        frac = (self.n_total / self.n_excess) * (1.0 - p)

        if frac <= 0:
            var_p = u
        else:
            var_p = u + (sig / xi) * ((frac ** (-xi)) - 1.0)

        if xi >= 1.0:
            # First moment undefined
            es_p = np.inf
        else:
            es_p = (var_p + sig - xi * u) / (1.0 - xi)

        return float(var_p), float(es_p)
