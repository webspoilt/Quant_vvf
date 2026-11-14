"""
QUANT_VVF — Dynamic Conditional Correlation (DCC-GARCH).
Reference:
- Engle (2002): Dynamic Conditional Correlation: A Simple Class of Multivariate Generalized Autoregressive Conditional Heteroskedasticity Models.
  Journal of Business & Economic Statistics, 20(3), 339-350.
  DOI: 10.1198/073500102288618487
"""

from __future__ import annotations
import math
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from scipy.optimize import minimize
from research_math.volatility.advanced_volatility import GarchModel


class DccGarchModel:
    """
    Engle (2002) Dynamic Conditional Correlation (DCC-GARCH):
    Stage 1: Univariate GARCH(1,1) per asset i:
             sigma_{i,t}^2 = omega_i + alpha_i * r_{i,t-1}^2 + beta_i * sigma_{i,t-1}^2
             standardized residuals: z_{i,t} = r_{i,t} / sigma_{i,t}
    Stage 2: Quasi-correlation evolution:
             Q_t = (1 - a - b) * \\bar{Q} + a * (z_{t-1} z_{t-1}^T) + b * Q_{t-1}
             R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}
    Constraints: a >= 0, b >= 0, a + b < 1.0.
    """

    def __init__(self) -> None:
        self.univariate_models: List[GarchModel] = []
        self.a: float = 0.0
        self.b: float = 0.0
        self.q_bar: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, returns_matrix: np.ndarray) -> Dict[str, Any]:
        """
        returns_matrix: shape (T, N) where T >= 50, N >= 2.
        """
        R = np.asarray(returns_matrix, dtype=np.float64)
        T, N = R.shape
        if T < 50 or N < 2:
            raise ValueError(f"DCC-GARCH requires T >= 50 and N >= 2, got ({T}, {N})")

        # Stage 1: Fit univariate GARCH models and obtain standardized residuals
        self.univariate_models = []
        std_residuals = np.empty((T, N), dtype=np.float64)
        sigmas = np.empty((T, N), dtype=np.float64)

        for i in range(N):
            garch_i = GarchModel()
            garch_i.fit(R[:, i])
            self.univariate_models.append(garch_i)

            # Compute in-sample conditional standard deviations
            sig2 = np.empty(T, dtype=np.float64)
            sig2[0] = float(np.var(R[:, i]))
            for t in range(1, T):
                sig2[t] = garch_i.omega + garch_i.alpha * (R[t - 1, i] ** 2) + garch_i.beta * sig2[t - 1]
            sigmas[:, i] = np.sqrt(np.maximum(1e-8, sig2))
            std_residuals[:, i] = R[:, i] / sigmas[:, i]

        # Stage 2: Fit DCC parameters (a, b)
        self.q_bar = np.cov(std_residuals, rowvar=False)

        x0 = np.array([0.05, 0.90], dtype=np.float64)
        bounds = [(1e-5, 0.50), (1e-5, 0.99)]

        def dcc_neg_log_lik(params: np.ndarray) -> float:
            a, b = params
            if a + b >= 0.9999 or a <= 0 or b <= 0:
                return 1e10

            Q = np.copy(self.q_bar)
            ll = 0.0

            for t in range(1, T):
                z_prev = std_residuals[t - 1 : t, :].T  # (N, 1)
                Q = (1.0 - a - b) * self.q_bar + a * (z_prev @ z_prev.T) + b * Q

                # Rescale to correlation matrix R_t
                d_inv = 1.0 / np.sqrt(np.maximum(1e-8, np.diag(Q)))
                R_t = np.outer(d_inv, d_inv) * Q

                # Likelihood term: -0.5 * (log(det(R_t)) + z_t^T R_t^{-1} z_t - z_t^T z_t)
                z_curr = std_residuals[t : t + 1, :].T
                try:
                    sign, logdet = np.linalg.slogdet(R_t)
                    if sign <= 0:
                        return 1e10
                    inv_R = np.linalg.inv(R_t)
                    quad_term = float((z_curr.T @ inv_R @ z_curr).item())
                    norm_term = float((z_curr.T @ z_curr).item())
                    ll += -0.5 * (logdet + quad_term - norm_term)
                except np.linalg.LinAlgError:
                    return 1e10

            return -float(ll) if np.isfinite(ll) else 1e10

        cons = {"type": "ineq", "fun": lambda p: 0.9999 - (p[0] + p[1])}
        res = minimize(dcc_neg_log_lik, x0, method="SLSQP", bounds=bounds, constraints=cons)

        self.a = float(res.x[0])
        self.b = float(res.x[1])
        self.is_fitted = True

        return {
            "a": self.a,
            "b": self.b,
            "persistence": self.a + self.b,
            "unconditional_corr": self.q_bar,
            "converged": bool(res.success),
        }

    def predict_next_covariance(self, recent_returns: np.ndarray) -> np.ndarray:
        """Forecast 1-step ahead conditional covariance matrix Sigma_{T+1} = D_{T+1} R_{T+1} D_{T+1}."""
        if not self.is_fitted:
            raise RuntimeError("DccGarchModel must be fitted before prediction.")

        R = np.asarray(recent_returns, dtype=np.float64)
        T, N = R.shape

        # Forecast univariate std devs
        fc_sigmas = np.empty(N, dtype=np.float64)
        for i, model in enumerate(self.univariate_models):
            fc_sigmas[i] = math.sqrt(model.forecast(R[:, i], horizon=1)[0])

        # 1-step ahead Q
        z_last = (R[-1, :] / fc_sigmas).reshape(-1, 1)
        Q_next = (1.0 - self.a - self.b) * self.q_bar + self.a * (z_last @ z_last.T) + self.b * self.q_bar
        d_inv = 1.0 / np.sqrt(np.maximum(1e-8, np.diag(Q_next)))
        R_next = np.outer(d_inv, d_inv) * Q_next

        # Covariance Sigma = D * R * D
        D = np.diag(fc_sigmas)
        return D @ R_next @ D
