"""
QUANT_VVF — Advanced Volatility Models & Forecast Comparison.
References:
- Bollerslev (1986): GARCH(1,1)
- Nelson (1991): EGARCH
- Glosten, Jagannathan & Runkle (1993): GJR-GARCH
- Corsi (2009): HAR-RV
- Patton (2011): QLIKE and forecast evaluation loss functions
"""

from __future__ import annotations
import math
from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy.optimize import minimize


class GarchModel:
    """
    Standard GARCH(1,1) model:
    sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
    Constraints: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
    """

    def __init__(self) -> None:
        self.omega: float = 0.0
        self.alpha: float = 0.0
        self.beta: float = 0.0
        self.log_likelihood: float = -np.inf
        self.is_fitted: bool = False

    def fit(self, returns: np.ndarray, disp: bool = False) -> Dict[str, Any]:
        ret = np.asarray(returns, dtype=np.float64)
        n = len(ret)
        if n < 30:
            raise ValueError(f"Insufficient sample size for GARCH fitting: {n} < 30")

        # Initial sample variance
        var_sample = float(np.var(ret))
        if var_sample <= 0:
            var_sample = 1e-6

        # Parameter vector: [omega, alpha, beta]
        # Initial guess
        x0 = np.array([var_sample * 0.05, 0.10, 0.85], dtype=np.float64)
        bounds = [(1e-8, None), (1e-6, 0.999), (1e-6, 0.999)]

        def neg_log_lik(params: np.ndarray) -> float:
            omega, alpha, beta = params
            if alpha + beta >= 0.9999 or omega <= 0 or alpha < 0 or beta < 0:
                return 1e10

            sigma2 = np.empty(n, dtype=np.float64)
            sigma2[0] = var_sample

            for t in range(1, n):
                sigma2[t] = omega + alpha * (ret[t - 1] ** 2) + beta * sigma2[t - 1]
                if sigma2[t] <= 1e-12:
                    sigma2[t] = 1e-12

            # Gaussian log-likelihood
            ll = -0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sigma2) + (ret ** 2) / sigma2)
            if not np.isfinite(ll):
                return 1e10
            return -float(ll)

        # Stationarity constraint: 1 - alpha - beta >= 1e-4
        cons = {"type": "ineq", "fun": lambda p: 0.9999 - (p[1] + p[2])}

        res = minimize(
            neg_log_lik,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 300, "ftol": 1e-8, "disp": disp},
        )

        if not res.success and not np.isfinite(res.fun):
            raise RuntimeError(f"GARCH(1,1) optimization failed: {res.message}")

        self.omega = float(res.x[0])
        self.alpha = float(res.x[1])
        self.beta = float(res.x[2])
        self.log_likelihood = -float(res.fun)
        self.is_fitted = True

        return {
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "persistence": self.alpha + self.beta,
            "unconditional_variance": self.omega / max(1e-6, (1.0 - (self.alpha + self.beta))),
            "log_likelihood": self.log_likelihood,
            "converged": bool(res.success),
        }

    def forecast(self, returns: np.ndarray, horizon: int = 1) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("GarchModel must be fitted before forecasting.")
        ret = np.asarray(returns, dtype=np.float64)
        n = len(ret)
        sigma2 = np.empty(n, dtype=np.float64)
        sigma2[0] = float(np.var(ret)) if n > 0 else 1e-4

        for t in range(1, n):
            sigma2[t] = self.omega + self.alpha * (ret[t - 1] ** 2) + self.beta * sigma2[t - 1]

        # 1-step ahead forecast from last return
        h1 = self.omega + self.alpha * (ret[-1] ** 2) + self.beta * sigma2[-1]
        forecasts = np.empty(horizon, dtype=np.float64)
        forecasts[0] = h1

        uncond_var = self.omega / max(1e-6, (1.0 - (self.alpha + self.beta)))
        persistence = self.alpha + self.beta

        for h in range(1, horizon):
            forecasts[h] = uncond_var + (persistence ** h) * (h1 - uncond_var)

        return forecasts


class EgarchModel:
    """
    Exponential GARCH (Nelson 1991):
    ln(sigma_t^2) = omega + beta * ln(sigma_{t-1}^2) + alpha * (|z_{t-1}| - sqrt(2/pi)) + gamma * z_{t-1}
    where z_{t-1} = epsilon_{t-1} / sigma_{t-1}
    """

    def __init__(self) -> None:
        self.omega: float = 0.0
        self.alpha: float = 0.0
        self.beta: float = 0.0
        self.gamma: float = 0.0
        self.is_fitted: bool = False

    def fit(self, returns: np.ndarray) -> Dict[str, Any]:
        ret = np.asarray(returns, dtype=np.float64)
        n = len(ret)
        if n < 30:
            raise ValueError(f"Insufficient sample for EGARCH: {n} < 30")

        var_sample = max(1e-6, float(np.var(ret)))
        c_norm = math.sqrt(2.0 / math.pi)

        # params: [omega, alpha, beta, gamma]
        x0 = np.array([0.05 * math.log(var_sample), 0.15, 0.90, -0.05], dtype=np.float64)
        bounds = [(-10.0, 10.0), (-2.0, 2.0), (-0.999, 0.999), (-2.0, 2.0)]

        def neg_log_lik(params: np.ndarray) -> float:
            omega, alpha, beta, gamma = params
            log_sigma2 = np.empty(n, dtype=np.float64)
            log_sigma2[0] = math.log(var_sample)

            for t in range(1, n):
                sigma_prev = math.sqrt(max(1e-12, math.exp(log_sigma2[t - 1])))
                z_prev = ret[t - 1] / sigma_prev
                log_sigma2[t] = omega + beta * log_sigma2[t - 1] + alpha * (abs(z_prev) - c_norm) + gamma * z_prev
                if log_sigma2[t] > 20.0:
                    log_sigma2[t] = 20.0
                elif log_sigma2[t] < -20.0:
                    log_sigma2[t] = -20.0

            sigma2 = np.exp(log_sigma2)
            ll = -0.5 * np.sum(np.log(2.0 * np.pi) + log_sigma2 + (ret ** 2) / sigma2)
            return -float(ll) if np.isfinite(ll) else 1e10

        res = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300})
        self.omega, self.alpha, self.beta, self.gamma = res.x
        self.is_fitted = True
        return {
            "omega": float(self.omega),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "gamma": float(self.gamma),
            "leverage_effect": float(self.gamma),
        }

    def forecast(self, returns: np.ndarray, horizon: int = 1) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("EgarchModel must be fitted before forecasting.")
        ret = np.asarray(returns, dtype=np.float64)
        n = len(ret)
        var_sample = max(1e-6, float(np.var(ret)))
        c_norm = math.sqrt(2.0 / math.pi)

        log_sigma2 = np.empty(n, dtype=np.float64)
        log_sigma2[0] = math.log(var_sample)
        for t in range(1, n):
            sigma_prev = math.sqrt(max(1e-12, math.exp(log_sigma2[t - 1])))
            z_prev = ret[t - 1] / sigma_prev
            log_sigma2[t] = self.omega + self.beta * log_sigma2[t - 1] + self.alpha * (abs(z_prev) - c_norm) + self.gamma * z_prev

        sigma_last = math.sqrt(max(1e-12, math.exp(log_sigma2[-1])))
        z_last = ret[-1] / sigma_last
        h1_log = self.omega + self.beta * log_sigma2[-1] + self.alpha * (abs(z_last) - c_norm) + self.gamma * z_last

        forecasts = np.empty(horizon, dtype=np.float64)
        forecasts[0] = math.exp(h1_log)
        curr_log = h1_log
        for h in range(1, horizon):
            curr_log = self.omega + self.beta * curr_log
            forecasts[h] = math.exp(curr_log)

        return forecasts


class GjrGarchModel:
    """
    GJR-GARCH (Glosten, Jagannathan, Runkle 1993):
    sigma_t^2 = omega + (alpha + gamma * I_{t-1}) * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
    where I_{t-1} = 1 if epsilon_{t-1} < 0 else 0
    """

    def __init__(self) -> None:
        self.omega: float = 0.0
        self.alpha: float = 0.0
        self.gamma: float = 0.0
        self.beta: float = 0.0
        self.is_fitted: bool = False

    def fit(self, returns: np.ndarray) -> Dict[str, Any]:
        ret = np.asarray(returns, dtype=np.float64)
        n = len(ret)
        if n < 30:
            raise ValueError(f"Insufficient sample for GJR-GARCH: {n} < 30")

        var_sample = max(1e-6, float(np.var(ret)))
        x0 = np.array([var_sample * 0.05, 0.05, 0.10, 0.80], dtype=np.float64)
        bounds = [(1e-8, None), (1e-6, 0.99), (0.0, 0.99), (1e-6, 0.99)]

        def neg_log_lik(params: np.ndarray) -> float:
            omega, alpha, gamma, beta = params
            if alpha + 0.5 * gamma + beta >= 0.9999 or omega <= 0:
                return 1e10

            sigma2 = np.empty(n, dtype=np.float64)
            sigma2[0] = var_sample
            for t in range(1, n):
                eps2 = ret[t - 1] ** 2
                indicator = 1.0 if ret[t - 1] < 0.0 else 0.0
                sigma2[t] = omega + (alpha + gamma * indicator) * eps2 + beta * sigma2[t - 1]
                if sigma2[t] <= 1e-12:
                    sigma2[t] = 1e-12

            ll = -0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sigma2) + (ret ** 2) / sigma2)
            return -float(ll) if np.isfinite(ll) else 1e10

        cons = {"type": "ineq", "fun": lambda p: 0.9999 - (p[1] + 0.5 * p[2] + p[3])}
        res = minimize(neg_log_lik, x0, method="SLSQP", bounds=bounds, constraints=cons)

        self.omega, self.alpha, self.gamma, self.beta = res.x
        self.is_fitted = True
        return {
            "omega": float(self.omega),
            "alpha": float(self.alpha),
            "gamma": float(self.gamma),
            "beta": float(self.beta),
            "asymmetry": float(self.gamma),
        }

    def forecast(self, returns: np.ndarray, horizon: int = 1) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("GjrGarchModel must be fitted before forecasting.")
        ret = np.asarray(returns, dtype=np.float64)
        n = len(ret)
        var_sample = max(1e-6, float(np.var(ret)))

        sigma2 = np.empty(n, dtype=np.float64)
        sigma2[0] = var_sample
        for t in range(1, n):
            indicator = 1.0 if ret[t - 1] < 0.0 else 0.0
            sigma2[t] = self.omega + (self.alpha + self.gamma * indicator) * (ret[t - 1] ** 2) + self.beta * sigma2[t - 1]

        indicator_last = 1.0 if ret[-1] < 0.0 else 0.0
        h1 = self.omega + (self.alpha + self.gamma * indicator_last) * (ret[-1] ** 2) + self.beta * sigma2[-1]
        forecasts = np.empty(horizon, dtype=np.float64)
        forecasts[0] = h1

        persistence = self.alpha + 0.5 * self.gamma + self.beta
        uncond = self.omega / max(1e-6, (1.0 - persistence))
        for h in range(1, horizon):
            forecasts[h] = uncond + (persistence ** h) * (h1 - uncond)

        return forecasts


class HarRvModel:
    """
    Heterogeneous Autoregressive Realized Volatility Model (Corsi 2009):
    RV_{t+1d} = c + beta_d * RV_t^{(d)} + beta_w * RV_t^{(w)} + beta_m * RV_t^{(m)} + epsilon
    where components are daily, weekly (5/7 days), and monthly (22/30 days) rolling averages.
    """

    def __init__(self, weekly_window: int = 7, monthly_window: int = 30) -> None:
        self.weekly_window = weekly_window
        self.monthly_window = monthly_window
        self.c: float = 0.0
        self.beta_d: float = 0.0
        self.beta_w: float = 0.0
        self.beta_m: float = 0.0
        self.is_fitted: bool = False

    def fit(self, realized_variances: np.ndarray) -> Dict[str, Any]:
        rv = np.asarray(realized_variances, dtype=np.float64)
        n = len(rv)
        m = self.monthly_window
        if n <= m + 1:
            raise ValueError(f"Insufficient history for HAR-RV: {n} <= {m + 1}")

        # Compute rolling components
        rv_d = rv[m - 1 : -1]  # lagged daily
        rv_w = np.array([np.mean(rv[i - self.weekly_window : i]) for i in range(m, n)])
        rv_m = np.array([np.mean(rv[i - m : i]) for i in range(m, n)])
        y = rv[m:]  # target next day

        X = np.column_stack([np.ones_like(rv_d), rv_d, rv_w, rv_m])

        # Ordinary Least Squares
        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        self.c, self.beta_d, self.beta_w, self.beta_m = beta
        self.is_fitted = True

        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        ss_res = float(np.sum((y - X @ beta) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "c": float(self.c),
            "beta_d": float(self.beta_d),
            "beta_w": float(self.beta_w),
            "beta_m": float(self.beta_m),
            "r_squared": float(r2),
        }

    def predict(self, recent_rv: np.ndarray) -> float:
        if not self.is_fitted:
            raise RuntimeError("HarRvModel must be fitted before predict.")
        rv = np.asarray(recent_rv, dtype=np.float64)
        if len(rv) < self.monthly_window:
            raise ValueError(f"Need at least {self.monthly_window} periods of RV for prediction.")

        rv_d = float(rv[-1])
        rv_w = float(np.mean(rv[-self.weekly_window :]))
        rv_m = float(np.mean(rv[-self.monthly_window :]))

        pred = self.c + self.beta_d * rv_d + self.beta_w * rv_w + self.beta_m * rv_m
        return float(max(1e-8, pred))


def calculate_qlike(actual_variance: np.ndarray, forecast_variance: np.ndarray) -> float:
    """
    Patton (2011) Quasi-Likelihood (QLIKE) loss function:
    QLIKE = (actual / forecast) - log(actual / forecast) - 1
    Robust to microstructure noise in proxy volatility.
    """
    h = np.asarray(forecast_variance, dtype=np.float64)
    sigma2 = np.asarray(actual_variance, dtype=np.float64)
    ratio = np.maximum(1e-12, sigma2) / np.maximum(1e-12, h)
    qlike = ratio - np.log(ratio) - 1.0
    return float(np.mean(qlike))


def calculate_forecast_metrics(actual: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
    """Calculate standard forecast evaluation metrics: QLIKE, MAE, RMSE."""
    act = np.asarray(actual, dtype=np.float64)
    fc = np.asarray(forecast, dtype=np.float64)
    err = act - fc
    return {
        "qlike": calculate_qlike(act, fc),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }
