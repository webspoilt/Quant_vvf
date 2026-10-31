"""Corsi (2009) HAR-RV Model, Asymmetric QLIKE Loss, and Forecasting Competitors.

References:
Corsi, F. (2009), "A simple approximate long-memory model of realized volatility"
Patton, A. J. (2011), "Data-based ranking of volatility models"
Diebold, F. X. & Mariano, R. S. (1995), "Comparing predictive accuracy"
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import statsmodels.api as sm


def qlike_loss(
    actual_var: np.ndarray | float,
    forecast_var: np.ndarray | float,
) -> np.ndarray | float:
    """Evaluate Patton (2011) asymmetric QLIKE loss function.

    QLIKE(h, h_hat) = (h / h_hat) - ln(h / h_hat) - 1
    where h is realized variance, h_hat is forecasted variance.

    Properties:
    - Minimum is 0 when h_hat == h.
    - Strictly positive when h_hat != h.
    - Robust strictly proper scoring rule for volatility estimators.
    - Asymmetric: penalizes under-forecasting more heavily than over-forecasting.
    """
    h = np.asarray(actual_var, dtype=np.float64)
    h_hat = np.asarray(forecast_var, dtype=np.float64)
    # Fail-closed (V-20): `<= 0` comparisons are False for NaN, which would
    # otherwise pass validation and emit a NaN loss into ranking logic.
    if np.any(~np.isfinite(h)) or np.any(~np.isfinite(h_hat)):
        raise ValueError("Both actual and forecasted variances must be finite numbers")
    if np.any(h <= 0.0) or np.any(h_hat <= 0.0):
        raise ValueError("Both actual and forecasted variances must be strictly positive")

    ratio = h / h_hat
    loss = ratio - np.log(ratio) - 1.0
    if np.isscalar(actual_var) and np.isscalar(forecast_var):
        return float(loss)
    return loss


@dataclass
class HARParameters:
    beta_0: float
    beta_d: float
    beta_w: float
    beta_m: float
    r_squared: float


class CorsiHAR_RV:
    """Corsi (2009) Heterogeneous Autoregressive Realized Volatility model.

    Supports both:
    1. Standard linear HAR-RV (Corsi 2009, Eq. 1):
       RV_{t+1}^{(d)} = beta_0 + beta_d * RV_t^{(d)} + beta_w * RV_t^{(w)} + beta_m * RV_t^{(m)} + eps_{t+1}
    2. Log-HAR-RV (Corsi 2009, Section 3.2):
       ln(RV_{t+1}^{(d)}) = beta_0 + beta_d * ln(RV_t^{(d)}) + beta_w * ln(RV_t^{(w)}) + beta_m * ln(RV_t^{(m)}) + eps_{t+1}
       with Jensen convexity correction RV_hat = exp(ln_pred + 0.5 * sigma_eps^2).
       Guarantees strictly positive variance forecasts and minimizes asymmetric QLIKE loss.
    """

    def __init__(self, log_transform: bool = True) -> None:
        self.log_transform = log_transform
        self.params: HARParameters | None = None
        self._beta_vec: np.ndarray | None = None
        self._sigma2_residual: float = 0.0

    @staticmethod
    def construct_features(rv_daily: np.ndarray, log_transform: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Construct HAR feature matrix [1, RV_d, RV_w, RV_m] and 1-step ahead targets.

        Requires at least 23 daily observations.
        Returns:
            (X, y) where X has shape (N - 22, 4) and y has shape (N - 22,)
        """
        arr = np.asarray(rv_daily, dtype=np.float64)
        n = len(arr)
        if n < 23:
            raise ValueError(f"Need at least 23 daily observations to construct HAR features, got {n}")
        if not np.isfinite(arr).all():
            raise ValueError("rv_daily contains non-finite values (NaN/Inf) — fail-closed")

        t_indices = np.arange(21, n - 1)
        k = len(t_indices)

        X = np.empty((k, 4), dtype=np.float64)
        y = np.empty(k, dtype=np.float64)

        for i, t in enumerate(t_indices):
            target = arr[t + 1]
            rv_d = arr[t]
            rv_w = float(np.mean(arr[t - 4 : t + 1]))
            rv_m = float(np.mean(arr[t - 21 : t + 1]))

            if log_transform:
                y[i] = np.log(max(1e-12, target))
                X[i] = [
                    1.0,
                    np.log(max(1e-12, rv_d)),
                    np.log(max(1e-12, rv_w)),
                    np.log(max(1e-12, rv_m)),
                ]
            else:
                y[i] = target
                X[i] = [1.0, rv_d, rv_w, rv_m]

        return X, y

    def fit(self, rv_daily: np.ndarray) -> CorsiHAR_RV:
        """Fit HAR-RV via OLS with Newey-West standard errors."""
        X, y = self.construct_features(rv_daily, log_transform=self.log_transform)
        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

        b0, bd, bw, bm = model.params
        self.params = HARParameters(
            beta_0=float(b0),
            beta_d=float(bd),
            beta_w=float(bw),
            beta_m=float(bm),
            r_squared=float(model.rsquared),
        )
        self._beta_vec = np.array([b0, bd, bw, bm], dtype=np.float64)
        self._sigma2_residual = float(np.var(model.resid, ddof=4))
        return self

    def predict_one_step(self, rv_history: np.ndarray) -> float:
        """Predict 1-step ahead RV given the most recent 22+ observations."""
        if self._beta_vec is None:
            raise RuntimeError("Model must be fitted before predict_one_step")
        arr = np.asarray(rv_history, dtype=np.float64)
        if len(arr) < 22:
            raise ValueError(f"Need at least 22 days of history, got {len(arr)}")
        if not np.isfinite(arr).all():
            raise ValueError("rv_history contains non-finite values (NaN/Inf) — fail-closed")

        rv_d = arr[-1]
        rv_w = float(np.mean(arr[-5:]))
        rv_m = float(np.mean(arr[-22:]))

        if self.log_transform:
            feat = np.array([
                1.0,
                np.log(max(1e-12, rv_d)),
                np.log(max(1e-12, rv_w)),
                np.log(max(1e-12, rv_m)),
            ], dtype=np.float64)
            log_pred = float(np.dot(self._beta_vec, feat))
            # Unbiased expectation with Jensen correction: exp(mu + 0.5 * sigma^2)
            pred = float(np.exp(log_pred + 0.5 * self._sigma2_residual))
        else:
            feat = np.array([1.0, rv_d, rv_w, rv_m], dtype=np.float64)
            pred = float(np.dot(self._beta_vec, feat))

        return max(1e-6, pred)


class GARCH_EWMA_RV:
    """RiskMetrics Exponentially Weighted Moving Average (EWMA) RV forecaster."""

    def __init__(self, decay_lambda: float = 0.94) -> None:
        if not (math.isfinite(decay_lambda) and 0.0 < decay_lambda < 1.0):
            raise ValueError(f"decay_lambda must be in (0, 1), got {decay_lambda!r}")
        self.decay_lambda = decay_lambda

    def predict_one_step(self, rv_history: np.ndarray) -> float:
        arr = np.asarray(rv_history, dtype=np.float64)
        if len(arr) == 0:
            raise ValueError("History cannot be empty")
        if not np.isfinite(arr).all():
            raise ValueError("rv_history contains non-finite values (NaN/Inf) — fail-closed")
        # Compute EWMA with decay lambda
        weights = (1.0 - self.decay_lambda) * (self.decay_lambda ** np.arange(len(arr) - 1, -1, -1))
        weights /= np.sum(weights)
        pred = float(np.sum(weights * arr))
        return max(1e-6, pred)


class NaiveRandomWalkRV:
    """Naive Random Walk benchmark: RV_{t+1} = RV_t."""

    def predict_one_step(self, rv_history: np.ndarray) -> float:
        arr = np.asarray(rv_history, dtype=np.float64)
        if len(arr) == 0:
            raise ValueError("History cannot be empty")
        if not np.isfinite(arr).all():
            raise ValueError("rv_history contains non-finite values (NaN/Inf) — fail-closed")
        return max(1e-6, float(arr[-1]))


class SimpleMovingAverageRV:
    """Simple 20-day Moving Average benchmark."""

    def __init__(self, window: int = 20) -> None:
        if not isinstance(window, int) or window <= 0:
            raise ValueError(f"window must be a positive integer, got {window!r}")
        self.window = window

    def predict_one_step(self, rv_history: np.ndarray) -> float:
        arr = np.asarray(rv_history, dtype=np.float64)
        if arr.size and not np.isfinite(arr).all():
            raise ValueError("rv_history contains non-finite values (NaN/Inf) — fail-closed")
        if len(arr) < self.window:
            return max(1e-6, float(np.mean(arr)))
        return max(1e-6, float(np.mean(arr[-self.window :])))


def diebold_mariano_test(
    loss_benchmark: np.ndarray,
    loss_candidate: np.ndarray,
    max_lags: int = 5,
) -> tuple[float, float]:
    """Diebold-Mariano test of equal predictive accuracy with Newey-West HAC standard errors.

    Loss differential:
        d_t = loss_candidate_t - loss_benchmark_t
    If d_t > 0, benchmark has LOWER loss (benchmark outperforms candidate).

    Returns:
        (dm_statistic, p_value)
    """
    loss_bench = np.asarray(loss_benchmark, dtype=np.float64)
    loss_cand = np.asarray(loss_candidate, dtype=np.float64)
    if len(loss_bench) != len(loss_cand):
        raise ValueError("Loss arrays must have the same length")

    d = loss_cand - loss_bench
    n = len(d)
    if n < 5:
        raise ValueError("At least 5 out-of-sample predictions required for DM test")

    # Fit constant OLS on d to obtain Newey-West HAC t-statistic
    X = np.ones((n, 1), dtype=np.float64)
    model = sm.OLS(d, X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
    dm_stat = float(model.tvalues[0])
    p_val = float(model.pvalues[0])
    return dm_stat, p_val
