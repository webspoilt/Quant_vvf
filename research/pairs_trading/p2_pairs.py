"""
P2 — Pairs Trading and Cointegration Engine
OLS Hedge Ratios, Engle-Granger Cointegration, ADF Testing, Rolling Z-Scores,
Ornstein-Uhlenbeck Half-Life, and Structural Break Diagnostics.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


@dataclass
class CointegrationResult:
    is_cointegrated: bool
    adf_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    hedge_ratio: float
    intercept: float
    half_life_days: float
    mean_spread: float
    std_spread: float


class PairsTradingEngine:
    """Rigorous statistical arbitrage and cointegration testing engine."""

    @staticmethod
    def calculate_ols_hedge_ratio(series_y: pd.Series, series_x: pd.Series) -> Tuple[float, float, pd.Series]:
        """Calculates Ordinary Least Squares (OLS) hedge ratio: y = alpha + beta * x + epsilon."""
        y = np.asarray(series_y, dtype=np.float64)
        x = np.asarray(series_x, dtype=np.float64)
        n = len(y)
        if n != len(x) or n < 20:
            raise ValueError("Series lengths must match and exceed 20 observations")

        X = np.vstack([x, np.ones(n)]).T
        beta, alpha = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - (beta * x + alpha)
        return float(beta), float(alpha), pd.Series(residuals, index=series_y.index)

    @classmethod
    def test_engle_granger_cointegration(
        cls,
        series_y: pd.Series,
        series_x: pd.Series,
        p_value_threshold: float = 0.05
    ) -> CointegrationResult:
        """Performs two-step Engle-Granger cointegration test with ADF and OU half-life."""
        beta, alpha, spread = cls.calculate_ols_hedge_ratio(series_y, series_x)

        # Augmented Dickey-Fuller test on spread residuals
        adf_res = adfuller(spread.values, autolag="AIC")
        stat = float(adf_res[0])
        p_val = float(adf_res[1])
        crit_vals = {k: float(v) for k, v in adf_res[4].items()}

        # Ornstein-Uhlenbeck Half-Life estimation: delta_s_t = -lambda * (s_{t-1} - mu) + e
        spread_diff = spread.diff().dropna().values
        spread_lag = spread.shift(1).dropna().values

        X_lag = np.vstack([spread_lag, np.ones(len(spread_lag))]).T
        theta = np.linalg.lstsq(X_lag, spread_diff, rcond=None)[0][0]

        # Half-life = -ln(2) / lambda (where lambda = ln(1 + theta))
        if theta < 0:
            half_life = -np.log(2.0) / theta
        else:
            half_life = np.inf

        return CointegrationResult(
            is_cointegrated=bool(p_val <= p_value_threshold),
            adf_statistic=stat,
            p_value=p_val,
            critical_values=crit_vals,
            hedge_ratio=beta,
            intercept=alpha,
            half_life_days=float(half_life),
            mean_spread=float(np.mean(spread)),
            std_spread=float(np.std(spread, ddof=1)),
        )

    @staticmethod
    def calculate_rolling_z_score(spread: pd.Series, window: int = 60) -> pd.Series:
        """Calculates rolling Z-score: (spread - rolling_mean) / rolling_std."""
        roll_mean = spread.rolling(window=window).mean()
        roll_std = spread.rolling(window=window).std()
        return (spread - roll_mean) / roll_std
