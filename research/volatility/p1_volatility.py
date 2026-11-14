"""
P1 — Volatility Forecasting Engine
GARCH(1,1), Rolling RV, Gradient Boosting, Hybrid Models, and Walk-Forward Evaluation.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class GARCHParams:
    omega: float
    alpha: float
    beta: float
    unconditional_vol: float
    log_likelihood: float
    converged: bool


class VolatilityResearchEngine:
    """Quantitative research engine for multi-horizon volatility modeling."""

    def __init__(self, trading_days: int = 365):
        self.trading_days = trading_days

    @staticmethod
    def calculate_realized_volatility(returns: pd.Series, window: int, trading_days: int = 365) -> pd.Series:
        """Calculates rolling annualized realized volatility from log returns."""
        return returns.rolling(window=window).std() * np.sqrt(trading_days)

    @staticmethod
    def fit_garch11(returns: np.ndarray, trading_days: int = 365) -> GARCHParams:
        """Fits GARCH(1,1) model using Maximum Likelihood Estimation (MLE)."""
        rets = np.asarray(returns, dtype=np.float64)
        n = len(rets)
        if n < 30:
            raise ValueError("Insufficient sample size for GARCH(1,1) estimation")

        sample_var = np.var(rets, ddof=1)
        if sample_var <= 0:
            sample_var = 1e-6

        # Objective: Negative Log-Likelihood under Gaussian innovations
        def neg_log_likelihood(params: np.ndarray) -> float:
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 1.0:
                return 1e10

            h = np.zeros(n, dtype=np.float64)
            h[0] = sample_var
            for t in range(1, n):
                h[t] = omega + alpha * (rets[t - 1] ** 2) + beta * h[t - 1]
                if h[t] <= 0:
                    return 1e10

            ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + (rets ** 2) / h)
            return -ll if np.isfinite(ll) else 1e10

        init_params = np.array([sample_var * 0.05, 0.10, 0.85])
        bounds = [(1e-8, None), (1e-6, 0.999), (1e-6, 0.999)]

        res = minimize(
            neg_log_likelihood,
            init_params,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-8}
        )

        omega, alpha, beta = res.x
        denom = 1.0 - (alpha + beta)
        uncond_var = (omega / denom) if denom > 1e-6 else sample_var
        uncond_vol = np.sqrt(uncond_var * trading_days)

        return GARCHParams(
            omega=float(omega),
            alpha=float(alpha),
            beta=float(beta),
            unconditional_vol=float(uncond_vol),
            log_likelihood=float(-res.fun),
            converged=bool(res.success)
        )

    @staticmethod
    def forecast_garch11_horizon(params: GARCHParams, current_res_sq: float, current_var: float, horizon: int) -> float:
        """Forecasts multi-step cumulative variance using GARCH(1,1) analytical recursion."""
        omega, alpha, beta = params.omega, params.alpha, params.beta
        persist = alpha + beta
        if persist >= 1.0:
            persist = 0.999

        uncond_var = omega / (1.0 - persist) if (1.0 - persist) > 1e-6 else current_var
        h_1 = omega + alpha * current_res_sq + beta * current_var

        # Cumulative expectation over horizon steps
        total_forecast_var = 0.0
        h_t = h_1
        for _ in range(horizon):
            total_forecast_var += h_t
            h_t = uncond_var + (persist) * (h_t - uncond_var)

        return total_forecast_var / horizon

    def fit_evaluate_walk_forward(
        self,
        returns: pd.Series,
        train_window: int = 252,
        forecast_horizon: int = 5
    ) -> Dict[str, float]:
        """Walk-forward evaluation comparing Historical RV, GARCH(1,1), and Gradient Boosting."""
        rets = returns.values
        n = len(rets)
        if n < train_window + forecast_horizon + 20:
            raise ValueError("Time series too short for walk-forward evaluation")

        actuals = []
        hist_rv_preds = []
        garch_preds = []
        ml_preds = []

        # Train ML regressor on lag features
        df_feat = pd.DataFrame({
            "ret": returns,
            "ret_sq": returns ** 2,
            "rv5": returns.rolling(5).std() * np.sqrt(self.trading_days),
            "rv21": returns.rolling(21).std() * np.sqrt(self.trading_days),
        }).dropna()

        # Target: Forward 5-day realized volatility
        forward_target = returns.rolling(forecast_horizon).std().shift(-forecast_horizon) * np.sqrt(self.trading_days)
        df_feat["target"] = forward_target
        df_feat = df_feat.dropna()

        eval_indices = range(train_window, len(df_feat) - forecast_horizon, max(1, (len(df_feat) - train_window) // 25))

        for idx in eval_indices:
            train_data = df_feat.iloc[idx - train_window : idx]
            test_row = df_feat.iloc[idx]

            # 1. Historical RV baseline (trailing 21-day)
            hist_pred = test_row["rv21"]

            # 2. GARCH(1,1) fit on rolling window
            garch_fit = self.fit_garch11(train_data["ret"].values, trading_days=self.trading_days)
            last_res_sq = train_data["ret_sq"].iloc[-1]
            last_var = (train_data["rv5"].iloc[-1] / np.sqrt(self.trading_days)) ** 2
            garch_daily_var = self.forecast_garch11_horizon(garch_fit, last_res_sq, last_var, forecast_horizon)
            garch_pred = np.sqrt(garch_daily_var * self.trading_days)

            # 3. Machine Learning (Gradient Boosting)
            feature_cols = ["ret_sq", "rv5", "rv21"]
            X_train = train_data[feature_cols]
            y_train = train_data["target"]
            gbr = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            gbr.fit(X_train, y_train)
            ml_pred = float(gbr.predict(pd.DataFrame([test_row[feature_cols]]))[0])

            actual = test_row["target"]
            actuals.append(actual)
            hist_rv_preds.append(hist_pred)
            garch_preds.append(garch_pred)
            ml_preds.append(ml_pred)

        y_true = np.array(actuals)
        p_hist = np.array(hist_rv_preds)
        p_garch = np.array(garch_preds)
        p_ml = np.array(ml_preds)

        return {
            "hist_rv_rmse": float(np.sqrt(np.mean((y_true - p_hist) ** 2))),
            "hist_rv_mae": float(np.mean(np.abs(y_true - p_hist))),
            "garch_rmse": float(np.sqrt(np.mean((y_true - p_garch) ** 2))),
            "garch_mae": float(np.mean(np.abs(y_true - p_garch))),
            "ml_rmse": float(np.sqrt(np.mean((y_true - p_ml) ** 2))),
            "ml_mae": float(np.mean(np.abs(y_true - p_ml))),
            "sample_count": len(y_true),
        }
