"""Strategy Layer: Target Greek Profile Generator from (IV - RV) Variance Spread.

Non-Negotiable Core Rule 2 (Instruments Over Weights):
Never use standard Markowitz/CVaR solvers to assign raw capital weights to options legs.
The strategy layer outputs a TargetGreekProfile (Delta bounds, target vega, max gamma),
and the execution layer solves for discrete strikes/expiries under margin bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math
import numpy as np

from libs.core.har_rv import CorsiHAR_RV
from libs.core.svi_surface import SVISurface
from risk.tail_triggers import RiskVetoDecision


@dataclass(frozen=True)
class TargetGreekProfile:
    """Target Greek exposure emitted by the strategy layer."""
    target_vega_btc: float          # Negative for short volatility harvesting
    min_delta_btc: float            # Delta lower boundary (e.g. -0.15)
    max_delta_btc: float            # Delta upper boundary (e.g. +0.15)
    max_gamma: float                # Gamma ceiling to avoid tail convexity acceleration
    min_dte_days: float             # Expiry window lower bound
    max_dte_days: float             # Expiry window upper bound
    max_margin_utilization: float   # Margin utilization limit
    iv_atm: float                   # Current SVI ATM IV
    rv_forecast: float              # Current HAR-RV forecasted volatility
    variance_spread: float          # IV^2 - RV^2
    is_active: bool                 # False if vetoed or spread <= hurdle
    reason: str                     # Diagnostic status note


class VRPProfileGenerator:
    """Computes VRP signal and translates it into an explicit TargetGreekProfile."""

    def __init__(
        self,
        target_dte_days: float = 14.0,
        min_dte_days: float = 7.0,
        max_dte_days: float = 35.0,
        delta_tolerance_btc: float = 0.15,
        max_portfolio_gamma: float = 0.005,
        max_margin_utilization: float = 0.70,
        max_vega_btc: float = 30.0,
        min_vol_spread_hurdle: float = 0.04,  # Minimum 4 vol points spread
    ) -> None:
        self.target_dte = target_dte_days
        self.min_dte = min_dte_days
        self.max_dte = max_dte_days
        self.delta_tol = delta_tolerance_btc
        self.max_gamma = max_portfolio_gamma
        self.max_margin = max_margin_utilization
        self.max_vega = max_vega_btc
        self.spread_hurdle = min_vol_spread_hurdle

    def generate_profile(
        self,
        svi_surface: SVISurface,
        har_model: CorsiHAR_RV,
        rv_history: np.ndarray,
        risk_veto: RiskVetoDecision | None = None,
    ) -> TargetGreekProfile:
        """Evaluate market conditions and construct the TargetGreekProfile.

        Args:
            svi_surface: Calibrated SVI surface for the current timestamp
            har_model: Fitted Corsi HAR-RV forecasting model
            rv_history: Recent daily realized variance history
            risk_veto: Decision from independent risk veto authority

        Returns:
            TargetGreekProfile with Greek boundaries
        """
        tau_target = self.target_dte / 365.0
        atm_iv = svi_surface.atm_implied_vol(tau_target)

        # 1-step ahead forecasted RV from HAR-RV
        rv_var_pred = har_model.predict_one_step(rv_history)

        # Fail-closed input integrity (V-12): NaN or non-positive variance/IV
        # must deactivate the profile, not slip past the hurdle check (a NaN
        # spread yields `NaN < hurdle` -> False -> is_active=True on garbage).
        if not math.isfinite(atm_iv) or not math.isfinite(rv_var_pred) or rv_var_pred <= 0.0:
            return TargetGreekProfile(
                target_vega_btc=0.0,
                min_delta_btc=-self.delta_tol,
                max_delta_btc=self.delta_tol,
                max_gamma=self.max_gamma,
                min_dte_days=self.min_dte,
                max_dte_days=self.max_dte,
                max_margin_utilization=self.max_margin,
                iv_atm=float(atm_iv) if math.isfinite(atm_iv) else float("nan"),
                rv_forecast=float("nan"),
                variance_spread=float("nan"),
                is_active=False,
                reason=(
                    "Fail-closed: invalid variance inputs, no profile generated "
                    f"(atm_iv={atm_iv!r}, rv_variance={rv_var_pred!r})"
                ),
            )

        rv_vol_pred = float(np.sqrt(rv_var_pred * 365.0))

        vol_spread = atm_iv - rv_vol_pred
        var_spread = (atm_iv ** 2) - (rv_vol_pred ** 2)

        # 1. Independent Risk Veto Check
        if risk_veto is not None and risk_veto.veto_active:
            return TargetGreekProfile(
                target_vega_btc=0.0,
                min_delta_btc=-self.delta_tol,
                max_delta_btc=self.delta_tol,
                max_gamma=0.0,
                min_dte_days=self.min_dte,
                max_dte_days=self.max_dte,
                max_margin_utilization=self.max_margin,
                iv_atm=atm_iv,
                rv_forecast=rv_vol_pred,
                variance_spread=var_spread,
                is_active=False,
                reason=f"Risk Veto Active: {risk_veto.action} ({', '.join(risk_veto.reasons)})",
            )

        # 2. Hurdle Check
        if vol_spread < self.spread_hurdle:
            return TargetGreekProfile(
                target_vega_btc=0.0,
                min_delta_btc=-self.delta_tol,
                max_delta_btc=self.delta_tol,
                max_gamma=self.max_gamma,
                min_dte_days=self.min_dte,
                max_dte_days=self.max_dte,
                max_margin_utilization=self.max_margin,
                iv_atm=atm_iv,
                rv_forecast=rv_vol_pred,
                variance_spread=var_spread,
                is_active=False,
                reason=f"Spread {vol_spread:.4f} below hurdle {self.spread_hurdle:.4f}",
            )

        # 3. Dynamic Vega Sizing (scaled linearly by spread strength, capped at max_vega)
        # Sizing factor: e.g. 10 vol points above hurdle -> 100% of max vega
        excess_spread = max(0.0, vol_spread - self.spread_hurdle)
        sizing_ratio = min(1.0, excess_spread / 0.10)
        target_vega = -float(sizing_ratio * self.max_vega)

        return TargetGreekProfile(
            target_vega_btc=target_vega,
            min_delta_btc=-self.delta_tol,
            max_delta_btc=self.delta_tol,
            max_gamma=self.max_gamma,
            min_dte_days=self.min_dte,
            max_dte_days=self.max_dte,
            max_margin_utilization=self.max_margin,
            iv_atm=atm_iv,
            rv_forecast=rv_vol_pred,
            variance_spread=var_spread,
            is_active=True,
            reason="Active VRP harvesting profile",
        )
