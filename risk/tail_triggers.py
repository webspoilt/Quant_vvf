"""Independent Risk Veto Authority: Deterministic Non-Parametric Tail Triggers.

Implements pre-registered tail triggers that possess unilateral veto authority
over all strategy layer execution:
1. RV Shock: Current 24h Realized Volatility > 3.0x 30-day baseline.
2. Skew Spike: 25-delta Put/Call IV Skew > 95th percentile of rolling 90-day window.
3. Funding Rate: Deribit BTC Perpetual funding rate z-score > 2.5 (or |z| > 2.5).

When any trigger fires:
- Immediate veto issued: HALT_NEW_ENTRIES / FLATTEN_SHORT_GAMMA.
- Strict cooldown period enforced before normal harvesting can resume.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence
import numpy as np

from libs.core.validation import non_finite_fields
from risk.risk_constants import (
    FUNDING_Z_SCORE_LIMIT,
    RV_SHOCK_MULTIPLIER_LIMIT,
    SKEW_SPIKE_PERCENTILE_LIMIT,
    RiskAction,
    RiskCheckResult,
    RiskGatekeeper,
)


@dataclass(frozen=True)
class RiskVetoDecision:
    """Immutable verdict emitted by the risk veto authority."""
    veto_active: bool
    action: Literal["ALL_CLEAR", "HALT_NEW_ENTRIES", "FLATTEN_SHORT_GAMMA"]
    reasons: tuple[str, ...]
    rv_ratio: float
    skew_percentile: float
    funding_z_score: float

    @property
    def is_vetoed(self) -> bool:
        return self.veto_active


from scipy import stats


def deflated_sharpe_ratio(
    returns: np.ndarray | Sequence[float],
    trial_budget_n: int = 3,
    benchmark_sr: float = 0.0,
    annualization_factor: float = 365.0,
) -> float:
    """Compute Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio (DSR).

    Adjusts estimated Sharpe ratio for selection bias across N trials,
    sample length T, skewness, and kurtosis.

    Returns:
        Probability (0 to 1) that the true Sharpe ratio is greater than benchmark_sr.
        Values > 0.95 indicate statistical significance at alpha = 0.05.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    t_len = len(r)
    if t_len < 10:
        return 0.50

    mean_r = float(np.mean(r))
    std_r = float(np.std(r, ddof=1))
    if std_r <= 1e-9:
        return 0.50

    # Daily Sharpe ratio
    sr_daily = mean_r / std_r
    sr_annual = sr_daily * np.sqrt(annualization_factor)

    # Higher moments
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))  # Pearson kurtosis (normal = 3)

    # Asymptotic variance of Sharpe ratio estimator
    sr_var = (1.0 - skew * sr_daily + 0.25 * (kurt - 1.0) * (sr_daily ** 2)) / (t_len - 1.0)
    sr_std = np.sqrt(max(1e-12, sr_var))

    # Expected maximum Sharpe ratio under the null across N independent trials
    # Euler-Mascheroni constant ~ 0.5772156649
    euler_gamma = 0.5772156649
    if trial_budget_n > 1:
        log_n = np.log(float(trial_budget_n))
        z_n = np.sqrt(2.0 * log_n) + euler_gamma / np.sqrt(2.0 * log_n)
        expected_max_sr_daily = z_n * sr_std
    else:
        expected_max_sr_daily = benchmark_sr / np.sqrt(annualization_factor)

    # Standardized z-score
    z_stat = (sr_daily - expected_max_sr_daily) / sr_std
    dsr_prob = float(stats.norm.cdf(z_stat))
    return dsr_prob


class TailTriggerMonitor:
    """Deterministic circuit breaker tracking market regime distress."""

    def __init__(
        self,
        rv_multiplier_threshold: float = RV_SHOCK_MULTIPLIER_LIMIT,
        skew_percentile_threshold: float = SKEW_SPIKE_PERCENTILE_LIMIT,
        funding_z_threshold: float = FUNDING_Z_SCORE_LIMIT,
        cooldown_steps: int = 12,  # e.g., 2 days at 4-hour steps
    ) -> None:
        self.rv_mult_threshold = rv_multiplier_threshold
        self.skew_pct_threshold = skew_percentile_threshold
        self.funding_z_threshold = funding_z_threshold
        self.cooldown_steps = cooldown_steps
        self.cooldown_remaining = 0
        self.gatekeeper = RiskGatekeeper()

    def veto_strategy_output(
        self,
        proposed_orders: Sequence[Any],
        proposed_vega_btc: float,
        proposed_delta_btc: float,
        proposed_gamma: float,
        proposed_margin_btc: float,
        total_equity_btc: float,
        last_veto_decision: RiskVetoDecision | None = None,
    ) -> tuple[bool, str, tuple[Any, ...]]:
        """Programmatic veto authority over proposed strategy orders.

        Inspired by the AstraQuant fail-closed risk architecture:
        - Absolute veto authority over strategy outputs
        - If any tail trigger is active -> Orders are vetoed outright
        - If proposed Greek or margin exposure breaches constants -> Vetoed
        - If an unexpected exception occurs -> Fail-closed rejection

        Returns:
            (is_approved, reason, approved_orders)
        """
        # 1. Independent Tail Trigger veto check
        if last_veto_decision is not None and last_veto_decision.veto_active:
            reasons_str = "; ".join(last_veto_decision.reasons)
            return False, f"VETOED by TailTrigger: {last_veto_decision.action} ({reasons_str})", ()

        # 2. Gatekeeper hard constants validation (fail-closed)
        check = self.gatekeeper.validate_proposed_execution(
            proposed_vega_btc=proposed_vega_btc,
            proposed_delta_btc=proposed_delta_btc,
            proposed_gamma=proposed_gamma,
            proposed_margin_btc=proposed_margin_btc,
            total_equity_btc=total_equity_btc,
        )

        if not check.approved:
            return False, f"VETOED by RiskGatekeeper: {'; '.join(check.reasons)}", ()

        return True, "APPROVED: All programmatic risk constants satisfied", tuple(proposed_orders)

    def evaluate(
        self,
        current_24h_rv: float,
        baseline_30d_rv: float,
        current_25d_skew: float,
        historical_90d_skew: np.ndarray | Sequence[float],
        current_funding_rate: float,
        historical_28d_funding: np.ndarray | Sequence[float],
    ) -> RiskVetoDecision:
        """Evaluate market conditions against immutable tail triggers.

        Args:
            current_24h_rv: Current annualized 24h realized volatility
            baseline_30d_rv: 30-day baseline realized volatility
            current_25d_skew: Current 25-delta (IV_put - IV_call)
            historical_90d_skew: Rolling array of daily 25-delta skews
            current_funding_rate: Current 8-hour perp funding rate
            historical_28d_funding: Rolling array of 8-hour funding rates

        Returns:
            RiskVetoDecision with veto action and diagnostic values
        """
        reasons: list[str] = []

        # 0. Fail-closed data integrity gate. IEEE 754 NaN defeats every `>`
        # comparison below, so corrupt feeds during a market shock would
        # otherwise produce ALL_CLEAR precisely when limits matter most.
        sk_hist = np.asarray(historical_90d_skew, dtype=np.float64)
        f_hist = np.asarray(historical_28d_funding, dtype=np.float64)
        scalar_corrupt = non_finite_fields(
            {
                "current_24h_rv": current_24h_rv,
                "baseline_30d_rv": baseline_30d_rv,
                "current_25d_skew": current_25d_skew,
                "current_funding_rate": current_funding_rate,
            }
        )
        array_corrupt: list[str] = []
        if not np.isfinite(sk_hist).all():
            array_corrupt.append("historical_90d_skew")
        if not np.isfinite(f_hist).all():
            array_corrupt.append("historical_28d_funding")
        if scalar_corrupt or array_corrupt:
            # Corrupt telemetry is distinct from a live tail event: veto the
            # strategy immediately but do NOT arm the post-shock cooldown,
            # which carries different semantics (shock decay, not data health).
            fields = scalar_corrupt + [f"{name}: contains non-finite values" for name in array_corrupt]
            return RiskVetoDecision(
                veto_active=True,
                action="HALT_NEW_ENTRIES",
                reasons=tuple(f"CORRUPT_INPUT: {field}" for field in fields),
                rv_ratio=float("nan"),
                skew_percentile=float("nan"),
                funding_z_score=float("nan"),
            )

        # 1. Realized Volatility Shock
        if baseline_30d_rv <= 0.0:
            reasons.append(f"RV Baseline Invalid: baseline_30d_rv={baseline_30d_rv:.4g} <= 0")
        base_rv = max(1e-4, baseline_30d_rv)
        rv_ratio = float(current_24h_rv / base_rv)
        if rv_ratio > self.rv_mult_threshold:
            reasons.append(f"RV Shock Triggered: {rv_ratio:.2f}x > {self.rv_mult_threshold:.1f}x threshold")

        # 2. Skew Spike (Panic Left-Tail Put Demand)
        if len(sk_hist) > 10:
            skew_percentile = float(np.mean(sk_hist <= current_25d_skew))
        else:
            skew_percentile = 0.50

        if skew_percentile > self.skew_pct_threshold:
            reasons.append(f"Skew Spike Triggered: {skew_percentile * 100:.1f}th pct > {self.skew_pct_threshold * 100:.1f}th pct")

        # 3. Perpetual Funding Rate Imbalance (Crowded Leverage)
        if len(f_hist) > 10:
            f_mean = float(np.mean(f_hist))
            f_std = max(1e-4, float(np.std(f_hist)))
            funding_z = float((current_funding_rate - f_mean) / f_std)
        else:
            funding_z = 0.0

        if abs(funding_z) > self.funding_z_threshold:
            reasons.append(f"Funding Imbalance Triggered: |z|={abs(funding_z):.2f} > {self.funding_z_threshold:.1f}")

        # Cooldown management
        any_trigger = len(reasons) > 0
        if any_trigger:
            self.cooldown_remaining = self.cooldown_steps
            return RiskVetoDecision(
                veto_active=True,
                action="FLATTEN_SHORT_GAMMA",
                reasons=tuple(reasons),
                rv_ratio=rv_ratio,
                skew_percentile=skew_percentile,
                funding_z_score=funding_z,
            )

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return RiskVetoDecision(
                veto_active=True,
                action="HALT_NEW_ENTRIES",
                reasons=(f"Post-shock Cooldown ({self.cooldown_remaining} steps remaining)",),
                rv_ratio=rv_ratio,
                skew_percentile=skew_percentile,
                funding_z_score=funding_z,
            )

        return RiskVetoDecision(
            veto_active=False,
            action="ALL_CLEAR",
            reasons=(),
            rv_ratio=rv_ratio,
            skew_percentile=skew_percentile,
            funding_z_score=funding_z,
        )
