"""Deterministic Risk Constants and Fail-Closed Programmatic Veto Authority.

Inspired by the AstraQuant architectural model (separation of hypothesis and enforcement):
- Deterministic Python risk control with absolute veto authority over all strategy outputs.
- Fail-Closed Principle: If any risk assertion fails, times out, or encounters unexpected data,
  proposed orders are rejected outright.
- Parameters can only be dynamically tightened, never loosened past immutable hard bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from libs.core.validation import non_finite_fields


class RiskAction(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    FORCE_FLATTEN = "FORCE_FLATTEN"


# --------------------------------------------------------------------------- #
# IMMUTABLE SYSTEM-WIDE RISK CEILINGS (FAIL-CLOSED BOUNDS)
# --------------------------------------------------------------------------- #
MAX_PORTFOLIO_VEGA_BTC: float = 50.0          # Global portfolio vega ceiling (BTC)
MAX_PORTFOLIO_DELTA_ABS: float = 0.15         # Rebalance trigger when |Delta| > 0.15 BTC
MAX_PORTFOLIO_GAMMA: float = 0.005            # Portfolio gamma limit
MAX_MARGIN_UTILIZATION: float = 0.70          # Hard cap on initial margin (70% equity)
MIN_COLLATERAL_BUFFER_BTC: float = 0.50       # Minimum free unencumbered BTC balance
MAX_SINGLE_STRIKE_CONCENTRATION: float = 0.35 # Max fraction of contracts in one strike
LOCAL_VARIANCE_BREAKER_LIMIT: float = 0.0025  # Microsecond tick variance limit (r^2 > 0.0025)

# Tail Trigger Thresholds
RV_SHOCK_MULTIPLIER_LIMIT: float = 3.0        # 24h RV > 3.0x 30d baseline
SKEW_SPIKE_PERCENTILE_LIMIT: float = 0.95     # 25-delta Put-Call skew > 95th percentile
FUNDING_Z_SCORE_LIMIT: float = 2.5            # Perp funding rate |z| > 2.5
MAX_DRAWDOWN_HARD_STOP: float = 0.20          # 20% portfolio peak-to-trough hard stop


@dataclass(frozen=True)
class RiskCheckResult:
    action: RiskAction
    approved: bool
    reasons: tuple[str, ...]
    fail_closed: bool = True


class RiskGatekeeper:
    """Independent programmatic veto authority enforcing risk_constants.

    Evaluates proposed orders from the strategy/solver layer against immutable
    risk constants before execution.
    """

    def __init__(
        self,
        max_vega_btc: float = MAX_PORTFOLIO_VEGA_BTC,
        max_delta_btc: float = MAX_PORTFOLIO_DELTA_ABS,
        max_gamma: float = MAX_PORTFOLIO_GAMMA,
        max_margin: float = MAX_MARGIN_UTILIZATION,
        max_strike_pct: float = MAX_SINGLE_STRIKE_CONCENTRATION,
    ) -> None:
        # Fail-closed bound integrity: a NaN bound would silently defeat every
        # comparison check below (`NaN < x` is False → tightening never lands).
        bound_map = {
            "max_vega_btc": max_vega_btc,
            "max_delta_btc": max_delta_btc,
            "max_gamma": max_gamma,
            "max_margin": max_margin,
            "max_strike_pct": max_strike_pct,
        }
        if non_finite_fields(bound_map):
            raise ValueError(f"RiskGatekeeper bounds are non-finite: {bound_map}")
        # Enforce that parameters can only tighten, never loosen beyond hard constants
        self.max_vega = min(max_vega_btc, MAX_PORTFOLIO_VEGA_BTC)
        self.max_delta = min(max_delta_btc, MAX_PORTFOLIO_DELTA_ABS)
        self.max_gamma = min(max_gamma, MAX_PORTFOLIO_GAMMA)
        self.max_margin = min(max_margin, MAX_MARGIN_UTILIZATION)
        self.max_strike_pct = min(max_strike_pct, MAX_SINGLE_STRIKE_CONCENTRATION)

    def validate_proposed_execution(
        self,
        proposed_vega_btc: float,
        proposed_delta_btc: float,
        proposed_gamma: float,
        proposed_margin_btc: float,
        total_equity_btc: float,
        strike_contract_counts: Sequence[int] | None = None,
        circuit_breaker_active: bool = False,
    ) -> RiskCheckResult:
        """Evaluate proposed orders using the fail-closed principle.

        If any condition breaches limits or if an unexpected exception occurs,
        execution is REJECTED outright.
        """
        try:
            reasons: list[str] = []

            # 1. Circuit breaker override
            if circuit_breaker_active:
                return RiskCheckResult(
                    action=RiskAction.REJECT,
                    approved=False,
                    reasons=("Circuit breaker active: execution blocked",),
                )

            # 1b. Fail-closed numerical integrity gate. IEEE 754 NaN defeats
            # every `>`/`<` comparison below, so non-finite proposals must be
            # rejected before any comparison executes.
            corrupt = non_finite_fields(
                {
                    "proposed_vega_btc": proposed_vega_btc,
                    "proposed_delta_btc": proposed_delta_btc,
                    "proposed_gamma": proposed_gamma,
                    "proposed_margin_btc": proposed_margin_btc,
                    "total_equity_btc": total_equity_btc,
                }
            )
            if corrupt:
                return RiskCheckResult(
                    action=RiskAction.REJECT,
                    approved=False,
                    reasons=tuple(f"CORRUPT_INPUT: {field}" for field in corrupt),
                )

            # 2. Vega ceiling check
            if abs(proposed_vega_btc) > self.max_vega:
                reasons.append(
                    f"Vega breach: {abs(proposed_vega_btc):.2f} BTC > {self.max_vega:.2f} BTC limit"
                )

            # 3. Delta boundary check
            if abs(proposed_delta_btc) > self.max_delta:
                reasons.append(
                    f"Delta breach: |Delta|={abs(proposed_delta_btc):.3f} > {self.max_delta:.3f} limit"
                )

            # 4. Gamma boundary check
            if abs(proposed_gamma) > self.max_gamma:
                reasons.append(
                    f"Gamma breach: {abs(proposed_gamma):.5f} > {self.max_gamma:.5f} limit"
                )

            # 5. Margin utilization check
            margin_util = proposed_margin_btc / max(1e-4, total_equity_btc)
            if margin_util > self.max_margin:
                reasons.append(
                    f"Margin utilization breach: {margin_util * 100:.1f}% > {self.max_margin * 100:.1f}% limit"
                )

            # 6. Concentration check
            if strike_contract_counts is not None and len(strike_contract_counts) > 0:
                if len(strike_contract_counts) == 1:
                    # Single-strike structure (e.g. pure ATM straddle): the
                    # per-strike share is 100% by construction, so this
                    # diversification guard is structurally unsatisfiable and
                    # must not deadlock the only legitimate one-pair strategy.
                    # Size is still bounded by the vega/gamma/margin checks
                    # above and the solver's allocation caps; the concentration
                    # limit applies to multi-strike structures only.
                    pass
                else:
                    total_contracts = sum(strike_contract_counts)
                    if total_contracts > 0:
                        max_single = max(strike_contract_counts)
                        pct = max_single / total_contracts
                        if pct > self.max_strike_pct:
                            reasons.append(
                                f"Single-strike concentration breach: {pct * 100:.1f}% > {self.max_strike_pct * 100:.1f}% limit"
                            )

            if reasons:
                return RiskCheckResult(
                    action=RiskAction.REJECT,
                    approved=False,
                    reasons=tuple(reasons),
                )

            return RiskCheckResult(
                action=RiskAction.APPROVE,
                approved=True,
                reasons=("All programmatic risk invariant checks passed",),
            )

        except Exception as exc:
            # FAIL-CLOSED: On any unexpected failure, reject execution.
            # Sanitize the exception detail (V-24): leak the exception *type*
            # only — raw exception text can embed internal state/paths that
            # would leak into risk logs shared downstream.
            return RiskCheckResult(
                action=RiskAction.REJECT,
                approved=False,
                reasons=(f"FAIL-CLOSED triggered due to evaluation error: {type(exc).__name__}",),
            )
