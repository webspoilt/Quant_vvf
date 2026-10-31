"""Real-time Portfolio Delta/Vega/Gamma Surface Tracker and Margin Reconciliation.

Accounts for Deribit BTC-settled inverse numeraire mechanics:
- USD-equivalent vs BTC-numeraire Greeks
- 1/S^2 inverse perpetual hedge drift tracking
- Margin utilization and buffer monitoring
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import math
import numpy as np

from libs.core.inverse_payoffs import (
    inverse_perp_btc_delta,
    inverse_perp_hedge_rescale_factor,
    inverse_perp_usd_delta,
)


@dataclass(frozen=True)
class PositionLeg:
    instrument_id: str
    quantity: float            # Signed: > 0 long, < 0 short
    strike: float
    is_option: bool
    option_type: str | None    # "call", "put", or None for perp
    delta_usd: float           # Per-contract USD delta
    gamma: float               # Per-contract gamma
    vega_btc: float            # Per-contract vega in BTC
    theta_usd: float           # Per-contract daily theta in USD
    margin_btc: float          # Per-contract initial margin in BTC


@dataclass(frozen=True)
class PortfolioGreekState:
    """Aggregated portfolio risk and Greek surface."""
    total_delta_usd: float
    total_delta_btc: float
    total_gamma: float
    total_vega_btc: float
    total_theta_usd_daily: float
    total_margin_used_btc: float
    total_equity_btc: float
    margin_utilization: float
    spot_price: float
    inverse_hedge_rescale_factor: float
    is_delta_neutral: bool
    delta_rebalance_required: bool


class GreeksMonitor:
    """Real-time portfolio Greek surface tracker."""

    def __init__(
        self,
        delta_neutral_tolerance_btc: float = 0.15,
        max_margin_utilization: float = 0.70,
    ) -> None:
        self.delta_tolerance = delta_neutral_tolerance_btc
        self.max_margin = max_margin_utilization
        self._entry_spot: float | None = None

    def update_state(
        self,
        positions: Mapping[str, PositionLeg],
        perp_signed_notional_usd: float,
        spot_price: float,
        total_equity_btc: float,
        entry_spot: float | None = None,
    ) -> PortfolioGreekState:
        """Compute live portfolio Greeks and hedge drift.

        Args:
            positions: Map of instrument_id to PositionLeg
            perp_signed_notional_usd: Signed USD notional of inverse perpetual hedge
            spot_price: Current BTC spot index price
            total_equity_btc: Current equity balance in BTC
            entry_spot: Initial entry spot for hedge drift tracking

        Returns:
            PortfolioGreekState snapshot

        Raises:
            ValueError: fail-closed on non-finite spot/equity/leg data or
                non-positive equity (V-17 — clamping equity to 1e-4 BTC used
                to *mask* insolvency during liquidations by producing an
                enormous-but-finite margin utilization instead of aborting).
        """
        corrupt = (
            not math.isfinite(spot_price)
            or not math.isfinite(total_equity_btc)
            or not math.isfinite(perp_signed_notional_usd)
        )
        if not corrupt:
            for leg in positions.values():
                corrupt |= any(
                    not math.isfinite(getattr(leg, field))
                    for field in ("quantity", "strike", "delta_usd", "gamma", "vega_btc", "theta_usd", "margin_btc")
                )
        if corrupt:
            raise ValueError("GreeksMonitor.update_state: non-finite portfolio inputs (fail-closed)")
        if total_equity_btc <= 0.0:
            raise ValueError(
                f"GreeksMonitor.update_state: insolvent equity total_equity_btc={total_equity_btc!r} <= 0"
            )
        if spot_price <= 0.0:
            raise ValueError(f"GreeksMonitor.update_state: invalid spot_price={spot_price!r} <= 0")

        # V-16 fix: the entry spot may be refreshed explicitly on each call;
        # it latches only when no explicit anchor is provided (previously the
        # latch ignored all later anchors, freezing 1/S^2 hedge-drift tracking
        # at the very first spot ever passed in).
        if entry_spot is not None:
            if not math.isfinite(entry_spot) or entry_spot <= 0.0:
                raise ValueError(f"GreeksMonitor.update_state: invalid entry_spot={entry_spot!r}")
            self._entry_spot = entry_spot
        elif self._entry_spot is None:
            self._entry_spot = spot_price

        # Aggregate option book Greeks
        opt_delta_usd = 0.0
        opt_gamma = 0.0
        opt_vega = 0.0
        opt_theta = 0.0
        total_margin = 0.0

        for leg in positions.values():
            q = leg.quantity
            opt_delta_usd += q * leg.delta_usd
            opt_gamma += q * leg.gamma
            opt_vega += q * leg.vega_btc
            opt_theta += q * leg.theta_usd
            total_margin += abs(q) * leg.margin_btc

        # Inverse perpetual hedge Greeks
        # USD delta is linear: C / P_entry
        perp_delta_usd = inverse_perp_usd_delta(perp_signed_notional_usd, self._entry_spot)
        perp_delta_btc = inverse_perp_btc_delta(perp_signed_notional_usd, spot_price)

        # Total portfolio delta
        total_delta_usd = opt_delta_usd + perp_delta_usd
        total_delta_btc = (opt_delta_usd / max(1.0, spot_price)) + perp_delta_btc

        # Inverse hedge rescale drift: (S0 / S1)^2
        rescale_factor = inverse_perp_hedge_rescale_factor(self._entry_spot, spot_price)

        # Margin utilization (equity > 0 guaranteed; insolvency raises above)
        margin_util = float(total_margin / total_equity_btc)

        is_neutral = abs(total_delta_btc) <= self.delta_tolerance
        rebalance_needed = not is_neutral

        return PortfolioGreekState(
            total_delta_usd=total_delta_usd,
            total_delta_btc=total_delta_btc,
            total_gamma=opt_gamma,
            total_vega_btc=opt_vega,
            total_theta_usd_daily=opt_theta,
            total_margin_used_btc=total_margin,
            total_equity_btc=total_equity_btc,
            margin_utilization=margin_util,
            spot_price=spot_price,
            inverse_hedge_rescale_factor=rescale_factor,
            is_delta_neutral=is_neutral,
            delta_rebalance_required=rebalance_needed,
        )
