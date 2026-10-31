"""Discrete Greek Solver: Resolves Discrete Strikes and Expiries under Margin Bounds.

Non-Negotiable Core Rule 2 (Instruments Over Weights):
Never use standard Markowitz/CVaR solvers to assign raw capital weights to options legs.
The strategy layer outputs a TargetGreekProfile, and the execution layer solves for
discrete strikes/expiries under margin bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence
import numpy as np

from libs.core.validation import non_finite_fields
from strategies.vrp_profile import TargetGreekProfile


@dataclass(frozen=True)
class OptionChainQuote:
    """Market quote and Greek profile for a discrete Deribit option instrument."""
    instrument_id: str
    strike: float
    expiry_tau: float
    option_type: Literal["call", "put"]
    bid_price_btc: float
    ask_price_btc: float
    mid_price_btc: float
    delta_btc: float
    gamma: float
    vega_btc: float
    initial_margin_btc: float


@dataclass(frozen=True)
class SolvedOrder:
    """Discrete executable order."""
    instrument_id: str
    quantity: int                 # Absolute contract count
    side: Literal["BUY", "SELL"]
    price_btc: float
    strike: float
    option_type: Literal["call", "put"]


@dataclass(frozen=True)
class GreekSolverResult:
    """Output of the discrete Greek solver."""
    orders: tuple[SolvedOrder, ...]
    achieved_vega_btc: float
    achieved_delta_btc: float
    achieved_gamma: float
    total_margin_btc: float
    success: bool
    message: str


class DiscreteGreekSolver:
    """Selects discrete integer contracts to fit a TargetGreekProfile within hard bounds."""

    def __init__(
        self,
        max_single_strike_pct: float = 0.35,
        max_slippage_bps: float = 25.0,
    ) -> None:
        self.max_strike_pct = max_single_strike_pct
        self.max_slippage_bps = max_slippage_bps

    def solve(
        self,
        profile: TargetGreekProfile,
        chain: Sequence[OptionChainQuote],
        spot_price: float,
        available_margin_btc: float,
    ) -> GreekSolverResult:
        """Solve for discrete strikes and contract quantities matching the Greek profile.

        Args:
            profile: Target Greek profile emitted by strategy layer
            chain: Available option quotes from Deribit chain
            spot_price: Current BTC spot index price
            available_margin_btc: Free margin available for collateralization

        Returns:
            GreekSolverResult with discrete integer orders and achieved Greek totals
        """
        if not profile.is_active or abs(profile.target_vega_btc) < 1e-4:
            return GreekSolverResult(
                orders=(),
                achieved_vega_btc=0.0,
                achieved_delta_btc=0.0,
                achieved_gamma=0.0,
                total_margin_btc=0.0,
                success=True,
                message=f"No orders generated: {profile.reason}",
            )

        # Fail-closed numeric sanity: corrupt profile/market inputs must abort
        # the solve rather than propagate NaN through sizing arithmetic.
        corrupt = non_finite_fields(
            {
                "target_vega_btc": profile.target_vega_btc,
                "max_gamma": profile.max_gamma,
                "max_margin_utilization": profile.max_margin_utilization,
                "spot_price": spot_price,
                "available_margin_btc": available_margin_btc,
            }
        )
        if corrupt:
            return GreekSolverResult(
                orders=(),
                achieved_vega_btc=0.0,
                achieved_delta_btc=0.0,
                achieved_gamma=0.0,
                total_margin_btc=0.0,
                success=False,
                message=f"CORRUPT_INPUT: {'; '.join(corrupt)}",
            )
        if spot_price <= 0.0:
            return GreekSolverResult(
                orders=(),
                achieved_vega_btc=0.0,
                achieved_delta_btc=0.0,
                achieved_gamma=0.0,
                total_margin_btc=0.0,
                success=False,
                message=f"INVALID_SPOT_PRICE: spot_price={spot_price!r} <= 0",
            )

        # Filter chain by DTE window
        eligible: list[OptionChainQuote] = []
        for q in chain:
            dte_days = q.expiry_tau * 365.0
            if profile.min_dte_days <= dte_days <= profile.max_dte_days:
                # Valid bid-ask spread check
                if q.bid_price_btc > 0 and q.ask_price_btc >= q.bid_price_btc:
                    eligible.append(q)

        if not eligible:
            return GreekSolverResult(
                orders=(),
                achieved_vega_btc=0.0,
                achieved_delta_btc=0.0,
                achieved_gamma=0.0,
                total_margin_btc=0.0,
                success=False,
                message="No eligible options quotes within DTE and liquidity bounds",
            )

        # Group by expiry and pair ATM/near-the-money Call + Put vertical straddles/strangles
        # to guarantee delta neutrality while fitting target vega.

        # V-10 fix: quotes must be paired within the SAME expiry. A flat
        # ``{strike: quote}`` dict over expiries lets the LAST quote for a
        # strike win, producing diagonal spreads (call from tau1, put from
        # tau2) whose Greeks/margins don't match the hedges they claim.
        expiries: dict[float, dict[str, dict[float, OptionChainQuote]]] = {}
        for q in eligible:
            book = expiries.setdefault(q.expiry_tau, {"call": {}, "put": {}})
            if q.option_type not in ("call", "put"):
                continue
            book[q.option_type][q.strike] = q

        # Candidate ATM pairs = (tau, strike) tuples matching on both legs
        # within one expiry; select the pair whose strike is closest to spot,
        # tie-broken toward the nearer expiry structure.
        best_key: tuple[float, float] | None = None
        best_call: OptionChainQuote | None = None
        best_put: OptionChainQuote | None = None
        for tau, book in expiries.items():
            common_strikes = sorted(set(book["call"]) & set(book["put"]))
            if not common_strikes:
                continue
            strike = min(common_strikes, key=lambda k: (abs(k - spot_price), k))
            key = (abs(strike - spot_price), tau)
            if best_key is None or key < best_key:
                best_key = key
                best_call = book["call"][strike]
                best_put = book["put"][strike]

        if best_call is None or best_put is None:
            return GreekSolverResult(
                orders=(),
                achieved_vega_btc=0.0,
                achieved_delta_btc=0.0,
                achieved_gamma=0.0,
                total_margin_btc=0.0,
                success=False,
                message="No matching call/put strike pairs found in chain (within any single expiry)",
            )

        atm_call = best_call
        atm_put = best_put

        # Straddle vega per 1 contract pair (1 Call + 1 Put)
        pair_vega = atm_call.vega_btc + atm_put.vega_btc
        pair_delta = atm_call.delta_btc + atm_put.delta_btc
        pair_gamma = atm_call.gamma + atm_put.gamma
        pair_margin = atm_call.initial_margin_btc + atm_put.initial_margin_btc

        if pair_vega <= 1e-6:
            return GreekSolverResult(
                orders=(),
                achieved_vega_btc=0.0,
                achieved_delta_btc=0.0,
                achieved_gamma=0.0,
                total_margin_btc=0.0,
                success=False,
                message="Pair vega too small",
            )

        # Short volatility target: target_vega is negative, selling pairs gives negative vega
        # n_pairs * (-pair_vega) = target_vega_btc
        desired_pairs = int(round(abs(profile.target_vega_btc) / pair_vega))
        if desired_pairs <= 0:
            # Never force a whole straddle pair that the target does not fund.
            return GreekSolverResult(
                orders=(),
                achieved_vega_btc=0.0,
                achieved_delta_btc=0.0,
                achieved_gamma=0.0,
                total_margin_btc=0.0,
                success=False,
                message=(
                    "Target vega below one-pair granularity: "
                    f"|target|={abs(profile.target_vega_btc):.4f} < pair_vega/2={pair_vega / 2.0:.4f}"
                ),
            )

        # Margin and Gamma constraints (V-06 fix: `max(1, limit)`, which
        # forced >= 1 pair even when the budget/capacity was zero or negative,
        # is forbidden — insufficient capacity must produce ZERO orders).
        max_margin_budget = available_margin_btc * profile.max_margin_utilization
        if pair_margin > 0:
            margin_pair_limit = int(max_margin_budget // pair_margin)
            if margin_pair_limit <= 0:
                return GreekSolverResult(
                    orders=(),
                    achieved_vega_btc=0.0,
                    achieved_delta_btc=0.0,
                    achieved_gamma=0.0,
                    total_margin_btc=0.0,
                    success=False,
                    message=f"Insufficient margin: budget={max_margin_budget:.6f} < pair_margin={pair_margin:.6f}",
                )
            desired_pairs = min(desired_pairs, margin_pair_limit)

        if pair_gamma > 0 and profile.max_gamma > 0:
            gamma_pair_limit = int(profile.max_gamma // pair_gamma)
            if gamma_pair_limit <= 0:
                return GreekSolverResult(
                    orders=(),
                    achieved_vega_btc=0.0,
                    achieved_delta_btc=0.0,
                    achieved_gamma=0.0,
                    total_margin_btc=0.0,
                    success=False,
                    message=f"Gamma ceiling exceeded: max_gamma={profile.max_gamma:.6f} < pair_gamma={pair_gamma:.6f}",
                )
            desired_pairs = min(desired_pairs, gamma_pair_limit)

        # Build orders (Selling short straddle to harvest positive VRP)
        orders = (
            SolvedOrder(
                instrument_id=atm_call.instrument_id,
                quantity=desired_pairs,
                side="SELL",
                price_btc=atm_call.bid_price_btc,
                strike=atm_call.strike,
                option_type="call",
            ),
            SolvedOrder(
                instrument_id=atm_put.instrument_id,
                quantity=desired_pairs,
                side="SELL",
                price_btc=atm_put.bid_price_btc,
                strike=atm_put.strike,
                option_type="put",
            ),
        )

        achieved_vega = -float(desired_pairs * pair_vega)
        achieved_delta = -float(desired_pairs * pair_delta)
        achieved_gamma = float(desired_pairs * pair_gamma)
        total_margin = float(desired_pairs * pair_margin)

        return GreekSolverResult(
            orders=orders,
            achieved_vega_btc=achieved_vega,
            achieved_delta_btc=achieved_delta,
            achieved_gamma=achieved_gamma,
            total_margin_btc=total_margin,
            success=True,
            message=(
                f"Solved {desired_pairs} straddle pairs at strike {atm_call.strike} "
                f"(tau={atm_call.expiry_tau:.4f})"
            ),
        )
