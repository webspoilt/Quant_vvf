"""Low-Latency Execution Layer: NautilusTrader Actor with Microsecond Circuit Breakers.

Features:
1. Subclass of nautilus_trader.common.actor.Actor
2. Microsecond local variance circuit breaker (r^2 > 0.0025 halts execution instantly)
3. Continuous delta rebalancing when |Delta| > 0.15 via Deribit inverse perpetual
4. Integrates DiscreteGreekSolver under hard margin and risk boundaries
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import logging
import math
import numpy as np
from nautilus_trader.common.actor import Actor, ActorConfig

from strategies.greek_solver import DiscreteGreekSolver, GreekSolverResult, OptionChainQuote
from strategies.vrp_profile import TargetGreekProfile
from risk.risk_constants import RiskGatekeeper

_FALLBACK_LOGGER = logging.getLogger(__name__)


class VRPActorConfig(ActorConfig):
    """Configuration for the VRP Options Execution Actor."""
    local_variance_limit: float = 0.0025          # r^2 > 0.0025 stops execution
    delta_rebalance_trigger: float = 0.15         # |Delta| > 0.15 triggers rebalance
    delta_target_neutral: float = 0.0             # Neutral target delta
    perp_instrument_id: str = "BTC-PERPETUAL"     # Deribit inverse perpetual
    enable_cloud_brackets: bool = True            # Immediate venue-native bracket submission
    cloud_stop_multiplier: float = 2.0            # Stop loss at 2x premium for short options
    cloud_take_profit_pct: float = 0.50           # Take profit at 50% premium capture


@dataclass(frozen=True)
class CircuitBreakerEvent:
    timestamp_ns: int
    spot_price: float
    last_price: float
    tick_log_return: float
    tick_variance: float
    limit: float
    action: str


@dataclass(frozen=True)
class DeltaRebalanceOrder:
    timestamp_ns: int
    instrument_id: str
    target_hedge_usd: float
    trade_notional_usd: float
    side: str
    spot_price: float
    pre_rebalance_delta_btc: float


@dataclass(frozen=True)
class CloudBracketOrder:
    """Cloud bracket descriptor for a protective stop/take-profit pair.

    ACKNOWLEDGED LIMITATION (V-07): as long as no venue submission is wired
    into the Nautilus execution client, instances of this dataclass describe
    *local intent only* — nothing is registered on the exchange, and the
    position is NOT protected against client network disconnects. The
    ``venue_native`` flag is reserved for the wired implementation and must
    not be read as a guarantee of exchange-side execution.
    """
    parent_fill_id: str
    instrument_id: str
    quantity: int
    close_side: str
    stop_loss_trigger_price_btc: float
    take_profit_price_btc: float
    trigger_type: str = "mark_price"
    venue_native: bool = True
    timestamp_ns: int = 0


class VRPOptionsExecutionActor(Actor):
    """NautilusTrader Actor executing VRP profiles with high-frequency risk stops."""

    def __init__(self, config: VRPActorConfig | None = None) -> None:
        super().__init__(config=config or VRPActorConfig())
        self._cfg: VRPActorConfig = config or VRPActorConfig()

        # Circuit breaker state
        self.circuit_breaker_tripped: bool = False
        self.circuit_breaker_events: list[CircuitBreakerEvent] = []
        self._last_tick_price: float | None = None
        self._last_tick_ts_ns: int = 0

        # Delta hedging state
        self.current_perp_hedge_usd: float = 0.0
        self.rebalance_orders: list[DeltaRebalanceOrder] = []
        # Hedge orders whose fill has NOT been confirmed yet. Hedge state
        # advances only on fill confirmation (V-15 state desync fix).
        self.pending_hedge_orders: list[DeltaRebalanceOrder] = []

        # Discrete Greek solver & Programmatic Gatekeeper
        self.solver = DiscreteGreekSolver()
        self.gatekeeper = RiskGatekeeper()
        self.active_solved_orders: list[Any] = []
        self.active_cloud_brackets: dict[str, CloudBracketOrder] = {}

    def on_tick(self, timestamp_ns: int, price: float) -> bool:
        """Process an incoming high-frequency market tick.

        Evaluates the local variance circuit breaker:
            r_t = ln(P_t / P_{t-1})
            r^2 > 0.0025 -> TRIP CIRCUIT BREAKER

        Fail-closed: non-finite (NaN/±Inf) or non-positive prices trip the
        circuit breaker with action VERY ``HALT_CORRUPT_TICK_DATA``. IEEE 754
        defines ``NaN > x`` as False, so an unguarded NaN would bypass the
        variance computation and then poison ``_last_tick_price`` for every
        subsequent tick.

        Returns:
            True if tick is valid and actor is healthy, False if circuit breaker tripped.
        """
        if not math.isfinite(price) or price <= 0.0:
            self.circuit_breaker_tripped = True
            if self._last_tick_price is not None and math.isfinite(self._last_tick_price):
                last_price = self._last_tick_price
            else:
                last_price = 0.0
            self.circuit_breaker_events.append(
                CircuitBreakerEvent(
                    timestamp_ns=timestamp_ns,
                    spot_price=price,
                    last_price=last_price,
                    tick_log_return=float("nan"),
                    tick_variance=float("nan"),
                    limit=self._cfg.local_variance_limit,
                    action="HALT_CORRUPT_TICK_DATA",
                )
            )
            # Do NOT update _last_tick_price with non-finite data.
            return False

        if self._last_tick_price is not None and self._last_tick_price > 0:
            log_ret = float(np.log(price / self._last_tick_price))
            local_var = log_ret * log_ret

            if local_var > self._cfg.local_variance_limit:
                self.circuit_breaker_tripped = True
                event = CircuitBreakerEvent(
                    timestamp_ns=timestamp_ns,
                    spot_price=price,
                    last_price=self._last_tick_price,
                    tick_log_return=log_ret,
                    tick_variance=local_var,
                    limit=self._cfg.local_variance_limit,
                    action="HALT_EXECUTION_AND_CANCEL_ALL",
                )
                self.circuit_breaker_events.append(event)
                return False

        self._last_tick_price = price
        self._last_tick_ts_ns = timestamp_ns
        return not self.circuit_breaker_tripped
    def check_and_rebalance_delta(
        self,
        portfolio_delta_btc: float,
        spot_price: float,
        timestamp_ns: int,
    ) -> DeltaRebalanceOrder | None:
        """Continuously rebalance portfolio delta if absolute delta exceeds 0.15 BTC.

        Calculates the required inverse perpetual hedge increment to restore neutrality.
        """
        if self.circuit_breaker_tripped:
            return None

        abs_delta = abs(portfolio_delta_btc)
        if abs_delta <= self._cfg.delta_rebalance_trigger:
            return None  # Within acceptable neutrality band

        # Delta to eliminate: portfolio_delta_btc -> need perp delta = -portfolio_delta_btc
        # For inverse perp, delta_btc = C / S^2 -> C = delta_btc * S^2
        # To add -portfolio_delta_btc:
        target_hedge_usd = -portfolio_delta_btc * spot_price
        trade_usd = target_hedge_usd - self.current_perp_hedge_usd

        side = "BUY" if trade_usd > 0 else "SELL"
        order = DeltaRebalanceOrder(
            timestamp_ns=timestamp_ns,
            instrument_id=self._cfg.perp_instrument_id,
            target_hedge_usd=target_hedge_usd,
            trade_notional_usd=abs(trade_usd),
            side=side,
            spot_price=spot_price,
            pre_rebalance_delta_btc=portfolio_delta_btc,
        )

        # V-15 fix: hedge STATE is not optimistically advanced at order time.
        # The realized hedge position may be rejected at the venue; recording
        # the target as already-executed would desync hedge accounting. State
        # advances in on_fill() when the hedge fill is confirmed.
        self.rebalance_orders.append(order)
        self.pending_hedge_orders.append(order)
        return order

    def execute_greek_profile(
        self,
        profile: TargetGreekProfile,
        chain: Sequence[OptionChainQuote],
        spot_price: float,
        available_margin_btc: float,
    ) -> GreekSolverResult:
        """Resolve discrete option legs from Greek profile and dispatch orders."""
        if self.circuit_breaker_tripped:
            return GreekSolverResult(
                orders=(),
                achieved_vega_btc=0.0,
                achieved_delta_btc=0.0,
                achieved_gamma=0.0,
                total_margin_btc=0.0,
                success=False,
                message="Cannot execute: local variance circuit breaker is tripped",
            )

        result = self.solver.solve(profile, chain, spot_price, available_margin_btc)
        if not result.success:
            return result

        # Fail-closed programmatic risk gatekeeper validation (risk_constants model)
        total_eq = available_margin_btc / max(1e-4, profile.max_margin_utilization)
        # V-14 fix: aggregate contracts per STRIKE, not per order leg. Passing
        # per-leg quantities dilutes one strike's share when a call+put pair is
        # quoted at the same strike (two 50% 'legs' instead of one 100% strike)
        # and misreports concentration for multi-leg structures.
        strike_totals: Counter[float] = Counter()
        for order in result.orders:
            strike_totals[order.strike] += order.quantity
        risk_check = self.gatekeeper.validate_proposed_execution(
            proposed_vega_btc=result.achieved_vega_btc,
            proposed_delta_btc=result.achieved_delta_btc,
            proposed_gamma=result.achieved_gamma,
            proposed_margin_btc=result.total_margin_btc,
            total_equity_btc=total_eq,
            strike_contract_counts=list(strike_totals.values()),
            circuit_breaker_active=self.circuit_breaker_tripped,
        )

        if not risk_check.approved:
            return GreekSolverResult(
                orders=(),
                achieved_vega_btc=0.0,
                achieved_delta_btc=0.0,
                achieved_gamma=0.0,
                total_margin_btc=0.0,
                success=False,
                message=f"VETOED by RiskGatekeeper: {'; '.join(risk_check.reasons)}",
            )

        self.active_solved_orders.extend(result.orders)
        return result

    def _warn(self, message: str) -> None:
        """Log a warning via the Nautilus logger, falling back to stdlib when
        the actor is instantiated outside a Nautilus kernel (unit tests)."""
        try:
            self.log.warning(message)
        except Exception:  # nautilus logger unavailable without kernel context
            _FALLBACK_LOGGER.warning(message)

    def on_fill(
        self,
        fill_id: str,
        instrument_id: str,
        quantity: int,
        side: str,
        fill_price_btc: float,
        timestamp_ns: int,
    ) -> CloudBracketOrder | None:
        """Handle execution fill confirmation.

        SECURITY NOTE (V-07): the bracket object built here is registered
        LOCALLY only — no venue order is submitted yet. Until adaptive stop
        submission is wired into the Nautilus execution client, the position
        is UNPROTECTED against client network disconnects; this method emits a
        loud warning instead of silently claiming venue-native safety.

        Also reconciles hedge accounting: a confirmed fill on the perpetual
        instrument advances the realized hedge state (V-15 — no optimistic
        state updates at order time).
        """
        # 1. Hedge fill reconciliation (perp), independent of bracket feature
        if instrument_id == self._cfg.perp_instrument_id and self.pending_hedge_orders:
            hedge_order = self.pending_hedge_orders.pop(0)
            self.current_perp_hedge_usd = hedge_order.target_hedge_usd

        if not self._cfg.enable_cloud_brackets:
            return None

        # 2. For a short option (side == 'SELL'), close_side is 'BUY'.
        # Stop-loss triggers if mark price rises to e.g. 2.0x entry price.
        # Take-profit triggers if mark price decays to e.g. 0.50x entry price.
        close_side = "BUY" if side == "SELL" else "SELL"
        if side == "SELL":
            sl_price = fill_price_btc * self._cfg.cloud_stop_multiplier
            tp_price = fill_price_btc * self._cfg.cloud_take_profit_pct
        else:
            sl_price = fill_price_btc * (1.0 / max(1.1, self._cfg.cloud_stop_multiplier))
            tp_price = fill_price_btc * (1.0 + self._cfg.cloud_take_profit_pct)

        bracket = CloudBracketOrder(
            parent_fill_id=fill_id,
            instrument_id=instrument_id,
            quantity=quantity,
            close_side=close_side,
            stop_loss_trigger_price_btc=sl_price,
            take_profit_price_btc=tp_price,
            trigger_type="mark_price",
            venue_native=True,
            timestamp_ns=timestamp_ns,
        )

        self.active_cloud_brackets[fill_id] = bracket
        self._warn(
            f"SAFETY NOTICE: bracket for fill {fill_id} ({instrument_id}) registered "
            "LOCALLY only — no exchange-side cloud conditional stop is active. "
            "Position is UNPROTECTED against client network disconnects until "
            "venue-native submission is wired into the execution client."
        )
        return bracket

    def reset_circuit_breaker(self) -> None:
        """Manually reset the local variance circuit breaker after risk clearance.

        Also clears the last-tick state: a stale price from before the halt is
        considered corrupt by definition (item 27 — breaker reset hygiene); a
        fresh healthy tick must re-anchor the variance reference.
        """
        self.circuit_breaker_tripped = False
        self._last_tick_price = None
        self._last_tick_ts_ns = 0
