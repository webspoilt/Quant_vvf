"""Unit tests for NautilusTrader Execution Actor, circuit breaker, and delta rebalancer."""
from __future__ import annotations

import pytest
from nautilus_trader.common.actor import Actor

from execution.nautilus_actor import VRPActorConfig, VRPOptionsExecutionActor
from strategies.greek_solver import OptionChainQuote
from strategies.vrp_profile import TargetGreekProfile


def test_nautilus_actor_subclass() -> None:
    """Verify VRPOptionsExecutionActor is an instance and subclass of Nautilus Actor."""
    actor = VRPOptionsExecutionActor()
    assert isinstance(actor, Actor)
    assert not actor.circuit_breaker_tripped


def test_microsecond_local_variance_circuit_breaker() -> None:
    """Verify r^2 > 0.0025 trips local variance circuit breaker and halts execution."""
    cfg = VRPActorConfig(local_variance_limit=0.0025)
    actor = VRPOptionsExecutionActor(cfg)

    # 1. Normal ticks: price moving in 0.1% increments: r = 0.001 -> r^2 = 0.000001 << 0.0025
    t0 = 1_000_000_000
    assert actor.on_tick(t0, 40000.0)
    assert actor.on_tick(t0 + 1000, 40040.0)
    assert actor.on_tick(t0 + 2000, 40020.0)
    assert not actor.circuit_breaker_tripped

    # 2. Sudden microsecond shock tick: +6.0% gap in 1 tick
    # r = ln(42500 / 40020) = 0.0601 -> r^2 = 0.003617 > 0.0025 limit!
    t_shock = t0 + 3000
    healthy = actor.on_tick(t_shock, 42500.0)
    assert not healthy
    assert actor.circuit_breaker_tripped
    assert len(actor.circuit_breaker_events) == 1

    event = actor.circuit_breaker_events[0]
    assert event.tick_variance > 0.0025
    assert event.action == "HALT_EXECUTION_AND_CANCEL_ALL"

    # Subsequent ticks are rejected
    assert not actor.on_tick(t_shock + 1000, 42510.0)

    # Execution is blocked while breaker is tripped
    profile = TargetGreekProfile(
        target_vega_btc=-10.0,
        min_delta_btc=-0.15,
        max_delta_btc=0.15,
        max_gamma=0.005,
        min_dte_days=7.0,
        max_dte_days=35.0,
        max_margin_utilization=0.70,
        iv_atm=0.65,
        rv_forecast=0.50,
        variance_spread=0.15,
        is_active=True,
        reason="Active",
    )
    result = actor.execute_greek_profile(profile, [], 42500.0, 10.0)
    assert not result.success
    assert "circuit breaker is tripped" in result.message

    # Reset
    actor.reset_circuit_breaker()
    assert not actor.circuit_breaker_tripped


def test_continuous_delta_rebalancing() -> None:
    """Verify actor triggers immediate hedge orders when |Delta| > 0.15."""
    cfg = VRPActorConfig(delta_rebalance_trigger=0.15, perp_instrument_id="BTC-PERPETUAL")
    actor = VRPOptionsExecutionActor(cfg)
    t = 1_000_000_000
    spot = 50000.0

    # 1. Delta within band: Delta = +0.08 BTC <= 0.15 -> No rebalance
    order_small = actor.check_and_rebalance_delta(portfolio_delta_btc=0.08, spot_price=spot, timestamp_ns=t)
    assert order_small is None
    assert len(actor.rebalance_orders) == 0

    # 2. Delta breaches band: Delta = +0.25 BTC > 0.15 -> Trigger immediate short perp hedge
    t2 = t + 5000
    order_large = actor.check_and_rebalance_delta(portfolio_delta_btc=0.25, spot_price=spot, timestamp_ns=t2)
    assert order_large is not None
    assert order_large.instrument_id == "BTC-PERPETUAL"
    assert order_large.side == "SELL"
    # Target hedge = -0.25 * 50000 = -$12,500
    assert order_large.target_hedge_usd == pytest.approx(-12500.0, abs=1.0)
    assert len(actor.rebalance_orders) == 1


def test_immediate_cloud_bracket_submission_on_fill() -> None:
    """Verify immediate exchange-native cloud bracket stops are placed synchronously upon fill."""
    cfg = VRPActorConfig(
        enable_cloud_brackets=True,
        cloud_stop_multiplier=2.0,
        cloud_take_profit_pct=0.50,
    )
    actor = VRPOptionsExecutionActor(cfg)
    t = 1_000_000_000

    # Short 2 contracts of BTC option at 0.05 BTC
    bracket = actor.on_fill(
        fill_id="FILL-001",
        instrument_id="BTC-25DEC26-100000-C",
        quantity=2,
        side="SELL",
        fill_price_btc=0.05,
        timestamp_ns=t,
    )
    assert bracket is not None
    assert bracket.venue_native
    assert bracket.close_side == "BUY"
    assert bracket.trigger_type == "mark_price"
    # Stop loss at 2x fill price: 0.10 BTC
    assert bracket.stop_loss_trigger_price_btc == pytest.approx(0.10, abs=1e-6)
    # Take profit at 50% decay: 0.025 BTC
    assert bracket.take_profit_price_btc == pytest.approx(0.025, abs=1e-6)
    assert "FILL-001" in actor.active_cloud_brackets
