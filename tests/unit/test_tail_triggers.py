"""Unit tests for deterministic tail triggers and portfolio risk limits."""
from __future__ import annotations

import numpy as np
import pytest

from risk.limits import PortfolioRiskMonitor, RiskLimitsConfig
from risk.tail_triggers import RiskVetoDecision, TailTriggerMonitor


def test_tail_trigger_rv_shock() -> None:
    """RV Shock > 3.0x baseline must trigger immediate FLATTEN_SHORT_GAMMA veto."""
    monitor = TailTriggerMonitor(rv_multiplier_threshold=3.0, cooldown_steps=4)

    # Normal conditions
    hist_skew = np.linspace(0.01, 0.08, 90)
    hist_funding = np.full(84, 0.0001)
    normal = monitor.evaluate(
        current_24h_rv=0.50,
        baseline_30d_rv=0.45,
        current_25d_skew=0.04,
        historical_90d_skew=hist_skew,
        current_funding_rate=0.0001,
        historical_28d_funding=hist_funding,
    )
    assert not normal.veto_active
    assert normal.action == "ALL_CLEAR"

    # Sudden RV explosion: 1.80 vs baseline 0.45 (4.0x shock)
    shock = monitor.evaluate(
        current_24h_rv=1.80,
        baseline_30d_rv=0.45,
        current_25d_skew=0.04,
        historical_90d_skew=hist_skew,
        current_funding_rate=0.0001,
        historical_28d_funding=hist_funding,
    )
    assert shock.veto_active
    assert shock.action == "FLATTEN_SHORT_GAMMA"
    assert any("RV Shock Triggered" in r for r in shock.reasons)


def test_tail_trigger_skew_spike() -> None:
    """Skew > 95th percentile must trigger veto."""
    monitor = TailTriggerMonitor(skew_percentile_threshold=0.95)
    hist_skew = np.linspace(0.01, 0.10, 100)

    # 98th percentile skew (0.12 > max of hist)
    spike = monitor.evaluate(
        current_24h_rv=0.50,
        baseline_30d_rv=0.50,
        current_25d_skew=0.12,
        historical_90d_skew=hist_skew,
        current_funding_rate=0.0,
        historical_28d_funding=np.zeros(84),
    )
    assert spike.veto_active
    assert any("Skew Spike Triggered" in r for r in spike.reasons)


def test_tail_trigger_funding_z_score() -> None:
    """Perpetual funding rate |z| > 2.5 must trigger veto."""
    monitor = TailTriggerMonitor(funding_z_threshold=2.5)
    # Mean 0.0001, std ~0.0002
    hist_funding = np.random.normal(0.0001, 0.0002, 84)

    # Extreme funding rate 0.0010 (z > 4.0)
    extreme_funding = monitor.evaluate(
        current_24h_rv=0.50,
        baseline_30d_rv=0.50,
        current_25d_skew=0.04,
        historical_90d_skew=np.linspace(0.01, 0.08, 90),
        current_funding_rate=0.0010,
        historical_28d_funding=hist_funding,
    )
    assert extreme_funding.veto_active
    assert any("Funding Imbalance Triggered" in r for r in extreme_funding.reasons)


def test_cooldown_behavior() -> None:
    """Monitor must enforce cooldown before clearing veto."""
    monitor = TailTriggerMonitor(rv_multiplier_threshold=3.0, cooldown_steps=2)
    hist_skew = np.linspace(0.01, 0.08, 90)
    hist_funding = np.full(84, 0.0001)

    # Shock (RV = 1.80 > 3.0 * 0.45 = 1.35)
    s = monitor.evaluate(1.80, 0.45, 0.04, hist_skew, 0.0001, hist_funding)
    assert s.veto_active
    assert s.action == "FLATTEN_SHORT_GAMMA"

    # Step 1 post-shock: conditions back to normal, but cooldown active
    c1 = monitor.evaluate(0.45, 0.45, 0.04, hist_skew, 0.0001, hist_funding)
    assert c1.veto_active
    assert c1.action == "HALT_NEW_ENTRIES"

    # Step 2 post-shock: cooldown step 2
    c2 = monitor.evaluate(0.45, 0.45, 0.04, hist_skew, 0.0001, hist_funding)
    assert c2.veto_active
    assert c2.action == "HALT_NEW_ENTRIES"

    # Step 3 post-shock: cooldown complete -> ALL_CLEAR
    c3 = monitor.evaluate(0.45, 0.45, 0.04, hist_skew, 0.0001, hist_funding)
    assert not c3.veto_active
    assert c3.action == "ALL_CLEAR"


def test_portfolio_limits_monitor() -> None:
    """Verify limit checking for delta, vega, gamma, and margin ceilings."""
    cfg = RiskLimitsConfig(
        max_portfolio_delta_abs=0.15,
        max_portfolio_vega_btc=50.0,
        max_margin_utilization=0.70,
        min_collateral_buffer_btc=0.50,
    )
    monitor = PortfolioRiskMonitor(cfg)

    # Within bounds:
    # 40000 spot, 2000 USD delta = 0.05 BTC delta < 0.15
    # 20 BTC vega < 50
    # 2.0 BTC margin on 5.0 BTC equity = 40% margin < 70%
    # Free collateral = 3.0 BTC > 0.50 BTC
    res = monitor.check_portfolio(
        portfolio_delta_usd=2000.0,
        spot_price=40000.0,
        portfolio_vega_btc=20.0,
        portfolio_gamma=0.001,
        current_margin_used_btc=2.0,
        total_equity_btc=5.0,
    )
    assert res.passed
    assert len(res.breaches) == 0

    # Margin breach: 4.0 BTC used on 5.0 BTC equity = 80% > 70%
    res_margin = monitor.check_portfolio(
        portfolio_delta_usd=2000.0,
        spot_price=40000.0,
        portfolio_vega_btc=20.0,
        portfolio_gamma=0.001,
        current_margin_used_btc=4.0,
        total_equity_btc=5.0,
    )
    assert not res_margin.passed
    assert any("Margin breach" in b for b in res_margin.breaches)
