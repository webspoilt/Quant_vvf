"""Unit tests for RiskGatekeeper and programmatic veto authority (risk_constants model)."""
from __future__ import annotations

import pytest

from risk.risk_constants import (
    MAX_MARGIN_UTILIZATION,
    MAX_PORTFOLIO_DELTA_ABS,
    MAX_PORTFOLIO_GAMMA,
    MAX_PORTFOLIO_VEGA_BTC,
    RiskAction,
    RiskGatekeeper,
)
from risk.tail_triggers import RiskVetoDecision, TailTriggerMonitor


def test_risk_gatekeeper_approves_safe_orders() -> None:
    """Gatekeeper must approve orders within all hard boundaries."""
    gatekeeper = RiskGatekeeper()

    result = gatekeeper.validate_proposed_execution(
        proposed_vega_btc=-20.0,
        proposed_delta_btc=0.05,
        proposed_gamma=0.001,
        proposed_margin_btc=2.0,
        total_equity_btc=10.0,  # 20% margin < 70%
        strike_contract_counts=[4, 4, 4, 3],  # Max single is 4/15 = 26.7% < 35%
        circuit_breaker_active=False,
    )
    assert result.approved
    assert result.action == RiskAction.APPROVE


def test_risk_gatekeeper_vetoes_vega_and_delta_breaches() -> None:
    """Gatekeeper must reject executions breaching vega or delta ceilings."""
    gatekeeper = RiskGatekeeper()

    # Vega breach: 60 BTC > 50 BTC ceiling
    res_vega = gatekeeper.validate_proposed_execution(
        proposed_vega_btc=-60.0,
        proposed_delta_btc=0.05,
        proposed_gamma=0.001,
        proposed_margin_btc=1.0,
        total_equity_btc=10.0,
    )
    assert not res_vega.approved
    assert res_vega.action == RiskAction.REJECT
    assert any("Vega breach" in r for r in res_vega.reasons)

    # Delta breach: 0.25 BTC > 0.15 BTC ceiling
    res_delta = gatekeeper.validate_proposed_execution(
        proposed_vega_btc=-20.0,
        proposed_delta_btc=0.25,
        proposed_gamma=0.001,
        proposed_margin_btc=1.0,
        total_equity_btc=10.0,
    )
    assert not res_delta.approved
    assert any("Delta breach" in r for r in res_delta.reasons)


def test_risk_gatekeeper_fail_closed_on_error() -> None:
    """Fail-Closed Principle: unexpected exceptions must reject execution outright."""
    gatekeeper = RiskGatekeeper()

    # Pass invalid equity to trigger potential division / calculation anomaly
    result = gatekeeper.validate_proposed_execution(
        proposed_vega_btc=-10.0,
        proposed_delta_btc=0.0,
        proposed_gamma=0.0,
        proposed_margin_btc=1.0,
        total_equity_btc=-1.0,  # Invalid negative equity
    )
    # Must fail or reject
    assert not result.approved
    assert result.action == RiskAction.REJECT


def test_tail_trigger_monitor_programmatic_veto() -> None:
    """TailTriggerMonitor must exercise programmatic veto authority over proposed orders."""
    monitor = TailTriggerMonitor()

    proposed_orders = [("BTC-CALL", 5), ("BTC-PUT", 5)]

    # 1. Normal all-clear
    approved, reason, executable = monitor.veto_strategy_output(
        proposed_orders=proposed_orders,
        proposed_vega_btc=-10.0,
        proposed_delta_btc=0.02,
        proposed_gamma=0.001,
        proposed_margin_btc=1.0,
        total_equity_btc=10.0,
    )
    assert approved
    assert len(executable) == 2

    # 2. Tail trigger fired veto
    tail_decision = RiskVetoDecision(
        veto_active=True,
        action="FLATTEN_SHORT_GAMMA",
        reasons=("RV Shock > 3.0x",),
        rv_ratio=3.5,
        skew_percentile=0.98,
        funding_z_score=2.8,
    )
    approved_veto, reason_veto, executable_veto = monitor.veto_strategy_output(
        proposed_orders=proposed_orders,
        proposed_vega_btc=-10.0,
        proposed_delta_btc=0.02,
        proposed_gamma=0.001,
        proposed_margin_btc=1.0,
        total_equity_btc=10.0,
        last_veto_decision=tail_decision,
    )
    assert not approved_veto
    assert len(executable_veto) == 0
    assert "VETOED by TailTrigger" in reason_veto
