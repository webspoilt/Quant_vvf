"""Unit tests for TargetGreekProfile generation, DiscreteGreekSolver, and GreeksMonitor."""
from __future__ import annotations

import numpy as np
import pytest

from libs.core.har_rv import CorsiHAR_RV
from libs.core.svi_surface import SVIParameters, SVISurface
from monitoring.greeks_monitor import GreeksMonitor, PositionLeg
from risk.tail_triggers import RiskVetoDecision
from simulation.walk_forward_sim import PurgedWalkForwardEngine
from strategies.greek_solver import DiscreteGreekSolver, OptionChainQuote
from strategies.vrp_profile import TargetGreekProfile, VRPProfileGenerator


def test_vrp_profile_generator_active_and_veto() -> None:
    """Verify TargetGreekProfile generation and risk veto override."""
    surface = SVISurface()
    tau = 14.0 / 365.0
    # SVI with 65% ATM IV
    w0 = (0.65 ** 2) * tau
    p = SVIParameters(a=w0 - 0.1 * w0 * 0.1, b=0.1 * w0, rho=-0.2, m=0.0, sigma=0.10)
    surface.add_slice(tau, p, 40000.0)

    # HAR model forecasting 50% RV (variance ~ 0.000685 daily)
    # IV (65%) - RV (50%) = 15% spread > 4% hurdle
    np.random.seed(42)
    daily_rv = np.clip((0.50 ** 2) / 365.0 + np.random.normal(0, 0.00005, 50), 1e-5, 0.01)
    har = CorsiHAR_RV(log_transform=True).fit(daily_rv)

    generator = VRPProfileGenerator(
        target_dte_days=14.0,
        delta_tolerance_btc=0.15,
        max_portfolio_gamma=0.005,
        max_vega_btc=30.0,
        min_vol_spread_hurdle=0.04,
    )

    # 1. Normal active market
    profile = generator.generate_profile(surface, har, daily_rv)
    assert profile.is_active
    assert profile.target_vega_btc < -5.0  # Selling volatility
    assert profile.min_delta_btc == -0.15
    assert profile.max_delta_btc == +0.15
    assert profile.max_gamma == 0.005

    # 2. Risk veto active
    veto = RiskVetoDecision(
        veto_active=True,
        action="FLATTEN_SHORT_GAMMA",
        reasons=("RV Shock Triggered",),
        rv_ratio=4.0,
        skew_percentile=0.98,
        funding_z_score=3.0,
    )
    profile_vetoed = generator.generate_profile(surface, har, daily_rv, risk_veto=veto)
    assert not profile_vetoed.is_active
    assert profile_vetoed.target_vega_btc == 0.0
    assert profile_vetoed.max_gamma == 0.0
    assert "Risk Veto Active" in profile_vetoed.reason


def test_discrete_greek_solver() -> None:
    """Verify DiscreteGreekSolver selects integer contracts matching target vega."""
    solver = DiscreteGreekSolver()
    tau = 14.0 / 365.0
    spot = 40000.0

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

    # Mock Deribit chain quotes around 40k strike
    chain = [
        OptionChainQuote(
            instrument_id="BTC-14D-40000-C",
            strike=40000.0,
            expiry_tau=tau,
            option_type="call",
            bid_price_btc=0.045,
            ask_price_btc=0.046,
            mid_price_btc=0.0455,
            delta_btc=0.52,
            gamma=0.00002,
            vega_btc=0.05,
            initial_margin_btc=0.15,
        ),
        OptionChainQuote(
            instrument_id="BTC-14D-40000-P",
            strike=40000.0,
            expiry_tau=tau,
            option_type="put",
            bid_price_btc=0.044,
            ask_price_btc=0.045,
            mid_price_btc=0.0445,
            delta_btc=-0.48,
            gamma=0.00002,
            vega_btc=0.05,
            initial_margin_btc=0.15,
        ),
    ]

    result = solver.solve(profile, chain, spot_price=spot, available_margin_btc=100.0)
    assert result.success
    assert len(result.orders) == 2
    # Verify integer quantities and short side
    for order in result.orders:
        assert isinstance(order.quantity, int)
        assert order.quantity > 0
        assert order.side == "SELL"

    # Achieved vega matches desired direction
    assert result.achieved_vega_btc < 0.0


def test_greeks_monitor_inverse_numeraire() -> None:
    """Verify portfolio Greek aggregation and 1/S^2 hedge drift tracking."""
    monitor = GreeksMonitor(delta_neutral_tolerance_btc=0.15)
    spot = 50000.0

    # Short 1 ATM straddle: short call + short put
    positions = {
        "BTC-CALL": PositionLeg(
            instrument_id="BTC-CALL",
            quantity=-1.0,
            strike=50000.0,
            is_option=True,
            option_type="call",
            delta_usd=25000.0,  # 0.5 delta * 50k
            gamma=0.00002,
            vega_btc=0.05,
            theta_usd=-200.0,
            margin_btc=0.15,
        ),
        "BTC-PUT": PositionLeg(
            instrument_id="BTC-PUT",
            quantity=-1.0,
            strike=50000.0,
            is_option=True,
            option_type="put",
            delta_usd=-25000.0,  # -0.5 delta * 50k
            gamma=0.00002,
            vega_btc=0.05,
            theta_usd=-200.0,
            margin_btc=0.15,
        ),
    }

    state = monitor.update_state(
        positions=positions,
        perp_signed_notional_usd=0.0,
        spot_price=spot,
        total_equity_btc=2.0,
        entry_spot=50000.0,
    )
    assert state.is_delta_neutral
    assert state.total_vega_btc == pytest.approx(-0.10, abs=1e-6)

    # Test spot gap +30% to 65000 -> hedge rescale factor == (50000/65000)^2
    state_gap = monitor.update_state(
        positions=positions,
        perp_signed_notional_usd=10000.0,
        spot_price=65000.0,
        total_equity_btc=2.0,
        entry_spot=50000.0,
    )
    expected_rescale = (50000.0 / 65000.0) ** 2
    assert state_gap.inverse_hedge_rescale_factor == pytest.approx(expected_rescale, abs=1e-9)


def test_purged_walk_forward_splits() -> None:
    """Verify CPCV / walk-forward engine eliminates overlap and enforces embargo."""
    engine = PurgedWalkForwardEngine(holding_period_steps=14, embargo_pct=0.02)
    splits = list(engine.generate_purged_walk_forward_splits(n_samples=200, n_splits=4, min_train_size=60))

    assert len(splits) == 4
    for fold in splits:
        # Assert no intersection between train and test
        assert len(set(fold.train_indices) & set(fold.test_indices)) == 0
        # Assert test start is after min_train_size
        assert fold.test_indices[0] >= 60
        # Assert purging: last pre-train index + 14 <= test_indices[0]
        pre_train = fold.train_indices[fold.train_indices < fold.test_indices[0]]
        if len(pre_train) > 0:
            assert fold.test_indices[0] - pre_train[-1] >= 14
