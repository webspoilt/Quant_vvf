"""Unit tests for realized volatility measures, bipower variation, and Corsi HAR-RV."""
from __future__ import annotations

import numpy as np
import pytest

from libs.core.har_rv import (
    CorsiHAR_RV,
    GARCH_EWMA_RV,
    NaiveRandomWalkRV,
    SimpleMovingAverageRV,
    diebold_mariano_test,
    qlike_loss,
)
from libs.core.realized_measures import (
    annualize_rv,
    bipower_variation,
    compute_log_returns,
    jump_variation,
    realized_variance_from_returns,
    subsampled_realized_variance_5min,
)


def test_log_returns_and_realized_variance() -> None:
    """Verify log return calculation and basic RV identity."""
    prices = np.array([100.0, 105.0, 102.0, 108.0])
    returns = compute_log_returns(prices)
    assert len(returns) == 3
    expected_r0 = np.log(105.0 / 100.0)
    assert returns[0] == pytest.approx(expected_r0, abs=1e-12)

    rv = realized_variance_from_returns(returns)
    assert rv == pytest.approx(float(np.sum(returns ** 2)), abs=1e-12)

    # Annualized vol
    ann_vol = annualize_rv(rv, trading_days=365)
    assert ann_vol == pytest.approx(np.sqrt(rv * 365), abs=1e-12)


def test_subsampled_realized_variance_5min() -> None:
    """Verify 5-minute subsampling across 1-minute interleaved subgrids."""
    # Create 1-minute timestamps across 2 hours (120 minutes = 7200 sec)
    timestamps = np.arange(0, 7201, 60)
    np.random.seed(42)
    # Price path with microsecond noise
    increments = np.random.normal(0, 0.001, size=len(timestamps))
    log_p = np.cumsum(increments)
    prices = 30000.0 * np.exp(log_p)

    rv_sub = subsampled_realized_variance_5min(timestamps, prices, interval_sec=300, subgrid_offset_sec=60)
    assert rv_sub > 0.0
    # Sanity check magnitude
    assert 1e-6 < rv_sub < 0.1


def test_bipower_variation_and_jump_robustness() -> None:
    """Bipower variation must remain robust when a large discrete jump occurs."""
    np.random.seed(123)
    # 288 5-minute returns in a trading day
    continuous_returns = np.random.normal(0, 0.003, size=288)

    rv_clean = realized_variance_from_returns(continuous_returns)
    bv_clean = bipower_variation(continuous_returns)
    # For Gaussian continuous returns, BV is close to RV: E[BV] = E[RV]
    assert bv_clean == pytest.approx(rv_clean, rel=0.15)

    # Inject a large discrete price jump (+10%) at index 144
    jump_returns = continuous_returns.copy()
    jump_returns[144] += 0.10

    rv_jump = realized_variance_from_returns(jump_returns)
    bv_jump = bipower_variation(jump_returns)

    # RV must explode due to the jump
    assert rv_jump > rv_clean + 0.009  # 0.10^2 = 0.01

    # BV must remain close to the continuous baseline (jump robust)
    assert abs(bv_jump - bv_clean) < 0.003
    assert bv_jump < 0.5 * rv_jump

    # Jump variation correctly isolates the jump
    j = jump_variation(rv_jump, bv_jump)
    assert j > 0.007


def test_qlike_loss_properties() -> None:
    """Verify strictly proper scoring rule and asymmetry of QLIKE loss."""
    h = 0.04  # 20% vol squared

    # Zero loss at perfect forecast
    assert qlike_loss(h, h) == pytest.approx(0.0, abs=1e-12)

    # Positive loss away from target
    loss_over = qlike_loss(h, 0.06)
    loss_under = qlike_loss(h, 0.02)
    assert loss_over > 0.0
    assert loss_under > 0.0

    # Asymmetry: under-forecasting is penalized more heavily than equal relative over-forecasting
    # h = 0.04. Compare h_hat = 0.02 (factor of 0.5) vs h_hat = 0.08 (factor of 2.0)
    # loss(0.04, 0.02) = 2 - ln(2) - 1 = 1 - 0.693 = 0.307
    # loss(0.04, 0.08) = 0.5 - ln(0.5) - 1 = -0.5 + 0.693 = 0.193
    l_under = qlike_loss(0.04, 0.02)
    l_over = qlike_loss(0.04, 0.08)
    assert l_under > l_over, "QLIKE must penalize under-forecasting more heavily than over-forecasting"


def test_har_rv_model_fit_and_predict() -> None:
    """Test Corsi HAR-RV feature construction, fitting, and prediction."""
    np.random.seed(42)
    # 200 days of daily realized variance
    base_rv = 0.0009  # ~3% daily vol squared
    rv_series = np.clip(base_rv + 0.0003 * np.random.randn(200), 1e-5, 0.01)

    model = CorsiHAR_RV().fit(rv_series)
    assert model.params is not None
    assert model.params.r_squared > 0.0
    assert model.params.beta_d >= -0.5

    # Predict one step ahead
    pred = model.predict_one_step(rv_series[-30:])
    assert pred > 0.0
    assert 1e-4 < pred < 0.02


def test_diebold_mariano_test() -> None:
    """Test Diebold-Mariano test functionality."""
    np.random.seed(42)
    n = 100
    loss_bench = np.random.exponential(1.0, size=n)
    # Candidate with systematically higher loss
    loss_cand = loss_bench + 0.3 + np.random.normal(0, 0.1, size=n)

    dm_stat, p_val = diebold_mariano_test(loss_bench, loss_cand)
    assert dm_stat > 0.0  # Benchmark has lower loss
    assert p_val < 0.01   # Statistically significant
