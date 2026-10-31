"""Unit tests for Gatheral's Quasi-Explicit SVI surface and arbitrage verification."""
from __future__ import annotations

import numpy as np
import pytest

from libs.core.svi_surface import (
    SVIParameters,
    SVISurface,
    calibrate_svi_quasi_explicit,
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
    durrleman_density_factor,
    svi_first_derivative,
    svi_implied_vol,
    svi_second_derivative,
    svi_total_variance,
)


def test_svi_parameter_validation() -> None:
    """Invalid parameter bounds must raise ValueError."""
    # b < 0
    with pytest.raises(ValueError, match="b must be non-negative"):
        SVIParameters(a=0.04, b=-0.1, rho=0.0, m=0.0, sigma=0.1)

    # |rho| > 1
    with pytest.raises(ValueError, match="rho must be in"):
        SVIParameters(a=0.04, b=0.1, rho=1.2, m=0.0, sigma=0.1)

    # sigma <= 0
    with pytest.raises(ValueError, match="sigma must be positive"):
        SVIParameters(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=-0.05)


def test_svi_derivatives_finite_difference() -> None:
    """Verify analytical w' and w'' match central finite differences."""
    params = SVIParameters(a=0.04, b=0.15, rho=-0.3, m=0.05, sigma=0.12)
    k_eval = np.array([-0.5, -0.2, 0.0, 0.1, 0.4])
    eps = 1e-6

    # Analytical
    w_prime = svi_first_derivative(k_eval, params)
    w_second = svi_second_derivative(k_eval, params)

    # Numerical w'
    w_plus = svi_total_variance(k_eval + eps, params)
    w_minus = svi_total_variance(k_eval - eps, params)
    num_prime = (w_plus - w_minus) / (2.0 * eps)
    np.testing.assert_allclose(w_prime, num_prime, rtol=1e-5, atol=1e-6)

    # Numerical w'' via central difference of analytical first derivative (avoids catastrophic cancellation)
    eps_d2 = 1e-5
    w_p_plus = svi_first_derivative(k_eval + eps_d2, params)
    w_p_minus = svi_first_derivative(k_eval - eps_d2, params)
    num_second = (w_p_plus - w_p_minus) / (2.0 * eps_d2)
    np.testing.assert_allclose(w_second, num_second, rtol=1e-5, atol=1e-6)


def test_butterfly_arbitrage_detection() -> None:
    """Verify butterfly arbitrage detector flags valid vs invalid parameters."""
    # Valid typical market smile
    valid_params = SVIParameters(a=0.04, b=0.10, rho=-0.25, m=0.02, sigma=0.15)
    is_valid, min_g = check_butterfly_arbitrage(valid_params)
    assert is_valid
    assert min_g >= 0.0

    # Pathological parameters creating negative density / butterfly arbitrage
    invalid_params = SVIParameters(a=0.001, b=0.8, rho=-0.8, m=0.1, sigma=0.02)
    is_invalid, min_g_bad = check_butterfly_arbitrage(invalid_params)
    assert not is_invalid
    assert min_g_bad < 0.0


def test_calendar_arbitrage_detection() -> None:
    """Check detection of calendar spread arbitrage across expiries."""
    # Early expiry: 14 days, lower variance
    p_early = SVIParameters(a=0.02, b=0.08, rho=-0.2, m=0.0, sigma=0.1)
    tau_early = 14.0 / 365.0

    # Late expiry: 30 days, higher variance
    p_late = SVIParameters(a=0.045, b=0.10, rho=-0.2, m=0.0, sigma=0.12)
    tau_late = 30.0 / 365.0

    is_free, min_spread = check_calendar_arbitrage(p_early, tau_early, p_late, tau_late)
    assert is_free
    assert min_spread >= 0.0

    # Inverted: late has lower variance (calendar arbitrage)
    is_free_inv, min_spread_inv = check_calendar_arbitrage(p_late, tau_early, p_early, tau_late)
    assert not is_free_inv
    assert min_spread_inv < 0.0


def test_svi_quasi_explicit_calibration() -> None:
    """Calibrate SVI to synthetic noisy market smile and verify fit."""
    true_params = SVIParameters(a=0.035, b=0.12, rho=-0.35, m=0.02, sigma=0.15)
    k_grid = np.linspace(-0.6, 0.6, 25)
    w_clean = svi_total_variance(k_grid, true_params)

    np.random.seed(42)
    noise = np.random.normal(0.0, 0.0005, size=len(k_grid))
    w_noisy = np.maximum(w_clean + noise, 0.001)

    calib_params = calibrate_svi_quasi_explicit(k_grid, w_noisy)

    # Check fit quality
    w_calib = svi_total_variance(k_grid, calib_params)
    rmse = np.sqrt(np.mean((w_calib - w_noisy) ** 2))
    assert rmse < 0.005, f"Calibration RMSE {rmse} too high"

    # Check arbitrage-free
    is_free, min_g = check_butterfly_arbitrage(calib_params)
    assert is_free, f"Calibrated SVI violates butterfly arbitrage: min_g={min_g}"


def test_svi_surface_multislice_interpolation() -> None:
    """Test SVISurface multi-slice registration and interpolation."""
    surface = SVISurface()
    tau1 = 7.0 / 365.0
    tau2 = 30.0 / 365.0
    fwd = 50_000.0

    p1 = SVIParameters(a=0.015, b=0.08, rho=-0.2, m=0.0, sigma=0.1)
    p2 = SVIParameters(a=0.050, b=0.12, rho=-0.2, m=0.0, sigma=0.12)

    surface.add_slice(tau1, p1, fwd)
    surface.add_slice(tau2, p2, fwd)

    # Validate surface calendar monotonicity
    is_calendar_free, spread = surface.validate_calendar_arbitrage()
    assert is_calendar_free

    # ATM Implied vols
    iv1 = surface.atm_implied_vol(tau1)
    iv2 = surface.atm_implied_vol(tau2)
    assert 0.3 < iv1 < 1.5
    assert 0.3 < iv2 < 1.5

    # Interpolate at intermediate tau = 14 days
    tau_mid = 14.0 / 365.0
    iv_mid = surface.atm_implied_vol(tau_mid)
    assert min(iv1, iv2) <= iv_mid <= max(iv1, iv2)
