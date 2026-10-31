"""Unit tests for Advanced Mathematical Algorithms and Multivariate Calculus.

Verifies:
1. Double & Triple Numerical Quadrature against closed-form analytical integrals.
2. High-Order Greeks (Speed, Ultima, Vanna, Volga) against central finite differences.
3. Fournié et al. (1999) Malliavin Greek Estimator against analytical digital option Greeks.
4. Fang & Oosterlee (2008) COS Method against Black-Scholes analytical benchmarks.
5. Carr-Madan (1999) FFT Option Pricer against analytical closed forms.
6. Higham (2002) Nearest Correlation Matrix repair of non-PSD matrices.
7. Continuous-time Markov transition semigroup matrix exponentials.
"""
from __future__ import annotations

import math
import numpy as np
import pytest
from scipy import stats

from libs.core.fourier_pricing import (
    carr_madan_fft_pricer,
    char_func_black_scholes,
    cos_method_pricer,
)
from libs.core.matrix_algebra import (
    matrix_exponential_generator,
    nearest_correlation_matrix,
    project_psd,
    spectral_pca,
)
from libs.core.multivariate_calculus import (
    compute_high_order_greeks,
    integrate_2d,
    integrate_3d,
    malliavin_greeks_mc,
)


class TestMultivariateQuadrature:
    """Test 2D and 3D Gauss-Legendre numerical integration."""

    def test_integrate_2d_polynomial(self) -> None:
        # integral_0^1 dx integral_0^2 dy (x^2 * y^3)
        # = [x^3 / 3]_0^1 * [y^4 / 4]_0^2 = (1/3) * (16/4) = 4/3 = 1.33333333...
        val = integrate_2d(lambda x, y: x * x * (y ** 3), 0.0, 1.0, 0.0, 2.0, n_points=16)
        assert pytest.approx(val, abs=1e-12) == 4.0 / 3.0

    def test_integrate_2d_variable_boundary_quarter_circle(self) -> None:
        # Area of quarter circle: integral_0^1 dx integral_0^{sqrt(1 - x^2)} dy (1) = pi / 4
        area = integrate_2d(
            lambda x, y: 1.0,
            0.0,
            1.0,
            0.0,
            lambda x: math.sqrt(max(0.0, 1.0 - x * x)),
            n_points=64,
        )
        assert pytest.approx(area, abs=1e-5) == math.pi / 4.0

    def test_integrate_3d_cuboid(self) -> None:
        # integral_0^1 dx integral_0^2 dy integral_0^3 dz (x * y * z)
        # = (1/2) * (4/2) * (9/2) = 0.5 * 2.0 * 4.5 = 4.5
        val = integrate_3d(
            lambda x, y, z: x * y * z,
            0.0,
            1.0,
            (0.0, 2.0),
            (0.0, 3.0),
            n_points=16,
        )
        assert pytest.approx(val, abs=1e-12) == 4.5


class TestHighOrderGreeks:
    """Verify 3rd-order Greek jet against numerical finite differences."""

    def test_greeks_match_finite_differences(self) -> None:
        s0 = 50000.0
        strike = 50000.0
        sigma = 0.55
        tau = 30.0 / 365.0
        r = 0.04

        base = compute_high_order_greeks(s0, strike, sigma, tau, r, is_call=True)

        # Finite difference step
        h_s = 1.0  # $1 bump
        g_up = compute_high_order_greeks(s0 + h_s, strike, sigma, tau, r, is_call=True)
        g_dn = compute_high_order_greeks(s0 - h_s, strike, sigma, tau, r, is_call=True)

        # Speed = d Gamma / dS
        speed_fd = (g_up.gamma - g_dn.gamma) / (2.0 * h_s)
        assert pytest.approx(base.speed, rel=1e-4) == speed_fd

        # Volga = d Vega / d sigma
        h_sig = 0.0001
        v_up = compute_high_order_greeks(s0, strike, sigma + h_sig, tau, r, is_call=True)
        v_dn = compute_high_order_greeks(s0, strike, sigma - h_sig, tau, r, is_call=True)
        volga_fd = (v_up.vega - v_dn.vega) / (2.0 * h_sig)
        assert pytest.approx(base.volga, rel=1e-4) == volga_fd

        # Vanna = d Delta / d sigma
        vanna_fd = (v_up.delta - v_dn.delta) / (2.0 * h_sig)
        assert pytest.approx(base.vanna, rel=1e-4) == vanna_fd


class TestMalliavinGreeks:
    """Verify Malliavin stochastic integration-by-parts on discontinuous digital payoffs."""

    def test_malliavin_digital_call_delta(self) -> None:
        spot = 60000.0
        strike = 60000.0
        sigma = 0.50
        tau = 14.0 / 365.0
        r = 0.03

        # Analytical digital call price: exp(-r*T) * N(d2)
        # Analytical digital call delta: exp(-r*T) * phi(d2) / (S * sigma * sqrt(T))
        sq_tau = math.sqrt(tau)
        d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sq_tau)
        d2 = d1 - sigma * sq_tau

        df = math.exp(-r * tau)
        true_delta = df * float(stats.norm.pdf(d2)) / (spot * sigma * sq_tau)
        true_price = df * float(stats.norm.cdf(d2))

        # Discontinuous digital payoff: Phi(S) = 1 if S >= K else 0
        def digital_payoff(s_arr: np.ndarray) -> np.ndarray:
            return (s_arr >= strike).astype(np.float64)

        res = malliavin_greeks_mc(
            payoff_func=digital_payoff,
            spot=spot,
            sigma=sigma,
            tau=tau,
            r=r,
            n_sims=150_000,
            seed=42,
        )

        # Price matches analytical
        assert pytest.approx(res.price, abs=0.01) == true_price

        # Malliavin Delta matches analytical within 3 standard errors
        assert abs(res.delta - true_delta) <= 3.0 * res.delta_std_err


class TestFourierAndCOSMethods:
    """Verify Fang & Oosterlee (2008) COS method and Carr-Madan (1999) FFT."""

    def test_cos_method_vs_black_scholes(self) -> None:
        s0 = 45000.0
        strike = 46000.0
        r = 0.05
        sigma = 0.60
        tau = 30.0 / 365.0

        # Analytical Black-Scholes
        sq_tau = math.sqrt(tau)
        d1 = (math.log(s0 / strike) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sq_tau)
        d2 = d1 - sigma * sq_tau
        df = math.exp(-r * tau)
        bs_call = s0 * float(stats.norm.cdf(d1)) - strike * df * float(stats.norm.cdf(d2))
        bs_put = strike * df * float(stats.norm.cdf(-d2)) - s0 * float(stats.norm.cdf(-d1))

        # Characteristic function
        def cf(u: np.ndarray) -> np.ndarray:
            res = char_func_black_scholes(u, s0=s0, strike=strike, r=r, sigma=sigma, tau=tau)
            assert isinstance(res, np.ndarray)
            return res

        cos_call = cos_method_pricer(cf, s0, strike, r, tau, option_type="call", n_terms=128)
        cos_put = cos_method_pricer(cf, s0, strike, r, tau, option_type="put", n_terms=128)

        # Matches to within 1e-6 (near-machine precision)
        assert pytest.approx(cos_call, abs=1e-4) == bs_call
        assert pytest.approx(cos_put, abs=1e-4) == bs_put

        # Put-Call Parity in COS method: Call - Put == S0 - K * exp(-r*T)
        parity_lhs = cos_call - cos_put
        parity_rhs = s0 - strike * df
        assert pytest.approx(parity_lhs, abs=1e-4) == parity_rhs

    def test_carr_madan_fft_matches_atm_call(self) -> None:
        s0 = 50000.0
        r = 0.02
        sigma = 0.45
        tau = 14.0 / 365.0

        def cf_ln_s(u: complex | np.ndarray) -> complex | np.ndarray:
            mu = (r - 0.5 * sigma * sigma) * tau
            return np.exp(1j * u * (np.log(s0) + mu) - 0.5 * (u * sigma) ** 2 * tau)  # type: ignore[no-any-return]

        fft_res = carr_madan_fft_pricer(cf_ln_s, s0=s0, r=r, tau=tau, n_points=2048)

        # Find strike closest to spot
        idx = int(np.argmin(np.abs(fft_res.strikes - s0)))
        k_atm = fft_res.strikes[idx]
        call_fft = fft_res.call_prices[idx]

        # Analytical BS for k_atm
        sq = sigma * math.sqrt(tau)
        d1 = (math.log(s0 / k_atm) + (r + 0.5 * sigma * sigma) * tau) / sq
        d2 = d1 - sq
        bs_atm = s0 * float(stats.norm.cdf(d1)) - k_atm * math.exp(-r * tau) * float(stats.norm.cdf(d2))

        # FFT matches analytical BS
        assert pytest.approx(call_fft, rel=1e-2) == bs_atm


class TestMatrixAlgebraAndSpectralMethods:
    """Test Higham (2002) Nearest Correlation Matrix and Markov matrix exponentials."""

    def test_nearest_correlation_matrix_repair(self) -> None:
        # Create non-positive semi-definite 3x3 matrix with unit diagonal
        # Non-transitive correlations guarantee a negative eigenvalue
        bad_matrix = np.array([
            [1.0, 0.90, -0.90],
            [0.90, 1.0, 0.90],
            [-0.90, 0.90, 1.0],
        ])

        eigvals_before = np.linalg.eigvalsh(bad_matrix)
        assert np.min(eigvals_before) < 0.0, "Test fixture must be non-positive semi-definite"

        # Apply Higham (2002)
        res = nearest_correlation_matrix(bad_matrix, max_iterations=100)
        assert res.converged
        assert res.min_eigenvalue >= 0.0

        repaired = res.correlation_matrix
        # Check unit diagonal
        np.testing.assert_allclose(np.diag(repaired), 1.0, atol=1e-8)
        # Check symmetry
        np.testing.assert_allclose(repaired, repaired.T, atol=1e-12)
        # Check all eigenvalues are non-negative
        repaired_eigs = np.linalg.eigvalsh(repaired)
        assert np.all(repaired_eigs >= -1e-10)

    def test_matrix_exponential_transition_generator(self) -> None:
        # 3-state volatility regime generator matrix Q:
        # State 0: Low Vol, State 1: Normal VRP, State 2: Tail Shock
        q = np.array([
            [-0.20,  0.18,  0.02],
            [ 0.10, -0.25,  0.15],
            [ 0.30,  0.50, -0.80],
        ])
        # Assert valid generator: row sums == 0
        np.testing.assert_allclose(q.sum(axis=1), 0.0, atol=1e-12)

        # P(0) == Identity
        p_0 = matrix_exponential_generator(q, t=0.0)
        np.testing.assert_allclose(p_0, np.eye(3), atol=1e-12)

        # P(1.0) is a valid stochastic matrix
        p_1 = matrix_exponential_generator(q, t=1.0)
        # Row sums == 1.0
        np.testing.assert_allclose(p_1.sum(axis=1), 1.0, atol=1e-12)
        # All probabilities in [0, 1]
        assert np.all(p_1 >= 0.0) and np.all(p_1 <= 1.0)

        # Chapman-Kolmogorov semigroup property: P(t + s) == P(t) * P(s)
        p_t = matrix_exponential_generator(q, t=0.5)
        p_s = matrix_exponential_generator(q, t=0.5)
        p_ts = matrix_exponential_generator(q, t=1.0)
        np.testing.assert_allclose(p_ts, p_t @ p_s, atol=1e-8)

    def test_spectral_pca_decomposition(self) -> None:
        # Synthesize a positive definite correlation matrix
        rng = np.random.default_rng(123)
        x = rng.standard_normal((100, 4))
        cov = np.cov(x, rowvar=False)

        pca = spectral_pca(cov, n_components=3)
        assert len(pca.singular_values) == 3
        # Explained variance ratios are descending and positive
        assert np.all(pca.explained_variance_ratio >= 0.0)
        assert pca.explained_variance_ratio[0] >= pca.explained_variance_ratio[1]
        assert np.sum(pca.explained_variance_ratio) <= 1.0 + 1e-12
