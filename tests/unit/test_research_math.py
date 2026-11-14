"""
QUANT_VVF — Comprehensive Unit and Numerical Tests for Mathematical Research Modules.
Tests:
- GARCH(1,1), EGARCH, GJR-GARCH, HAR-RV, QLIKE
- Heston pricing, Fourier inversion, Call-Put Parity, SVI Arbitrage diagnostics
- Hawkes Process stability & intensity, Queue imbalance, Microprice, Almgren-Chriss
- Kalman Filter hedge ratio tracking, Latent volatility filtering
- Ledoit-Wolf shrinkage, CVaR loss threshold optimization, ERC, HRP
"""

import math
import numpy as np
import pytest
from research_math.volatility.advanced_volatility import (
    GarchModel,
    EgarchModel,
    GjrGarchModel,
    HarRvModel,
    calculate_qlike,
    calculate_forecast_metrics,
)
from research_math.stochastic_options.heston_svi_inverse import (
    black_scholes_price,
    black_scholes_greeks,
    implied_volatility,
    HestonModel,
    SviSurface,
    coin_margined_option_payoff,
    inverse_delta_rescaling,
)
from research_math.microstructure.orderbook_hawkes import (
    BivariateHawkesProcess,
    calculate_queue_imbalance,
    calculate_microprice,
    almgren_chriss_execution_cost,
)
from research_math.state_estimation.kalman_filters import (
    KalmanHedgeRatioEstimator,
    LatentVolatilityFilter,
)
from research_math.portfolio.advanced_portfolio import (
    ledoit_wolf_shrinkage,
    cvar_portfolio_optimization,
    risk_parity_weights,
    hierarchical_risk_parity_weights,
)


class TestVolatilityModels:
    def test_garch_fit_and_forecast(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, size=300)
        model = GarchModel()
        res = model.fit(returns)
        assert res["converged"]
        assert 0.0 < res["alpha"] < 1.0
        assert 0.0 < res["beta"] < 1.0
        assert res["alpha"] + res["beta"] < 1.0

        fc = model.forecast(returns, horizon=5)
        assert len(fc) == 5
        assert np.all(fc > 0)

    def test_egarch_fit_and_forecast(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, size=200)
        model = EgarchModel()
        res = model.fit(returns)
        assert "leverage_effect" in res
        fc = model.forecast(returns, horizon=3)
        assert len(fc) == 3
        assert np.all(fc > 0)

    def test_gjr_garch_fit(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, size=200)
        model = GjrGarchModel()
        res = model.fit(returns)
        assert "asymmetry" in res
        fc = model.forecast(returns, horizon=2)
        assert len(fc) == 2
        assert np.all(fc > 0)

    def test_har_rv_fit_and_predict(self):
        np.random.seed(42)
        rv = 0.0004 + np.random.exponential(0.0001, size=100)
        har = HarRvModel(weekly_window=7, monthly_window=30)
        res = har.fit(rv)
        assert res["r_squared"] >= 0.0
        pred = har.predict(rv[-30:])
        assert pred > 0

    def test_qlike_metric(self):
        actual = np.array([0.04, 0.05, 0.03])
        forecast = np.array([0.04, 0.045, 0.032])
        qlike = calculate_qlike(actual, forecast)
        assert qlike >= 0.0
        # Zero loss if identical
        assert math.isclose(calculate_qlike(actual, actual), 0.0, abs_tol=1e-10)


class TestStochasticOptions:
    def test_black_scholes_analytical_and_greeks(self):
        price = black_scholes_price(spot=100.0, strike=100.0, time_to_expiry=1.0, risk_free=0.05, volatility=0.2, is_call=True)
        assert 10.0 < price < 11.0  # standard textbook ATM call is ~10.45

        greeks = black_scholes_greeks(spot=100.0, strike=100.0, time_to_expiry=1.0, risk_free=0.05, volatility=0.2, is_call=True)
        assert 0.60 < greeks["delta"] < 0.65
        assert greeks["gamma"] > 0
        assert greeks["vega"] > 0

    def test_heston_model_pricing_and_parity(self):
        heston = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        assert heston.feller_satisfied

        call = heston.price(spot=100.0, strike=100.0, time_to_expiry=0.5, risk_free=0.03, is_call=True)
        put = heston.price(spot=100.0, strike=100.0, time_to_expiry=0.5, risk_free=0.03, is_call=False)
        assert call > 0
        assert put > 0

        # Put-Call Parity check: C - P = S - K*exp(-r*T)
        parity_diff = (call - put) - (100.0 - 100.0 * math.exp(-0.03 * 0.5))
        assert math.isclose(parity_diff, 0.0, abs_tol=1e-3)

    def test_svi_surface_durrleman_arbitrage(self):
        # Arbitrage-free SVI slice
        svi = SviSurface(a=0.04, b=0.1, rho=-0.5, m=0.0, sigma=0.1)
        k_range = np.linspace(-0.5, 0.5, 20)
        assert not svi.has_butterfly_arbitrage(k_range)

    def test_coin_margined_inverse_payoff_and_rescaling(self):
        btc_call, usd_call = coin_margined_option_payoff(spot_terminal=60000.0, strike=50000.0, is_call=True)
        assert math.isclose(usd_call, 10000.0)
        assert math.isclose(btc_call, 10000.0 / 60000.0)

        # Inverse delta (S0/S1)^2 rescaling
        rescaled = inverse_delta_rescaling(spot=60000.0, spot_ref=50000.0, bs_usd_delta=0.5)
        expected = 0.5 * ((50000.0 / 60000.0) ** 2)
        assert math.isclose(rescaled, expected)


class TestMicrostructure:
    def test_bivariate_hawkes_stability(self):
        mu = np.array([0.5, 0.5])
        alpha = np.array([[0.2, 0.1], [0.1, 0.2]])
        beta = np.array([[1.0, 1.0], [1.0, 1.0]])
        hawkes = BivariateHawkesProcess(mu, alpha, beta)
        assert hawkes.is_stable
        assert hawkes.spectral_radius < 1.0

        t1 = np.array([0.1, 0.5, 1.2])
        t2 = np.array([0.3, 0.8, 1.5])
        l1, l2 = hawkes.compute_intensities(t1, t2, eval_time=2.0)
        assert l1 > 0 and l2 > 0

    def test_microprice_and_imbalance(self):
        imbalance = calculate_queue_imbalance(bid_qty=10.0, ask_qty=5.0)
        assert math.isclose(imbalance, (10 - 5) / 15)

        micro = calculate_microprice(bid_price=50000.0, ask_price=50002.0, bid_qty=10.0, ask_qty=5.0)
        # Higher bid quantity pulls microprice closer to ask price (buyer pressure)
        assert micro > 50001.0

    def test_almgren_chriss_costs(self):
        costs = almgren_chriss_execution_cost(trade_size=5.0, daily_volume=1000.0, volatility=0.6)
        assert costs["total_slippage_bps"] > 0


class TestStateEstimation:
    def test_kalman_dynamic_hedge_ratio(self):
        kf = KalmanHedgeRatioEstimator(
            initial_state=np.array([0.0, 1.0]),
            initial_cov=np.eye(2) * 0.1,
            process_noise=np.eye(2) * 1e-4,
            measurement_noise=0.01,
        )
        intercept, beta, _ = kf.update(x=100.0, y=100.5)
        assert np.isfinite(intercept)
        assert np.isfinite(beta)

    def test_latent_vol_filter(self):
        lvf = LatentVolatilityFilter(phi=0.95, q_var=0.01, r_var=0.05)
        h = lvf.filter_step(obs_log_vol=-3.0)
        assert np.isfinite(h)


class TestPortfolioMathematics:
    def test_ledoit_wolf_shrinkage(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, size=(100, 4))
        shrunk_cov, intensity = ledoit_wolf_shrinkage(returns)
        assert shrunk_cov.shape == (4, 4)
        assert 0.0 <= intensity <= 1.0
        # Positive definite eigenvalues
        eigvals = np.linalg.eigvals(shrunk_cov)
        assert np.all(eigvals > 0)

    def test_cvar_loss_threshold_optimization(self):
        np.random.seed(42)
        scenarios = np.random.normal(0.001, 0.02, size=(200, 3))
        w, eta, cvar = cvar_portfolio_optimization(scenarios, alpha=0.95)
        assert len(w) == 3
        assert math.isclose(np.sum(w), 1.0, abs_tol=1e-5)
        assert np.all(w >= -1e-6)
        assert cvar > 0

    def test_risk_parity_and_hrp(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, size=(150, 4))
        cov = np.cov(returns, rowvar=False)

        w_erc = risk_parity_weights(cov)
        assert len(w_erc) == 4
        assert math.isclose(np.sum(w_erc), 1.0, abs_tol=1e-5)

        w_hrp = hierarchical_risk_parity_weights(returns)
        assert len(w_hrp) == 4
        assert math.isclose(np.sum(w_hrp), 1.0, abs_tol=1e-5)
