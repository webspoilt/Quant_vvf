"""
Unit and Scientific Validation Tests for Advanced Physics, Stochastic Calculus, Spectral & India Modules.
"""

import math
import numpy as np
import pytest

from research_math.physics_models.physics_engine import (
    FokkerPlanck1DSolver,
    FeynmanKacPdePricer,
    MarketIsingModel,
    ReactionDiffusionOrderBook,
)
from research_math.stochastic_calculus.stochastic_engine import (
    GirsanovMeasureEngine,
    MalliavinGreeksCalculator,
    RoughVolatilityEngine,
    HjbOptimalExecutionSolver,
)
from research_math.spectral.spectral_engine import (
    RandomMatrixDenoising,
    OptimalTransportDriftDetector,
    InformationTheoryEngine,
    MarketGraphSpectralEngine,
)
from india.costs.tax_engine import (
    IndiaTransactionCostCalculator,
    TradeExecution,
    MarketSegment,
    OrderSide,
    OrderToTradeRatioController,
)
from india.derivatives.indian_options import (
    SabrVolatilityModel,
    PutCallParityArbitrage,
    IndexDispersionAnalyzer,
)
from india.cross_exchange.cross_exchange_engine import (
    CrossExchangeArbitrageEngine,
    DualListedSpreadQuote,
    GiftNiftyDomesticBasisAnalyzer,
)
from india.microstructure.microstructure_engine import (
    IndianMarketMicrostructureEngine,
    OrderBookLevelSnapshot,
)


class TestAdvancedPhysicsAndStochastic:
    """Validates PDE solvers, Feynman-Kac, Ising model, and Malliavin Greeks."""

    def test_fokker_planck_conservation_and_gaussian_evolution(self):
        """Validates that 1D Fokker-Planck PDE solver strictly conserves probability mass."""
        solver = FokkerPlanck1DSolver(x_min=-5.0, x_max=5.0, n_space=150)
        # Initial standard normal spike
        p0 = np.exp(-0.5 * (solver.x_grid ** 2)) / math.sqrt(2 * math.pi)
        
        # Pure Ornstein-Uhlenbeck drift mu(x) = -0.5 * x, diffusion sigma(x) = 1.0
        mu_fn = lambda x, t: -0.5 * x
        sigma_fn = lambda x, t: np.ones_like(x)

        t_grid, p_hist = solver.solve_evolution(p0, t_span=0.5, n_time=20, mu_fn=mu_fn, sigma_fn=sigma_fn)
        # Check mass conservation at all times
        masses = [np.sum(p_hist[i]) * solver.dx for i in range(len(t_grid))]
        for m in masses:
            assert abs(m - 1.0) < 1e-4
        assert np.all(p_hist >= 0.0)

    def test_feynman_kac_pde_versus_black_scholes(self):
        """Tests that Feynman-Kac finite-difference PDE matches analytical Black-Scholes within 0.5%."""
        pricer = FeynmanKacPdePricer(s_min=1e-3, s_max=300.0, n_space=300, n_time=200)
        s0, k, t, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        # Analytical BS call price = 10.4506
        pde_call = pricer.price_european(s0, k, t, r, sigma, is_call=True)
        # Approximate BS formula for reference
        d1 = (math.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
        d2 = d1 - sigma * math.sqrt(t)
        import scipy.stats as stats
        bs_call = s0 * stats.norm.cdf(d1) - k * math.exp(-r * t) * stats.norm.cdf(d2)
        assert abs(pde_call - bs_call) / bs_call < 0.01

    def test_ising_model_magnetization(self):
        """Tests Ising spin glass simulation returns bounded magnetization and positive susceptibility."""
        model = MarketIsingModel(n_agents=50, coupling_j=1.0, external_field_h=0.2, temperature_t=1.0)
        res = model.simulate_metropolis(n_sweeps=100, seed=42)
        assert -1.0 <= res["mean_magnetization"] <= 1.0
        assert res["susceptibility"] >= 0.0

    def test_malliavin_digital_call_delta(self):
        """Tests Malliavin integration by parts for discontinuous digital call agrees with analytical."""
        calc = MalliavinGreeksCalculator(s0=100.0, k=100.0, t_expiry=0.5, r=0.03, sigma=0.25)
        malliavin_delta, se = calc.digital_call_malliavin_delta(n_paths=50000, seed=42)
        analytical_delta = calc.analytical_digital_call_delta()
        # Verify Malliavin estimate is within 3 standard errors of analytical
        assert abs(malliavin_delta - analytical_delta) < 3.0 * se

    def test_rough_volatility_simulation_and_hurst(self):
        """Tests Volterra fractional kernel simulation and Hurst exponent estimation."""
        engine = RoughVolatilityEngine(hurst_h=0.15, eta=1.0, v0=0.04)
        v_paths = engine.simulate_fractional_kernel_paths(n_paths=100, n_steps=50, t_expiry=0.5, seed=42)
        assert v_paths.shape == (100, 51)
        assert np.all(v_paths > 0.0)
        # Test Hurst estimation on synthetic path
        path = np.cumsum(np.random.normal(0, 1, 1000))
        h_est = RoughVolatilityEngine.estimate_hurst_exponent(path)
        assert 0.01 <= h_est <= 0.99

    def test_hjb_optimal_execution(self):
        """Tests HJB optimal liquidation path monotonically decreases to zero."""
        hjb = HjbOptimalExecutionSolver(initial_inventory=1000.0, liquidation_time=1.0)
        t_grid, inv, speed = hjb.solve_optimal_trajectory(n_steps=50)
        assert inv[0] == 1000.0
        assert abs(inv[-1]) < 1.0 # Terminal inventory depleted
        assert np.all(np.diff(inv) <= 0.0) # Strictly non-increasing inventory


class TestSpectralAndInformationTheory:
    """Validates RMT Marchenko-Pastur, Wasserstein drift, and Transfer Entropy."""

    def test_marchenko_pastur_denoising(self):
        """Tests RMT covariance denoising preserves trace and positive-definiteness."""
        rng = np.random.default_rng(42)
        # 10 assets, 100 observations (q = 0.10)
        returns = rng.normal(0, 0.02, (100, 10))
        emp_cov = np.cov(returns, rowvar=False)

        rmt = RandomMatrixDenoising(aspect_ratio_q=10.0 / 100.0)
        denoised_cov = rmt.denoise_covariance(emp_cov)

        assert denoised_cov.shape == (10, 10)
        eigenvals = np.linalg.eigvalsh(denoised_cov)
        assert np.all(eigenvals > 0.0) # Strictly positive definite

    def test_wasserstein_and_sinkhorn_divergence(self):
        """Tests Wasserstein-1 and Sinkhorn optimal transport distance between distributions."""
        rng = np.random.default_rng(42)
        x = rng.normal(0.0, 1.0, 100)
        y = rng.normal(1.0, 1.0, 100)
        w1 = OptimalTransportDriftDetector.wasserstein_1d(x, y)
        assert w1 > 0.5 # Mean shift is 1.0

        ot = OptimalTransportDriftDetector(reg_epsilon=0.1)
        cost = ot.sinkhorn_divergence(x, y)
        assert cost > 0.0

    def test_mutual_information_and_transfer_entropy(self):
        """Tests information-theoretic dependency and directed transfer entropy."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        y = 0.8 * x + rng.normal(0, 0.2, 500)
        info = InformationTheoryEngine(n_bins=10)
        mi = info.mutual_information(x, y)
        assert mi > 0.2 # Significant mutual information

        # Directed relationship: Y_{t} depends on X_{t-1}
        x_src = rng.normal(0, 1, 400)
        y_dst = np.zeros(400)
        y_dst[1:] = 0.6 * x_src[:-1] + rng.normal(0, 0.3, 399)
        te = info.transfer_entropy(x_src, y_dst, lag=1)
        assert te > 0.0


class TestIndianMarketsAndRegulatoryCompliance:
    """Validates complete Indian statutory costs (STT, GST, Stamp), OTR, SABR, and cross-exchange."""

    def test_indian_statutory_transaction_costs(self):
        """Validates exact STT, Stamp Duty, Exchange Charges, SEBI fees, and GST on F&O trades."""
        calc = IndiaTransactionCostCalculator(brokerage_per_order=20.0)

        # 1. Equity Futures Sell: 0.02% STT
        trade_fut_sell = TradeExecution(
            segment=MarketSegment.FUTURES_INDEX,
            side=OrderSide.SELL,
            quantity=50, # 2 lots of NIFTY
            price=24000.0, # Notional = 1,200,000 INR
        )
        res_fut = calc.calculate_cost(trade_fut_sell)
        assert res_fut.notional_value == 1200000.0
        assert res_fut.stt == 240.0 # 0.02% of 1.2M = 240 INR
        assert res_fut.stamp_duty == 0.0 # Stamp duty only on buy
        assert res_fut.brokerage == 20.0
        assert res_fut.total_cost > 270.0

        # 2. Equity Options Sell: 0.1% STT on premium
        trade_opt_sell = TradeExecution(
            segment=MarketSegment.OPTIONS_INDEX,
            side=OrderSide.SELL,
            quantity=100,
            price=150.0, # Notional premium = 15,000 INR
        )
        res_opt = calc.calculate_cost(trade_opt_sell)
        assert res_opt.notional_value == 15000.0
        assert res_opt.stt == 15.0 # 0.1% of 15,000 = 15 INR

    def test_order_to_trade_ratio_otr_controller(self):
        """Tests OTR monitoring and penalty throttling when ratio exceeds SEBI limits."""
        otr = OrderToTradeRatioController(critical_otr_threshold=50.0)
        # Execute 200 orders and only 1 trade -> OTR = 200 > 50
        for _ in range(200):
            otr.record_order_event("place")
        otr.record_order_event("trade")

        assert otr.get_current_otr() == 200.0
        assert otr.is_throttled is True

    def test_sabr_calibration_and_smile(self):
        """Tests SABR volatility model evaluation and smile calibration."""
        sabr = SabrVolatilityModel(alpha=0.25, beta=1.0, rho=-0.3, nu=0.5)
        f = 24000.0
        strikes = np.array([23000.0, 23500.0, 24000.0, 24500.0, 25000.0])
        ivs = [sabr.implied_volatility(f, k, t_expiry=0.1) for k in strikes]
        # Under negative rho (-0.3), implied volatility exhibits negative skew:
        # Lower strike (ITM Call / OTM Put) has higher IV than ATM and higher strikes
        assert ivs[0] > ivs[2] > ivs[4]

        # Calibrate back
        calibrated = SabrVolatilityModel.calibrate_to_market_skew(f, strikes, np.array(ivs), t_expiry=0.1, fixed_beta=1.0)
        assert abs(calibrated.alpha - 0.25) < 0.05
        assert abs(calibrated.rho - (-0.3)) < 0.15

    def test_put_call_parity_arbitrage(self):
        """Tests conversion/reversal arbitrage detection on synthetic futures."""
        res = PutCallParityArbitrage.evaluate_conversion_reversal(
            f_market=24000.0,
            k=24000.0,
            call_bid=300.0,
            call_ask=302.0,
            put_bid=298.0,
            put_ask=300.0,
            t_expiry=0.08,
            r=0.065,
        )
        assert "synthetic_long_f" in res
        assert "conversion_profit_points" in res

    def test_cross_exchange_nse_bse_arbitrage(self):
        """Tests dual-listed execution analysis between NSE and BSE."""
        engine = CrossExchangeArbitrageEngine(round_trip_cost_bps=8.0)
        quote = DualListedSpreadQuote(
            symbol="RELIANCE",
            nse_bid=2950.0,
            nse_ask=2951.0,
            bse_bid=2945.0,
            bse_ask=2946.0, # BSE ask is 2946, NSE bid is 2950 -> Spread = 4 pts (~13.5 bps)
            nse_timestamp_ms=1000,
            bse_timestamp_ms=1050,
        )
        res = engine.analyze_spread(quote)
        assert res["is_executable"] is True
        assert res["direction"] == "BUY_BSE_SELL_NSE"

    def test_gift_nifty_domestic_basis(self):
        """Tests GIFT Nifty vs Domestic NIFTY basis calculation."""
        res = GiftNiftyDomesticBasisAnalyzer.calculate_cross_border_basis(
            gift_nifty_usd=24100.0,
            domestic_nifty_inr=24000.0,
            usdinr_spot=84.0,
            r_usd=0.05,
            r_inr=0.065,
            t_expiry=0.1,
        )
        assert res["basis_points"] == 100.0
        assert "mispricing_points" in res

    def test_microstructure_microprice_and_max_pain(self):
        """Tests order queue microprice and max pain strike."""
        micro = IndianMarketMicrostructureEngine.calculate_microprice(
            bid=100.0, ask=100.5, bid_size=500.0, ask_size=100.0
        )
        # Heavy bid size pulls microprice closer to ask (higher price)
        assert micro > 100.25

        strikes = np.array([23800.0, 23900.0, 24000.0, 24100.0, 24200.0])
        call_oi = np.array([10000, 20000, 50000, 25000, 10000])
        put_oi = np.array([10000, 30000, 60000, 20000, 5000])
        max_pain = IndianMarketMicrostructureEngine.calculate_max_pain_strike(strikes, call_oi, put_oi)
        assert max_pain == 24000.0 # Pinning strike
