"""
Remediation and Rigorous Mathematical Validation Test Suite.
"""

from datetime import date, timedelta
import math
import numpy as np
import pytest

from india.costs.tax_engine import (
    IndiaStatutoryCostEngine,
    TradeExecution,
    ExchangeVenue,
    MarketSegment,
    OrderSide,
    NseBseSurveillanceOtrController,
)
from research_math.physics_models.physics_engine import (
    FeynmanKacPdePricer,
    FokkerPlanck1DSolver,
    CurieWeissMeanFieldIsing,
    GeneralSpinGlassModel,
)
from research_math.stochastic_calculus.stochastic_engine import (
    GirsanovMeasureEngine,
    MalliavinGreeksCalculator,
    RoughVolatilityEngine,
)
from research_math.spectral.spectral_engine import (
    RandomMatrixDenoising,
    OptimalTransportDriftDetector,
    InformationTheoryEngine,
)
from india.cross_exchange.cross_exchange_engine import (
    CrossExchangeArbitrageEngine,
    DualListedSpreadQuote,
)


class TestRemediatedStatutoryCostsAndOTR:
    """Validates April 1, 2026 STT boundary, intrinsic exercise, and NSE 30-day OTR surveillance."""

    def test_stt_schedule_boundary_april_2026(self):
        """Verifies STT rates before and after April 1, 2026 boundary."""
        engine = IndiaStatutoryCostEngine(brokerage_per_order=20.0)

        # 1. Futures Sale on March 31, 2026 (0.02% STT)
        trade_pre = TradeExecution(
            exchange=ExchangeVenue.NSE,
            segment=MarketSegment.FUTURES_INDEX,
            side=OrderSide.SELL,
            quantity_units=50,
            price=24000.0, # Notional = 1,200,000 INR
            trade_date=date(2026, 3, 31),
        )
        res_pre = engine.calculate_cost(trade_pre)
        assert res_pre.policy_version == "OCT_2024_TO_MAR_2026"
        assert res_pre.stt == 240.0 # 0.02% of 1.2M = 240 INR

        # 2. Futures Sale on April 1, 2026 (0.05% STT under Finance Act 2026)
        trade_post = TradeExecution(
            exchange=ExchangeVenue.NSE,
            segment=MarketSegment.FUTURES_INDEX,
            side=OrderSide.SELL,
            quantity_units=50,
            price=24000.0, # Notional = 1,200,000 INR
            trade_date=date(2026, 4, 1),
        )
        res_post = engine.calculate_cost(trade_post)
        assert res_post.policy_version == "APR_2026_ONWARD"
        assert res_post.stt == 600.0 # 0.05% of 1.2M = 600 INR

    def test_options_premium_and_intrinsic_exercise_taxable_base(self):
        """Validates that exercised options apply STT to INTRINSIC value, not strike notional."""
        engine = IndiaStatutoryCostEngine(brokerage_per_order=20.0)

        # Option exercise on April 15, 2026 (Call exercised: Strike 24,000, Settle 24,200 -> Intrinsic 200 INR)
        trade_exercise = TradeExecution(
            exchange=ExchangeVenue.NSE,
            segment=MarketSegment.OPTIONS_INDEX,
            side=OrderSide.BUY,
            quantity_units=100,
            price=150.0,
            strike_price=24000.0,
            settlement_price=24200.0,
            is_exercise=True,
            trade_date=date(2026, 4, 15),
        )
        res = engine.calculate_cost(trade_exercise)
        # Taxable base = 100 units * 200 intrinsic = 20,000 INR (NOT 100 * 24,000 = 2.4M)
        assert res.taxable_stt_base == 20000.0
        # Post April 2026 exercise STT is 0.15% on intrinsic = 30 INR
        assert res.stt == 30.0

    def test_nse_official_surveillance_rolling_30_days(self):
        """Tests that official NSE OTR surveillance triggers ONLY upon >= 3 instances of OTR >= 2000."""
        controller = NseBseSurveillanceOtrController()
        base_d = date(2026, 4, 1)

        # Instance 1: High OTR on Day 1
        for _ in range(2500):
            controller.record_order_event("place", base_d)
        controller.record_order_event("trade", base_d)
        is_breached, count, _ = controller.check_nse_official_surveillance_breach()
        assert not is_breached
        assert count == 1

        # Instance 2: High OTR on Day 5
        d2 = base_d + timedelta(days=4)
        for _ in range(2100):
            controller.record_order_event("place", d2)
        controller.record_order_event("trade", d2)
        is_breached, count, _ = controller.check_nse_official_surveillance_breach()
        assert not is_breached
        assert count == 2

        # Instance 3: High OTR on Day 10 (Surveillance Breached!)
        d3 = base_d + timedelta(days=9)
        for _ in range(2200):
            controller.record_order_event("place", d3)
        controller.record_order_event("trade", d3)
        is_breached, count, dates = controller.check_nse_official_surveillance_breach()
        assert is_breached
        assert count == 3
        assert len(dates) == 3


class TestRemediatedPhysicsAndStochastics:
    """Validates Feynman-Kac PDE, Girsanov density normalization, and Debiased Sinkhorn."""

    def test_feynman_kac_pde_across_strikes_and_maturities(self):
        """Validates Feynman-Kac PDE against Black-Scholes across multiple strikes and spots."""
        pricer = FeynmanKacPdePricer(s_min=0.0, s_max=300.0, n_space=300, n_time=200)
        import scipy.stats as stats

        for s0 in [80.0, 100.0, 120.0]:
            for k in [90.0, 100.0, 110.0]:
                t, r, sigma = 0.5, 0.05, 0.25
                pde_val = pricer.price_european(s0, k, t, r, sigma, is_call=True)

                d1 = (math.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
                d2 = d1 - sigma * math.sqrt(t)
                bs_val = s0 * stats.norm.cdf(d1) - k * math.exp(-r * t) * stats.norm.cdf(d2)

                assert abs(pde_val - bs_val) < 0.25 # Within 25 cents across strikes

    def test_girsanov_density_normalization_under_mc(self):
        """Tests that E_P[Z_T] == 1.0 (Radon-Nikodym true martingale property)."""
        engine = GirsanovMeasureEngine(lambda_market_price_of_risk=0.25)
        assert engine.check_novikov_condition(t_horizon=1.0) is True

        mean_z, se = engine.verify_density_normalization(n_paths=50000, t_horizon=1.0)
        assert abs(mean_z - 1.0) < 3.0 * se # Within 3 standard errors of 1.0

    def test_debiased_sinkhorn_divergence_self_distance(self):
        """Tests that debiased Sinkhorn divergence satisfies S_eps(X, X) == 0."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        ot = OptimalTransportDriftDetector(reg_epsilon=0.1)

        # Self-divergence should be exactly 0.0
        self_div = ot.debiased_sinkhorn_divergence(x, x)
        assert abs(self_div) < 1e-6

        # Cross-distribution divergence should be strictly positive
        y = rng.normal(1.0, 1.0, 100)
        cross_div = ot.debiased_sinkhorn_divergence(x, y)
        assert cross_div > 0.0

    def test_spin_models_curie_weiss_and_general(self):
        """Tests Curie-Weiss mean-field and general spin glass models."""
        cw = CurieWeissMeanFieldIsing(n_agents=40, coupling_j=1.0, external_field_h=0.1)
        res_cw = cw.simulate(n_sweeps=100)
        assert "mean_magnetization" in res_cw

        # General spin glass with arbitrary J_ij
        j_mat = np.random.normal(0, 0.5, (20, 20))
        j_mat = 0.5 * (j_mat + j_mat.T) # Symmetric
        np.fill_diagonal(j_mat, 0.0)
        h_vec = np.zeros(20)

        sg = GeneralSpinGlassModel(coupling_matrix=j_mat, external_fields=h_vec)
        res_sg = sg.simulate_metropolis(n_sweeps=100)
        assert "susceptibility" in res_sg

    def test_cross_exchange_executable_with_statutory_costs(self):
        """Tests that cross-exchange dual arbitrage uses exact leg costs, rejecting unprofitable spreads."""
        engine = CrossExchangeArbitrageEngine(max_latency_skew_ms=100)

        # Spread that is too thin to cover both statutory legs
        thin_quote = DualListedSpreadQuote(
            symbol="INFY",
            nse_bid=1500.20, nse_ask=1500.30, nse_bid_qty=100, nse_ask_qty=100,
            bse_bid=1500.00, bse_ask=1500.10, bse_bid_qty=100, bse_ask_qty=100, # Gross spread = 10 paise on 1500 (<1 bps)
            nse_timestamp_ms=1000, bse_timestamp_ms=1020,
        )
        res = engine.analyze_spread(thin_quote)
        assert res["is_executable"] is False
