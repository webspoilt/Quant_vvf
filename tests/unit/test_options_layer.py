"""
Unit and Integration Tests for Options Pricing, VRP Analytics, and Delta Hedging.
"""

import numpy as np
import pandas as pd
import pytest

from options.pricing.black_scholes import BlackScholesEngine, OptionGreeks
from options.vrp.vrp_analyzer import VRPAnalyzer, VRPAnalysisResult
from options.hedging.delta_hedger import DynamicDeltaHedger, HedgingSimulationResult


class TestOptionsAndDerivativesLayer:

    def test_black_scholes_greeks_and_parity(self):
        S = 65000.0
        K = 65000.0
        T = 30.0 / 365.0
        r = 0.05
        sigma = 0.50

        call = BlackScholesEngine.calculate_greeks(S, K, T, r, sigma, option_type="call")
        put = BlackScholesEngine.calculate_greeks(S, K, T, r, sigma, option_type="put")

        assert isinstance(call, OptionGreeks)
        assert isinstance(put, OptionGreeks)

        # Call-Put Parity: C - P = S - K * exp(-r * T)
        parity_diff = (call.price - put.price) - (S - K * np.exp(-r * T))
        assert np.isclose(parity_diff, 0.0, atol=1e-5)

        # Delta relationship: Delta_call - Delta_put = 1.0
        assert np.isclose(call.delta - put.delta, 1.0, atol=1e-5)

        # Identical Gamma & Vega
        assert np.isclose(call.gamma, put.gamma, atol=1e-7)
        assert np.isclose(call.vega, put.vega, atol=1e-7)

    def test_implied_volatility_inversion(self):
        S = 65000.0
        K = 67000.0
        T = 0.25
        r = 0.04
        true_sigma = 0.62

        greeks = BlackScholesEngine.calculate_greeks(S, K, T, r, true_sigma, option_type="call")
        solved_sigma = BlackScholesEngine.implied_volatility(greeks.price, S, K, T, r, option_type="call")
        assert np.isclose(solved_sigma, true_sigma, atol=1e-4)

    def test_vrp_analytics(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        iv = pd.Series(0.60 + np.random.normal(0, 0.02, 100), index=dates)
        rv = pd.Series(0.50 + np.random.normal(0, 0.03, 100), index=dates)

        res = VRPAnalyzer.analyze_historical_vrp(iv, rv)
        assert isinstance(res, VRPAnalysisResult)
        assert res.mean_vrp > 0.0
        assert res.is_positive_premium_dominant

        skew = VRPAnalyzer.calculate_skew_25delta(0.65, 0.58, 0.60)
        assert skew > 0.0

    def test_dynamic_delta_hedging_simulation(self):
        n = 100
        spot = 65000.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
        deltas = 0.50 + 0.05 * np.sin(np.linspace(0, 3.14, n))

        hedger = DynamicDeltaHedger(rebalance_threshold_btc=0.10)
        res = hedger.simulate_hedging_path(spot, deltas)

        assert isinstance(res, HedgingSimulationResult)
        assert res.total_rebalances > 0
        assert res.total_hedging_costs_usd > 0.0
        assert res.max_unhedged_delta_btc >= 0.0
