"""
Unit Tests for Portfolio Construction & Optimization Layer.
"""

import numpy as np
import pandas as pd
import pytest

from portfolio.optimizer import PortfolioOptimizer, AllocationResult


class TestPortfolioLayer:

    @pytest.fixture
    def multi_asset_returns(self):
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        r_btc = pd.Series(np.random.normal(0, 0.03, n), index=dates)
        r_eth = r_btc * 0.7 + np.random.normal(0, 0.02, n)
        r_sol = r_btc * 0.5 + np.random.normal(0, 0.04, n)
        return pd.DataFrame({"BTC": r_btc, "ETH": r_eth, "SOL": r_sol})

    def test_ledoit_wolf_shrinkage(self, multi_asset_returns):
        cov, delta = PortfolioOptimizer.ledoit_wolf_shrinkage(multi_asset_returns)
        assert cov.shape == (3, 3)
        assert 0.0 <= delta <= 1.0
        # Positive semi-definite check
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0.0)

    def test_risk_parity_allocation(self, multi_asset_returns):
        res = PortfolioOptimizer.risk_parity_allocation(multi_asset_returns)
        assert isinstance(res, AllocationResult)
        assert len(res.weights) == 3
        assert np.isclose(sum(res.weights.values()), 1.0, atol=1e-4)
        assert res.expected_portfolio_volatility > 0.0
        assert res.diversification_ratio >= 1.0

    def test_hierarchical_risk_parity(self, multi_asset_returns):
        res = PortfolioOptimizer.hierarchical_risk_parity(multi_asset_returns)
        assert isinstance(res, AllocationResult)
        assert len(res.weights) == 3
        assert np.isclose(sum(res.weights.values()), 1.0, atol=1e-4)
        assert res.expected_portfolio_volatility > 0.0
