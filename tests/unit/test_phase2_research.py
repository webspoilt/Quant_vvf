"""
Comprehensive Unit & Regression Tests for Phase 2 Research Modules (P1–P5).
"""

import numpy as np
import pandas as pd
import pytest

from research.volatility.p1_volatility import VolatilityResearchEngine, GARCHParams
from research.pairs_trading.p2_pairs import PairsTradingEngine, CointegrationResult
from research.regimes.p3_regimes import MarketRegimeEngine, RegimeAnalysisResult
from research.pca.p4_pca import PCAFactorEngine, PCAResult
from research.directional.p5_direction import DirectionalSignificanceEngine, DirectionalEvaluationResult


@pytest.fixture
def synthetic_market_data():
    np.random.seed(42)
    n = 400
    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    # Generate synthetic stationary log returns with GARCH-like clustering
    eps = np.random.normal(0, 1, n)
    sigma = np.zeros(n)
    sigma[0] = 0.02
    for t in range(1, n):
        sigma[t] = np.sqrt(0.00002 + 0.10 * (sigma[t-1] * eps[t-1])**2 + 0.85 * (sigma[t-1]**2))
    returns_btc = pd.Series(sigma * eps, index=dates, name="BTC")

    # Cointegrated asset: Log prices are cointegrated
    p_btc = pd.Series(100.0 * np.exp(np.cumsum(returns_btc)), index=dates)
    # Price of ETH is linearly cointegrated with BTC price + stationary mean-reverting noise
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.7 * noise[t-1] + np.random.normal(0, 0.5)
    p_eth = pd.Series(0.05 * p_btc + 10.0 + noise, index=dates)

    return {
        "returns_btc": returns_btc,
        "prices_btc": p_btc,
        "prices_eth": p_eth,
    }


class TestPhase2ResearchModules:

    def test_p1_garch11_mle_and_walk_forward(self, synthetic_market_data):
        rets = synthetic_market_data["returns_btc"]
        engine = VolatilityResearchEngine(trading_days=365)

        # 1. GARCH fit
        params = engine.fit_garch11(rets.values)
        assert isinstance(params, GARCHParams)
        assert params.alpha > 0.0
        assert params.beta > 0.0
        assert params.alpha + params.beta < 1.0
        assert params.unconditional_vol > 0.0
        assert np.isfinite(params.log_likelihood)

        # 2. Walk-forward evaluation
        res = engine.fit_evaluate_walk_forward(rets, train_window=200, forecast_horizon=5)
        assert "garch_rmse" in res
        assert "hist_rv_rmse" in res
        assert "ml_rmse" in res
        assert res["sample_count"] > 0

    def test_p2_engle_granger_cointegration(self, synthetic_market_data):
        p_btc = synthetic_market_data["prices_btc"]
        p_eth = synthetic_market_data["prices_eth"]

        c_res = PairsTradingEngine.test_engle_granger_cointegration(p_eth, p_btc)
        assert isinstance(c_res, CointegrationResult)
        assert np.isfinite(c_res.adf_statistic)
        assert 0.0 <= c_res.p_value <= 1.0
        assert c_res.hedge_ratio > 0.0
        assert c_res.half_life_days > 0.0

        # Rolling Z-score check
        beta, alpha, spread = PairsTradingEngine.calculate_ols_hedge_ratio(p_eth, p_btc)
        z = PairsTradingEngine.calculate_rolling_z_score(spread, window=30)
        assert len(z.dropna()) > 0

    def test_p3_market_regime_gmm(self, synthetic_market_data):
        rets = synthetic_market_data["returns_btc"]
        engine = MarketRegimeEngine(n_regimes=2, random_state=42)
        res = engine.fit_predict(rets, rolling_vol_window=14)

        assert isinstance(res, RegimeAnalysisResult)
        assert res.converged
        assert len(res.regime_labels) > 0
        assert res.transition_matrix.shape == (2, 2)
        # Check transition matrix stochastic property (rows sum to 1)
        np.testing.assert_allclose(res.transition_matrix.sum(axis=1), np.ones(2), rtol=1e-5)

    def test_p4_pca_spectral_decomposition(self, synthetic_market_data):
        r_btc = synthetic_market_data["returns_btc"]
        # Create a 3-asset return matrix
        r_eth = r_btc * 0.8 + np.random.normal(0, 0.01, len(r_btc))
        r_sol = r_btc * 1.2 + np.random.normal(0, 0.02, len(r_btc))
        matrix = pd.DataFrame({"BTC": r_btc, "ETH": r_eth, "SOL": r_sol})

        res = PCAFactorEngine.decompose_returns(matrix, n_components=3)
        assert isinstance(res, PCAResult)
        assert len(res.eigenvalues) == 3
        # Eigenvalues must be sorted descending
        assert res.eigenvalues[0] >= res.eigenvalues[1] >= res.eigenvalues[2]
        # Cumulative variance of all components must sum to 1.0
        assert np.isclose(res.cumulative_variance[-1], 1.0, atol=1e-5)
        assert res.factor_loadings.shape == (3, 3)

    def test_p5_directional_significance(self, synthetic_market_data):
        rets = synthetic_market_data["returns_btc"]
        # Construct predictive feature table
        features = pd.DataFrame({
            "lag1": rets.shift(1),
            "lag2": rets.shift(2),
            "rv5": rets.rolling(5).std(),
        }).dropna()

        res = DirectionalSignificanceEngine.evaluate_classifier_chronological(
            features,
            rets.loc[features.index],
            train_ratio=0.70
        )
        assert isinstance(res, DirectionalEvaluationResult)
        assert 0.0 <= res.accuracy <= 1.0
        assert 0.0 <= res.binomial_p_value <= 1.0
        assert res.confidence_interval_95[0] <= res.accuracy <= res.confidence_interval_95[1]
        assert res.n_test_samples > 0
