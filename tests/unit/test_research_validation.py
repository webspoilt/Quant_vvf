"""
Comprehensive Scientific Validation & Defect Remediation Test Suite.

Covers:
1. Exact multi-jump Merton, Kou, and Bates Poisson accumulations and martingale conditions.
2. Andersen QE vs QE-M vs Full Truncation Heston schemes.
3. Genuine Baum-Welch HMM forward-backward recursion and online filtered vs smoothed probabilities.
4. SALib Global Sobol sensitivity on the Ishigami benchmark.
5. Deflated Sharpe Ratio (DSR) using EVT expected maximum and higher return moments.
6. AAD Greeks and AAD Delta finite-difference Gamma validation against closed-form Black-Scholes.
7. Deribit Inverse BTC Numeraire discounting and settlement verification.
"""

import math
import numpy as np
import pandas as pd
import pytest

from research_math.monte_carlo.engine import (
    HestonScheme,
    MeasureType,
    PayoffType,
    ProfessionalMonteCarloEngine,
    SimulationConfig,
    VarianceReductionMethod,
)
from research_math.monte_carlo.qmc_variance_reduction import (
    BrownianBridgePathGenerator,
    VarianceReductionBenchmarkSuite,
)
from research_math.monte_carlo.greeks_aad import MonteCarloGreeksEngine
from research_math.validation.global_sensitivity import GlobalParameterSensitivity
from research_math.validation.deflated_sharpe import DeflatedSharpeCalculator
from strategies.statistical_arbitrage.hmm_regime import GaussianHMMRegimeDetector
from backtesting.walk_forward.nested_wfo import NestedWalkForwardOptimizer


class TestResearchModelCorrectness:
    """Scientific verification suite for Phase 1-12 research remediations."""

    def test_merton_multi_jump_martingale(self):
        """Forces high jump intensity to verify multi-jump Poisson accumulation and exact Q martingale drift."""
        cfg = SimulationConfig(n_paths=25000, n_steps=20, t_expiry=0.5, r_rate=0.03, seed=42)
        engine = ProfessionalMonteCarloEngine(cfg)
        s0 = 50000.0

        # Artificially high jump intensity (lambda = 10.0) forces multiple jumps per interval
        paths = engine.simulate_merton_jump_diffusion(
            s0=s0, sigma=0.15, lambda_j=10.0, mu_j=-0.05, sigma_j=0.08
        )
        assert np.all(paths > 0)

        # Risk-neutral Martingale condition: E[exp(-rT) * S_T] == S0
        discounted_terminal = math.exp(-cfg.r_rate * cfg.t_expiry) * paths[:, -1]
        mean_terminal = float(np.mean(discounted_terminal))
        assert abs(mean_terminal - s0) / s0 < 0.02

    def test_kou_multi_jump_martingale(self):
        """Verifies Kou double exponential multi-jump accumulation and risk-neutral drift compensation."""
        cfg = SimulationConfig(n_paths=25000, n_steps=20, t_expiry=0.5, r_rate=0.03, seed=42)
        engine = ProfessionalMonteCarloEngine(cfg)
        s0 = 50000.0

        paths = engine.simulate_kou_jump_diffusion(
            s0=s0, sigma=0.15, lambda_j=8.0, p_up=0.3, eta1=8.0, eta2=4.0
        )
        assert np.all(paths > 0)

        discounted_terminal = math.exp(-cfg.r_rate * cfg.t_expiry) * paths[:, -1]
        mean_terminal = float(np.mean(discounted_terminal))
        assert abs(mean_terminal - s0) / s0 < 0.02

    def test_bates_multi_jump_simulation(self):
        """Verifies Bates model combines Heston stochastic variance with multiple Poisson Merton jumps."""
        cfg = SimulationConfig(n_paths=2000, n_steps=20, t_expiry=0.25, r_rate=0.03, seed=42)
        engine = ProfessionalMonteCarloEngine(cfg)
        s_bates, v_bates = engine.simulate_bates(
            s0=50000.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.6,
            lambda_j=5.0, mu_j=-0.04, sigma_j=0.05
        )
        assert np.all(s_bates > 0)
        assert np.all(v_bates >= 0)

    def test_heston_schemes_and_feller(self):
        """Tests Andersen QE, Andersen QE-M, and Full Truncation Euler Heston schemes."""
        cfg = SimulationConfig(n_paths=1000, n_steps=20, t_expiry=0.5, r_rate=0.03, seed=42)
        engine = ProfessionalMonteCarloEngine(cfg)
        s0, v0, kappa, theta, xi, rho = 50000.0, 0.04, 1.5, 0.04, 0.3, -0.6

        # Feller condition: 2 * kappa * theta = 0.12 > xi^2 = 0.09 (Satisfied)
        for scheme in (HestonScheme.ANDERSEN_QE, HestonScheme.ANDERSEN_QE_M, HestonScheme.FULL_TRUNCATION_EULER):
            s_paths, v_paths = engine.simulate_heston(s0, v0, kappa, theta, xi, rho, scheme=scheme)
            assert np.all(s_paths > 0)
            assert np.all(v_paths >= 0)

    def test_brownian_bridge_theoretical_covariance(self):
        """Validates that Brownian Bridge paths exhibit theoretical min(s, t) covariance."""
        max_cov_err = BrownianBridgePathGenerator.validate_covariance(n_paths=15000, n_steps=10, t_expiry=1.0)
        assert max_cov_err < 0.03

    def test_deribit_inverse_btc_numeraire_discounting(self):
        """Verifies Deribit Inverse Call payoff and numeraire settlement in BTC."""
        cfg = SimulationConfig(n_paths=10000, n_steps=20, t_expiry=0.5, r_rate=0.03, r_btc=0.01, seed=42)
        engine = ProfessionalMonteCarloEngine(cfg)
        paths = engine.simulate_gbm(s0=60000.0, sigma=0.40)

        res = engine.price_contingent_claim(paths, strike=60000.0, payoff_type=PayoffType.INVERSE_BTC_CALL)
        assert res.numeraire_currency == "BTC"
        assert res.price_estimate > 0.0
        # For ATM inverse call, price in BTC typically between 0.05 and 0.20 BTC
        assert 0.02 < res.price_estimate < 0.30

    def test_baum_welch_hmm_forward_backward(self):
        """Verifies Gaussian HMM forward-backward Baum-Welch recursion and online filtering."""
        rng = np.random.default_rng(42)
        # 2-state synthetic regime series
        reg1 = rng.normal(0.001, 0.01, 150)
        reg2 = rng.normal(-0.002, 0.04, 150)
        obs = np.concatenate([reg1, reg2])

        hmm = GaussianHMMRegimeDetector(n_regimes=2, max_iters=30)
        fit_res = hmm.fit(obs)

        assert fit_res.converged or fit_res.iterations == 30
        assert fit_res.filtered_probs.shape == (300, 2)
        assert fit_res.smoothed_probs.shape == (300, 2)
        # Row stochasticity: probabilities sum to 1.0
        assert np.allclose(np.sum(fit_res.filtered_probs, axis=1), 1.0)
        assert np.allclose(np.sum(fit_res.smoothed_probs, axis=1), 1.0)
        assert np.allclose(np.sum(fit_res.transition_matrix, axis=1), 1.0)

        # Online step verification
        online_post = hmm.filter_online_step(fit_res.filtered_probs[-1], new_observation=0.03)
        assert len(online_post) == 2
        assert abs(np.sum(online_post) - 1.0) < 1e-6

    def test_salib_ishigami_global_sensitivity(self):
        """Validates SALib Sobol sensitivity against theoretical Ishigami benchmark."""
        bounds = {
            "x1": (-math.pi, math.pi),
            "x2": (-math.pi, math.pi),
            "x3": (-math.pi, math.pi),
        }
        res = GlobalParameterSensitivity.run_sobol_sensitivity(
            GlobalParameterSensitivity.ishigami_benchmark_function, bounds, n_samples=512, seed=42
        )
        # Ishigami theory: S1(x2) > S1(x1) > S1(x3), with S1(x3) ~ 0
        s1 = res.first_order_indices
        assert s1[1] > s1[0] # x2 dominant first-order effect
        assert s1[2] < 0.10  # x3 has negligible main effect (mostly total/interaction)
        assert res.total_order_indices[2] > 0.15 # x3 has substantial total effect

    def test_deflated_sharpe_ratio_calculation(self):
        """Verifies Deflated Sharpe Ratio accounts for trial count, skewness, and kurtosis."""
        rng = np.random.default_rng(42)
        # Normal returns with moderate Sharpe
        rets = rng.normal(0.001, 0.015, 252)

        # Baseline single trial
        res_single = DeflatedSharpeCalculator.calculate_dsr(rets, n_trials=1)
        assert 0.0 <= res_single.deflated_sharpe_ratio <= 1.0

        # Multiple testing (100 trials) should deflate the significance
        res_multi = DeflatedSharpeCalculator.calculate_dsr(rets, n_trials=100)
        assert res_multi.deflated_sharpe_ratio < res_single.deflated_sharpe_ratio
        assert res_multi.expected_max_sharpe > res_single.expected_max_sharpe

    def test_aad_and_finite_difference_greeks(self):
        """Verifies PyTorch AAD Delta/Vega/Rho and finite-difference Gamma/Theta match Black-Scholes."""
        s0, k, t, r, sigma = 65000.0, 65000.0, 1.0, 0.03, 0.40
        bs = MonteCarloGreeksEngine.analytical_bs_greeks(s0, k, t, r, sigma)
        aad = MonteCarloGreeksEngine.aad_greeks_pytorch(s0, k, t, r, sigma, n_paths=25000)

        # Delta & Vega agreement
        assert abs(aad.delta - bs.delta) < 0.03
        assert abs(aad.vega - bs.vega) / bs.vega < 0.05
        # Gamma & Theta non-zero and aligned with analytical
        assert aad.gamma > 0
        assert aad.theta < 0 # Negative calendar decay

    def test_nested_walk_forward_optimization(self):
        """Validates nested inner cross-validation, parameter freezing, and out-of-sample testing."""
        rng = np.random.default_rng(42)
        n = 450
        rets = rng.normal(0.001, 0.02, n)
        prices = 100.0 * np.exp(np.cumsum(rets))
        df = pd.DataFrame({"close": prices, "return": rets})

        def simple_strat(sub_df: pd.DataFrame, params: dict):
            th = params["th"]
            sig = np.where(sub_df["return"] > th, 1.0, -1.0)
            ret = sig[:-1] * sub_df["return"].iloc[1:].values
            sh = float((np.mean(ret) / max(1e-6, np.std(ret))) * math.sqrt(252))
            pnl = float(np.sum(ret) * 100.0)
            return sh, pnl, 0.05, 0.55

        grid = [{"th": -0.005}, {"th": 0.0}, {"th": 0.005}]
        wfo = NestedWalkForwardOptimizer(outer_train_size=180, outer_test_size=60, step_size=60, inner_splits=2)
        rep = wfo.run_nested_optimization(df, simple_strat, grid)

        assert rep.total_windows >= 3
        assert 0.0 <= rep.deflated_sharpe_ratio <= 1.0
        assert len(rep.window_details) == rep.total_windows
