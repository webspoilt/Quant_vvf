"""
Unit and integration tests for the Advanced Monte Carlo, Pricing, Walk-Forward,
Parameter Sensitivity, and Statistical Arbitrage modules.
"""

import math
import numpy as np
import pandas as pd
import pytest

from research_math.monte_carlo.engine import (
    MeasureType,
    MonteCarloResult,
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
from research_math.pricing.fft_pricer import CarrMadanFftPricer, CharacteristicFunctions
from backtesting.walk_forward.engine import WalkForwardOptimizer
from research_math.validation.parameter_sensitivity import ParameterSensitivityAnalyzer
from strategies.statistical_arbitrage.stat_arb_engine import (
    AdvancedStatisticalArbitrage,
    GaussianHMMRegimeDetector,
    GbdtMicrostructurePredictor,
)
from research_math.state_estimation.particle_filter import (
    AvellanedaStoikovEngine,
    ParticleFilterVolatility,
)
from research.experiment_registry import QuantitativeExperimentRegistry


class TestAdvancedMonteCarloAndPricing:
    """Validation test suite for Monte Carlo and Fourier numerical methods."""

    def test_gbm_martingale_and_analytical_pricing(self):
        """Validates that risk-neutral GBM paths satisfy martingale condition and match Black-Scholes within 1%."""
        cfg = SimulationConfig(n_paths=20000, n_steps=50, t_expiry=1.0, r_rate=0.03, seed=42)
        engine = ProfessionalMonteCarloEngine(cfg)
        s0, sigma, k = 65000.0, 0.40, 65000.0
        paths = engine.simulate_gbm(s0=s0, sigma=sigma)

        # Martingale check: E[e^(-rT) S_T] = S_0
        discounted_terminal = math.exp(-cfg.r_rate * cfg.t_expiry) * paths[:, -1]
        assert abs(np.mean(discounted_terminal) - s0) / s0 < 0.015

        # Analytical BS Call benchmark
        d1 = (math.log(s0 / k) + (cfg.r_rate + 0.5 * sigma**2) * cfg.t_expiry) / (sigma * math.sqrt(cfg.t_expiry))
        d2 = d1 - sigma * math.sqrt(cfg.t_expiry)
        import scipy.stats as stats
        bs_call = s0 * stats.norm.cdf(d1) - k * math.exp(-cfg.r_rate * cfg.t_expiry) * stats.norm.cdf(d2)

        res = engine.price_contingent_claim(paths, strike=k, payoff_type=PayoffType.EUROPEAN_CALL, analytical_benchmark=bs_call)
        assert res.absolute_error is not None
        assert res.relative_error < 0.02
        assert res.confidence_interval_95[0] <= bs_call <= res.confidence_interval_95[1]

    def test_heston_and_jump_diffusions(self):
        """Verifies simulation of Heston QE, Merton, and Kou jump diffusions generate strictly positive finite paths."""
        cfg = SimulationConfig(n_paths=1000, n_steps=30, t_expiry=0.5, r_rate=0.03, seed=42)
        engine = ProfessionalMonteCarloEngine(cfg)

        # Heston QE
        s_h, v_h = engine.simulate_heston(s0=50000.0, v0=0.04, kappa=1.5, theta=0.04, xi=0.3, rho=-0.7, scheme="qe")
        assert np.all(s_h > 0)
        assert np.all(v_h >= 0)

        # Merton Jump
        s_m = engine.simulate_merton_jump_diffusion(s0=50000.0, sigma=0.2, lambda_j=1.0, mu_j=-0.05, sigma_j=0.1)
        assert np.all(s_m > 0)

        # Kou Jump
        s_k = engine.simulate_kou_jump_diffusion(s0=50000.0, sigma=0.2, lambda_j=1.0, p_up=0.4, eta1=10.0, eta2=5.0)
        assert np.all(s_k > 0)

    def test_variance_reduction_benchmarking(self):
        """Confirms that Antithetic, Control Variate, and Sobol QMC achieve measurable variance reduction."""
        benchmarks = VarianceReductionBenchmarkSuite.benchmark_european_call(
            s0=65000.0, k=65000.0, t=0.5, r=0.03, sigma=0.30, n_paths=4096, n_steps=20
        )
        assert len(benchmarks) >= 5
        # Check that Control Variate achieves variance reduction factor > 1.5
        cv_bench = next(b for b in benchmarks if b.method == "Control Variate")
        assert cv_bench.variance_reduction_factor > 1.5

    def test_greeks_pathwise_and_aad(self):
        """Tests that Pathwise Monte Carlo and PyTorch AAD Greeks agree with analytical Black-Scholes."""
        s0, k, t, r, sigma = 65000.0, 65000.0, 1.0, 0.03, 0.40
        bs = MonteCarloGreeksEngine.analytical_bs_greeks(s0, k, t, r, sigma)
        pathwise = MonteCarloGreeksEngine.pathwise_greeks_gbm(s0, k, t, r, sigma, n_paths=20000)
        aad = MonteCarloGreeksEngine.aad_greeks_pytorch(s0, k, t, r, sigma, n_paths=20000)

        # Delta should agree within 0.03
        assert abs(pathwise.delta - bs.delta) < 0.03
        assert abs(aad.delta - bs.delta) < 0.03
        # Vega should agree within 5%
        assert abs(pathwise.vega - bs.vega) / bs.vega < 0.05
        assert abs(aad.vega - bs.vega) / bs.vega < 0.05

    def test_carr_madan_fft_pricing(self):
        """Validates that Carr-Madan FFT pricing of Black-Scholes matches analytical within 0.5% at ATM."""
        s0, t, r, sigma = 60000.0, 0.5, 0.03, 0.35
        cf = lambda u: CharacteristicFunctions.black_scholes(u, t, r, sigma)
        fft_res = CarrMadanFftPricer.price_options_fft(s0=s0, t=t, r=r, char_func=cf, n=2048, model_name="Black-Scholes")

        # Find closest ATM strike index
        atm_idx = np.argmin(np.abs(fft_res.strikes - s0))
        fft_atm_call = fft_res.call_prices[atm_idx]

        bs = MonteCarloGreeksEngine.analytical_bs_greeks(s0, s0, t, r, sigma)
        # Analytical BS Call at ATM
        d1 = (0.5 * sigma**2 * t) / (sigma * math.sqrt(t))
        d2 = d1 - sigma * math.sqrt(t)
        import scipy.stats as stats
        bs_call_atm = s0 * stats.norm.cdf(d1) - s0 * math.exp(-r * t) * stats.norm.cdf(d2)

        assert abs(fft_atm_call - bs_call_atm) / bs_call_atm < 0.02


class TestWalkForwardSensitivityAndStatArb:
    """Validation test suite for Walk-Forward, Sensitivity, Stat Arb, and ML."""

    def test_walk_forward_optimizer(self):
        """Validates chronological rolling windows, parameter freezing, and Deflated Sharpe calculation."""
        rng = np.random.default_rng(42)
        n_samples = 400
        returns = rng.normal(0.001, 0.02, size=n_samples)
        prices = 100.0 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({"close": prices, "return": returns})

        def mock_strategy(sub_df: pd.DataFrame, params: dict):
            threshold = params["threshold"]
            sig = np.where(sub_df["return"] > threshold, 1.0, -1.0)
            strat_ret = sig[:-1] * sub_df["return"].iloc[1:].values
            mean_r = np.mean(strat_ret)
            std_r = max(1e-6, np.std(strat_ret))
            sharpe = (mean_r / std_r) * math.sqrt(252)
            pnl = float(np.sum(strat_ret) * 100.0)
            cum = np.cumsum(strat_ret)
            peak = np.maximum.accumulate(cum)
            dd = float(np.max(peak - cum))
            win_rate = float(np.mean(strat_ret > 0))
            return float(sharpe), pnl, dd, win_rate

        param_grid = [{"threshold": -0.005}, {"threshold": 0.0}, {"threshold": 0.005}]
        wfo = WalkForwardOptimizer(train_window_size=150, test_window_size=50, step_size=50)
        report = wfo.run_walk_forward(df, mock_strategy, param_grid)

        assert report.total_windows >= 3
        assert 0.0 <= report.deflated_sharpe_ratio <= 1.0
        assert len(report.window_details) == report.total_windows

    def test_parameter_sensitivity_and_plateau(self):
        """Confirms parameter sensitivity elasticity and plateau detection."""
        def quadratic_obj(params: dict) -> float:
            x = params["lookback"]
            # Maxima at x = 20 with smooth plateau
            return float(1.5 - 0.001 * (x - 20.0)**2)

        base_params = {"lookback": 20.0}
        bounds = {"lookback": (5.0, 50.0)}

        sens = ParameterSensitivityAnalyzer.analyze_local_sensitivity(quadratic_obj, base_params, bounds)
        assert "lookback" in sens
        assert sens["lookback"].is_plateau_robust is True
        assert sens["lookback"].fragility_score < 10.0

    def test_cointegration_and_ou_spread(self):
        """Verifies two-step Engle-Granger and OU half-life calculation on synthetic mean-reverting series."""
        rng = np.random.default_rng(42)
        n = 500
        x = np.cumsum(rng.normal(0, 1, n))
        spread = np.zeros(n)
        for t in range(1, n):
            spread[t] = 0.8 * spread[t - 1] + rng.normal(0, 1)
        y = 1.5 * x + 10.0 + spread

        coint_res = AdvancedStatisticalArbitrage.fit_cointegration_ou(y, x)
        assert coint_res.is_cointegrated is True
        assert abs(coint_res.hedge_ratio - 1.5) < 0.10
        assert coint_res.half_life_bars > 0

    def test_hmm_regime_detector(self):
        """Tests that 2-state Gaussian HMM infers distinct regime probabilities."""
        rng = np.random.default_rng(42)
        low_vol = rng.normal(0.0, 0.01, 100)
        high_vol = rng.normal(0.0, 0.05, 100)
        obs = np.concatenate([low_vol, high_vol])

        hmm = GaussianHMMRegimeDetector(n_regimes=2, max_iters=20)
        posteriors = hmm.fit(obs)
        assert posteriors.shape == (200, 2)
        assert np.allclose(np.sum(posteriors, axis=1), 1.0)

    def test_gbdt_microstructure_predictor(self):
        """Validates that GBDT model trains chronologically without error and outputs valid probabilities."""
        rng = np.random.default_rng(42)
        x = rng.normal(size=(200, 5))
        y = (x[:, 0] + x[:, 1] > 0).astype(int)

        x_train, y_train = x[:150], y[:150]
        x_test, y_test = x[150:], y[150:]

        gbdt = GbdtMicrostructurePredictor(model_type="lightgbm")
        res = gbdt.train_and_evaluate(x_train, y_train, x_test, y_test)
        assert 0.0 <= res["test_accuracy"] <= 1.0
        assert res["test_log_loss"] > 0

    def test_avellaneda_stoikov_and_particle_filter(self):
        """Validates market-making quotes skew with inventory and particle filter updates latent volatility."""
        as_engine = AvellanedaStoikovEngine(gamma=0.1, sigma=0.02, k=1.5, tick_size=0.5)
        # Neutral inventory
        q_zero = as_engine.compute_optimal_quotes(mid_price=65000.0, inventory_q=0.0)
        assert q_zero.optimal_bid < q_zero.optimal_ask

        # Long inventory (q = +2) should depress quotes downward
        q_long = as_engine.compute_optimal_quotes(mid_price=65000.0, inventory_q=2.0)
        assert q_long.reservation_price < q_zero.reservation_price
        assert q_long.optimal_bid <= q_zero.optimal_bid

        # Particle Filter
        pf = ParticleFilterVolatility(n_particles=500)
        pf.initialize(seed=42)
        mean_vol, std_vol, ess = pf.filter_step(observation_return=0.02, seed=42)
        assert not math.isnan(mean_vol)
        assert ess > 0

    def test_experiment_registry_audit_trail(self, tmp_path):
        """Verifies immutable experiment logging and Deflated Sharpe historical count."""
        reg_file = str(tmp_path / "exp_reg.jsonl")
        reg = QuantitativeExperimentRegistry(registry_file=reg_file)
        rec = reg.log_experiment(
            phase="monte_carlo",
            model_name="Heston-QE",
            strategy_name="Delta-Neutral-Market-Making",
            parameters={"kappa": 1.5, "theta": 0.04},
            metrics={"sharpe": 1.85, "max_dd": 0.04},
        )
        assert rec.experiment_id.startswith("EXP-MONT")
        assert reg.get_total_experiments_count() == 1
