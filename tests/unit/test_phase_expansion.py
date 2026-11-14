"""
QUANT_VVF — Unit and Numerical Tests for Advanced Phase 1–5 Research Expansion.
Tests:
- DCC-GARCH dynamic covariance estimation
- Merton jump-diffusion pricing and Kou density
- Generalized Pareto EVT tail estimation & VaR/ES
- Clayton and Gumbel copula CDF & tail dependence
- Avellaneda-Stoikov reservation prices and Obizhaeva-Wang impact
- Deflated Sharpe Ratio calculation
- iTransformer forward pass and variate tokenization
- N-BEATS residual basis expansion forward pass
- JARVIS Quantitative Research Supervisor synthesis
"""

import math
import numpy as np
import pytest
import torch
from research_math.volatility.dcc_garch import DccGarchModel
from research_math.stochastic_processes.jump_diffusion import (
    merton_jump_diffusion_price,
    kou_jump_diffusion_density,
)
from research_math.extreme_value.evt_tail_risk import GeneralizedParetoTailEstimator
from research_math.copulas.copula_dependence import ClaytonCopula, GumbelCopula
from research_math.execution.advanced_execution import (
    AvellanedaStoikovMarketMaker,
    ObizhaevaWangExecution,
)
from research_math.validation.statistical_significance import deflated_sharpe_ratio
from deep_learning.itransformer.model import iTransformer
from deep_learning.nbeats.model import NBeats
from model_registry.registry import ModelRegistry
from ai_orchestrator.agents.research_supervisor import QuantitativeResearchSupervisor


class TestAdvancedMathematicalFinance:
    def test_dcc_garch_fit_and_predict(self):
        np.random.seed(42)
        # Bivariate return series
        r1 = np.random.normal(0, 0.02, size=120)
        r2 = 0.6 * r1 + np.random.normal(0, 0.015, size=120)
        R = np.column_stack([r1, r2])

        dcc = DccGarchModel()
        res = dcc.fit(R)
        assert res["a"] > 0
        assert res["b"] > 0
        assert res["persistence"] < 1.0

        cov_next = dcc.predict_next_covariance(R[-30:])
        assert cov_next.shape == (2, 2)
        # Check symmetry and positive definiteness
        assert math.isclose(cov_next[0, 1], cov_next[1, 0], abs_tol=1e-8)
        assert np.all(np.linalg.eigvals(cov_next) > 0)

    def test_merton_jump_diffusion(self):
        p_no_jump = merton_jump_diffusion_price(
            spot=100.0, strike=100.0, time_to_expiry=1.0, risk_free=0.05, volatility=0.2,
            jump_intensity=0.0, jump_mean=0.0, jump_std=0.05, is_call=True
        )
        p_with_jump = merton_jump_diffusion_price(
            spot=100.0, strike=100.0, time_to_expiry=1.0, risk_free=0.05, volatility=0.2,
            jump_intensity=2.0, jump_mean=-0.1, jump_std=0.15, is_call=True
        )
        assert p_no_jump > 0
        assert p_with_jump > 0
        # Jump presence adds option value
        assert p_with_jump > p_no_jump

    def test_kou_density(self):
        d_up = kou_jump_diffusion_density(x=0.05, p_up=0.4, eta_up=10.0, eta_down=5.0)
        d_down = kou_jump_diffusion_density(x=-0.05, p_up=0.4, eta_up=10.0, eta_down=5.0)
        assert d_up > 0 and d_down > 0

    def test_evt_tail_risk(self):
        np.random.seed(42)
        losses = np.abs(np.random.standard_t(df=3, size=300)) * 100.0  # heavy-tailed losses
        evt = GeneralizedParetoTailEstimator(threshold_percentile=90.0)
        res = evt.fit(losses)
        assert res["xi_shape"] > 0  # fat tails confirmed

        var_99, es_99 = evt.estimate_var_and_es(confidence_level=0.99)
        assert var_99 > res["threshold"]
        assert es_99 > var_99

    def test_copulas(self):
        clayton = ClaytonCopula(theta=2.5)
        assert 0.0 < clayton.lower_tail_dependence < 1.0
        samples_c = clayton.sample(50)
        assert samples_c.shape == (50, 2)

        gumbel = GumbelCopula(theta=2.0)
        assert 0.0 < gumbel.upper_tail_dependence < 1.0
        val = gumbel.cdf(0.5, 0.5)
        assert 0.0 <= val <= 1.0

    def test_avellaneda_stoikov_and_obizhaeva(self):
        mm = AvellanedaStoikovMarketMaker(gamma=0.1, kappa=1.5)
        quotes = mm.compute_quotes(mid_price=50000.0, inventory=2.0, volatility=0.5, time_remaining=0.5)
        # Positive inventory -> reservation price below mid price (desire to sell)
        assert quotes["reservation_price"] < 50000.0
        assert quotes["bid_quote"] < 50000.0 < quotes["ask_quote"]

        ow = ObizhaevaWangExecution(impact_lambda=1e-3, resilience_rho=1.5)
        i0 = ow.transient_impact(trade_size=10.0, elapsed_time=0.0)
        i1 = ow.transient_impact(trade_size=10.0, elapsed_time=1.0)
        assert i0 > i1 > 0

    def test_deflated_sharpe_ratio(self):
        # 1 trial -> simple normal test
        dsr_single = deflated_sharpe_ratio(observed_sr=2.0, num_trials=1, sample_length=252)
        assert dsr_single > 0.95

        # 500 multiple backtest trials -> deflates significance
        dsr_multiple = deflated_sharpe_ratio(observed_sr=2.0, num_trials=500, sample_length=252)
        assert dsr_multiple < dsr_single


class TestNextGenDeepLearningAndSupervisor:
    def test_itransformer_forward(self):
        model = iTransformer(num_features=5, seq_len=30, forecast_horizon=4, d_model=16, n_heads=2)
        x = torch.randn(2, 30, 5)  # batch=2, seq_len=30, features=5
        out = model(x)
        assert out.shape == (2, 4, 5)

    def test_nbeats_forward(self):
        model = NBeats(backcast_len=30, forecast_len=4, num_blocks=2, hidden_dim=32)
        x = torch.randn(3, 30)
        out = model(x)
        assert out.shape == (3, 4)

    def test_jarvis_research_supervisor(self):
        registry = ModelRegistry()
        supervisor = QuantitativeResearchSupervisor(registry)
        synth = supervisor.generate_research_synthesis()
        assert len(synth["active_agents"]) == 9
        assert "FAIL-CLOSED" in synth["risk_authority_boundary"]
