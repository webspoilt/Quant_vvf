"""
QUANT_VVF — Deep Learning Architecture Tests & Validation Suite.
Tests:
- DeepLOB forward, shape checks, probability & uncertainty output
- TFT multi-horizon forecasting, attention interpretability, variable importance
- Deep Hedging trajectory simulation, transaction cost penalty, risk-aversion loss
- PatchTST channel-independent patching, long-term forecasting
- Model Registry lifecycle transitions & JARVIS advisory checks
- Model Comparison benchmarking
"""

import math
import numpy as np
import pytest
import torch
from deep_learning.deeplob.model import DeepLOB
from deep_learning.tft.model import TemporalFusionTransformer, quantile_loss
from deep_learning.deep_hedging.model import DeepHedgingEngine, DeepHedgingPolicy
from deep_learning.patchtst.model import PatchTST
from deep_learning.common.training import set_seed, chronological_split, purged_walk_forward_splits
from model_registry.registry import (
    ModelRegistry,
    ModelMetadata,
    ModelPrediction,
    ModelLifecycleState,
    JarvisModelAdvisor,
)
from model_registry.comparison import ModelComparisonService


class TestDeepLearningModels:
    def test_deeplob_forward_and_uncertainty(self):
        set_seed(42)
        model = DeepLOB(num_classes=3, seq_len=100, num_features=40, lstm_hidden=32)
        dummy_lob = torch.randn(4, 100, 40)  # batch=4, seq_len=100, 10 levels of bid/ask price/vol

        logits = model(dummy_lob)
        assert logits.shape == (4, 3)

        preds = model.predict_probabilities(dummy_lob)
        assert preds["probabilities"].shape == (4, 3)
        assert len(preds["predicted_class"]) == 4
        assert len(preds["uncertainty_entropy"]) == 4
        assert np.all(preds["uncertainty_entropy"] >= 0.0)

    def test_tft_multihorizon_and_attention(self):
        set_seed(42)
        model = TemporalFusionTransformer(
            num_features=8, forecast_horizon=5, d_model=16, num_heads=2, quantiles=(0.10, 0.50, 0.90)
        )
        dummy_inputs = torch.randn(2, 30, 8)  # batch=2, seq_len=30, features=8

        q_preds, var_weights, attn_weights = model(dummy_inputs)
        assert q_preds.shape == (2, 5, 3)  # 5 horizons, 3 quantiles
        assert var_weights.shape == (2, 30, 8)  # variable selection weights
        assert attn_weights.shape == (2, 30, 30)  # temporal attention map

        # Quantile loss check
        dummy_target = torch.randn(2, 5)
        loss = quantile_loss(q_preds, dummy_target)
        assert loss.item() >= 0.0

    def test_deep_hedging_simulation_and_loss(self):
        set_seed(42)
        engine = DeepHedgingEngine(cost_bps=3.0, risk_aversion=1.0)

        # 50 simulated Geometric Brownian Motion paths
        num_paths = 50
        num_steps = 10
        spot_paths = torch.ones(num_paths, num_steps + 1) * 50000.0
        # Add random drift
        spot_paths += torch.cumsum(torch.randn(num_paths, num_steps + 1) * 200.0, dim=1)

        hedge_err, costs, trajectories = engine.simulate_and_evaluate(
            spot_paths=spot_paths, strike=50000.0, time_to_expiry=0.1, volatility=0.5, is_call=True
        )

        assert hedge_err.shape == (num_paths,)
        assert costs.shape == (num_paths,)
        assert trajectories.shape == (num_paths, num_steps)
        assert torch.all(costs >= 0.0)

        total_loss = engine.compute_loss(hedge_err)
        assert torch.isfinite(total_loss)

    def test_patchtst_channel_independence(self):
        set_seed(42)
        model = PatchTST(
            num_features=6, seq_len=64, forecast_horizon=5, patch_len=16, stride=8, d_model=16, n_heads=2
        )
        dummy_ts = torch.randn(2, 64, 6)

        out = model(dummy_ts)
        assert out.shape == (2, 5, 6)

    def test_chronological_splits_and_purging(self):
        data = np.arange(100)
        train, val, test = chronological_split(data, 0.7, 0.15)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
        assert train[-1] < val[0] < test[0]

        splits = list(purged_walk_forward_splits(total_length=100, train_window=30, test_window=10, purge_window=2))
        assert len(splits) > 0
        for tr, te in splits:
            # Assert purge gap between train and test
            assert te[0] >= tr[-1] + 2


class TestRegistryAndComparison:
    def test_registry_and_jarvis_advisory(self):
        registry = ModelRegistry()
        meta = ModelMetadata(
            model_name="DeepLOB_v1",
            research_reference="Zhang et al. (2019)",
            architecture="DeepLOB (Conv2D+Inception+LSTM)",
            version="1.0.0",
            feature_schema=["L1_bid_px", "L1_ask_px"],
            training_period="2025-01-01 to 2026-06-30",
            validation_metrics={"accuracy": 0.68, "brier_score": 0.18},
            artifact_checksum="abc123sha256",
            deployment_status=ModelLifecycleState.APPROVED_FOR_BACKTEST,
            limitations="Sensitive to sudden spread blowouts during volatility spikes",
            last_updated_epoch=1700000000.0,
        )
        registry.register_model(meta)
        assert registry.get_model("DeepLOB_v1") is not None

        advisor = JarvisModelAdvisor(registry)
        pred = ModelPrediction(
            timestamp=1700000050.0,
            model_name="DeepLOB_v1",
            model_version="1.0.0",
            forecast_horizon=5,
            predicted_value="UP",
            uncertainty=0.12,
            input_freshness_seconds=5.0,
            is_valid=True,
            status_reason="OK",
        )
        res = advisor.evaluate_prediction(pred)
        assert res["approved"]

        # Reject stale data
        stale_pred = ModelPrediction(
            timestamp=1700000050.0,
            model_name="DeepLOB_v1",
            model_version="1.0.0",
            forecast_horizon=5,
            predicted_value="UP",
            uncertainty=0.12,
            input_freshness_seconds=120.0,  # exceeds 60s
            is_valid=True,
            status_reason="OK",
        )
        res_stale = advisor.evaluate_prediction(stale_pred)
        assert not res_stale["approved"]

    def test_model_comparison_service(self):
        np.random.seed(42)
        train_ret = np.random.normal(0, 0.02, 100)
        test_ret = np.random.normal(0, 0.02, 20)
        train_rv = 0.0004 + np.random.exponential(0.0001, 100)
        test_rv = 0.0004 + np.random.exponential(0.0001, 20)

        svc = ModelComparisonService()
        bench = svc.benchmark_volatility_forecasting(train_ret, test_ret, train_rv, test_rv)
        assert "Historical_Average" in bench
        assert "GARCH_1_1" in bench
        assert "HAR_RV" in bench
        for k, v in bench.items():
            assert "qlike" in v
            assert "mae" in v
            assert "rmse" in v
