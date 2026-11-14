"""
Unit Tests for Quant_vvf Predictive Intelligence & Forward Validation Engine (PIFVE).
"""

import numpy as np
import pandas as pd
import pytest

from research.pifve.prediction_ledger.unconditional_ledger import (
    UnconditionalPredictionLedger,
    ForecastRecord,
    DecisionPolicyAction,
    OutcomeBarrier,
)
from research.pifve.triple_barrier.labeler import (
    TripleBarrierLabeler,
    BarrierSpecification,
)
from research.pifve.probability_evaluation.evaluator import (
    ProbabilityEvaluator,
    ProbabilityEvaluationReport,
)
from research.pifve.forward_validation.validation_engine import (
    ForwardValidationEngine,
    EconomicExpectedValueResult,
)


class TestPredictiveIntelligenceAndForwardValidation:
    """Validates UPL logging, Triple-Barrier labeling, Brier/LogLoss scoring, and Economic EV."""

    def test_unconditional_prediction_ledger_recording(self):
        """Validates that predictions are logged unconditionally before policy rejection."""
        ledger = UnconditionalPredictionLedger()
        
        # Log 1: LightGBM forecast rejected due to negative expected value
        rec1 = ledger.log_forecast(
            model_id="LightGBM_VRP_Classifier",
            model_version="2.1.0",
            market="DERIBIT",
            instrument="BTC-PERPETUAL",
            forecast_horizon_seconds=7200,
            raw_probabilities={"upper": 0.65, "lower": 0.20, "vertical": 0.15},
            policy_action=DecisionPolicyAction.REJECTED_EXPECTED_VALUE,
            rejection_reason="Expected return 12 bps < Cost threshold 15 bps",
            trade_attempted=False,
        )
        assert rec1.prediction_id is not None
        assert rec1.policy_action == DecisionPolicyAction.REJECTED_EXPECTED_VALUE
        assert rec1.realized_barrier == OutcomeBarrier.UNMATURED

        # Log 2: Operational failure (e.g. timeout / NaN)
        rec2 = ledger.log_operational_failure(
            model_id="DeepLOB_PyTorch",
            market="NSE",
            instrument="NIFTY-FUT",
            error_message="Feature snapshot buffer timeout (>100ms)",
        )
        assert rec2.is_operational_failure is True
        assert rec2.policy_action == DecisionPolicyAction.FAILED_OPERATIONAL

        # Outcome resolution once horizon matures
        success = ledger.resolve_outcome(
            prediction_id=rec1.prediction_id,
            realized_barrier=OutcomeBarrier.UPPER_TOUCHED,
            realized_return_pct=0.012,
            realized_pnl_bps=120.0,
        )
        assert success is True
        matured = ledger.get_matured_records("LightGBM_VRP_Classifier")
        assert len(matured) == 1
        assert matured[0].realized_barrier == OutcomeBarrier.UPPER_TOUCHED

    def test_triple_barrier_labeler(self):
        """Tests that triple-barrier touches identify upper, lower, and vertical barriers."""
        labeler = TripleBarrierLabeler(BarrierSpecification(upper_profit_mult=1.0, lower_loss_mult=1.0, max_horizon_bars=5))
        
        # Upward price path hitting upper barrier
        df_up = pd.DataFrame({"close": [100.0, 101.0, 102.5, 101.5, 101.0, 100.5]})
        barrier, bars, ret = labeler.label_event(df_up, start_idx=0, rolling_volatility=0.02)
        assert barrier == OutcomeBarrier.UPPER_TOUCHED
        assert bars == 2
        assert ret > 0.0

        # Downward price path hitting lower barrier
        df_down = pd.DataFrame({"close": [100.0, 99.0, 97.0, 98.0, 98.5, 99.0]})
        barrier_dn, bars_dn, ret_dn = labeler.label_event(df_down, start_idx=0, rolling_volatility=0.02)
        assert barrier_dn == OutcomeBarrier.LOWER_TOUCHED
        assert ret_dn < 0.0

        # Neutral horizontal path hitting vertical expiry
        df_flat = pd.DataFrame({"close": [100.0, 100.1, 100.0, 99.9, 100.1, 100.0]})
        barrier_fl, bars_fl, ret_fl = labeler.label_event(df_flat, start_idx=0, rolling_volatility=0.02)
        assert barrier_fl == OutcomeBarrier.VERTICAL_TOUCHED

    def test_probability_evaluator_scoring_rules(self):
        """Tests Multi-class Brier Score, Log Loss, and Dynamic Calibration Blending."""
        ledger = UnconditionalPredictionLedger()
        
        # Create 10 mock matured predictions
        for i in range(10):
            # Perfect predictor: predicts high upper probability on upper events
            rec = ledger.log_forecast(
                model_id="Test_Hawkes",
                model_version="1.0",
                market="DERIBIT",
                instrument="BTC-PERPETUAL",
                forecast_horizon_seconds=3600,
                raw_probabilities={"upper": 0.80, "lower": 0.10, "vertical": 0.10},
                historical_base_rates={"upper": 0.33, "lower": 0.33, "vertical": 0.34},
            )
            ledger.resolve_outcome(rec.prediction_id, OutcomeBarrier.UPPER_TOUCHED, 0.015, 150.0)

        matured = ledger.get_matured_records("Test_Hawkes")
        report = ProbabilityEvaluator.evaluate_matured_forecasts(matured)

        assert report.n_samples == 10
        assert report.brier_score < 0.10 # High skill gives very low Brier score
        assert report.brier_skill_score > 0.50 # Beats base rate significantly
        assert report.log_loss < 0.30
        assert 0.0 <= report.expected_calibration_error <= 1.0

    def test_cost_aware_economic_expected_value(self):
        """Tests that signal execution requires gross expected return to exceed venue costs."""
        # Scenario A: High probability of upper target (+1.5%), lower risk (-1.0%)
        # Gross = 0.70 * 150 bps - 0.20 * 100 bps - 0.10 * 0 = 85 bps
        # Costs = 24 bps (STT/Fees) -> Net EV = 61 bps > Hurdle (5 bps) -> Viable
        res_viable = ForwardValidationEngine.calculate_cost_aware_ev(
            probabilities={"upper": 0.70, "lower": 0.20, "vertical": 0.10},
            target_upper_pct=0.015,
            target_lower_pct=-0.010,
            vertical_exit_pct=0.0,
            statutory_cost_bps=22.0,
            slippage_bps=2.0,
            hurdle_rate_bps=5.0,
        )
        assert res_viable.is_economically_viable is True
        assert res_viable.expected_value_bps > 50.0

        # Scenario B: Low probability edge, gross returns eaten by fees
        res_unviable = ForwardValidationEngine.calculate_cost_aware_ev(
            probabilities={"upper": 0.40, "lower": 0.40, "vertical": 0.20},
            target_upper_pct=0.005, # 50 bps
            target_lower_pct=-0.005, # -50 bps
            vertical_exit_pct=0.0,
            statutory_cost_bps=25.0,
            slippage_bps=5.0,
            hurdle_rate_bps=5.0,
        )
        assert res_unviable.is_economically_viable is False
        assert res_unviable.expected_value_bps < 0.0

    def test_champion_challenger_model_arena(self):
        """Tests head-to-head comparison between Champion and Challenger models."""
        ledger = UnconditionalPredictionLedger()
        champ_recs = []
        chall_recs = []

        for i in range(5):
            # Model 1 (Champion): Accurate
            r1 = ledger.log_forecast("DeepLOB", "1.0", "DERIBIT", "BTC", 3600, {"upper": 0.75, "lower": 0.15, "vertical": 0.10})
            ledger.resolve_outcome(r1.prediction_id, OutcomeBarrier.UPPER_TOUCHED, 0.01, 100.0)
            champ_recs.append(r1)

            # Model 2 (Challenger): Inaccurate
            r2 = ledger.log_forecast("Naif_MLP", "1.0", "DERIBIT", "BTC", 3600, {"upper": 0.10, "lower": 0.80, "vertical": 0.10})
            ledger.resolve_outcome(r2.prediction_id, OutcomeBarrier.UPPER_TOUCHED, 0.01, 100.0)
            chall_recs.append(r2)

        arena = ForwardValidationEngine.champion_challenger_comparison(champ_recs, chall_recs)
        assert arena["winner"] == "CHAMPION"
        assert arena["champion_brier"] < arena["challenger_brier"]
