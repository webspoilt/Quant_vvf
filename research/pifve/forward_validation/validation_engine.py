"""
Quant_vvf Predictive Intelligence & Forward Validation Engine (PIFVE)
Subsystem 4: Forward Validation, Independent Horizon Accounting & Economic Expected Value

Key Modules:
1. Independent Horizon Accounting:
   - Purges overlapping forecast windows to report effective independent sample size N_eff.
2. Cost-Aware Expected Value Calculator:
   - EV = P(Upper) * E[R_Upper] + P(Lower) * E[R_Lower] + P(Vertical) * E[R_Vertical] - E[Costs]
   - Evaluates whether predicted probabilities justify crossing the bid-ask spread under statutory costs.
3. Champion / Challenger Model Arena:
   - Evaluates multiple competing models (e.g., DeepLOB vs LightGBM vs Hawkes) on the exact same event stream.
"""

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from research.pifve.prediction_ledger.unconditional_ledger import ForecastRecord, OutcomeBarrier


@dataclass
class EconomicExpectedValueResult:
    expected_value_bps: float
    expected_return_bps: float
    expected_statutory_cost_bps: float
    is_economically_viable: bool
    hurdle_rate_bps: float


class ForwardValidationEngine:
    """
    Integrates model scoring with true economic viability and independent window filtering.
    """
    @staticmethod
    def filter_independent_horizons(records: List[ForecastRecord]) -> List[ForecastRecord]:
        """
        Extracts non-overlapping forecast events to eliminate autocorrelation dependency in statistical tests.
        """
        sorted_recs = sorted(records, key=lambda r: r.timestamp_utc)
        independent_recs: List[ForecastRecord] = []
        last_matured_ts: Optional[datetime] = None

        for rec in sorted_recs:
            rec_ts = datetime.fromisoformat(rec.timestamp_utc)
            if last_matured_ts is None or rec_ts >= last_matured_ts:
                independent_recs.append(rec)
                # Horizon duration
                dur_secs = rec.forecast_horizon_seconds
                last_matured_ts = datetime.fromtimestamp(rec_ts.timestamp() + dur_secs, tz=rec_ts.tzinfo)

        return independent_recs

    @staticmethod
    def calculate_cost_aware_ev(
        probabilities: Dict[str, float],
        target_upper_pct: float,
        target_lower_pct: float,
        vertical_exit_pct: float,
        statutory_cost_bps: float,
        slippage_bps: float = 2.0,
        hurdle_rate_bps: float = 5.0, # Minimum economic alpha hurdle
    ) -> EconomicExpectedValueResult:
        """
        EV = P(Upper) * R_Upper + P(Lower) * R_Lower + P(Vertical) * R_Vertical - Total_Costs
        """
        p_u = probabilities.get("upper", 0.333)
        p_l = probabilities.get("lower", 0.333)
        p_v = probabilities.get("vertical", 0.334)

        # Convert returns to basis points (1% = 100 bps)
        r_u_bps = target_upper_pct * 10000.0
        r_l_bps = target_lower_pct * 10000.0
        r_v_bps = vertical_exit_pct * 10000.0

        expected_gross_bps = (p_u * r_u_bps) + (p_l * r_l_bps) + (p_v * r_v_bps)
        total_costs_bps = statutory_cost_bps + slippage_bps
        net_ev_bps = expected_gross_bps - total_costs_bps

        is_viable = net_ev_bps > hurdle_rate_bps

        return EconomicExpectedValueResult(
            expected_value_bps=round(float(net_ev_bps), 2),
            expected_return_bps=round(float(expected_gross_bps), 2),
            expected_statutory_cost_bps=round(float(total_costs_bps), 2),
            is_economically_viable=is_viable,
            hurdle_rate_bps=hurdle_rate_bps,
        )

    @staticmethod
    def champion_challenger_comparison(
        records_champ: List[ForecastRecord],
        records_chall: List[ForecastRecord],
    ) -> Dict[str, Any]:
        """
        Directly compares two models evaluated across identical forecast events.
        """
        from research.pifve.probability_evaluation.evaluator import ProbabilityEvaluator
        eval_champ = ProbabilityEvaluator.evaluate_matured_forecasts(records_champ)
        eval_chall = ProbabilityEvaluator.evaluate_matured_forecasts(records_chall)

        # Winner based on lower Brier Score and higher Brier Skill Score
        champ_score = eval_champ.brier_score
        chall_score = eval_chall.brier_score

        winner = "CHAMPION" if champ_score <= chall_score else "CHALLENGER"
        diff_pct = ((chall_score - champ_score) / max(1e-6, champ_score)) * 100.0

        return {
            "winner": winner,
            "champion_brier": champ_score,
            "challenger_brier": chall_score,
            "champion_skill_score": eval_champ.brier_skill_score,
            "challenger_skill_score": eval_chall.brier_skill_score,
            "brier_difference_pct": round(float(diff_pct), 2),
            "champion_samples": eval_champ.n_samples,
            "challenger_samples": eval_chall.n_samples,
        }
