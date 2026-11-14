"""
Quant_vvf Predictive Intelligence & Forward Validation Engine (PIFVE)
Subsystem 3: Probability Evaluation, Scoring Rules & Calibration

Metrics:
1. Multi-class Brier Score:
   BS = 1/N * sum_i sum_k (p_ik - y_ik)^2
2. Log Loss (Cross-Entropy):
   LL = -1/N * sum_i sum_k y_ik * log(max(eps, p_ik))
3. Brier Skill Score (BSS) against historical base rate:
   BSS = 1 - (BS_model / BS_base)
4. Expected Calibration Error (ECE) & Reliability Diagrams.
5. Dynamic Chronological Calibration Blending:
   P_calibrated = w * P_model + (1 - w) * P_base
   Fitted strictly on previously matured historical outcomes.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from research.pifve.prediction_ledger.unconditional_ledger import ForecastRecord, OutcomeBarrier


@dataclass
class ProbabilityEvaluationReport:
    n_samples: int
    brier_score: float
    base_rate_brier_score: float
    brier_skill_score: float
    log_loss: float
    expected_calibration_error: float
    reliability_bins_confidence: List[float]
    reliability_bins_accuracy: List[float]
    outcome_distribution: Dict[str, float]
    fitted_base_weight: float = 0.85 # w in w*P_model + (1-w)*P_base


class ProbabilityEvaluator:
    """
    Evaluates probabilistic forecasts and fits chronological calibration shrinkage.
    """
    BARRIER_MAP = {
        OutcomeBarrier.UPPER_TOUCHED: 0,
        OutcomeBarrier.LOWER_TOUCHED: 1,
        OutcomeBarrier.VERTICAL_TOUCHED: 2,
    }

    @classmethod
    def evaluate_matured_forecasts(
        cls,
        records: List[ForecastRecord],
        n_bins: int = 10,
    ) -> ProbabilityEvaluationReport:
        valid_recs = [r for r in records if r.realized_barrier in cls.BARRIER_MAP]
        n = len(valid_recs)
        if n == 0:
            return ProbabilityEvaluationReport(
                n_samples=0,
                brier_score=0.0,
                base_rate_brier_score=0.0,
                brier_skill_score=0.0,
                log_loss=0.0,
                expected_calibration_error=0.0,
                reliability_bins_confidence=[],
                reliability_bins_accuracy=[],
                outcome_distribution={},
            )

        # Build one-hot matrix Y and predicted matrix P (N, 3)
        p_matrix = np.empty((n, 3), dtype=np.float64)
        p_base_matrix = np.empty((n, 3), dtype=np.float64)
        y_matrix = np.zeros((n, 3), dtype=np.float64)

        for i, rec in enumerate(valid_recs):
            true_idx = cls.BARRIER_MAP[rec.realized_barrier]
            y_matrix[i, true_idx] = 1.0

            # Map dict probabilities: upper, lower, vertical
            p_u = rec.raw_probabilities.get("upper", 0.333)
            p_l = rec.raw_probabilities.get("lower", 0.333)
            p_v = rec.raw_probabilities.get("vertical", 0.334)
            tot = p_u + p_l + p_v
            p_matrix[i] = [p_u / tot, p_l / tot, p_v / tot]

            bu = rec.historical_base_rates.get("upper", 0.333)
            bl = rec.historical_base_rates.get("lower", 0.333)
            bv = rec.historical_base_rates.get("vertical", 0.334)
            btot = bu + bl + bv
            p_base_matrix[i] = [bu / btot, bl / btot, bv / btot]

        # 1. Brier Score
        bs = float(np.mean(np.sum((p_matrix - y_matrix) ** 2, axis=1)))
        bs_base = float(np.mean(np.sum((p_base_matrix - y_matrix) ** 2, axis=1)))
        bss = float(1.0 - (bs / max(1e-6, bs_base)))

        # 2. Log Loss
        eps = 1e-12
        p_clipped = np.clip(p_matrix, eps, 1.0 - eps)
        ll = float(-np.mean(np.sum(y_matrix * np.log(p_clipped), axis=1)))

        # 3. Expected Calibration Error (ECE) for top predicted class
        confidences = np.max(p_matrix, axis=1)
        predictions = np.argmax(p_matrix, axis=1)
        accuracies = (predictions == np.argmax(y_matrix, axis=1)).astype(np.float64)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        rel_conf = []
        rel_acc = []

        for b in range(n_bins):
            bin_mask = (confidences > bin_edges[b]) & (confidences <= bin_edges[b + 1])
            if np.sum(bin_mask) > 0:
                bin_acc = float(np.mean(accuracies[bin_mask]))
                bin_conf = float(np.mean(confidences[bin_mask]))
                bin_prop = np.sum(bin_mask) / n
                ece += bin_prop * abs(bin_acc - bin_conf)
                rel_conf.append(bin_conf)
                rel_acc.append(bin_acc)
            else:
                rel_conf.append(0.5 * (bin_edges[b] + bin_edges[b + 1]))
                rel_acc.append(0.0)

        # 4. Fit optimal calibration blending weight w in [0, 1] minimizing Brier Score on matured records
        # P_blend = w * P_model + (1 - w) * P_base
        # Derivative d(BS)/dw = 2 * sum (P_blend - Y) * (P_model - P_base) = 0
        diff = p_matrix - p_base_matrix
        target = y_matrix - p_base_matrix
        denom = np.sum(diff ** 2)
        if denom > 1e-8:
            w_opt = float(np.clip(np.sum(diff * target) / denom, 0.0, 1.0))
        else:
            w_opt = 0.5

        outcome_counts = {
            "upper": float(np.mean(y_matrix[:, 0])),
            "lower": float(np.mean(y_matrix[:, 1])),
            "vertical": float(np.mean(y_matrix[:, 2])),
        }

        return ProbabilityEvaluationReport(
            n_samples=n,
            brier_score=round(bs, 4),
            base_rate_brier_score=round(bs_base, 4),
            brier_skill_score=round(bss, 4),
            log_loss=round(ll, 4),
            expected_calibration_error=round(ece, 4),
            reliability_bins_confidence=[round(c, 3) for c in rel_conf],
            reliability_bins_accuracy=[round(a, 3) for a in rel_acc],
            outcome_distribution={k: round(v, 4) for k, v in outcome_counts.items()},
            fitted_base_weight=round(w_opt, 3),
        )
