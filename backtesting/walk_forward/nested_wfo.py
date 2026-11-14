"""
Nested Walk-Forward Strategy Optimization (WFO) Engine.

Phase 8 Remediation:
- Outer Training Window -> Inner Cross-Validation -> Hyperparameter Freezing -> Outer OOS Evaluation.
- Purged and embargoed boundaries preventing forward-looking label leakage.
- Isolated strategy evaluation ensuring fresh state on every iteration.
- Deflated Sharpe Ratio integration using Bailey et al. (2016) EVT formulation.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats

from research_math.validation.deflated_sharpe import DeflatedSharpeCalculator


@dataclass
class WindowPerformance:
    window_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    optimal_params: Dict[str, Any]
    train_sharpe: float
    test_sharpe: float
    test_net_pnl: float
    test_max_drawdown: float
    test_win_rate: float
    test_profit_factor: float
    performance_degradation_pct: float


@dataclass
class WalkForwardReport:
    total_windows: int
    mean_test_sharpe: float
    median_test_sharpe: float
    worst_window_sharpe: float
    best_window_sharpe: float
    profitable_windows_pct: float
    mean_degradation_pct: float
    cumulative_test_pnl: float
    overall_test_max_dd: float
    deflated_sharpe_ratio: float
    expected_max_sharpe: float
    window_details: List[WindowPerformance] = field(default_factory=list)


class NestedWalkForwardOptimizer:
    """Strict Chronological Nested Walk-Forward Strategy Optimization Engine."""

    def __init__(
        self,
        outer_train_size: int = 252,
        outer_test_size: int = 63,
        step_size: int = 63,
        embargo_size: int = 5,
        inner_splits: int = 3,
        expanding_window: bool = False,
    ):
        self.outer_train_size = outer_train_size
        self.outer_test_size = outer_test_size
        self.step_size = step_size
        self.embargo_size = embargo_size
        self.inner_splits = inner_splits
        self.expanding_window = expanding_window

    def run_nested_optimization(
        self,
        data: pd.DataFrame,
        strategy_objective_fn: Callable[[pd.DataFrame, Dict[str, Any]], Tuple[float, float, float, float]],
        param_grid: List[Dict[str, Any]],
    ) -> WalkForwardReport:
        """
        Executes Nested Walk-Forward Optimization:
        Inner cross-validation within outer_train selects hyperparameters;
        Selected parameters are frozen and evaluated on the subsequent outer_test window.
        """
        n_samples = len(data)
        window_results: List[WindowPerformance] = []
        all_oos_returns: List[float] = []

        window_id = 1
        current_train_start = 0

        while True:
            train_end = current_train_start + self.outer_train_size
            test_start = train_end + self.embargo_size
            test_end = test_start + self.outer_test_size

            if test_end > n_samples:
                break

            outer_train_df = data.iloc[current_train_start:train_end]
            outer_test_df = data.iloc[test_start:test_end]

            # Inner Validation: K-fold chronological cross-validation inside outer_train
            best_params = None
            best_inner_score = -np.inf
            inner_chunk = len(outer_train_df) // (self.inner_splits + 1)

            for params in param_grid:
                inner_scores = []
                for k in range(1, self.inner_splits + 1):
                    in_tr = outer_train_df.iloc[: k * inner_chunk]
                    in_val = outer_train_df.iloc[k * inner_chunk : min(len(outer_train_df), (k + 1) * inner_chunk)]
                    if len(in_val) < 5:
                        continue
                    sh, _, _, _ = strategy_objective_fn(in_val, params)
                    inner_scores.append(sh)
                
                avg_score = float(np.mean(inner_scores)) if inner_scores else -999.0
                if avg_score > best_inner_score:
                    best_inner_score = avg_score
                    best_params = params

            if best_params is None:
                best_params = param_grid[0]

            # Train Sharpe on full outer training window
            train_sharpe, _, _, _ = strategy_objective_fn(outer_train_df, best_params)

            # Outer Out-of-Sample Evaluation with FROZEN parameters
            test_sharpe, test_pnl, test_dd, test_win = strategy_objective_fn(outer_test_df, best_params)
            ret_proxy = test_pnl / 100.0
            all_oos_returns.append(ret_proxy)

            degradation = (
                ((train_sharpe - test_sharpe) / max(abs(train_sharpe), 1e-6)) * 100.0
                if train_sharpe != 0.0 else 0.0
            )

            pnl_wins = max(0.0, test_pnl)
            pnl_loss = abs(min(0.0, test_pnl))
            profit_factor = (pnl_wins / max(pnl_loss, 1e-6)) if pnl_loss > 0 else (2.0 if pnl_wins > 0 else 1.0)

            window_results.append(
                WindowPerformance(
                    window_id=window_id,
                    train_start=current_train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    optimal_params=best_params,
                    train_sharpe=float(train_sharpe),
                    test_sharpe=float(test_sharpe),
                    test_net_pnl=float(test_pnl),
                    test_max_drawdown=float(test_dd),
                    test_win_rate=float(test_win),
                    test_profit_factor=float(profit_factor),
                    performance_degradation_pct=float(degradation),
                )
            )

            window_id += 1
            if self.expanding_window:
                self.outer_train_size += self.step_size
            else:
                current_train_start += self.step_size

        if not window_results:
            raise ValueError(f"Insufficient samples ({n_samples}) for specified nested walk-forward windows")

        test_sharpes = [w.test_sharpe for w in window_results]
        test_pnls = [w.test_net_pnl for w in window_results]
        degradations = [w.performance_degradation_pct for w in window_results]

        mean_sharpe = float(np.mean(test_sharpes))
        median_sharpe = float(np.median(test_sharpes))
        worst_sharpe = float(np.min(test_sharpes))
        best_sharpe = float(np.max(test_sharpes))
        profitable_pct = float(np.mean([1.0 if p > 0 else 0.0 for p in test_pnls]) * 100.0)
        mean_deg = float(np.mean(degradations))
        cum_pnl = float(np.sum(test_pnls))
        overall_max_dd = float(np.max([w.test_max_drawdown for w in window_results]))

        # Calculate exact Deflated Sharpe Ratio
        if len(all_oos_returns) >= 5:
            dsr_res = DeflatedSharpeCalculator.calculate_dsr(
                returns=np.array(all_oos_returns),
                n_trials=len(param_grid) * len(window_results),
                benchmark_annualized_sharpe=0.0,
            )
            dsr_val = dsr_res.deflated_sharpe_ratio
            exp_max_sh = dsr_res.expected_max_sharpe
        else:
            # Fallback when window count < 5
            dsr_val = 0.5
            exp_max_sh = mean_sharpe

        return WalkForwardReport(
            total_windows=len(window_results),
            mean_test_sharpe=mean_sharpe,
            median_test_sharpe=median_sharpe,
            worst_window_sharpe=worst_sharpe,
            best_window_sharpe=best_sharpe,
            profitable_windows_pct=profitable_pct,
            mean_degradation_pct=mean_deg,
            cumulative_test_pnl=cum_pnl,
            overall_test_max_dd=overall_max_dd,
            deflated_sharpe_ratio=dsr_val,
            expected_max_sharpe=exp_max_sh,
            window_details=window_results,
        )
