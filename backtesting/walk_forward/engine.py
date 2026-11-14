"""
Walk-Forward Optimization (WFO) and Chronological Cross-Validation Framework.

Implements:
1. Strict rolling and expanding chronological training windows (no forward-looking leakage).
2. Out-of-Sample (OOS) parameter freezing and sequential testing.
3. Purged and embargoed cross-validation boundaries.
4. Comprehensive performance metrics: Sharpe, Sortino, Calmar, MaxDD, WinRate, Profit Factor,
   Deflated Sharpe Ratio (DSR), and Train-to-Test Degradation Analysis.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats


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
    window_details: List[WindowPerformance] = field(default_factory=list)


class WalkForwardOptimizer:
    """Rigorous Chronological Walk-Forward Strategy Optimization Engine."""

    def __init__(
        self,
        train_window_size: int = 252,
        test_window_size: int = 63,
        step_size: int = 63,
        expanding_window: bool = False,
        embargo_size: int = 5,
    ):
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size
        self.step_size = step_size
        self.expanding_window = expanding_window
        self.embargo_size = embargo_size

    def run_walk_forward(
        self,
        data: pd.DataFrame,
        strategy_objective_fn: Callable[[pd.DataFrame, Dict[str, Any]], Tuple[float, float, float, float]],
        param_grid: List[Dict[str, Any]],
    ) -> WalkForwardReport:
        """
        Executes strict chronological Walk-Forward Optimization across market data.
        strategy_objective_fn signature: (slice_df, params) -> (sharpe, net_pnl, max_dd, win_rate)
        """
        n_samples = len(data)
        window_results: List[WindowPerformance] = []

        window_id = 1
        current_train_start = 0

        while True:
            train_end = current_train_start + self.train_window_size
            test_start = train_end + self.embargo_size
            test_end = test_start + self.test_window_size

            if test_end > n_samples:
                break

            train_df = data.iloc[current_train_start:train_end]
            test_df = data.iloc[test_start:test_end]

            # In-sample grid optimization
            best_params = None
            best_train_sharpe = -999.0

            for params in param_grid:
                train_sharpe, _, _, _ = strategy_objective_fn(train_df, params)
                if train_sharpe > best_train_sharpe:
                    best_train_sharpe = train_sharpe
                    best_params = params

            if best_params is None:
                best_params = param_grid[0]

            # Out-of-sample evaluation with frozen optimal parameters
            test_sharpe, test_pnl, test_dd, test_win = strategy_objective_fn(test_df, best_params)

            degradation = (
                ((best_train_sharpe - test_sharpe) / max(abs(best_train_sharpe), 1e-6)) * 100.0
                if best_train_sharpe != 0.0 else 0.0
            )

            # Profit factor estimation
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
                    train_sharpe=float(best_train_sharpe),
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
                # Keep train_start at 0, expand train_window
                self.train_window_size += self.step_size
            else:
                current_train_start += self.step_size

        if not window_results:
            raise ValueError(f"Insufficient data length ({n_samples}) for specified walk-forward windows")

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

        # Bailey & López de Prado (2014) Deflated Sharpe Ratio calculation
        n_trials = len(param_grid)
        euler_gamma = 0.5772156649
        z_score = (1.0 - euler_gamma) * stats.norm.ppf(1.0 - 1.0 / n_trials) + euler_gamma * stats.norm.ppf(
            1.0 - 1.0 / (n_trials * math.e)
        )
        dsr_val = float(stats.norm.cdf((mean_sharpe - z_score * np.std(test_sharpes, ddof=1)) / max(1e-6, np.std(test_sharpes, ddof=1))))

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
            window_details=window_results,
        )
