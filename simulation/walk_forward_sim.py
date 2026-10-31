"""Purged Walk-Forward Engine.

Implements Lopez de Prado (2018) "Advances in Financial Machine Learning"
walk-forward splitting semantics:
- Walk-forward: training data strictly SURVIVES BEFORE the test window
  (post-test observations are never re-used in training — V-11 look-ahead fix)
- Purging: removes training labels whose holding period overlaps test windows
- Runs identical production mathematical kernels and risk monitors

(The combinatorial CPCV variant is intentionally NOT used here: it trains on
post-test observations, which is look-ahead bias under walk-forward semantics.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitFold:
    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray


class PurgedWalkForwardEngine:
    """Purged Walk-Forward split engine (no post-test data in training)."""

    def __init__(
        self,
        holding_period_steps: int = 14,  # 14 days holding period
        embargo_pct: float = 0.01,       # % of samples reserved as embargo buffer
    ) -> None:
        self.holding_period = holding_period_steps
        self.embargo_pct = embargo_pct

    def generate_purged_walk_forward_splits(
        self,
        n_samples: int,
        n_splits: int = 5,
        min_train_size: int = 60,
    ) -> Generator[SplitFold, None, None]:
        """Generate strictly sequential walk-forward splits with purging.

        Walk-forward semantics (V-11 fix): every training observation MUST
        chronologically precede the test window. Including post-test data in
        training (as in combinatorial CPCV) is look-ahead bias under a
        walk-forward / live-preservation evaluation, because the fold pretends
        the model was fitted before the test period started.

        Guarantees:
        - Zero overlap between train and test indices.
        - Purging: training labels whose holding window would reach into the
          test window are dropped (gap >= holding_period_steps).
        """
        indices = np.arange(n_samples)
        test_size = (n_samples - min_train_size) // n_splits

        for fold in range(n_splits):
            test_start = min_train_size + fold * test_size
            test_end = min(n_samples, test_start + test_size)
            if test_end <= test_start:
                break

            test_idx = indices[test_start:test_end]

            # Purging: drop training observations whose holding window reaches
            # into test_start. Training data strictly pre-dates the test fold;
            # NO post-test observations enter training (walk-forward).
            purge_train_end = max(0, test_start - self.holding_period)
            train_idx = indices[:purge_train_end]

            yield SplitFold(
                fold_id=fold,
                train_indices=train_idx,
                test_indices=test_idx,
            )

    def evaluate_oos_metrics(
        self,
        equity_curve: np.ndarray,
        returns: np.ndarray,
    ) -> dict[str, float]:
        """Compute standard Out-of-Sample metrics: Sharpe, MaxDD, and Calmar."""
        r = np.asarray(returns, dtype=np.float64)
        r = r[~np.isnan(r)]
        if len(r) < 2:
            return {"sharpe": 0.0, "max_drawdown": 0.0, "calmar": 0.0}

        mean_r = float(np.mean(r))
        std_r = float(np.std(r, ddof=1))
        sharpe = (mean_r / std_r) * np.sqrt(365.0) if std_r > 0 else 0.0

        eq = np.asarray(equity_curve, dtype=np.float64)
        running_max = np.maximum.accumulate(eq)
        drawdowns = (running_max - eq) / running_max
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        n_days = len(eq)
        cagr = float((eq[-1] / eq[0]) ** (365.0 / max(1, n_days)) - 1.0) if eq[0] > 0 else 0.0
        calmar = cagr / max(1e-4, max_dd)

        return {
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "cagr": cagr,
        }
