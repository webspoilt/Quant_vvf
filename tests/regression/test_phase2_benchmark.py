"""PHASE 2 falsification gate: Realized Volatility Forecasting Benchmark Competition.

Kill-gate criteria (verbatim from agent directive):
  1. Implement 5-minute subsampled log-return realized variance in libs/core/realized_measures.py.
  2. Implement Corsi (2009) HAR-RV model and asymmetric QLIKE loss in libs/core/har_rv.py.
  3. Benchmark candidate forecasting approaches against Corsi HAR-RV.
  4. VERIFY: Drop any model that fails to beat HAR-RV on QLIKE out-of-sample (p < 0.05).
"""
from __future__ import annotations

import numpy as np
import pytest

from libs.core.har_rv import (
    CorsiHAR_RV,
    GARCH_EWMA_RV,
    NaiveRandomWalkRV,
    SimpleMovingAverageRV,
    diebold_mariano_test,
    qlike_loss,
)
from libs.core.realized_measures import (
    annualize_rv,
    bipower_variation,
    compute_log_returns,
    jump_variation,
    realized_variance_from_returns,
    subsampled_realized_variance_5min,
)


def generate_btc_rv_history(n_days: int = 420, seed: int = 42) -> np.ndarray:
    """Generate multi-scale persistent realized variance series with jumps (BTC characteristics).

    Features:
    - Daily, weekly, and monthly volatility persistence (heterogeneous market hypothesis)
    - Stochastic volatility clustering
    - Intermittent jump shocks
    """
    np.random.seed(seed)
    rv_series = np.zeros(n_days)
    base_rv = 0.0009  # ~3% daily vol squared (~57% annualized)

    comp_m = base_rv
    comp_w = base_rv
    comp_d = base_rv

    for t in range(n_days):
        comp_m = comp_m + 0.02 * (base_rv - comp_m) + 0.00008 * np.random.randn()
        comp_m = max(1e-5, comp_m)

        comp_w = comp_w + 0.10 * (comp_m - comp_w) + 0.00015 * np.random.randn()
        comp_w = max(1e-5, comp_w)

        comp_d = comp_d + 0.40 * (comp_w - comp_d) + 0.00025 * np.random.randn()
        comp_d = max(1e-5, comp_d)

        jump = 0.002 * (np.random.rand() < 0.05) * np.random.exponential(1.0)
        rv_series[t] = comp_d + jump

    return rv_series


# --------------------------------------------------------------------------- #
# Verification table plumbing
# --------------------------------------------------------------------------- #
_PHASE2_TABLE: list[str] = []


def _row(metric: str, value: float, expectation: str, p_val: float | None, ok: bool) -> None:
    p_str = f"p={p_val:.4e}" if p_val is not None else "n/a"
    _PHASE2_TABLE.append(
        f"{metric:<44} {value:>+14.6f}  {expectation:<30} {p_str:<14} {'PASS' if ok else 'FAIL'}"
    )


def _print_phase2_table(title: str) -> None:
    print(f"\n=== {title} ===")
    for line in _PHASE2_TABLE:
        print(line.encode("ascii", "replace").decode("ascii"))
    print(f"=== {title}: total checks={len(_PHASE2_TABLE)} ===\n")


def test_phase2_forecasting_benchmark_kill_gate() -> None:
    """Kill Gate 2: Benchmark candidate forecasting models against Corsi HAR-RV.

    Evaluates out-of-sample 1-step ahead variance forecasts:
    - Benchmark: Corsi (2009) HAR-RV
    - Candidate 1: Naive Random Walk (RW)
    - Candidate 2: Simple 20-day Moving Average (SMA)
    - Candidate 3: RiskMetrics EWMA (GARCH-type)

    Kill-Gate Criteria:
    - Corsi HAR-RV must achieve lowest QLIKE out-of-sample loss.
    - Any candidate failing to beat HAR-RV on QLIKE (p < 0.05) is DROPPED.
    """
    _PHASE2_TABLE.clear()

    n_total = 420
    n_train_min = 120
    rv_data = generate_btc_rv_history(n_days=n_total, seed=42)

    oos_actuals: list[float] = []
    preds_har: list[float] = []
    preds_naive: list[float] = []
    preds_sma: list[float] = []
    preds_ewma: list[float] = []

    naive_model = NaiveRandomWalkRV()
    sma_model = SimpleMovingAverageRV(window=20)
    ewma_model = GARCH_EWMA_RV(decay_lambda=0.94)

    # Rolling / expanding out-of-sample forecast loop (300 OOS days)
    for t in range(n_train_min, n_total):
        history = rv_data[:t]
        actual = float(rv_data[t])
        oos_actuals.append(actual)

        # Fit Log-HAR-RV on history up to t
        har_model = CorsiHAR_RV(log_transform=True).fit(history)
        preds_har.append(har_model.predict_one_step(history))

        # Candidates
        preds_naive.append(naive_model.predict_one_step(history))
        preds_sma.append(sma_model.predict_one_step(history))
        preds_ewma.append(ewma_model.predict_one_step(history))

    actuals = np.array(oos_actuals)
    h_har = np.array(preds_har)
    h_naive = np.array(preds_naive)
    h_sma = np.array(preds_sma)
    h_ewma = np.array(preds_ewma)

    # Out-of-sample QLIKE losses
    qlike_har = np.asarray(qlike_loss(actuals, h_har))
    qlike_naive = np.asarray(qlike_loss(actuals, h_naive))
    qlike_sma = np.asarray(qlike_loss(actuals, h_sma))
    qlike_ewma = np.asarray(qlike_loss(actuals, h_ewma))

    mean_loss_har = float(np.mean(qlike_har))
    mean_loss_naive = float(np.mean(qlike_naive))
    mean_loss_sma = float(np.mean(qlike_sma))
    mean_loss_ewma = float(np.mean(qlike_ewma))

    # Diebold-Mariano tests against HAR-RV:
    # d_t = loss_candidate_t - loss_HAR_t.
    # Positive DM indicates HAR-RV has lower loss than candidate.
    dm_naive, p_naive = diebold_mariano_test(qlike_har, qlike_naive)
    dm_sma, p_sma = diebold_mariano_test(qlike_har, qlike_sma)
    dm_ewma, p_ewma = diebold_mariano_test(qlike_har, qlike_ewma)

    # Assertions
    _row("qlike_loss_har_benchmark", mean_loss_har, "lowest loss", None, True)
    _row("qlike_loss_naive_rw", mean_loss_naive, "> HAR loss", None, mean_loss_naive > mean_loss_har)
    _row("qlike_loss_sma20", mean_loss_sma, "> HAR loss", None, mean_loss_sma > mean_loss_har)
    _row("qlike_loss_ewma94", mean_loss_ewma, "> HAR loss", None, mean_loss_ewma > mean_loss_har)

    assert mean_loss_har < mean_loss_naive, "HAR-RV must achieve lower loss than Naive RW"
    assert mean_loss_har < mean_loss_sma, "HAR-RV must achieve lower loss than SMA"
    assert mean_loss_har < mean_loss_ewma, "HAR-RV must achieve lower loss than EWMA"

    # Evaluation of Candidate Superiority:
    # A candidate beats HAR-RV iff candidate has LOWER loss than HAR with p < 0.05.
    # Since DM > 0 (HAR has lower loss), candidate beats HAR is False.
    # Therefore, each candidate fails to beat HAR-RV and is DROPPED.
    naive_beats_har = (dm_naive < 0) and (p_naive < 0.05)
    sma_beats_har = (dm_sma < 0) and (p_sma < 0.05)
    ewma_beats_har = (dm_ewma < 0) and (p_ewma < 0.05)

    _row("drop_naive_rw_fails_to_beat_har", float(dm_naive), "drop (DM > 0, beats=False)", p_naive, not naive_beats_har)
    _row("drop_sma_fails_to_beat_har", float(dm_sma), "drop (DM > 0, beats=False)", p_sma, not sma_beats_har)
    _row("drop_ewma_fails_to_beat_har", float(dm_ewma), "drop (DM > 0, beats=False)", p_ewma, not ewma_beats_har)

    assert not naive_beats_har, "Naive RW unexpectedly beat HAR-RV"
    assert not sma_beats_har, "SMA unexpectedly beat HAR-RV"
    assert not ewma_beats_har, "EWMA unexpectedly beat HAR-RV"

    # Statistically significant underperformance of naive and SMA
    assert dm_naive > 0 and p_naive < 0.05, f"Naive RW not significantly worse: DM={dm_naive}, p={p_naive}"
    assert dm_sma > 0 and p_sma < 0.05, f"SMA not significantly worse: DM={dm_sma}, p={p_sma}"

    _print_phase2_table("PHASE 2 FORECASTING BENCHMARK COMPETITION KILL GATE")
