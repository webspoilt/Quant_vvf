"""Realized Volatility and High-Frequency Return Measures.

Implements:
1. 5-minute subsampled log-return realized variance (Zhang, Mykland, & Aït-Sahalia 2005; Hansen & Lunde 2006)
   Averages across interleaved subgrids to mitigate microstructure noise and quote bounce.
2. Realized Bipower Variation (BV) (Barndorff-Nielsen & Shephard 2004)
   Jump-robust continuous volatility component.
3. Jump variation extraction: J = max(0, RV - BV).
4. Annualized volatility estimators.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd


def compute_log_returns(prices: np.ndarray | Sequence[float]) -> np.ndarray:
    """Compute logarithmic returns r_t = ln(P_t / P_{t-1}).

    V-21: prices must be strictly positive AND finite — NaN passes the `<= 0`
    comparison and would silently propagate NaN through every downstream RV.
    """
    p = np.asarray(prices, dtype=np.float64)
    if np.any(~np.isfinite(p)) or np.any(p <= 0.0):
        raise ValueError("Prices must be finite and strictly positive")
    if len(p) < 2:
        return np.empty(0, dtype=np.float64)
    return np.diff(np.log(p))


def realized_variance_from_returns(log_returns: np.ndarray | Sequence[float]) -> float:
    """Compute standard realized variance RV = sum(r_t^2)."""
    r = np.asarray(log_returns, dtype=np.float64)
    if len(r) == 0:
        return 0.0
    return float(np.sum(r * r))


def subsampled_realized_variance_5min(
    timestamps_sec: np.ndarray,
    prices: np.ndarray,
    interval_sec: int = 300,        # 5-minute interval
    subgrid_offset_sec: int = 60,   # 1-minute subgrid offsets
) -> float:
    """Compute 5-minute subsampled realized variance.

    Partitions the time series into K = (interval_sec / subgrid_offset_sec)
    interleaved 5-minute subgrids, calculates realized variance on each subgrid,
    and returns their arithmetic mean:
        RV_sub = (1 / K) * sum_{k=0}^{K-1} RV^{(k)}

    Args:
        timestamps_sec: 1D array of monotonic second timestamps
        prices: 1D array of positive price observations
        interval_sec: Target return sampling interval (300 sec = 5 min)
        subgrid_offset_sec: Spacing between subgrid starts (60 sec = 1 min)

    Returns:
        Subsampled realized variance float
    """
    ts = np.asarray(timestamps_sec, dtype=np.int64)
    p = np.asarray(prices, dtype=np.float64)
    if len(ts) != len(p):
        raise ValueError("timestamps and prices must have identical lengths")
    if len(p) < 2:
        return 0.0
    if np.any(~np.isfinite(p)) or np.any(p <= 0.0):
        raise ValueError("Prices must be finite and strictly positive")
    if interval_sec <= 0 or subgrid_offset_sec <= 0:
        raise ValueError(
            f"interval_sec and subgrid_offset_sec must be positive, got interval_sec={interval_sec}, subgrid_offset_sec={subgrid_offset_sec}"
        )
    if interval_sec % subgrid_offset_sec != 0:
        raise ValueError(
            f"interval_sec ({interval_sec}) must be a multiple of subgrid_offset_sec ({subgrid_offset_sec})"
        )
    # Monotonicity invariant (item 19): non-monotonic timestamps silently map
    # out-of-order prices onto the same grid, fabricating artificial returns.
    if np.any(np.diff(ts) < 0):
        raise ValueError("timestamps_sec must be monotonically non-decreasing")

    k_subgrids = max(1, interval_sec // subgrid_offset_sec)
    t_start = ts[0]
    t_end = ts[-1]
    total_duration = t_end - t_start

    if total_duration < interval_sec:
        # Fallback to simple RV if span is shorter than a single interval
        ret = compute_log_returns(p)
        return realized_variance_from_returns(ret)

    rv_subgrids: list[float] = []

    for k in range(k_subgrids):
        offset = k * subgrid_offset_sec
        grid_times = np.arange(t_start + offset, t_end + 1, interval_sec)
        if len(grid_times) < 2:
            continue

        # Forward-fill price search on grid times
        indices = np.searchsorted(ts, grid_times, side="right") - 1
        indices = np.clip(indices, 0, len(p) - 1)
        grid_prices = p[indices]

        # Valid returns on this subgrid
        sub_returns = compute_log_returns(grid_prices)
        if len(sub_returns) > 0:
            rv_subgrids.append(realized_variance_from_returns(sub_returns))

    if not rv_subgrids:
        return realized_variance_from_returns(compute_log_returns(p))

    return float(np.mean(rv_subgrids))


def bipower_variation(log_returns: np.ndarray | Sequence[float]) -> float:
    """Compute Realized Bipower Variation (BV) (Barndorff-Nielsen & Shephard 2004).

    BV = (pi / 2) * (N / (N - 1)) * sum_{i=2}^N |r_i| * |r_{i-1}|
    Asymptotically robust to rare jumps in price.
    """
    r = np.asarray(log_returns, dtype=np.float64)
    n = len(r)
    if n < 2:
        return 0.0
    abs_r = np.abs(r)
    # pi / 2 = 1.5707963267948966
    c = np.pi / 2.0 * (float(n) / float(n - 1))
    bv = c * np.sum(abs_r[1:] * abs_r[:-1])
    return float(bv)


def jump_variation(realized_var: float, bipower_var: float) -> float:
    """Extract jump variation component J = max(0.0, RV - BV)."""
    return float(max(0.0, realized_var - bipower_var))


def annualize_rv(rv_daily: float, trading_days: int = 365) -> float:
    """Convert daily realized variance to annualized realized volatility."""
    if not isinstance(trading_days, int) or trading_days <= 0:
        raise ValueError(f"trading_days must be a positive integer, got {trading_days!r}")
    if not np.isfinite(rv_daily):
        raise ValueError(f"rv_daily must be finite, got {rv_daily!r}")
    if rv_daily <= 0:
        return 0.0
    return float(np.sqrt(rv_daily * float(trading_days)))
