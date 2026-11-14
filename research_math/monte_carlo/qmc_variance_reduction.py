"""
Quasi-Monte Carlo (QMC), Digital Scrambling, and Variance Reduction Benchmarking.

Implements:
- Sobol low-discrepancy sequences with digital scramble (Owen / Matousek style).
- Brownian Bridge path construction for effective dimension reduction.
- Comprehensive empirical comparison against Standard Monte Carlo, Antithetic Variates,
  Moment Matching, and Control Variates.
"""

from dataclasses import dataclass
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats
import scipy.stats.qmc as qmc

from research_math.monte_carlo.engine import (
    MonteCarloResult,
    PayoffType,
    ProfessionalMonteCarloEngine,
    SimulationConfig,
    VarianceReductionMethod,
)


@dataclass
class VarianceReductionBenchmark:
    method: str
    price_estimate: float
    standard_error: float
    variance_reduction_factor: float
    computation_time_ms: float
    efficiency_ratio: float  # (Variance_baseline * Time_baseline) / (Variance_method * Time_method)


class BrownianBridgePathGenerator:
    """Constructs Brownian paths using the Brownian Bridge algorithm to concentrate variance in low dimensions."""

    @staticmethod
    def construct_paths(std_normal_increments: np.ndarray, t_expiry: float) -> np.ndarray:
        """
        Transforms independent N(0, 1) variates into Brownian paths via midpoint subdivision.
        std_normal_increments: shape (n_paths, n_steps)
        Returns: W(t) Brownian motion paths of shape (n_paths, n_steps + 1)
        """
        n_paths, n_steps = std_normal_increments.shape
        w = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        times = np.linspace(0.0, t_expiry, n_steps + 1)

        # Terminal point W(T)
        w[:, -1] = math.sqrt(t_expiry) * std_normal_increments[:, 0]

        # Recursive midpoint construction
        indices = [0, n_steps]
        step_idx = 1

        while len(indices) <= n_steps:
            new_indices = []
            for i in range(len(indices) - 1):
                left = indices[i]
                right = indices[i + 1]
                mid = (left + right) // 2
                if mid != left and mid != right and mid not in indices:
                    t_left = times[left]
                    t_right = times[right]
                    t_mid = times[mid]

                    bridge_mean = ((t_right - t_mid) * w[:, left] + (t_mid - t_left) * w[:, right]) / (t_right - t_left)
                    bridge_var = (t_mid - t_left) * (t_right - t_mid) / (t_right - t_left)
                    if step_idx < n_steps:
                        w[:, mid] = bridge_mean + math.sqrt(max(0.0, bridge_var)) * std_normal_increments[:, step_idx]
                        step_idx += 1
                    new_indices.append(mid)
            indices = sorted(indices + new_indices)
            if not new_indices:
                break

        return w


class VarianceReductionBenchmarkSuite:
    """Runs rigorous computational and variance benchmarks across variance-reduction techniques."""

    @staticmethod
    def benchmark_european_call(
        s0: float = 65000.0,
        k: float = 65000.0,
        t: float = 1.0,
        r: float = 0.03,
        sigma: float = 0.50,
        n_paths: int = 16384,
        n_steps: int = 50,
    ) -> List[VarianceReductionBenchmark]:
        """Compares Standard MC, Antithetic, Moment Matching, Control Variate, and Scrambled Sobol QMC."""
        # Analytical Black-Scholes benchmark
        d1 = (math.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
        d2 = d1 - sigma * math.sqrt(t)
        bs_call = s0 * stats.norm.cdf(d1) - k * math.exp(-r * t) * stats.norm.cdf(d2)

        methods = [
            ("Standard MC", VarianceReductionMethod.NONE),
            ("Antithetic Variates", VarianceReductionMethod.ANTITHETIC),
            ("Moment Matching", VarianceReductionMethod.MOMENT_MATCHING),
            ("Control Variate", VarianceReductionMethod.CONTROL_VARIATE),
            ("Sobol QMC", VarianceReductionMethod.SOBOL_QMC),
            ("Scrambled Sobol QMC", VarianceReductionMethod.SCRAMBLED_SOBOL_QMC),
        ]

        results: List[VarianceReductionBenchmark] = []
        baseline_var = 1.0
        baseline_time = 1.0

        for name, vr in methods:
            cfg = SimulationConfig(
                n_paths=n_paths,
                n_steps=n_steps,
                t_expiry=t,
                r_rate=r,
                variance_reduction=vr,
                seed=42,
            )
            engine = ProfessionalMonteCarloEngine(cfg)
            t_start = time.perf_counter()
            paths = engine.simulate_gbm(s0=s0, sigma=sigma)
            mc_res = engine.price_contingent_claim(
                paths, strike=k, payoff_type=PayoffType.EUROPEAN_CALL, analytical_benchmark=bs_call
            )
            dur_ms = (time.perf_counter() - t_start) * 1000.0

            if vr == VarianceReductionMethod.NONE:
                baseline_var = mc_res.sample_variance
                baseline_time = dur_ms
                vrf = 1.0
                eff = 1.0
            else:
                vrf = baseline_var / max(mc_res.sample_variance, 1e-12)
                eff = (baseline_var * baseline_time) / max(mc_res.sample_variance * dur_ms, 1e-12)

            results.append(
                VarianceReductionBenchmark(
                    method=name,
                    price_estimate=mc_res.price_estimate,
                    standard_error=mc_res.standard_error,
                    variance_reduction_factor=vrf,
                    computation_time_ms=dur_ms,
                    efficiency_ratio=eff,
                )
            )

        return results
