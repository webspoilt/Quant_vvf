"""
Quasi-Monte Carlo (QMC), Digital Scrambling, and Replicated Variance Reduction Benchmarking.

Phase 4 & 5 Remediation:
- Replicated Randomized Sobol QMC: Error estimation from independent scramble replicates
  (eliminating invalid IID assumption on deterministic low-discrepancy points).
- Brownian Bridge path generator integrated with QMC increments and covariance validation.
- Empirical benchmark suite recording unbiased estimator variance across equivalent sample budgets.
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
    efficiency_ratio: float
    n_replications: int = 1


class BrownianBridgePathGenerator:
    """Constructs Brownian paths using Brownian Bridge subdivision to concentrate variance in low dimensions."""

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

    @staticmethod
    def validate_covariance(n_paths: int = 20000, n_steps: int = 10, t_expiry: float = 1.0) -> float:
        """
        Validates empirical Brownian Bridge covariance against theoretical min(s, t).
        Returns max absolute covariance error across all grid pairs.
        """
        rng = np.random.default_rng(42)
        increments = rng.standard_normal((n_paths, n_steps))
        paths = BrownianBridgePathGenerator.construct_paths(increments, t_expiry)
        cov_emp = np.cov(paths.T)

        times = np.linspace(0.0, t_expiry, n_steps + 1)
        cov_theo = np.minimum.outer(times, times)
        return float(np.max(np.abs(cov_emp - cov_theo)))


class VarianceReductionBenchmarkSuite:
    """Runs computational and variance benchmarks across variance-reduction techniques."""

    @staticmethod
    def benchmark_european_call(
        s0: float = 65000.0,
        k: float = 65000.0,
        t: float = 1.0,
        r: float = 0.03,
        sigma: float = 0.40,
        n_paths: int = 16384,
        n_steps: int = 50,
        n_replicates: int = 8,
    ) -> List[VarianceReductionBenchmark]:
        """
        Compares Standard MC, Paired Antithetic, Moment Matching, Control Variate,
        and Independent Replicated Scrambled Sobol QMC.
        """
        d1 = (math.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
        d2 = d1 - sigma * math.sqrt(t)
        bs_call = s0 * stats.norm.cdf(d1) - k * math.exp(-r * t) * stats.norm.cdf(d2)

        methods = [
            ("Standard MC", VarianceReductionMethod.NONE),
            ("Antithetic Variates (Paired)", VarianceReductionMethod.ANTITHETIC),
            ("Moment Matching", VarianceReductionMethod.MOMENT_MATCHING),
            ("Control Variate", VarianceReductionMethod.CONTROL_VARIATE),
            ("Scrambled Sobol (Replicated)", VarianceReductionMethod.SCRAMBLED_SOBOL_QMC),
        ]

        results: List[VarianceReductionBenchmark] = []
        baseline_var = 1.0
        baseline_time = 1.0

        for name, vr in methods:
            t_start = time.perf_counter()

            if vr == VarianceReductionMethod.SCRAMBLED_SOBOL_QMC:
                # Rigorous replicated randomized QMC: R independent scrambles of N/R paths
                paths_per_rep = max(256, n_paths // n_replicates)
                rep_estimates = []
                for rep in range(n_replicates):
                    cfg = SimulationConfig(
                        n_paths=paths_per_rep,
                        n_steps=n_steps,
                        t_expiry=t,
                        r_rate=r,
                        variance_reduction=vr,
                        seed=100 + rep * 17,
                        sobol_scramble=True,
                    )
                    engine = ProfessionalMonteCarloEngine(cfg)
                    paths = engine.simulate_gbm(s0=s0, sigma=sigma)
                    res = engine.price_contingent_claim(paths, strike=k, payoff_type=PayoffType.EUROPEAN_CALL)
                    rep_estimates.append(res.price_estimate)

                mean_est = float(np.mean(rep_estimates))
                # Unbiased sample variance across independent scramble replicates
                sample_var = float(np.var(rep_estimates, ddof=1))
                se = float(math.sqrt(sample_var / n_replicates))
                dur_ms = (time.perf_counter() - t_start) * 1000.0
            else:
                cfg = SimulationConfig(
                    n_paths=n_paths,
                    n_steps=n_steps,
                    t_expiry=t,
                    r_rate=r,
                    variance_reduction=vr,
                    seed=42,
                )
                engine = ProfessionalMonteCarloEngine(cfg)
                paths = engine.simulate_gbm(s0=s0, sigma=sigma)
                mc_res = engine.price_contingent_claim(
                    paths, strike=k, payoff_type=PayoffType.EUROPEAN_CALL, analytical_benchmark=bs_call
                )
                mean_est = mc_res.price_estimate
                sample_var = mc_res.sample_variance
                se = mc_res.standard_error
                dur_ms = (time.perf_counter() - t_start) * 1000.0

            if vr == VarianceReductionMethod.NONE:
                baseline_var = sample_var
                baseline_time = dur_ms
                vrf = 1.0
                eff = 1.0
            else:
                vrf = float(baseline_var / max(sample_var, 1e-12))
                eff = float((baseline_var * baseline_time) / max(sample_var * dur_ms, 1e-12))

            results.append(
                VarianceReductionBenchmark(
                    method=name,
                    price_estimate=mean_est,
                    standard_error=se,
                    variance_reduction_factor=vrf,
                    computation_time_ms=dur_ms,
                    efficiency_ratio=eff,
                    n_replications=n_replicates if vr == VarianceReductionMethod.SCRAMBLED_SOBOL_QMC else 1,
                )
            )

        return results
