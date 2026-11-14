"""
Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR) Implementation.

Phase 9 Remediation:
- Implements Bailey & López de Prado (2014) and Bailey et al. (2016) formulations.
- Explicitly accounts for return sample size (N), skewness (gamma_3), and kurtosis (gamma_4).
- Models expected maximum Sharpe ratio under multiple testing over N_trials trials:
  E[max_k {SR_k}] = (1 - euler_gamma) * Z^{-1}(1 - 1/N) + euler_gamma * Z^{-1}(1 - 1/(N*e)).
- Prevents invalid derivation of DSR solely from cross-window standard deviations.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats


@dataclass
class DsrResult:
    observed_sharpe: float
    benchmark_sharpe: float
    probabilistic_sharpe_ratio: float # PSR(SR*)
    deflated_sharpe_ratio: float      # DSR(E[max SR])
    expected_max_sharpe: float        # SR* accounting for multiple testing
    skewness: float
    kurtosis: float
    n_samples: int
    n_trials: int
    is_statistically_significant: bool # True if DSR >= 0.95


class DeflatedSharpeCalculator:
    """Rigorous Multiple-Testing Deflated Sharpe Ratio (DSR) Engine."""

    @staticmethod
    def calculate_dsr(
        returns: np.ndarray,
        n_trials: int = 1,
        benchmark_annualized_sharpe: float = 0.0,
        annualization_factor: float = 252.0,
        trial_variance_sharpe: Optional[float] = None,
    ) -> DsrResult:
        """
        Calculates exact Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR).
        returns: 1D array of portfolio/strategy periodic returns.
        n_trials: Total number of parameters/strategies backtested across research.
        """
        n = len(returns)
        if n < 5:
            raise ValueError("At least 5 return observations required for DSR estimation")

        mean_r = float(np.mean(returns))
        std_r = float(np.std(returns, ddof=1))
        if std_r <= 1e-12:
            return DsrResult(
                observed_sharpe=0.0,
                benchmark_sharpe=benchmark_annualized_sharpe,
                probabilistic_sharpe_ratio=0.0,
                deflated_sharpe_ratio=0.0,
                expected_max_sharpe=0.0,
                skewness=0.0,
                kurtosis=3.0,
                n_samples=n,
                n_trials=n_trials,
                is_statistically_significant=False,
            )

        # Un-annualized period Sharpe ratio
        sr_period = mean_r / std_r
        sr_annual = sr_period * math.sqrt(annualization_factor)

        # Higher statistical moments
        skew = float(stats.skew(returns, bias=False))
        kurt = float(stats.kurtosis(returns, fisher=False, bias=False)) # Pearson kurtosis (Normal = 3.0)

        # Standard error of the period Sharpe ratio under non-normality (Mertens 2002 / Lo 2002)
        # Var(SR) = (1 / (N - 1)) * (1 - skew * SR + (kurt - 1)/4 * SR^2)
        sr_var = (1.0 / (n - 1.0)) * (1.0 - skew * sr_period + ((kurt - 1.0) / 4.0) * (sr_period**2))
        sr_std = math.sqrt(max(1e-12, sr_var))

        # 1. Probabilistic Sharpe Ratio PSR(SR*) against baseline
        sr_bench_period = benchmark_annualized_sharpe / math.sqrt(annualization_factor)
        z_psr = (sr_period - sr_bench_period) / sr_std
        psr = float(stats.norm.cdf(z_psr))

        # 2. Expected Maximum Sharpe ratio under N_trials independent tests
        if n_trials <= 1:
            sr_star_period = sr_bench_period
        else:
            euler_gamma = 0.5772156649
            # Exact Extreme Value Theory asymptotic mean
            z_term1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
            z_term2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
            z_max = (1.0 - euler_gamma) * z_term1 + euler_gamma * z_term2
            
            # Variance of Sharpe across trials (if not supplied, standard estimator is used)
            sigma_sr = math.sqrt(trial_variance_sharpe) if trial_variance_sharpe is not None else sr_std
            sr_star_period = sr_bench_period + sigma_sr * z_max

        # 3. Deflated Sharpe Ratio DSR = PSR(SR_max*)
        z_dsr = (sr_period - sr_star_period) / sr_std
        dsr = float(stats.norm.cdf(z_dsr))

        return DsrResult(
            observed_sharpe=sr_annual,
            benchmark_sharpe=benchmark_annualized_sharpe,
            probabilistic_sharpe_ratio=psr,
            deflated_sharpe_ratio=dsr,
            expected_max_sharpe=float(sr_star_period * math.sqrt(annualization_factor)),
            skewness=skew,
            kurtosis=kurt,
            n_samples=n,
            n_trials=n_trials,
            is_statistically_significant=(dsr >= 0.95),
        )
