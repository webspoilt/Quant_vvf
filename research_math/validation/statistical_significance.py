"""
QUANT_VVF — Deflated Sharpe Ratio (DSR) & Statistical Significance of Backtest Overfitting.
References:
- Bailey & López de Prado (2014): The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality.
  Journal of Portfolio Management, 40(5), 94-107.
- López de Prado (2018): Advances in Financial Machine Learning. Wiley.
"""

from __future__ import annotations
import math
from typing import Dict, Any, Tuple
import numpy as np
from scipy.stats import norm


def deflated_sharpe_ratio(
    observed_sr: float,
    num_trials: int,
    sample_length: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sr: float = 0.0,
) -> float:
    """
    Computes the Deflated Sharpe Ratio (Bailey & López de Prado 2014):
    DSR = Phi( (SR_hat - SR_0) * sqrt(T - 1) / sqrt(1 - skew * SR_hat + ((kurt - 1) / 4) * SR_hat^2) )
    where SR_0 is expected maximum Sharpe ratio under null hypothesis of zero true skill across N trials:
    SR_0 = sqrt(2 * ln(N)) + Euler_gamma / sqrt(2 * ln(N))
    """
    SR = observed_sr
    N = max(1, num_trials)
    T = max(2, sample_length)
    skew = skewness
    kurt = max(1.0, kurtosis)

    if N == 1:
        sr_0 = benchmark_sr
    else:
        # Expected max Sharpe ratio among N false strategies
        gamma = 0.5772156649  # Euler-Mascheroni constant
        z = math.sqrt(2.0 * math.log(N))
        sr_0 = z + gamma / z

    # Variance of Sharpe ratio estimator under non-normality
    var_sr = 1.0 - skew * SR + 0.25 * (kurt - 1.0) * (SR ** 2)
    var_sr = max(1e-6, var_sr)

    # Standardized test statistic
    z_stat = (SR - sr_0) * math.sqrt(T - 1.0) / math.sqrt(var_sr)

    # P-value / probability that observed SR is not a false positive
    dsr_prob = float(norm.cdf(z_stat))
    return dsr_prob
