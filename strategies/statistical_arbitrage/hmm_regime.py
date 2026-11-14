"""
Production-Grade Gaussian Hidden Markov Model (HMM) for Regime Detection.

Phase 6 Remediation:
- Genuine Forward-Backward recursion with numerical scaling.
- Filtered state probabilities P(S_t | y_{1:t}) using strictly historical data (online trading signal).
- Smoothed state probabilities P(S_t | y_{1:T}) using full sample information (offline regime analysis).
- Baum-Welch Expectation-Maximization updating transition matrix from expected transition counts xi_{t}(j, k).
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats


@dataclass
class HmmFitResult:
    log_likelihood: float
    converged: bool
    iterations: int
    filtered_probs: np.ndarray    # P(S_t = j | y_{1:t}) (Online, non-lookahead)
    smoothed_probs: np.ndarray    # P(S_t = j | y_{1:T}) (Offline full-sample)
    transition_matrix: np.ndarray # A_{j,k} = P(S_t = k | S_{t-1} = j)
    means: np.ndarray
    variances: np.ndarray


class GaussianHMMRegimeDetector:
    """
    Continuous Gaussian Hidden Markov Model with scaled Forward-Backward Baum-Welch EM.
    """

    def __init__(self, n_regimes: int = 2, max_iters: int = 100, tol: float = 1e-5):
        self.n_regimes = n_regimes
        self.max_iters = max_iters
        self.tol = tol
        self.pi: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.means: Optional[np.ndarray] = None
        self.variances: Optional[np.ndarray] = None

    def fit(self, observations: np.ndarray) -> HmmFitResult:
        """
        Fits Gaussian HMM using scaled Baum-Welch EM algorithm.
        observations: 1D array of asset log-returns or volatilities.
        """
        n = len(observations)
        k = self.n_regimes

        # Quantile-based deterministic initialization
        sorted_obs = np.sort(observations)
        q_steps = np.linspace(0, n, k + 1, dtype=int)
        self.means = np.array([np.mean(sorted_obs[q_steps[i] : q_steps[i + 1]]) for i in range(k)], dtype=np.float64)
        total_var = float(np.var(observations))
        self.variances = np.full(k, max(1e-6, total_var), dtype=np.float64)
        self.A = np.full((k, k), 1.0 / k, dtype=np.float64)
        np.fill_diagonal(self.A, 0.8)
        self.A = self.A / np.sum(self.A, axis=1, keepdims=True)
        self.pi = np.full(k, 1.0 / k, dtype=np.float64)

        prev_ll = -np.inf
        converged = False
        iteration_count = 0

        alpha = np.zeros((n, k), dtype=np.float64)
        beta = np.zeros((n, k), dtype=np.float64)
        c = np.zeros(n, dtype=np.float64) # Scaling factors

        for iteration in range(self.max_iters):
            iteration_count += 1
            # 1. Emission matrix B: B_{t, j} = f(y_t | S_t = j)
            B = np.zeros((n, k), dtype=np.float64)
            for j in range(k):
                std_j = math.sqrt(max(1e-8, self.variances[j]))
                B[:, j] = stats.norm.pdf(observations, loc=self.means[j], scale=std_j)
            B = np.maximum(B, 1e-15)

            # 2. Scaled Forward Recursion
            # alpha_hat_1(j) = pi(j) * B_1(j)
            alpha_raw_0 = self.pi * B[0]
            c[0] = 1.0 / np.sum(alpha_raw_0)
            alpha[0] = alpha_raw_0 * c[0]

            for t in range(1, n):
                alpha_raw_t = (alpha[t - 1] @ self.A) * B[t]
                c_t = np.sum(alpha_raw_t)
                c[t] = 1.0 / max(c_t, 1e-15)
                alpha[t] = alpha_raw_t * c[t]

            # Current log-likelihood: - sum(log(c_t))
            log_likelihood = -float(np.sum(np.log(c)))

            # 3. Scaled Backward Recursion
            # beta_hat_T(j) = c_T
            beta[n - 1] = c[n - 1]
            for t in range(n - 2, -1, -1):
                beta_raw_t = self.A @ (B[t + 1] * beta[t + 1])
                beta[t] = beta_raw_t * c[t]

            # 4. Smoothed State Posteriors: gamma_t(j) = P(S_t = j | y_{1:T})
            gamma = alpha * beta
            gamma = gamma / np.sum(gamma, axis=1, keepdims=True)

            # 5. Transition Joint Posteriors: xi_t(j, k) = P(S_t = j, S_{t+1} = k | y_{1:T})
            xi_sum = np.zeros((k, k), dtype=np.float64)
            for t in range(n - 1):
                xi_t = np.outer(alpha[t], B[t + 1] * beta[t + 1]) * self.A
                xi_sum += xi_t / np.sum(xi_t)

            # 6. M-Step Parameter Updates
            self.pi = gamma[0] / np.sum(gamma[0])
            self.A = xi_sum / np.maximum(1e-15, np.sum(xi_sum, axis=1, keepdims=True))

            gamma_sums = np.maximum(1e-15, np.sum(gamma, axis=0))
            for j in range(k):
                self.means[j] = np.sum(gamma[:, j] * observations) / gamma_sums[j]
                self.variances[j] = np.sum(gamma[:, j] * (observations - self.means[j])**2) / gamma_sums[j]
                self.variances[j] = max(1e-6, self.variances[j])

            # Convergence Check
            if abs(log_likelihood - prev_ll) < self.tol:
                converged = True
                break
            prev_ll = log_likelihood

        # Calculate final filtered probabilities P(S_t = j | y_{1:t}) strictly from forward variable
        filtered_probs = alpha / np.sum(alpha, axis=1, keepdims=True)

        return HmmFitResult(
            log_likelihood=prev_ll,
            converged=converged,
            iterations=iteration_count,
            filtered_probs=filtered_probs,
            smoothed_probs=gamma,
            transition_matrix=self.A,
            means=self.means,
            variances=self.variances,
        )

    def filter_online_step(self, prior_filtered: np.ndarray, new_observation: float) -> np.ndarray:
        """
        Online recursive update using strictly historical observations (no future information leakage).
        P(S_t = j | y_{1:t}) propto f(y_t | S_t = j) * sum_i [ P(S_{t-1} = i | y_{1:t-1}) * A_{i, j} ]
        """
        if self.A is None or self.means is None or self.variances is None:
            raise ValueError("HMM must be fitted before online step evaluation")

        k = self.n_regimes
        # Predict step: P(S_t = j | y_{1:t-1})
        predicted_state = prior_filtered @ self.A

        # Update step: Likelihood * Prior
        likelihoods = np.zeros(k, dtype=np.float64)
        for j in range(k):
            std_j = math.sqrt(self.variances[j])
            likelihoods[j] = stats.norm.pdf(new_observation, loc=self.means[j], scale=std_j)

        posterior = predicted_state * likelihoods
        sum_post = np.sum(posterior)
        return (posterior / sum_post) if sum_post > 0 else predicted_state
