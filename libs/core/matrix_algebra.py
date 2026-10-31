"""Advanced Linear Algebra and Spectral Matrix Methods for Quantitative Finance.

Implements:
1. Higham (2002) Nearest Correlation Matrix Algorithm:
   Solves min ||A - X||_F subject to X in S_+^n and diag(X) = I using
   Dykstra's alternating projection method on closed convex cones.
2. Positive Semi-Definite (PSD) Cone Projection:
   Orthogonal projection of symmetric matrices onto S_+^n via spectral decomposition.
3. Continuous-Time Markov Transition Semigroups:
   Matrix exponential P(t) = exp(t * Q) for intensity generators via scaling and squaring.
4. Spectral Dispersion & Principal Component Analysis (PCA):
   Variance term-structure factorization and factor risk budgeting.

References:
- Higham, N. J. (2002), "Computing the nearest correlation matrix - a problem from finance",
  IMA Journal of Numerical Analysis, 22(3), 329-343.
- Higham, N. J. (2005), "The scaling and squaring method for the matrix exponential revisited",
  SIAM Journal on Matrix Analysis and Applications, 26(4), 1179-1193.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast
import numpy as np

from libs.core.validation import assert_finite_scalar


@dataclass(frozen=True)
class NearestCorrelationResult:
    correlation_matrix: np.ndarray
    iterations: int
    converged: bool
    frobenius_distance: float
    min_eigenvalue: float


def project_psd(a: np.ndarray, min_eigenvalue: float = 1e-8) -> np.ndarray:
    """Project a symmetric matrix onto the cone of positive semi-definite matrices (S_+^n).

    Computes:
        P_{S_+^n}(A) = Q * diag(max(min_eig, lambda_i)) * Q^T

    Parameters
    ----------
    a : np.ndarray
        Real symmetric matrix (n x n)
    min_eigenvalue : float
        Strict positive floor for eigenvalues to guarantee positive definiteness.

    Returns
    -------
    np.ndarray
        Nearest positive semi-definite matrix.
    """
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be a square 2D matrix, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("Matrix contains non-finite elements (NaN or Inf)")

    # Symmetrize to eliminate machine precision asymmetries
    sym_a = 0.5 * (a + a.T)

    eigvals, eigvecs = np.linalg.eigh(sym_a)
    clipped_eigvals = np.maximum(eigvals, min_eigenvalue)

    res = eigvecs @ np.diag(clipped_eigvals) @ eigvecs.T
    return cast(np.ndarray, 0.5 * (res + res.T))


def nearest_correlation_matrix(
    a: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-7,
    min_eigenvalue: float = 1e-7,
) -> NearestCorrelationResult:
    """Compute the Nearest Correlation Matrix using Higham's (2002) alternating projections.

    Given an empirical, non-positive semi-definite or noisy correlation matrix A,
    finds the unique matrix X* satisfying:
        min 0.5 * ||A - X||_F^2
        subject to:
            X in S_+^n  (positive semi-definite)
            X_ii = 1.0  (unit diagonal)

    Parameters
    ----------
    a : np.ndarray
        Candidate symmetric correlation/covariance matrix (n x n)
    max_iterations : int
        Maximum Dykstra alternating projection cycles
    tolerance : float
        Stopping criterion on relative Frobenius norm: ||X_{k} - X_{k-1}||_F / ||X_k||_F
    min_eigenvalue : float
        Eigenvalue floor ensuring strict positive definiteness

    Returns
    -------
    NearestCorrelationResult
        Repaired correlation matrix, iteration count, and convergence metrics.
    """
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Input must be a square 2D matrix, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("Matrix contains non-finite elements (NaN or Inf)")

    n = a.shape[0]
    y_k = 0.5 * (a + a.T)
    delta_s = np.zeros((n, n), dtype=np.float64)

    converged = False
    iteration = 0

    for it in range(1, max_iterations + 1):
        iteration = it
        # 1. Dykstra correction
        r_k = y_k - delta_s

        # 2. Project onto S_+^n (PSD cone)
        x_k = project_psd(r_k, min_eigenvalue=min_eigenvalue)

        # 3. Update increment
        delta_s = x_k - r_k

        # 4. Project onto U (unit diagonal affine space)
        y_next = x_k.copy()
        np.fill_diagonal(y_next, 1.0)
        y_next = 0.5 * (y_next + y_next.T)

        # Convergence test
        diff = np.linalg.norm(y_next - y_k, ord="fro")
        scale = max(1.0, float(np.linalg.norm(y_next, ord="fro")))
        if (diff / scale) < tolerance:
            converged = True
            y_k = y_next
            break

        y_k = y_next

    # Final guarantee of unit diagonal and symmetry
    np.fill_diagonal(y_k, 1.0)
    final_matrix = 0.5 * (y_k + y_k.T)

    final_eigvals = np.linalg.eigvalsh(final_matrix)
    min_eig = float(np.min(final_eigvals))
    frob_dist = float(np.linalg.norm(final_matrix - a, ord="fro"))

    return NearestCorrelationResult(
        correlation_matrix=final_matrix,
        iterations=iteration,
        converged=converged,
        frobenius_distance=frob_dist,
        min_eigenvalue=min_eig,
    )


def matrix_exponential_generator(
    q_generator: np.ndarray,
    t: float,
) -> np.ndarray:
    """Compute the continuous-time Markov transition matrix P(t) = exp(t * Q).

    Uses the Padé approximation with scaling and squaring (Higham 2005).

    Parameters
    ----------
    q_generator : np.ndarray
        Infinitesimal transition rate matrix (Q_ij >= 0 for i != j, sum_j Q_ij = 0)
    t : float
        Time step (>= 0)

    Returns
    -------
    np.ndarray
        Stochastic transition probability matrix P(t) with row sums equal to 1.
    """
    assert_finite_scalar(t, "t")
    if t < 0:
        raise ValueError(f"Time t must be non-negative, got {t}")
    if q_generator.ndim != 2 or q_generator.shape[0] != q_generator.shape[1]:
        raise ValueError("Generator matrix must be square")

    # Use scipy.linalg.expm for high-accuracy Padé scaling-squaring
    from scipy.linalg import expm
    p = expm(q_generator * t)

    # Normalize row sums to enforce probability axioms P(t) 1 = 1
    p = np.clip(p, 0.0, 1.0)
    row_sums = p.sum(axis=1, keepdims=True)
    p = p / np.maximum(1e-12, row_sums)
    return cast(np.ndarray, p)


@dataclass(frozen=True)
class PCAResult:
    explained_variance_ratio: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray


def spectral_pca(cov_matrix: np.ndarray, n_components: int | None = None) -> PCAResult:
    """Perform exact Spectral PCA on a covariance or correlation matrix.

    Extracts principal volatility modes (Level, Slope, Curvature).
    """
    if cov_matrix.ndim != 2 or cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("Covariance matrix must be square")

    eigvals, eigvecs = np.linalg.eigh(0.5 * (cov_matrix + cov_matrix.T))

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    sorted_eigvals = np.maximum(0.0, eigvals[idx])
    sorted_eigvecs = eigvecs[:, idx]

    total_var = np.sum(sorted_eigvals)
    var_ratio = sorted_eigvals / max(1e-12, total_var)

    k = n_components or cov_matrix.shape[0]
    return PCAResult(
        explained_variance_ratio=var_ratio[:k],
        components=sorted_eigvecs[:, :k].T,
        singular_values=np.sqrt(sorted_eigvals[:k]),
    )
