"""
Spectral Linear Algebra, Random Matrix Theory, Optimal Transport & Information Theory.

Modules:
1. Random Matrix Theory (Marchenko-Pastur & Covariance Denoising):
   - Marchenko-Pastur bounds: lambda_pm = sigma^2 * (1 +- sqrt(q))^2 where q = N / T.
   - Spectral filtering / eigenvalue clipping / constant residual shrinkage.
2. Optimal Transport (Wasserstein Distances & Sinkhorn):
   - 1D Wasserstein-1 and Wasserstein-2 distances.
   - Entropic Regularized Optimal Transport via Sinkhorn algorithm for regime drift detection.
3. Information Theory (Entropy, Mutual Information & Transfer Entropy):
   - Shannon entropy, differential entropy.
   - Kraskov-Stögbauer-Grassberger (KSG) / binned Mutual Information.
   - Transfer Entropy: T(X -> Y | Z) measuring directed predictive information flow.
4. Graph Laplacians & Spectral Clustering:
   - Correlation graphs, degree matrices, unnormalized and normalized symmetric Laplacians:
     L_sym = D^{-1/2} (D - W) D^{-1/2} = I - D^{-1/2} W D^{-1/2}
   - Fiedler vector / spectral partitioning for cross-market asset clustering.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
import scipy.spatial.distance as sp_dist


@dataclass
class RandomMatrixDenoising:
    """
    Marchenko-Pastur spectral covariance filtering (Laloux et al. 1999, Bouchaud & Potters 2009).
    Filters noise eigenvalues below lambda_max = sigma^2 * (1 + sqrt(N/T))^2.
    """
    aspect_ratio_q: float # N / T (number of assets / sample observations)

    @staticmethod
    def marchenko_pastur_bounds(var: float, q: float) -> Tuple[float, float]:
        """Calculates theoretical Marchenko-Pastur lower and upper bounds."""
        lambda_minus = var * (1.0 - math.sqrt(q)) ** 2
        lambda_plus = var * (1.0 + math.sqrt(q)) ** 2
        return lambda_minus, lambda_plus

    def denoise_covariance(
        self,
        cov_matrix: np.ndarray,
        shrinkage_target: str = "constant_residual",
    ) -> np.ndarray:
        """
        Denoises empirical covariance matrix:
        1. Decompose into correlation and variances.
        2. Eigendecompose correlation matrix.
        3. Identify eigenvalues inside [lambda_minus, lambda_plus].
        4. Shrink/clip noise eigenvalues while preserving trace = N.
        """
        n = cov_matrix.shape[0]
        std = np.sqrt(np.diag(cov_matrix))
        std_outer = np.outer(std, std)
        corr = cov_matrix / np.maximum(std_outer, 1e-12)

        # Eigendecomposition (symmetric)
        eigenvals, eigenvecs = np.linalg.eigh(corr)
        # Sort descending
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        # Estimate noise variance sigma^2 by fitting eigenvalues below top factor
        # Typically the median or lower bulk is used
        noise_var = float(np.median(eigenvals))
        _, lambda_plus = self.marchenko_pastur_bounds(noise_var, self.aspect_ratio_q)

        # Identify noise eigenvalues
        noise_mask = eigenvals <= lambda_plus
        n_noise = np.sum(noise_mask)

        denoised_evals = np.copy(eigenvals)
        if n_noise > 0:
            if shrinkage_target == "constant_residual":
                # Replace noise eigenvalues with their mean to preserve trace
                avg_noise = float(np.mean(eigenvals[noise_mask]))
                denoised_evals[noise_mask] = avg_noise
            else:
                # Clipping
                denoised_evals[noise_mask] = lambda_plus

        # Reconstruct correlation matrix
        corr_denoised = eigenvecs @ np.diag(denoised_evals) @ eigenvecs.T
        # Rescale diagonal to 1.0
        d = np.diag(corr_denoised)
        corr_denoised = corr_denoised / np.sqrt(np.outer(d, d))
        np.fill_diagonal(corr_denoised, 1.0)

        # Rescale back to covariance
        cov_denoised = corr_denoised * std_outer
        return cov_denoised


@dataclass
class OptimalTransportDriftDetector:
    """
    Computes Wasserstein distances and Sinkhorn entropic divergence
    to detect distributional regime shifts between reference and current windows.
    """
    reg_epsilon: float = 0.05
    max_iter: int = 100

    @staticmethod
    def wasserstein_1d(u_samples: np.ndarray, v_samples: np.ndarray) -> float:
        """Computes exact 1D Wasserstein-1 (Earth Mover's) distance."""
        return float(stats.wasserstein_distance(u_samples, v_samples))

    def sinkhorn_divergence(
        self,
        x_samples: np.ndarray,
        y_samples: np.ndarray,
    ) -> float:
        """
        Entropic Regularized Optimal Transport via Sinkhorn-Knopp algorithm.
        Returns regularized transport cost between empirical distributions X and Y.
        """
        nx = len(x_samples)
        ny = len(y_samples)
        # Uniform empirical weights
        a = np.ones(nx) / nx
        b = np.ones(ny) / ny

        # Cost matrix: squared Euclidean distance
        if x_samples.ndim == 1:
            m_cost = sp_dist.cdist(x_samples[:, None], y_samples[:, None], metric="sqeuclidean")
        else:
            m_cost = sp_dist.cdist(x_samples, y_samples, metric="sqeuclidean")

        # Gibbs kernel
        k_kernel = np.exp(-m_cost / self.reg_epsilon)

        u = np.ones(nx) / nx
        for _ in range(self.max_iter):
            v = b / np.maximum(k_kernel.T @ u, 1e-12)
            u = a / np.maximum(k_kernel @ v, 1e-12)

        # Optimal transport matrix P = diag(u) * K * diag(v)
        p_opt = np.diag(u) @ k_kernel @ np.diag(v)
        cost = float(np.sum(p_opt * m_cost))
        return cost


@dataclass
class InformationTheoryEngine:
    """
    Computes Shannon Entropy, Mutual Information, and Transfer Entropy
    measuring nonlinear dependence and directed information flow between assets.
    """
    n_bins: int = 15

    def mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Binned Mutual Information I(X; Y) = H(X) + H(Y) - H(X, Y)."""
        c_xy, _, _ = np.histogram2d(x, y, bins=self.n_bins)
        p_xy = c_xy / np.sum(c_xy)
        p_x = np.sum(p_xy, axis=1) # Shape (n_bins,)
        p_y = np.sum(p_xy, axis=0) # Shape (n_bins,)

        # Marginal product P(X) * P(Y)
        p_prod = np.outer(p_x, p_y)

        # Mask non-zero joint probabilities
        nz = (p_xy > 0) & (p_prod > 0)
        mi = np.sum(p_xy[nz] * np.log(p_xy[nz] / p_prod[nz]))
        return max(0.0, float(mi))

    def transfer_entropy(
        self,
        x_source: np.ndarray,
        y_target: np.ndarray,
        lag: int = 1,
    ) -> float:
        """
        Computes directed Transfer Entropy T(X -> Y) = I(Y_t; X_{t-lag} | Y_{t-lag}).
        Measures predictive information flow from source X to destination Y.
        """
        n = len(x_source)
        if n <= lag + 5:
            return 0.0

        y_fut = y_target[lag:]
        y_past = y_target[:-lag]
        x_past = x_source[:-lag]

        # 3D Joint Histogram: (Y_fut, Y_past, X_past)
        bins = max(5, self.n_bins // 2)
        sample = np.vstack([y_fut, y_past, x_past]).T
        hist_3d, _ = np.histogramdd(sample, bins=bins)
        p_3d = hist_3d / np.sum(hist_3d)

        p_2d_y = np.sum(p_3d, axis=2) # P(Y_fut, Y_past)
        p_2d_cond = np.sum(p_3d, axis=0) # P(Y_past, X_past)
        p_1d_past = np.sum(p_2d_y, axis=0) # P(Y_past)

        te = 0.0
        for i in range(bins):
            for j in range(bins):
                for k in range(bins):
                    p_ijk = p_3d[i, j, k]
                    if p_ijk > 1e-12:
                        p_cond_yx = p_2d_cond[j, k]
                        p_y_fut_past = p_2d_y[i, j]
                        p_y_past = p_1d_past[j]
                        if p_cond_yx > 1e-12 and p_y_past > 1e-12:
                            arg = (p_ijk / p_cond_yx) / (p_y_fut_past / p_y_past)
                            if arg > 0:
                                te += p_ijk * math.log(arg)

        return max(0.0, float(te))


@dataclass
class MarketGraphSpectralEngine:
    """
    Constructs graph Laplacian from cross-market correlation networks:
    L = D - W, L_sym = D^{-1/2} L D^{-1/2}
    Eigenvectors provide spectral clustering and market lead-lag partitioning.
    """
    @staticmethod
    def construct_normalized_laplacian(adj_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates normalized symmetric graph Laplacian:
        Returns: (L_sym, eigenvalues, eigenvectors)
        """
        n = adj_matrix.shape[0]
        # Degrees
        degrees = np.sum(np.abs(adj_matrix), axis=1)
        deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
        d_mat_inv_sqrt = np.diag(deg_inv_sqrt)

        l_unnormalized = np.diag(degrees) - adj_matrix
        l_sym = d_mat_inv_sqrt @ l_unnormalized @ d_mat_inv_sqrt

        eigenvals, eigenvecs = np.linalg.eigh(l_sym)
        return l_sym, eigenvals, eigenvecs
