"""
P4 — Principal Component Analysis and Latent Factor Decomposition
Spectral Decomposition (eigh), Explained Variance, Factor Loadings, and Risk Exposures.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd


@dataclass
class PCAResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    factor_loadings: pd.DataFrame
    principal_components: pd.DataFrame


class PCAFactorEngine:
    """Rigorous latent factor decomposition for cross-sectional and term-structure surfaces."""

    @staticmethod
    def decompose_returns(returns_matrix: pd.DataFrame, n_components: int = 3) -> PCAResult:
        """Performs spectral decomposition via Hermitian/Symmetric eigenvalue solver (numpy.linalg.eigh)."""
        df_clean = returns_matrix.dropna()
        if df_clean.shape[0] < df_clean.shape[1] or df_clean.shape[1] < 2:
            raise ValueError("Data matrix requires observations >= features and at least 2 features")

        # Standardize (Z-score normalize features)
        means = df_clean.mean()
        stds = df_clean.std(ddof=1)
        stds[stds == 0] = 1.0
        standardized = (df_clean - means) / stds

        # Covariance / Correlation matrix
        cov_matrix = np.cov(standardized.values, rowvar=False)

        # Spectral decomposition
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)

        # Sort in descending order of explained variance
        idx = np.argsort(eigvals)[::-1]
        sorted_eigvals = eigvals[idx]
        sorted_eigvecs = eigvecs[:, idx]

        total_var = np.sum(sorted_eigvals)
        explained_ratio = sorted_eigvals / total_var
        cum_var = np.cumsum(explained_ratio)

        k = min(n_components, len(sorted_eigvals))
        top_eigenvalues = sorted_eigvals[:k]
        top_eigenvectors = sorted_eigvecs[:, :k]

        factor_loadings = pd.DataFrame(
            top_eigenvectors,
            index=returns_matrix.columns,
            columns=[f"PC{i+1}" for i in range(k)]
        )

        # Project standardized returns into latent component space
        components = standardized.values @ top_eigenvectors
        pc_df = pd.DataFrame(
            components,
            index=df_clean.index,
            columns=[f"PC{i+1}" for i in range(k)]
        )

        return PCAResult(
            eigenvalues=top_eigenvalues,
            eigenvectors=top_eigenvectors,
            explained_variance_ratio=explained_ratio[:k],
            cumulative_variance=cum_var[:k],
            factor_loadings=factor_loadings,
            principal_components=pc_df,
        )
