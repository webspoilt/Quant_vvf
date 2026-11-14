"""
P3 — Market Regime Detection Engine
Gaussian Mixture Models (GMM), Multi-State Unsupervised Clustering,
Transition Probability Matrix, and Regime-Conditioned Analysis.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


@dataclass
class RegimeAnalysisResult:
    regime_labels: np.ndarray
    transition_matrix: np.ndarray
    regime_means: Dict[int, float]
    regime_volatilities: Dict[int, float]
    converged: bool
    bic: float
    aic: float


class MarketRegimeEngine:
    """Detects and monitors macroeconomic volatility regimes (Calm vs Crisis)."""

    def __init__(self, n_regimes: int = 2, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model: GaussianMixture = GaussianMixture(
            n_components=n_regimes,
            covariance_type="full",
            max_iter=300,
            random_state=random_state
        )

    def fit_predict(self, returns: pd.Series, rolling_vol_window: int = 21) -> RegimeAnalysisResult:
        """Fits multi-dimensional Gaussian Mixture Model on returns and rolling realized volatility."""
        roll_vol = returns.rolling(window=rolling_vol_window).std() * np.sqrt(365)
        df_feat = pd.DataFrame({"ret": returns, "vol": roll_vol}).dropna()

        X = df_feat.values
        self.model.fit(X)
        labels = self.model.predict(X)

        # Transition Probability Matrix
        n_comp = self.n_regimes
        trans_counts = np.zeros((n_comp, n_comp), dtype=np.float64)
        for t in range(len(labels) - 1):
            trans_counts[labels[t], labels[t + 1]] += 1.0

        row_sums = trans_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        transition_matrix = trans_counts / row_sums

        regime_means = {}
        regime_vols = {}
        for r in range(n_comp):
            idx = (labels == r)
            regime_means[r] = float(np.mean(df_feat["ret"].values[idx])) if np.any(idx) else 0.0
            regime_vols[r] = float(np.mean(df_feat["vol"].values[idx])) if np.any(idx) else 0.0

        return RegimeAnalysisResult(
            regime_labels=labels,
            transition_matrix=transition_matrix,
            regime_means=regime_means,
            regime_volatilities=regime_vols,
            converged=bool(self.model.converged_),
            bic=float(self.model.bic(X)),
            aic=float(self.model.aic(X)),
        )
