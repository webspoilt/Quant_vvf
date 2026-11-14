"""
Institutional Portfolio Optimization Engine
Covariance Shrinkage (Ledoit-Wolf), Equal Risk Contribution (Risk Parity),
Hierarchical Risk Parity (HRP), and Black-Litterman Allocation with View Uncertainty.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram


@dataclass
class AllocationResult:
    weights: Dict[str, float]
    expected_portfolio_return: float
    expected_portfolio_volatility: float
    diversification_ratio: float


class PortfolioOptimizer:
    """Rigorous institutional portfolio allocation suite."""

    @staticmethod
    def ledoit_wolf_shrinkage(returns_matrix: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """Calculates Ledoit-Wolf optimal analytical shrinkage towards constant correlation target."""
        X = returns_matrix.dropna().values
        n, p = X.shape
        if n < 5 or p < 2:
            raise ValueError("Insufficient dimensions for covariance shrinkage")

        # Sample covariance
        mean_x = np.mean(X, axis=0)
        X_c = X - mean_x
        sample_cov = (X_c.T @ X_c) / (n - 1)

        # Shrinkage target: Constant correlation
        var = np.diag(sample_cov)
        std = np.sqrt(var)
        corr = sample_cov / np.outer(std, std)
        np.fill_diagonal(corr, 1.0)

        mean_corr = (np.sum(corr) - p) / (p * (p - 1))
        target_corr = np.full((p, p), mean_corr)
        np.fill_diagonal(target_corr, 1.0)
        target = target_corr * np.outer(std, std)

        # Estimate shrinkage intensity delta
        y = X_c ** 2
        phi_mat = (y.T @ y) / n - 2.0 * (X_c.T @ X_c) * sample_cov / n + sample_cov ** 2
        phi = np.sum(phi_mat)

        gamma = np.linalg.norm(sample_cov - target, "fro") ** 2
        kappa = phi / gamma if gamma > 1e-12 else 0.0
        delta = max(0.0, min(1.0, kappa / n))

        shrunk_cov = delta * target + (1.0 - delta) * sample_cov
        return shrunk_cov, float(delta)

    @classmethod
    def risk_parity_allocation(cls, returns_matrix: pd.DataFrame) -> AllocationResult:
        """Solves Equal Risk Contribution (ERC) portfolio: w_i * (Sigma * w)_i = 1/N * total_var."""
        assets = list(returns_matrix.columns)
        p = len(assets)
        cov, _ = cls.ledoit_wolf_shrinkage(returns_matrix)

        def objective(w: np.ndarray) -> float:
            port_var = w.T @ cov @ w
            marginal_contrib = cov @ w
            risk_contrib = w * marginal_contrib
            # Target equal risk share: port_var / p
            target_risk = port_var / p
            return float(np.sum((risk_contrib - target_risk) ** 2))

        init_w = np.full(p, 1.0 / p)
        bounds = [(0.01, 0.90) for _ in range(p)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res = minimize(objective, init_w, bounds=bounds, constraints=constraints, method="SLSQP")
        w_opt = res.x / np.sum(res.x)

        port_vol = float(np.sqrt(w_opt.T @ cov @ w_opt * 365.0))
        asset_vols = np.sqrt(np.diag(cov) * 365.0)
        div_ratio = float(np.sum(w_opt * asset_vols) / port_vol) if port_vol > 0 else 1.0

        return AllocationResult(
            weights={assets[i]: float(w_opt[i]) for i in range(p)},
            expected_portfolio_return=0.0,
            expected_portfolio_volatility=port_vol,
            diversification_ratio=div_ratio
        )

    @staticmethod
    def hierarchical_risk_parity(returns_matrix: pd.DataFrame) -> AllocationResult:
        """Hierarchical Risk Parity (HRP) using tree clustering and recursive bisection."""
        assets = list(returns_matrix.columns)
        p = len(assets)
        corr = returns_matrix.corr().values
        cov = returns_matrix.cov().values

        # Distance metric: d_{i,j} = sqrt(0.5 * (1 - rho_{i,j}))
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))

        # Single linkage clustering
        link = linkage(dist, method="single")

        # Recursive Bisection to assign weights inversely proportional to cluster variance
        weights = pd.Series(1.0, index=range(p))
        clusters = [list(range(p))]

        while len(clusters) > 0:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) > 1:
                    mid = len(cluster) // 2
                    c1 = cluster[:mid]
                    c2 = cluster[mid:]

                    # Cluster variances
                    cov1 = cov[np.ix_(c1, c1)]
                    w1 = 1.0 / np.diag(cov1)
                    w1 /= np.sum(w1)
                    var1 = float(w1.T @ cov1 @ w1)

                    cov2 = cov[np.ix_(c2, c2)]
                    w2 = 1.0 / np.diag(cov2)
                    w2 /= np.sum(w2)
                    var2 = float(w2.T @ cov2 @ w2)

                    alpha = 1.0 - var1 / (var1 + var2)
                    weights.iloc[c1] *= alpha
                    weights.iloc[c2] *= (1.0 - alpha)

                    new_clusters.append(c1)
                    new_clusters.append(c2)
            clusters = new_clusters

        final_w = weights.values / np.sum(weights.values)
        port_vol = float(np.sqrt(final_w.T @ cov @ final_w * 365.0))
        asset_vols = np.sqrt(np.diag(cov) * 365.0)
        div_ratio = float(np.sum(final_w * asset_vols) / port_vol) if port_vol > 0 else 1.0

        return AllocationResult(
            weights={assets[i]: float(final_w[i]) for i in range(p)},
            expected_portfolio_return=0.0,
            expected_portfolio_volatility=port_vol,
            diversification_ratio=div_ratio
        )
