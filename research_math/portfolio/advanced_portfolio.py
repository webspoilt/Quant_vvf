"""
QUANT_VVF — Portfolio Mathematics, CVaR Optimization & Covariance Shrinkage.
References:
- Rockafellar & Uryasev (2000): Optimization of Conditional Value-at-Risk
- Ledoit & Wolf (2004): Well-Conditioned Estimator for Large-Dimensional Covariance Matrices
- López de Prado (2016): Building Diversified Portfolios that Outperform Out-of-Sample (HRP)
- Maillard, Roncalli & Teïletche (2010): The Properties of Equally Weighted Risk Contributions
"""

from __future__ import annotations
import math
from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram


def ledoit_wolf_shrinkage(returns: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Ledoit-Wolf (2004) analytical optimal shrinkage towards constant correlation target.
    Returns (shrunk_cov_matrix, shrinkage_intensity).
    """
    X = np.asarray(returns, dtype=np.float64)
    T, N = X.shape
    if T < 5:
        raise ValueError("Sample size T must be at least 5 for shrinkage calculation.")

    # Demean
    mean = np.mean(X, axis=0)
    X = X - mean

    # Sample covariance S
    sample_cov = (X.T @ X) / T

    # Target: constant correlation F
    var = np.diag(sample_cov)
    std = np.sqrt(var)
    r_bar = (np.sum(sample_cov / np.outer(std, std)) - N) / (N * (N - 1))
    target = r_bar * np.outer(std, std)
    np.fill_diagonal(target, var)

    # Estimate pi (sum of asymptotic variances of sample covariance)
    X2 = X ** 2
    sample_cov2 = (X2.T @ X2) / T
    pi_mat = sample_cov2 - sample_cov ** 2
    pi_hat = np.sum(pi_mat)

    # Estimate gamma (misspecification of target)
    gamma_hat = np.linalg.norm(sample_cov - target, "fro") ** 2

    # Estimate rho (cross-terms)
    rho_diag = np.sum(np.diag(pi_mat))
    term1 = (X ** 3).T @ X / T
    term2 = var[:, None] * sample_cov
    theta_mat = term1 - term2
    rho_off = r_bar * np.sum(np.outer(1.0 / std, std) * theta_mat) - r_bar * np.sum(np.diag(theta_mat))
    rho_hat = rho_diag + rho_off

    # Optimal shrinkage intensity kappa
    kappa = (pi_hat - rho_hat) / gamma_hat
    shrinkage = max(0.0, min(1.0, kappa / T))

    shrunk_cov = shrinkage * target + (1.0 - shrinkage) * sample_cov
    return shrunk_cov, float(shrinkage)


def cvar_portfolio_optimization(
    scenario_returns: np.ndarray,
    alpha: float = 0.95,
    target_return: Optional[float] = None,
    allow_short: bool = False,
) -> Tuple[np.ndarray, float, float]:
    """
    Rockafellar & Uryasev (2000) linear/convex CVaR loss threshold optimization:
    minimize over (w, eta):
    eta + (1 / ((1 - alpha) * S)) * sum_{s=1}^S max( -w^T r_s - eta, 0 )
    subject to: sum(w) = 1, w >= 0 (if no short), E[w^T r] >= target_return.
    Returns (weights, optimal_VaR_eta, optimal_CVaR).
    """
    R = np.asarray(scenario_returns, dtype=np.float64)  # (S, N)
    S, N = R.shape
    if S < 10:
        raise ValueError("Need at least 10 scenarios for CVaR optimization.")

    expected_returns = np.mean(R, axis=0)

    # Decision variables: x = [w_1, ..., w_N, eta] -> dimension N + 1
    # Initial weights: equal weights, eta: 95th percentile sample loss
    w0 = np.ones(N, dtype=np.float64) / N
    losses_0 = -R @ w0
    eta0 = float(np.percentile(losses_0, alpha * 100.0))
    x0 = np.concatenate([w0, [eta0]])

    def objective(x: np.ndarray) -> float:
        w = x[:N]
        eta = x[N]
        losses = -R @ w  # portfolio loss per scenario
        excess = np.maximum(0.0, losses - eta)
        cvar = eta + (1.0 / ((1.0 - alpha) * S)) * np.sum(excess)
        return float(cvar)

    constraints = []
    # Budget constraint: sum(w) = 1
    constraints.append({"type": "eq", "fun": lambda x: np.sum(x[:N]) - 1.0})

    if target_return is not None:
        constraints.append({"type": "ineq", "fun": lambda x: float(x[:N] @ expected_returns) - target_return})

    bounds = [(0.0 if not allow_short else -1.0, 1.0) for _ in range(N)] + [(None, None)]

    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 300, "ftol": 1e-7})

    opt_w = res.x[:N]
    opt_eta = float(res.x[N])
    opt_cvar = float(res.fun)

    return opt_w, opt_eta, opt_cvar


def risk_parity_weights(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Equal Risk Contribution (ERC) portfolio (Maillard et al. 2010):
    sigma_i(w) = (w * (Sigma * w)_i) / (w^T Sigma w) = 1 / N for all i.
    """
    Sigma = np.asarray(cov_matrix, dtype=np.float64)
    N = Sigma.shape[0]
    w0 = np.ones(N, dtype=np.float64) / N

    def erc_objective(w: np.ndarray) -> float:
        w_col = w.reshape(-1, 1)
        port_var = float((w_col.T @ Sigma @ w_col).item())
        if port_var <= 1e-12:
            return 1e6
        mrc = (Sigma @ w_col).flatten()
        rc = w * mrc / port_var  # risk contributions
        target_rc = 1.0 / N
        return float(np.sum((rc - target_rc) ** 2))

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(1e-5, 1.0) for _ in range(N)]

    res = minimize(erc_objective, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 250})
    return res.x / np.sum(res.x)


def hierarchical_risk_parity_weights(returns: np.ndarray) -> np.ndarray:
    """
    Hierarchical Risk Parity (López de Prado 2016):
    Clustering -> Quasi-Diagonalization -> Recursive Bisection.
    """
    X = np.asarray(returns, dtype=np.float64)
    T, N = X.shape
    corr = np.corrcoef(X, rowvar=False)
    cov = np.cov(X, rowvar=False)

    # 1. Tree clustering distance
    dist = np.sqrt(0.5 * (1.0 - np.clip(corr, -1.0, 1.0)))
    np.fill_diagonal(dist, 0.0)

    # Linkage
    condensed_dist = dist[np.triu_indices(N, k=1)]
    link = linkage(condensed_dist, method="single")

    # 2. Quasi-diagonalization
    sort_order = [int(link[-1, 0]), int(link[-1, 1])]
    while len(sort_order) < N:
        new_order = []
        for idx in sort_order:
            if idx < N:
                new_order.append(idx)
            else:
                cluster_idx = idx - N
                new_order.append(int(link[cluster_idx, 0]))
                new_order.append(int(link[cluster_idx, 1]))
        sort_order = new_order

    # 3. Recursive bisection
    weights = np.ones(N, dtype=np.float64)
    clusters = [sort_order]

    while len(clusters) > 0:
        next_clusters = []
        for c in clusters:
            if len(c) > 1:
                mid = len(c) // 2
                c1 = c[:mid]
                c2 = c[mid:]

                cov1 = cov[np.ix_(c1, c1)]
                cov2 = cov[np.ix_(c2, c2)]

                # Inverse variance allocation between sub-clusters
                inv_var1 = 1.0 / np.trace(cov1)
                inv_var2 = 1.0 / np.trace(cov2)
                alpha1 = inv_var1 / (inv_var1 + inv_var2)
                alpha2 = 1.0 - alpha1

                weights[c1] *= alpha1
                weights[c2] *= alpha2

                next_clusters.extend([c1, c2])
        clusters = next_clusters

    return weights / np.sum(weights)
