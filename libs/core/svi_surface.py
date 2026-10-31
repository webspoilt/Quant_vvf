"""Gatheral's Quasi-Explicit SVI Surface Calibration and Arbitrage Validation.

Mathematical reference:
Gatheral, J. (2004), "A parsimonious arbitrage-free implied volatility parameterization"
Gatheral, J. & Jacquier, A. (2014), "Arbitrage-free SVI volatility surfaces"

Raw SVI parameterization:
    w(k; chi) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
where:
    k     = ln(K / F) is the log-moneyness relative to forward F
    w     = sigma_BS^2 * tau is total implied variance
    a     = vertical level of variance
    b     = angle between the asymptotes (b >= 0)
    rho   = orientation / skew parameter (|rho| < 1)
    m     = horizontal displacement of the smile minimum
    sigma = curvature / smoothness at the money (sigma > 0)

Arbitrage-free conditions:
1. Positivity: w(k) >= 0 for all k, requiring a + b * sigma * sqrt(1 - rho^2) >= 0.
2. Large-moneyness slope: b * (1 + |rho|) <= 2 (Lee's moment formula).
3. No butterfly arbitrage: Durrleman's risk-neutral density g(k) >= 0 for all k:
    g(k) = (1 - k*w'/(2*w))^2 - (w'^2 / 4) * (1/w + 1/4) + w'' / 2 >= 0
4. No calendar arbitrage across slices tau1 < tau2:
    w(k, tau1) <= w(k, tau2) for all k.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize, lsq_linear


@dataclass(frozen=True)
class SVIParameters:
    """Raw SVI parameter set for a single expiry slice."""
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        # V-22: NaN defeats every `b < 0` / `|rho| > 1` / `sigma <= 0`
        # comparison, letting corrupt parameters calibrate a nonsense surface.
        # Fail-closed: any non-finite parameter is a calibration corruption.
        for name in ("a", "b", "rho", "m", "sigma"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"SVI parameter {name} must be finite, got {value!r}")
        if self.b < 0:
            raise ValueError(f"SVI parameter b must be non-negative, got {self.b}")
        if abs(self.rho) > 1.0:
            raise ValueError(f"SVI parameter rho must be in [-1, 1], got {self.rho}")
        if self.sigma <= 0:
            raise ValueError(f"SVI parameter sigma must be positive, got {self.sigma}")

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.a, self.b, self.rho, self.m, self.sigma)


def svi_total_variance(k: np.ndarray | float, params: SVIParameters) -> np.ndarray | float:
    """Evaluate raw SVI total implied variance w(k)."""
    k_arr = np.asarray(k, dtype=np.float64)
    km = k_arr - params.m
    disc = np.sqrt(km * km + params.sigma * params.sigma)
    w = params.a + params.b * (params.rho * km + disc)
    if np.isscalar(k):
        return float(w)
    return w


def svi_implied_vol(k: np.ndarray | float, tau: float, params: SVIParameters) -> np.ndarray | float:
    """Evaluate Black-Scholes implied volatility sigma(k, tau) = sqrt(max(w, 0) / tau)."""
    if tau <= 0:
        raise ValueError(f"tau must be strictly positive, got {tau}")
    w = svi_total_variance(k, params)
    w_clipped = np.maximum(w, 0.0)
    return np.sqrt(w_clipped / tau)


def svi_first_derivative(k: np.ndarray | float, params: SVIParameters) -> np.ndarray | float:
    """First derivative of total variance w'(k)."""
    k_arr = np.asarray(k, dtype=np.float64)
    km = k_arr - params.m
    disc = np.sqrt(km * km + params.sigma * params.sigma)
    w_prime = params.b * (params.rho + km / disc)
    if np.isscalar(k):
        return float(w_prime)
    return w_prime


def svi_second_derivative(k: np.ndarray | float, params: SVIParameters) -> np.ndarray | float:
    """Second derivative of total variance w''(k)."""
    k_arr = np.asarray(k, dtype=np.float64)
    km = k_arr - params.m
    disc = np.sqrt(km * km + params.sigma * params.sigma)
    w_second = params.b * (params.sigma * params.sigma) / (disc ** 3)
    if np.isscalar(k):
        return float(w_second)
    return w_second


def durrleman_density_factor(k: np.ndarray | float, params: SVIParameters) -> np.ndarray | float:
    """Evaluate Durrleman's function g(k).

    For no butterfly arbitrage (positive risk-neutral density), g(k) >= 0 everywhere.
    g(k) = (1 - k*w'/(2*w))^2 - (w'^2 / 4) * (1/w + 1/4) + w'' / 2
    """
    w = np.asarray(svi_total_variance(k, params), dtype=np.float64)
    w = np.maximum(w, 1e-12)  # Avoid division by zero
    w_p = np.asarray(svi_first_derivative(k, params), dtype=np.float64)
    w_pp = np.asarray(svi_second_derivative(k, params), dtype=np.float64)
    k_arr = np.asarray(k, dtype=np.float64)

    term1 = (1.0 - (k_arr * w_p) / (2.0 * w)) ** 2
    term2 = (w_p * w_p / 4.0) * (1.0 / w + 0.25)
    term3 = 0.5 * w_pp
    g = term1 - term2 + term3
    if np.isscalar(k):
        return float(g)
    return g


def check_butterfly_arbitrage(
    params: SVIParameters,
    k_grid: np.ndarray | None = None,
) -> tuple[bool, float]:
    """Check if the SVI slice satisfies Gatheral's no-butterfly arbitrage condition.

    Returns:
        (is_arbitrage_free, min_g_value)
    """
    if k_grid is None:
        k_grid = np.linspace(-2.0, 2.0, 401)
    g = durrleman_density_factor(k_grid, params)
    min_g = float(np.min(g))
    w = svi_total_variance(k_grid, params)
    min_w = float(np.min(w))
    is_free = (min_g >= -1e-7) and (min_w >= 0.0)
    return is_free, min_g


def check_calendar_arbitrage(
    params_early: SVIParameters,
    tau_early: float,
    params_late: SVIParameters,
    tau_late: float,
    k_grid: np.ndarray | None = None,
) -> tuple[bool, float]:
    """Check that total implied variance is non-decreasing across expiries (tau1 < tau2).

    Returns:
        (is_arbitrage_free, min_variance_spread)
    """
    if tau_early >= tau_late:
        raise ValueError(f"tau_early ({tau_early}) must be < tau_late ({tau_late})")
    if k_grid is None:
        k_grid = np.linspace(-2.0, 2.0, 401)
    w_early = np.asarray(svi_total_variance(k_grid, params_early))
    w_late = np.asarray(svi_total_variance(k_grid, params_late))
    spread = w_late - w_early
    min_spread = float(np.min(spread))
    return (min_spread >= -1e-7), min_spread


def calibrate_svi_quasi_explicit(
    log_moneyness: np.ndarray,
    total_variance: np.ndarray,
    weights: np.ndarray | None = None,
    enforce_butterfly: bool = True,
) -> SVIParameters:
    """Calibrate raw SVI to a single expiry slice using Gatheral's Quasi-Explicit method.

    Given (m, sigma), the problem w(k) = a + b*rho*(k-m) + b*sqrt((k-m)^2 + sigma^2)
    is linear in (a, c, b) where c = b*rho:
        w(k) = a + c * y(k) + b * z(k)
    with bounds and linear constraints:
        b >= 0
        -b <= c <= b
        b * (1 + |c/b|) <= 2 (Lee's condition)
        a + b * sigma * sqrt(1 - (c/b)^2) >= 0 (positivity)

    Args:
        log_moneyness: 1D array of ln(K / F)
        total_variance: 1D array of sigma_BS^2 * tau
        weights: Optional observation weights (e.g. 1/vega or inverse bid-ask spread)
        enforce_butterfly: Whether to penalize butterfly arbitrage during calibration

    Returns:
        Calibrated SVIParameters
    """
    k = np.asarray(log_moneyness, dtype=np.float64)
    w_target = np.asarray(total_variance, dtype=np.float64)
    if len(k) != len(w_target):
        raise ValueError("k and total_variance must have the same length")
    if len(k) < 5:
        raise ValueError("At least 5 points required for SVI slice calibration")

    if weights is None:
        w_weights = np.ones_like(w_target)
    else:
        w_weights = np.asarray(weights, dtype=np.float64)
        w_weights = np.maximum(w_weights, 1e-4)

    sqrt_weights = np.sqrt(w_weights)
    w_weighted = w_target * sqrt_weights

    k_grid_eval = np.linspace(float(np.min(k)) - 0.5, float(np.max(k)) + 0.5, 50)

    def inner_fit(m_val: float, sigma_val: float) -> tuple[float, tuple[float, float, float]]:
        """Solve for (a, c, b) given (m, sigma)."""
        y = k - m_val
        z = np.sqrt(y * y + sigma_val * sigma_val)
        A = np.column_stack([np.ones_like(y), y, z])
        A_weighted = A * sqrt_weights[:, np.newaxis]

        # Bound on b: [0, 2.0]
        # Linear bounds for lsq_linear: a in [-1.0, 5.0], c in [-2.0, 2.0], b in [0.0, 2.0]
        res = lsq_linear(A_weighted, w_weighted, bounds=([-1.0, -2.0, 0.0], [5.0, 2.0, 2.0]), method="trf")
        a_opt, c_opt, b_opt = float(res.x[0]), float(res.x[1]), float(res.x[2])

        # Enforce |c| <= b (i.e. |rho| <= 1.0)
        if b_opt < 1e-6:
            c_opt = 0.0
            rho_opt = 0.0
        else:
            rho_opt = np.clip(c_opt / b_opt, -0.999, 0.999)
            c_opt = rho_opt * b_opt

        # Enforce minimum variance positivity: a + b * sigma * sqrt(1 - rho^2) >= 0
        min_w_val = a_opt + b_opt * sigma_val * np.sqrt(max(0.0, 1.0 - rho_opt * rho_opt))
        if min_w_val < 0:
            a_opt += (-min_w_val + 1e-5)

        w_pred = a_opt + c_opt * y + b_opt * z
        residual = np.sum(w_weights * ((w_pred - w_target) ** 2))

        # Butterfly arbitrage penalty
        if enforce_butterfly:
            p = SVIParameters(a=a_opt, b=b_opt, rho=rho_opt, m=m_val, sigma=sigma_val)
            _, min_g = check_butterfly_arbitrage(p, k_grid_eval)
            if min_g < 0:
                residual += 1e3 * (abs(min_g) ** 2)

        return residual, (a_opt, c_opt, b_opt)

    def outer_objective(theta: np.ndarray) -> float:
        m_val, sigma_val = float(theta[0]), float(theta[1])
        cost, _ = inner_fit(m_val, sigma_val)
        return cost

    # Multi-start initializations for (m, sigma)
    atm_idx = np.argmin(np.abs(k))
    atm_w = float(w_target[atm_idx])
    init_guesses = [
        (0.0, 0.1),
        (float(k[atm_idx]), max(0.05, float(np.std(k)))),
        (-0.05, 0.15),
        (0.05, 0.2),
    ]

    best_cost = float("inf")
    best_params: SVIParameters | None = None

    for m_init, sig_init in init_guesses:
        res = minimize(
            outer_objective,
            x0=np.array([m_init, sig_init]),
            bounds=[(-1.0, 1.0), (0.005, 1.5)],
            method="L-BFGS-B",
        )
        m_opt, sig_opt = float(res.x[0]), float(res.x[1])
        cost, (a_opt, c_opt, b_opt) = inner_fit(m_opt, sig_opt)
        if cost < best_cost:
            best_cost = cost
            rho_val = float(np.clip(c_opt / max(b_opt, 1e-9), -0.999, 0.999))
            best_params = SVIParameters(
                a=a_opt,
                b=b_opt,
                rho=rho_val,
                m=m_opt,
                sigma=sig_opt,
            )

    assert best_params is not None
    return best_params


class SVISurface:
    """Multi-slice SVI volatility surface container.

    Maintains calibrated SVI parameters across multiple expiries and enforces
    both slice-wise butterfly arbitrage-free and cross-slice calendar arbitrage-free bounds.
    """

    def __init__(self) -> None:
        self.slices: dict[float, SVIParameters] = {}  # tau -> SVIParameters
        self.forwards: dict[float, float] = {}       # tau -> forward price

    def add_slice(self, tau: float, params: SVIParameters, forward: float) -> None:
        """Register a calibrated expiry slice."""
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        self.slices[tau] = params
        self.forwards[tau] = forward

    @property
    def expiries(self) -> list[float]:
        return sorted(self.slices.keys())

    def validate_calendar_arbitrage(self, k_grid: np.ndarray | None = None) -> tuple[bool, float]:
        """Validate that total variance is non-decreasing across all sequential expiries."""
        taus = self.expiries
        if len(taus) < 2:
            return True, 0.0

        min_overall_spread = float("inf")
        for i in range(len(taus) - 1):
            t1, t2 = taus[i], taus[i + 1]
            is_free, spread = check_calendar_arbitrage(self.slices[t1], t1, self.slices[t2], t2, k_grid)
            if not is_free:
                return False, spread
            min_overall_spread = min(min_overall_spread, spread)

        return True, min_overall_spread

    def implied_vol(self, strike: float, tau: float) -> float:
        """Interpolate implied volatility for any strike K and maturity tau.

        Interpolates linearly in total implied variance w(k, tau) across nearest slices.
        """
        taus = self.expiries
        if not taus:
            raise RuntimeError("No slices registered in SVISurface")

        if tau in self.slices:
            fwd = self.forwards[tau]
            k = np.log(strike / fwd)
            return float(svi_implied_vol(k, tau, self.slices[tau]))

        # Interpolate total variance between adjacent slices
        if tau <= taus[0]:
            t_ref = taus[0]
            fwd = self.forwards[t_ref]
            k = np.log(strike / fwd)
            w = float(svi_total_variance(k, self.slices[t_ref]))
            return float(np.sqrt(max(w, 1e-12) / t_ref))
        elif tau >= taus[-1]:
            t_ref = taus[-1]
            fwd = self.forwards[t_ref]
            k = np.log(strike / fwd)
            w = float(svi_total_variance(k, self.slices[t_ref]))
            return float(np.sqrt(max(w, 1e-12) / t_ref))

        # Between two slices t1 < tau < t2
        idx = np.searchsorted(taus, tau)
        t1, t2 = taus[idx - 1], taus[idx]
        k1 = np.log(strike / self.forwards[t1])
        k2 = np.log(strike / self.forwards[t2])
        w1 = float(svi_total_variance(k1, self.slices[t1]))
        w2 = float(svi_total_variance(k2, self.slices[t2]))

        weight = (tau - t1) / (t2 - t1)
        w_interp = (1.0 - weight) * w1 + weight * w2
        return float(np.sqrt(max(w_interp, 1e-12) / tau))

    def atm_implied_vol(self, tau: float) -> float:
        """Get ATM implied volatility (k=0) for expiry tau."""
        taus = self.expiries
        if not taus:
            raise RuntimeError("No slices registered in SVISurface")
        if tau in self.slices:
            w0 = float(svi_total_variance(0.0, self.slices[tau]))
            return float(np.sqrt(max(w0, 1e-12) / tau))

        idx = np.clip(np.searchsorted(taus, tau), 1, len(taus) - 1)
        t1, t2 = taus[idx - 1], taus[idx]
        w1 = float(svi_total_variance(0.0, self.slices[t1]))
        w2 = float(svi_total_variance(0.0, self.slices[t2]))
        weight = (tau - t1) / (t2 - t1) if t2 > t1 else 0.0
        w_interp = (1.0 - weight) * w1 + weight * w2
        return float(np.sqrt(max(w_interp, 1e-12) / max(tau, 1e-6)))
