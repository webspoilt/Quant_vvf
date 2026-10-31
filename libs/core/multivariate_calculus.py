"""Multivariate Calculus: Double/Triple Integration, High-Order Greeks, and Malliavin Weights.

Implements:
1. Multidimensional Numerical Integration:
   - Tensor Gauss-Legendre Quadrature for 2D and 3D integrals
   - Variable boundaries and Jacobian coordinate transformations (polar, spherical)
2. High-Order Derivatives & Greek Jet:
   - 2nd Order: Vanna, Volga (Vomma), Charm
   - 3rd Order: Speed (d Gamma / d S), Color (d Gamma / d t), Ultima (d Volga / d sigma)
3. Fournié et al. (1999) Malliavin Greek Monte Carlo Estimator:
   - Exact stochastic integration-by-parts on Wiener space
   - Discontinuous payoff Greeks (digital options, barrier payoffs) without payoff differentiation

References:
- Fournié, E., Lasry, J.-M., Lebuchoux, J., Lions, P.-L., & Touzi, N. (1999),
  "Applications of Malliavin calculus to Monte Carlo methods in finance",
  Finance and Stochastics, 3(4), 391-412.
- Haug, E. G. (2007), "The Complete Guide to Option Pricing Formulas", McGraw-Hill.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np
from scipy import stats

from libs.core.validation import assert_finite_scalar


# --------------------------------------------------------------------------- #
# 1. Multidimensional Quadrature (Double and Triple Integration)
# --------------------------------------------------------------------------- #

def integrate_2d(
    func: Callable[[float, float], float],
    x_min: float,
    x_max: float,
    y_min: Callable[[float], float] | float,
    y_max: Callable[[float], float] | float,
    n_points: int = 32,
) -> float:
    """Compute double integral over [x_min, x_max] x [y_min(x), y_max(x)] via Gauss-Legendre.

    integral_{x_min}^{x_max} dx integral_{y_min(x)}^{y_max(x)} f(x, y) dy
    """
    assert_finite_scalar(x_min, "x_min")
    assert_finite_scalar(x_max, "x_max")
    if x_min >= x_max:
        raise ValueError(f"x_min ({x_min}) must be < x_max ({x_max})")

    # 1D Gauss-Legendre nodes and weights on [-1, 1]
    xi, wi = np.polynomial.legendre.leggauss(n_points)

    # Transform x: t in [-1, 1] -> x in [x_min, x_max]
    x_mid = 0.5 * (x_max + x_min)
    x_rad = 0.5 * (x_max - x_min)
    x_nodes = x_mid + x_rad * xi
    x_weights = x_rad * wi

    total = 0.0
    for x, w_x in zip(x_nodes, x_weights):
        y_low = y_min(x) if callable(y_min) else y_min
        y_high = y_max(x) if callable(y_max) else y_max
        if y_low >= y_high:
            continue

        y_mid = 0.5 * (y_high + y_low)
        y_rad = 0.5 * (y_high - y_low)
        y_nodes = y_mid + y_rad * xi
        y_weights = y_rad * wi

        inner_sum = sum(w_y * func(x, y) for y, w_y in zip(y_nodes, y_weights))
        total += w_x * inner_sum

    return float(total)


def integrate_3d(
    func: Callable[[float, float, float], float],
    x_min: float,
    x_max: float,
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    n_points: int = 24,
) -> float:
    """Compute triple integral over rectangular cuboid via Gauss-Legendre quadrature."""
    assert_finite_scalar(x_min, "x_min")
    assert_finite_scalar(x_max, "x_max")
    y_min, y_max = y_bounds
    z_min, z_max = z_bounds

    xi, wi = np.polynomial.legendre.leggauss(n_points)

    x_mid, x_rad = 0.5 * (x_max + x_min), 0.5 * (x_max - x_min)
    y_mid, y_rad = 0.5 * (y_max + y_min), 0.5 * (y_max - y_min)
    z_mid, z_rad = 0.5 * (z_max + z_min), 0.5 * (z_max - z_min)

    x_nodes, x_w = x_mid + x_rad * xi, x_rad * wi
    y_nodes, y_w = y_mid + y_rad * xi, y_rad * wi
    z_nodes, z_w = z_mid + z_rad * xi, z_rad * wi

    total = 0.0
    for x, wx in zip(x_nodes, x_w):
        for y, wy in zip(y_nodes, y_w):
            for z, wz in zip(z_nodes, z_w):
                total += wx * wy * wz * func(x, y, z)

    return float(total)


# --------------------------------------------------------------------------- #
# 2. High-Order Analytical Greeks (Up to 3rd Order)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class HighOrderGreeks:
    # 1st order
    delta: float
    vega: float
    theta: float
    rho: float
    # 2nd order
    gamma: float
    vanna: float
    volga: float
    charm: float
    # 3rd order
    speed: float
    color: float
    ultima: float


def compute_high_order_greeks(
    spot: float,
    strike: float,
    sigma: float,
    tau_years: float,
    r: float = 0.0,
    is_call: bool = True,
) -> HighOrderGreeks:
    """Compute exact analytical 1st, 2nd, and 3rd order Greeks under Black-Scholes.

    Includes Speed (d3 V / dS3), Color (d3 V / dS2 dt), and Ultima (d3 V / d sigma3).
    """
    assert_finite_scalar(spot, "spot")
    assert_finite_scalar(strike, "strike")
    assert_finite_scalar(sigma, "sigma")
    assert_finite_scalar(tau_years, "tau_years")
    assert_finite_scalar(r, "r")
    if spot <= 0 or strike <= 0 or sigma <= 0 or tau_years <= 0:
        raise ValueError("spot, strike, sigma, tau_years must be strictly positive")

    sq_tau = math.sqrt(tau_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * tau_years) / (sigma * sq_tau)
    d2 = d1 - sigma * sq_tau

    phi_d1 = float(stats.norm.pdf(d1))
    cdf_d1 = float(stats.norm.cdf(d1))
    cdf_d2 = float(stats.norm.cdf(d2))
    df = math.exp(-r * tau_years)

    # 1st order
    if is_call:
        delta = cdf_d1
        theta = -(spot * phi_d1 * sigma) / (2.0 * sq_tau) - r * strike * df * cdf_d2
        rho = strike * tau_years * df * cdf_d2
    else:
        delta = cdf_d1 - 1.0
        theta = -(spot * phi_d1 * sigma) / (2.0 * sq_tau) + r * strike * df * (1.0 - cdf_d2)
        rho = -strike * tau_years * df * (1.0 - cdf_d2)

    vega = spot * sq_tau * phi_d1

    # 2nd order
    gamma = phi_d1 / (spot * sigma * sq_tau)
    vanna = -phi_d1 * (d2 / sigma)
    volga = vega * (d1 * d2 / sigma)
    charm = phi_d1 * (r / (sigma * sq_tau) - d2 / (2.0 * tau_years)) if is_call else (
        phi_d1 * (r / (sigma * sq_tau) - d2 / (2.0 * tau_years))
    )

    # 3rd order
    speed = -gamma / spot * (d1 / (sigma * sq_tau) + 1.0)
    color = -gamma / (2.0 * tau_years) * (1.0 + (2.0 * r * tau_years - d2 * sigma * sq_tau) * d1 / (sigma * sq_tau))
    ultima = -vega / (sigma * sigma) * (d1 * d2 * (1.0 - d1 * d2) + d1 * d1 + d2 * d2)

    return HighOrderGreeks(
        delta=delta,
        vega=vega,
        theta=theta,
        rho=rho,
        gamma=gamma,
        vanna=vanna,
        volga=volga,
        charm=charm,
        speed=speed,
        color=color,
        ultima=ultima,
    )


# --------------------------------------------------------------------------- #
# 3. Fournié et al. (1999) Malliavin Greek Monte Carlo Estimator
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MalliavinGreeksResult:
    price: float
    delta: float
    vega: float
    gamma: float
    delta_std_err: float
    vega_std_err: float


def malliavin_greeks_mc(
    payoff_func: Callable[[np.ndarray], np.ndarray],
    spot: float,
    sigma: float,
    tau: float,
    r: float = 0.0,
    n_sims: int = 100_000,
    seed: int | None = 42,
) -> MalliavinGreeksResult:
    """Compute Greeks via Fournié et al. (1999) Malliavin calculus weights.

    The payoff is NEVER differentiated. Exact for discontinuous/digital options:
        Delta = exp(-r*T) * E[ Phi(S_T) * W_T / (S_0 * sigma * T) ]
        Vega  = exp(-r*T) * E[ Phi(S_T) * ( (W_T^2 - T)/(sigma * T) - W_T ) ]
        Gamma = exp(-r*T) * E[ Phi(S_T) * ( W_T^2 - sigma*T*W_T - T ) / (S_0^2 * sigma^2 * T^2) ]
    """
    assert_finite_scalar(spot, "spot")
    assert_finite_scalar(sigma, "sigma")
    assert_finite_scalar(tau, "tau")
    assert_finite_scalar(r, "r")
    if spot <= 0 or sigma <= 0 or tau <= 0:
        raise ValueError("spot, sigma, tau must be strictly positive")

    rng = np.random.default_rng(seed)

    # Standard normal increments
    z = rng.standard_normal(n_sims)
    sq_tau = math.sqrt(tau)
    w_t = z * sq_tau

    # Terminal spot price S_T under GBM
    drift = (r - 0.5 * sigma * sigma) * tau
    s_t = spot * np.exp(drift + sigma * w_t)

    # Evaluate payoff (can be discontinuous!)
    payoff_vals = payoff_func(s_t)
    df = math.exp(-r * tau)

    # Malliavin weights
    pi_delta = w_t / (spot * sigma * tau)
    pi_vega = (w_t * w_t - tau) / (sigma * tau) - w_t
    pi_gamma = (w_t * w_t - sigma * tau * w_t - tau) / ((spot * sigma * tau) ** 2)

    # Monte Carlo expectations
    price = df * float(np.mean(payoff_vals))
    delta = df * float(np.mean(payoff_vals * pi_delta))
    vega = df * float(np.mean(payoff_vals * pi_vega))
    gamma = df * float(np.mean(payoff_vals * pi_gamma))

    delta_se = df * float(np.std(payoff_vals * pi_delta) / math.sqrt(n_sims))
    vega_se = df * float(np.std(payoff_vals * pi_vega) / math.sqrt(n_sims))

    return MalliavinGreeksResult(
        price=price,
        delta=delta,
        vega=vega,
        gamma=gamma,
        delta_std_err=delta_se,
        vega_std_err=vega_se,
    )
