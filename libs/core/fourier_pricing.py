"""Fourier Transform and Fourier-Cosine (COS) Methods for Option Pricing.

Implements:
1. Fang & Oosterlee (2008) COS Method:
   Spectral Fourier-cosine expansion of characteristic functions achieving exponential
   convergence O(N^{-M}) and linear computational complexity O(N).
2. Carr-Madan (1999) Fast Fourier Transform (FFT) Pricing Method:
   Pricing over a full continuum of strikes in O(N log N) using dampened Fourier inversion.
3. Analytical Characteristic Functions:
   - Black-Scholes-Merton
   - Heston (1993) Stochastic Volatility Model

References:
- Fang, F., & Oosterlee, C. W. (2008), "A novel pricing method for European options
  based on Fourier-cosine series expansions", SIAM J. Sci. Comput., 31(2), 826-848.
- Carr, P., & Madan, D. (1999), "Option valuation using the fast Fourier transform",
  Journal of Computational Finance, 2(4), 61-73.
- Heston, S. L. (1993), "A closed-form solution for options with stochastic volatility",
  Review of Financial Studies, 6(2), 327-343.
"""
from __future__ import annotations

import cmath
from dataclasses import dataclass
from typing import Callable, Literal, cast
import numpy as np

from libs.core.validation import assert_finite_scalar


# --------------------------------------------------------------------------- #
# Characteristic Functions
# --------------------------------------------------------------------------- #

def char_func_black_scholes(
    u: complex | np.ndarray,
    s0: float,
    strike: float,
    r: float,
    sigma: float,
    tau: float,
) -> complex | np.ndarray:
    """Characteristic function of ln(S_T / K) under Black-Scholes-Merton.

    phi(u) = E[exp(i * u * x)], where x = ln(S_T / K)
    """
    assert_finite_scalar(s0, "s0")
    assert_finite_scalar(strike, "strike")
    assert_finite_scalar(r, "r")
    assert_finite_scalar(sigma, "sigma")
    assert_finite_scalar(tau, "tau")
    if s0 <= 0 or strike <= 0 or sigma <= 0 or tau <= 0:
        raise ValueError("s0, strike, sigma, tau must be strictly positive")

    mu = np.log(s0 / strike) + (r - 0.5 * sigma * sigma) * tau
    return cast(complex | np.ndarray, np.exp(1j * u * mu - 0.5 * (u * sigma) ** 2 * tau))


def char_func_heston(
    u: complex | np.ndarray,
    s0: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    tau: float,
) -> complex | np.ndarray:
    """Characteristic function of ln(S_T) under Heston (1993) stochastic volatility.

    Uses Albrecher et al. (2007) / Schmelzle (2010) numerically stable formulation
    avoiding branch cut discontinuities of the complex square root.
    """
    assert_finite_scalar(s0, "s0")
    assert_finite_scalar(v0, "v0")
    assert_finite_scalar(kappa, "kappa")
    assert_finite_scalar(theta, "theta")
    assert_finite_scalar(xi, "xi")
    assert_finite_scalar(rho, "rho")
    assert_finite_scalar(tau, "tau")

    # Parameters
    i_u = 1j * u
    # d = sqrt((kappa - i * rho * xi * u)^2 + (u^2 + i*u) * xi^2)
    gamma = kappa - rho * xi * i_u
    d = np.sqrt(gamma * gamma + (u * u + i_u) * (xi * xi))

    # Stable formulation: g = (gamma - d) / (gamma + d)
    g = (gamma - d) / (gamma + d)

    exp_neg_d_tau = np.exp(-d * tau)
    c_term = (r * i_u * tau) + (kappa * theta / (xi * xi)) * (
        (gamma - d) * tau - 2.0 * np.log((1.0 - g * exp_neg_d_tau) / (1.0 - g))
    )
    d_term = ((gamma - d) / (xi * xi)) * ((1.0 - exp_neg_d_tau) / (1.0 - g * exp_neg_d_tau))

    log_s0 = np.log(s0)
    return cast(complex | np.ndarray, np.exp(c_term + d_term * v0 + i_u * log_s0))


# --------------------------------------------------------------------------- #
# Fang & Oosterlee (2008) COS Method
# --------------------------------------------------------------------------- #

def cos_method_pricer(
    cf: Callable[[np.ndarray], np.ndarray],
    s0: float,
    strike: float,
    r: float,
    tau: float,
    option_type: Literal["call", "put"] = "call",
    n_terms: int = 128,
    l_bound: float = 10.0,
) -> float:
    """Price European options using the Fang & Oosterlee (2008) COS method.

    Parameters
    ----------
    cf : Callable
        Characteristic function of log-return x = ln(S_T / K): cf(u) = E[exp(i u x)]
    s0 : float
        Current spot price (> 0)
    strike : float
        Strike price (> 0)
    r : float
        Continuously compounded risk-free rate
    tau : float
        Time to maturity in years (> 0)
    option_type : "call" or "put"
        Contract payoff type
    n_terms : int
        Number of Fourier-cosine expansion terms (default 128 achieves machine precision)
    l_bound : float
        Integration truncation parameter defining [a, b] = [-L * sqrt(tau), L * sqrt(tau)]

    Returns
    -------
    float
        Option premium in numeraire currency.
    """
    assert_finite_scalar(s0, "s0")
    assert_finite_scalar(strike, "strike")
    assert_finite_scalar(r, "r")
    assert_finite_scalar(tau, "tau")
    if s0 <= 0 or strike <= 0 or tau <= 0:
        raise ValueError("s0, strike, tau must be strictly positive")
    if option_type not in ("call", "put"):
        raise ValueError(f"Invalid option_type: {option_type!r}, expected 'call' or 'put'")

    # Truncation interval [a, b] for x = ln(S_T / K)
    x0 = np.log(s0 / strike)
    half_width = l_bound * np.sqrt(tau)
    a = min(0.0, x0) - half_width
    b = max(0.0, x0) + half_width

    k = np.arange(0, n_terms)
    omega = k * np.pi / (b - a)

    # Characteristic function evaluated at omega
    cf_vals = cf(omega)

    # Fourier-cosine series coefficients A_k
    # A_k = 2 / (b - a) * Re( cf(k pi / (b - a)) * exp(-i * k * pi * a / (b - a)) )
    mat = np.exp(-1j * omega * a) * cf_vals
    a_k = (2.0 / (b - a)) * np.real(mat)

    # Analytical payoff coefficients U_k = chi_k - psi_k
    if option_type == "call":
        c_int = 0.0
        d_int = b
    else:
        c_int = a
        d_int = 0.0

    # Chi coefficients: chi_k(c, d) = integral_c^d exp(y) cos(k pi (y - a)/(b - a)) dy
    denom = 1.0 + omega * omega
    cos_d = np.cos(omega * (d_int - a))
    sin_d = np.sin(omega * (d_int - a))
    cos_c = np.cos(omega * (c_int - a))
    sin_c = np.sin(omega * (c_int - a))

    exp_d = np.exp(d_int)
    exp_c = np.exp(c_int)

    chi_k = (1.0 / denom) * (
        (cos_d + omega * sin_d) * exp_d - (cos_c + omega * sin_c) * exp_c
    )

    # Psi coefficients: psi_k(c, d) = integral_c^d cos(k pi (y - a)/(b - a)) dy
    psi_k = np.zeros(n_terms, dtype=np.float64)
    psi_k[0] = d_int - c_int
    if n_terms > 1:
        psi_k[1:] = (1.0 / omega[1:]) * (sin_d[1:] - sin_c[1:])

    if option_type == "call":
        u_k = chi_k - psi_k
    else:
        u_k = -chi_k + psi_k

    # First term k=0 weighted by 1/2
    a_k[0] *= 0.5

    # Sum of series
    val = np.exp(-r * tau) * strike * np.sum(a_k * u_k)
    return float(max(0.0, val))


# --------------------------------------------------------------------------- #
# Carr-Madan (1999) Fast Fourier Transform (FFT) Pricing
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CarrMadanFFTResult:
    strikes: np.ndarray
    call_prices: np.ndarray


def carr_madan_fft_pricer(
    cf_ln_s: Callable[[complex | np.ndarray], complex | np.ndarray],
    s0: float,
    r: float,
    tau: float,
    alpha: float = 1.25,
    n_points: int = 4096,
    eta: float = 0.05,
) -> CarrMadanFFTResult:
    """Price European call options across a grid of strikes via Carr-Madan (1999) FFT.

    Parameters
    ----------
    cf_ln_s : Callable
        Characteristic function of ln(S_T): cf(u) = E[exp(i u ln(S_T))]
    s0 : float
        Current spot price
    r : float
        Risk-free rate
    tau : float
        Maturity in years
    alpha : float
        Damping parameter (default 1.25 guarantees square-integrability)
    n_points : int
        Grid size for FFT (power of 2, typically 2048 or 4096)
    eta : float
        Integration grid spacing

    Returns
    -------
    CarrMadanFFTResult
        Strikes and corresponding European call option prices.
    """
    assert_finite_scalar(s0, "s0")
    assert_finite_scalar(r, "r")
    assert_finite_scalar(tau, "tau")
    assert_finite_scalar(alpha, "alpha")
    if s0 <= 0 or tau <= 0 or alpha <= 0:
        raise ValueError("s0, tau, alpha must be strictly positive")

    # Grid parameters
    # Nyquist relation: lambda * eta = 2 * pi / N
    lambda_param = (2.0 * np.pi) / (n_points * eta)
    b = 0.5 * n_points * lambda_param

    # Integration nodes v_j = j * eta
    j = np.arange(n_points)
    v_j = j * eta

    # Evaluated characteristic function at v_j - (alpha + 1) * i
    u_eval = v_j - (alpha + 1.0) * 1j
    phi_eval = cf_ln_s(u_eval)

    # Denominator: alpha^2 + alpha - v^2 + i * (2*alpha + 1) * v
    denom = (alpha * alpha + alpha - v_j * v_j) + 1j * ((2.0 * alpha + 1.0) * v_j)
    psi = (np.exp(-r * tau) * phi_eval) / denom

    # Simpson's rule quadrature weights
    simpson_weights = (3.0 + (-1.0) ** (j + 1)) / 3.0
    simpson_weights[0] = 1.0 / 3.0

    # FFT input array
    x_arr = np.exp(1j * b * v_j) * psi * eta * simpson_weights
    fft_res = np.fft.fft(x_arr)

    # Log-strikes: k_u = -b + u * lambda
    k_grid = -b + j * lambda_param
    strikes = np.exp(k_grid)

    call_prices = np.real((np.exp(-alpha * k_grid) / np.pi) * fft_res)
    call_prices = np.maximum(0.0, call_prices)

    return CarrMadanFFTResult(strikes=strikes, call_prices=call_prices)
