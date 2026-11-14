"""
QUANT_VVF — Jump-Diffusion & Stochastic Volatility with Jumps.
References:
- Merton (1976): Option pricing when underlying stock returns are discontinuous. Journal of Financial Economics, 3(1-2), 125-144.
- Kou (2002): A jump-diffusion model for option pricing. Management Science, 48(8), 1086-1101.
- Bates (1996): Jumps and stochastic volatility: Exchange rate processes implicit in Deutsche mark options.
  The Review of Financial Studies, 9(1), 69-107.
"""

from __future__ import annotations
import math
from typing import Dict, Any, Tuple
import numpy as np
from scipy.stats import norm
from research_math.stochastic_options.heston_svi_inverse import black_scholes_price


def merton_jump_diffusion_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free: float,
    volatility: float,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
    is_call: bool = True,
    n_terms: int = 40,
) -> float:
    """
    Merton (1976) compound Poisson jump-diffusion pricing via infinite series of Black-Scholes prices:
    dS_t / S_t = (r - q - lambda * k) dt + sigma dW_t + (Y - 1) dN_t
    where ln(Y) ~ N(mu_J, delta_J^2), k = exp(mu_J + delta_J^2 / 2) - 1.
    """
    S0 = spot
    K = strike
    T = time_to_expiry
    r = risk_free
    sigma = volatility
    lam = jump_intensity
    mu_J = jump_mean
    delta_J = jump_std

    if T <= 1e-6:
        return max(0.0, S0 - K) if is_call else max(0.0, K - S0)

    # Expected relative jump size
    k = math.exp(mu_J + 0.5 * delta_J ** 2) - 1.0
    lam_prime = lam * (1.0 + k)

    total_price = 0.0
    # Poisson summation
    for n in range(n_terms):
        # Probability of n jumps: P(N_T = n) = exp(-lam_prime * T) * (lam_prime * T)^n / n!
        weight = (math.exp(-lam_prime * T) * ((lam_prime * T) ** n)) / math.factorial(n)

        # Adjusted parameters for n jumps
        r_n = r - lam * k + (n * (mu_J + 0.5 * delta_J ** 2)) / T
        sigma_n = math.sqrt(sigma ** 2 + (n * (delta_J ** 2)) / T)

        bs_p = black_scholes_price(S0, K, T, r_n, sigma_n, is_call)
        total_price += weight * bs_p

    return float(max(0.0, total_price))


def kou_jump_diffusion_density(
    x: float, p_up: float, eta_up: float, eta_down: float
) -> float:
    """
    Kou (2002) asymmetric double-exponential jump density:
    f(y) = p * eta_1 * exp(-eta_1 * y) for y >= 0
         = (1 - p) * eta_2 * exp(eta_2 * y) for y < 0
    where eta_up > 1 captures memoryless upward tails, eta_down > 0 downward tails.
    """
    if x >= 0:
        return p_up * eta_up * math.exp(-eta_up * x)
    else:
        return (1.0 - p_up) * eta_down * math.exp(eta_down * x)


def bates_characteristic_function(
    u: complex,
    spot: float,
    time_to_expiry: float,
    risk_free: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
) -> complex:
    """
    Bates (1996) characteristic function: Heston stochastic volatility + Merton jumps.
    cf_Bates(u) = cf_Heston(u) * exp(T * lam * (exp(i*u*mu_J - 0.5*delta_J^2*u^2) - 1 - i*u*k))
    """
    from research_math.stochastic_options.heston_svi_inverse import HestonModel

    heston = HestonModel(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
    cf_h = heston.characteristic_function(u, spot, time_to_expiry, risk_free)

    k = math.exp(jump_mean + 0.5 * jump_std ** 2) - 1.0
    i = 1j
    jump_term = math.exp(time_to_expiry * jump_intensity * (cmath.exp(i * u * jump_mean - 0.5 * (jump_std ** 2) * (u ** 2)) - 1.0 - i * u * k))

    return cf_h * jump_term
