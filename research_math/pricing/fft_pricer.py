"""
Fast Option Pricing via Carr-Madan Fast Fourier Transform (FFT) and COS Method.

Supports:
- Heston Stochastic Volatility.
- Merton Jump Diffusion.
- Kou Double Exponential Jump Diffusion.
- Bates Model.
- Black-Scholes benchmark.
"""

from dataclasses import dataclass
import cmath
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats


@dataclass
class FftPricingResult:
    strikes: np.ndarray
    call_prices: np.ndarray
    put_prices: np.ndarray
    model_name: str
    computation_time_ms: float = 0.0


class CharacteristicFunctions:
    """Analytical characteristic functions phi(u) = E[exp(i * u * ln(S_T / S_0))] under risk-neutral measure Q."""

    @staticmethod
    def black_scholes(u: complex, t: float, r: float, sigma: float) -> complex:
        return cmath.exp(1j * u * (r - 0.5 * sigma**2) * t - 0.5 * (sigma**2) * (u**2) * t)

    @staticmethod
    def heston(u: complex, t: float, r: float, v0: float, kappa: float, theta: float, xi: float, rho: float) -> complex:
        # Albrecher et al. (2007) formulation (Little Heston Trap resolution)
        d = cmath.sqrt((kappa - 1j * rho * xi * u)**2 + (u**2 + 1j * u) * (xi**2))
        g = (kappa - 1j * rho * xi * u - d) / (kappa - 1j * rho * xi * u + d)

        c = (r * 1j * u * t) + (kappa * theta / (xi**2)) * (
            (kappa - 1j * rho * xi * u - d) * t - 2.0 * cmath.log((1.0 - g * cmath.exp(-d * t)) / (1.0 - g))
        )
        d_term = ((kappa - 1j * rho * xi * u - d) / (xi**2)) * ((1.0 - cmath.exp(-d * t)) / (1.0 - g * cmath.exp(-d * t)))
        return cmath.exp(c + d_term * v0)

    @staticmethod
    def merton_jump(
        u: complex, t: float, r: float, sigma: float, lambda_j: float, mu_j: float, sigma_j: float
    ) -> complex:
        k = math.exp(mu_j + 0.5 * sigma_j**2) - 1.0
        drift = r - 0.5 * sigma**2 - lambda_j * k
        diff_cf = cmath.exp(1j * u * drift * t - 0.5 * (u**2) * (sigma**2) * t)
        jump_cf = cmath.exp(lambda_j * t * (cmath.exp(1j * u * mu_j - 0.5 * (u**2) * (sigma_j**2)) - 1.0))
        return diff_cf * jump_cf

    @staticmethod
    def kou_jump(
        u: complex, t: float, r: float, sigma: float, lambda_j: float, p_up: float, eta1: float, eta2: float
    ) -> complex:
        k = p_up * eta1 / (eta1 - 1.0) + (1.0 - p_up) * eta2 / (eta2 + 1.0) - 1.0
        drift = r - 0.5 * sigma**2 - lambda_j * k
        diff_cf = cmath.exp(1j * u * drift * t - 0.5 * (u**2) * (sigma**2) * t)

        # Kou jump characteristic function: E[exp(i u Y)]
        jump_dist = p_up * eta1 / (eta1 - 1j * u) + (1.0 - p_up) * eta2 / (eta2 + 1j * u)
        jump_cf = cmath.exp(lambda_j * t * (jump_dist - 1.0))
        return diff_cf * jump_cf

    @staticmethod
    def bates(
        u: complex, t: float, r: float, v0: float, kappa: float, theta: float, xi: float, rho: float,
        lambda_j: float, mu_j: float, sigma_j: float
    ) -> complex:
        heston_part = CharacteristicFunctions.heston(u, t, r, v0, kappa, theta, xi, rho)
        k = math.exp(mu_j + 0.5 * sigma_j**2) - 1.0
        # Jump drift compensation term
        jump_cf = cmath.exp(
            lambda_j * t * (cmath.exp(1j * u * mu_j - 0.5 * (u**2) * (sigma_j**2)) - 1.0)
            - 1j * u * lambda_j * k * t
        )
        return heston_part * jump_cf


class CarrMadanFftPricer:
    """
    Carr-Madan (1999) Fast Fourier Transform option pricer.
    Evaluates European calls simultaneously across an entire strike grid.
    """

    @staticmethod
    def price_options_fft(
        s0: float,
        t: float,
        r: float,
        char_func: Callable[[complex], complex],
        alpha: float = 1.5,
        n: int = 4096,
        eta: float = 0.25,
        model_name: str = "Heston/Lévy",
    ) -> FftPricingResult:
        """
        Runs Carr-Madan FFT algorithm:
        C(k) = (exp(-alpha * k) / pi) * int_0^inf exp(-i v k) psi(v) dv
        where psi(v) = exp(-r * T) * phi(v - (alpha + 1)*i) / (alpha^2 + alpha - v^2 + i*(2*alpha + 1)*v)
        """
        import time
        t0 = time.perf_counter()

        # Step size in strike log space
        lambd = 2.0 * math.pi / (n * eta)
        b = 0.5 * n * lambd

        # Grid points
        u = np.arange(n)
        v_j = eta * u
        k_u = -b + lambd * u

        # Simpson's rule weights: 1/3, 4/3, 2/3, 4/3...
        simpson_weights = 3.0 + (-1.0)**(u + 1)
        simpson_weights[0] = 1.0
        simpson_weights[-1] = 1.0
        simpson_weights = simpson_weights / 3.0

        # Vectorized evaluation of modified characteristic function
        df = math.exp(-r * t)
        psi_vals = np.zeros(n, dtype=np.complex128)

        for j in range(n):
            v = v_j[j]
            # Argument to char_func: v - (alpha + 1)*1j
            arg = v - (alpha + 1.0) * 1j
            denom = alpha**2 + alpha - v**2 + 1j * (2.0 * alpha + 1.0) * v
            cf_val = char_func(arg)
            psi_vals[j] = (df * cf_val / denom) * cmath.exp(1j * v * b) * eta * simpson_weights[j]

        # Discrete Fourier Transform
        fft_vals = np.fft.fft(psi_vals)

        # Normalized option price (S0 = 1) across log moneyness k_u = ln(K / S0)
        c_norm = np.real((np.exp(-alpha * k_u) / math.pi) * fft_vals)

        # Rescale by spot price S0
        call_prices = s0 * np.maximum(c_norm, 0.0)
        strikes = s0 * np.exp(k_u)

        # Put-Call parity for put prices: P = C - S0 + K * e^(-r*T)
        put_prices = call_prices - s0 + strikes * df
        put_prices = np.maximum(put_prices, 0.0)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return FftPricingResult(
            strikes=strikes,
            call_prices=call_prices,
            put_prices=put_prices,
            model_name=model_name,
            computation_time_ms=elapsed_ms,
        )
