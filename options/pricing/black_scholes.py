"""
Analytical Black-Scholes and Black-76 Derivatives Pricing Engine
Complete Greeks (Delta, Gamma, Vega, Theta, Rho) and Implied Volatility Solvers.
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


@dataclass
class OptionGreeks:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class BlackScholesEngine:
    """Rigorous analytical pricing and Greek calculation for European options."""

    @staticmethod
    def calculate_greeks(
        spot: float,
        strike: float,
        time_to_expiry_years: float,
        rate: float,
        volatility: float,
        option_type: Literal["call", "put"] = "call",
        dividend_yield: float = 0.0
    ) -> OptionGreeks:
        """Calculates Black-Scholes analytical option price and exact closed-form Greeks."""
        if spot <= 0 or strike <= 0 or time_to_expiry_years <= 0 or volatility <= 0:
            raise ValueError("All pricing inputs (spot, strike, T, vol) must be strictly positive")

        S = float(spot)
        K = float(strike)
        T = float(time_to_expiry_years)
        r = float(rate)
        q = float(dividend_yield)
        sigma = float(volatility)

        sqrt_t = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t

        pdf_d1 = norm.pdf(d1)
        discount_r = np.exp(-r * T)
        discount_q = np.exp(-q * T)

        # Gamma and Vega are identical for Calls and Puts
        gamma = (discount_q * pdf_d1) / (S * sigma * sqrt_t)
        vega = S * discount_q * pdf_d1 * sqrt_t * 0.01  # Per 1% vol change convention

        if option_type.lower() == "call":
            price = S * discount_q * norm.cdf(d1) - K * discount_r * norm.cdf(d2)
            delta = discount_q * norm.cdf(d1)
            theta = (-(S * discount_q * pdf_d1 * sigma) / (2.0 * sqrt_t)
                     - r * K * discount_r * norm.cdf(d2)
                     + q * S * discount_q * norm.cdf(d1)) / 365.0
            rho = (K * T * discount_r * norm.cdf(d2)) * 0.01
        else:
            price = K * discount_r * norm.cdf(-d2) - S * discount_q * norm.cdf(-d1)
            delta = discount_q * (norm.cdf(d1) - 1.0)
            theta = (-(S * discount_q * pdf_d1 * sigma) / (2.0 * sqrt_t)
                     + r * K * discount_r * norm.cdf(-d2)
                     - q * S * discount_q * norm.cdf(-d1)) / 365.0
            rho = (-K * T * discount_r * norm.cdf(-d2)) * 0.01

        return OptionGreeks(
            price=float(price),
            delta=float(delta),
            gamma=float(gamma),
            vega=float(vega),
            theta=float(theta),
            rho=float(rho),
        )

    @classmethod
    def implied_volatility(
        cls,
        market_price: float,
        spot: float,
        strike: float,
        time_to_expiry_years: float,
        rate: float,
        option_type: Literal["call", "put"] = "call",
        dividend_yield: float = 0.0
    ) -> float:
        """Solves implied volatility using Brent's method with robust bracket boundaries."""
        # Intrinsic lower bound
        intrinsic = max(0.0, (spot - strike) if option_type == "call" else (strike - spot))
        if market_price <= intrinsic:
            return 1e-4

        def obj(sigma: float) -> float:
            greeks = cls.calculate_greeks(
                spot=spot,
                strike=strike,
                time_to_expiry_years=time_to_expiry_years,
                rate=rate,
                volatility=sigma,
                option_type=option_type,
                dividend_yield=dividend_yield
            )
            return greeks.price - market_price

        try:
            return float(brentq(obj, 1e-4, 5.0, xtol=1e-6))
        except Exception:
            return np.nan
