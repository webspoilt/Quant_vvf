"""
QUANT_VVF — Stochastic Volatility, Heston Pricing, SVI Calibration & Inverse Derivatives.
References:
- Heston (1993): A Closed-Form Solution for Options with Stochastic Volatility
- Albrecher et al. (2007): The Little Heston Trap (stable branch-cut free formulation)
- Gatheral & Jacquier (2014): Arbitrage-Free SVI Volatility Surfaces
"""

from __future__ import annotations
import cmath
import math
from typing import Dict, Any, Tuple, Optional, Callable
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm


def black_scholes_price(
    spot: float, strike: float, time_to_expiry: float, risk_free: float, volatility: float, is_call: bool = True
) -> float:
    """Analytical Black-Scholes benchmark European option price."""
    if time_to_expiry <= 0.0:
        return max(0.0, spot - strike) if is_call else max(0.0, strike - spot)
    if volatility <= 0.0:
        df = math.exp(-risk_free * time_to_expiry)
        return max(0.0, spot - strike * df) if is_call else max(0.0, strike * df - spot)

    d1 = (math.log(spot / strike) + (risk_free + 0.5 * volatility ** 2) * time_to_expiry) / (
        volatility * math.sqrt(time_to_expiry)
    )
    d2 = d1 - volatility * math.sqrt(time_to_expiry)

    df = math.exp(-risk_free * time_to_expiry)
    if is_call:
        return spot * norm.cdf(d1) - strike * df * norm.cdf(d2)
    else:
        return strike * df * norm.cdf(-d2) - spot * norm.cdf(-d1)


def black_scholes_greeks(
    spot: float, strike: float, time_to_expiry: float, risk_free: float, volatility: float, is_call: bool = True
) -> Dict[str, float]:
    """Analytical Greeks: Delta, Gamma, Vega, Theta, Rho."""
    if time_to_expiry <= 1e-6 or volatility <= 1e-6:
        delta = 1.0 if (spot > strike and is_call) else (-1.0 if (spot < strike and not is_call) else 0.0)
        return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (risk_free + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    df = math.exp(-risk_free * time_to_expiry)
    pdf_d1 = norm.pdf(d1)

    delta = norm.cdf(d1) if is_call else (norm.cdf(d1) - 1.0)
    gamma = pdf_d1 / (spot * volatility * sqrt_t)
    vega = spot * sqrt_t * pdf_d1
    theta = -(spot * pdf_d1 * volatility) / (2.0 * sqrt_t)
    if is_call:
        theta -= risk_free * strike * df * norm.cdf(d2)
        rho = strike * time_to_expiry * df * norm.cdf(d2)
    else:
        theta += risk_free * strike * df * norm.cdf(-d2)
        rho = -strike * time_to_expiry * df * norm.cdf(-d2)

    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega), "theta": float(theta), "rho": float(rho)}


def implied_volatility(
    price: float, spot: float, strike: float, time_to_expiry: float, risk_free: float, is_call: bool = True
) -> float:
    """Robust implied volatility inversion via Brent / Newton-Raphson."""
    intrinsic = max(0.0, spot - strike * math.exp(-risk_free * time_to_expiry)) if is_call else max(
        0.0, strike * math.exp(-risk_free * time_to_expiry) - spot
    )
    if price <= intrinsic:
        return 1e-4

    # Target objective
    def obj(sigma: float) -> float:
        return black_scholes_price(spot, strike, time_to_expiry, risk_free, sigma, is_call) - price

    # Bracket search
    low, high = 1e-4, 5.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        diff = obj(mid)
        if abs(diff) < 1e-7:
            return mid
        if diff > 0:
            high = mid
        else:
            low = mid
    return mid


class HestonModel:
    """
    Heston (1993) Stochastic Volatility Model:
    dS_t / S_t = (r - q) dt + sqrt(v_t) dW_t^S
    dv_t = kappa * (theta - v_t) dt + xi * sqrt(v_t) dW_t^v
    corr(dW_t^S, dW_t^v) = rho
    Using Albrecher et al. (2007) stable formulation without branch cuts.
    """

    def __init__(self, v0: float, kappa: float, theta: float, xi: float, rho: float) -> None:
        self.v0 = max(1e-6, float(v0))
        self.kappa = max(1e-6, float(kappa))
        self.theta = max(1e-6, float(theta))
        self.xi = max(1e-6, float(xi))
        self.rho = max(-0.999, min(0.999, float(rho)))

    @property
    def feller_satisfied(self) -> bool:
        """2 * kappa * theta > xi^2 guarantees strict positivity of variance."""
        return (2.0 * self.kappa * self.theta) > (self.xi ** 2)

    def characteristic_function(self, u: complex, spot: float, time_to_expiry: float, risk_free: float) -> complex:
        """Albrecher et al. stable characteristic function."""
        S0 = spot
        T = time_to_expiry
        r = risk_free
        kappa = self.kappa
        theta = self.theta
        xi = self.xi
        rho = self.rho
        v0 = self.v0

        i = 1j
        d = cmath.sqrt((rho * xi * u * i - kappa) ** 2 + (xi ** 2) * (u * i + u ** 2))
        g = (kappa - rho * xi * u * i - d) / (kappa - rho * xi * u * i + d)

        exp_neg_dT = cmath.exp(-d * T)
        C = (r * u * i * T) + (kappa * theta / (xi ** 2)) * (
            (kappa - rho * xi * u * i - d) * T - 2.0 * cmath.log((1.0 - g * exp_neg_dT) / (1.0 - g))
        )
        D = ((kappa - rho * xi * u * i - d) / (xi ** 2)) * ((1.0 - exp_neg_dT) / (1.0 - g * exp_neg_dT))

        return cmath.exp(C + D * v0 + i * u * math.log(S0))

    def price(
        self, spot: float, strike: float, time_to_expiry: float, risk_free: float, is_call: bool = True
    ) -> float:
        """Fourier inversion integration for European option pricing."""
        if time_to_expiry <= 1e-6:
            return max(0.0, spot - strike) if is_call else max(0.0, strike - spot)

        K = strike
        r = risk_free
        T = time_to_expiry

        def p1_integrand(phi: float) -> float:
            u = phi - 1j
            cf = self.characteristic_function(u, spot, T, r)
            denominator = phi * 1j * spot * math.exp(r * T)
            val = (cmath.exp(-1j * phi * math.log(K)) * cf / denominator).real
            return val

        def p2_integrand(phi: float) -> float:
            cf = self.characteristic_function(phi, spot, T, r)
            denominator = phi * 1j
            val = (cmath.exp(-1j * phi * math.log(K)) * cf / denominator).real
            return val

        # Numerical integration with adaptive quad
        int1, _ = quad(p1_integrand, 1e-5, 200.0, limit=1000)
        int2, _ = quad(p2_integrand, 1e-5, 200.0, limit=1000)

        P1 = 0.5 + (1.0 / math.pi) * int1
        P2 = 0.5 + (1.0 / math.pi) * int2

        call_price = spot * P1 - K * math.exp(-r * T) * P2
        call_price = max(0.0, call_price)

        if is_call:
            return float(call_price)
        else:
            # Put-Call Parity: P = C - S + K*exp(-r*T)
            put_price = call_price - spot + K * math.exp(-r * T)
            return float(max(0.0, put_price))

    @classmethod
    def calibrate(
        cls,
        market_quotes: list[dict[str, float]],
        spot: float,
        risk_free: float,
    ) -> HestonModel:
        """
        Calibrate Heston parameters [v0, kappa, theta, xi, rho] to market option prices.
        market_quotes: list of {'strike': float, 'expiry': float, 'price': float, 'is_call': bool}
        """
        # [v0, kappa, theta, xi, rho]
        x0 = np.array([0.04, 2.0, 0.04, 0.3, -0.6], dtype=np.float64)
        bounds = [(1e-4, 1.5), (1e-2, 10.0), (1e-4, 1.5), (1e-3, 3.0), (-0.99, 0.99)]

        def loss_fn(p: np.ndarray) -> float:
            model = cls(v0=p[0], kappa=p[1], theta=p[2], xi=p[3], rho=p[4])
            total_sq_err = 0.0
            for q in market_quotes:
                try:
                    m_price = model.price(spot, q["strike"], q["expiry"], risk_free, q.get("is_call", True))
                    err = (m_price - q["price"]) ** 2
                    total_sq_err += err
                except Exception:
                    total_sq_err += 1e4
            return float(total_sq_err)

        res = minimize(loss_fn, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 150})
        p = res.x
        return cls(v0=p[0], kappa=p[1], theta=p[2], xi=p[3], rho=p[4])


class SviSurface:
    """
    Raw SVI Volatility Surface (Gatheral 2004, Gatheral & Jacquier 2014):
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    where k = ln(K / F) is log-moneyness.
    """

    def __init__(self, a: float, b: float, rho: float, m: float, sigma: float) -> None:
        self.a = float(a)
        self.b = max(1e-6, float(b))
        self.rho = max(-0.999, min(0.999, float(rho)))
        self.m = float(m)
        self.sigma = max(1e-6, float(sigma))

    def total_variance(self, k: float | np.ndarray) -> float | np.ndarray:
        """Compute total variance w(k)."""
        km = k - self.m
        sqrt_term = np.sqrt(km ** 2 + self.sigma ** 2)
        return self.a + self.b * (self.rho * km + sqrt_term)

    def implied_vol(self, k: float | np.ndarray, time_to_expiry: float) -> float | np.ndarray:
        """Implied volatility sigma(k) = sqrt(w(k) / T)."""
        w = self.total_variance(k)
        return np.sqrt(np.maximum(1e-6, w) / max(1e-6, time_to_expiry))

    def durrleman_butterfly_density(self, k: float) -> float:
        """
        Durrleman (2004) non-arbitrage condition: g(k) >= 0 where
        g(k) = (1 - (k * w') / (2w))^2 - (w' / 4) * (1/w + 1/4) + (w'' / 2)
        """
        km = k - self.m
        disc = math.sqrt(km ** 2 + self.sigma ** 2)
        w = self.a + self.b * (self.rho * km + disc)
        if w <= 1e-8:
            return -1.0

        w_prime = self.b * (self.rho + km / disc)
        w_prime2 = self.b * (self.sigma ** 2) / (disc ** 3)

        d1 = 1.0 - (k * w_prime) / (2.0 * w)
        d2 = (w_prime ** 2 / 4.0) * (1.0 / w + 0.25)
        d3 = 0.5 * w_prime2

        return d1 ** 2 - d2 + d3

    def has_butterfly_arbitrage(self, k_range: np.ndarray) -> bool:
        """Returns True if any point exhibits g(k) < 0."""
        for k in k_range:
            if self.durrleman_butterfly_density(float(k)) < 0.0:
                return True
        return False


def coin_margined_option_payoff(
    spot_terminal: float, strike: float, is_call: bool = True
) -> Tuple[float, float]:
    """
    Deribit coin-margined inverse option payoff:
    Call: max(0, S_T - K) / S_T in BTC, max(0, S_T - K) in USD.
    Put:  max(0, K - S_T) / S_T in BTC, max(0, K - S_T) in USD.
    Returns (payoff_btc, payoff_usd).
    """
    ST = max(1e-8, spot_terminal)
    K = strike
    usd_val = max(0.0, ST - K) if is_call else max(0.0, K - ST)
    btc_val = usd_val / ST
    return btc_val, usd_val


def inverse_delta_rescaling(spot: float, spot_ref: float, bs_usd_delta: float) -> float:
    """
    Inverse numeraire (S0/S1)^2 delta transformation for coin-margined contracts.
    Ensures exact dollar hedge matching under spot movement.
    """
    ratio = spot_ref / max(1e-6, spot)
    return bs_usd_delta * (ratio ** 2)
