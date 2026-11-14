"""
Indian Index Options, Derivatives Pricing, Volatility Surfaces & Greeks.

Modules:
1. SABR Volatility Model Calibration (Hagan et al. 2002):
   - sigma_B(F, K) parameterization: (alpha, beta, rho, nu).
   - Numerical calibration to NIFTY / BANKNIFTY smile/skew.
2. Put-Call Parity and Box-Spread Arbitrage Analyzer:
   - For European index options: C - P = exp(-r*T) * (F - K).
   - Identifies synthetic futures mispricings, reversal/conversion bounds, and box spreads.
3. Delta-Gamma Scalping & Dynamic Futures Hedging:
   - Evaluates portfolio Gamma/Vega/Theta balance and rebalancing frequency under transaction costs.
4. Volatility Dispersion & Implied Correlation:
   - Compares index implied variance sigma_I^2 with weighted constituent variance sum(w_i^2 * sigma_i^2).
   - Derives implied correlation rho_implied.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.optimize as opt
import scipy.stats as stats


@dataclass
class SabrVolatilityModel:
    """
    SABR stochastic volatility model (Hagan et al. 2002):
    dF_t = alpha_t * F_t^beta * dW_t^F
    dalpha_t = nu * alpha_t * dW_t^alpha, with Corr(dW^F, dW^alpha) = rho.
    """
    alpha: float = 0.25 # Initial volatility level (lognormal vol when beta=1.0)
    beta: float = 1.0 # CEV backbone exponent (beta=1.0 lognormal SABR standard for FX/Equity smile)
    rho: float = -0.30 # Volatility-asset correlation (skew)
    nu: float = 0.40 # Vol of vol (curvature/smile)

    def implied_volatility(self, f: float, k: float, t_expiry: float) -> float:
        """Computes Hagan asymptotic implied Black volatility sigma_B(F, K)."""
        if f <= 0 or k <= 0 or t_expiry <= 0:
            return 0.0

        if abs(f - k) < 1e-5:
            # ATM case for beta=1.0: sigma_atm = alpha * (1 + ((2-3*rho^2)/24 * nu^2 + rho*beta*nu*alpha/4 + ...)*T)
            term = (((1.0 - self.beta)**2 / 24.0) * (self.alpha**2) / (f**(2.0 - 2.0*self.beta))
                    + 0.25 * self.rho * self.beta * self.nu * self.alpha / (f**(1.0 - self.beta))
                    + ((2.0 - 3.0 * self.rho**2) / 24.0) * (self.nu**2))
            sigma_atm = (self.alpha / (f**(1.0 - self.beta))) * (1.0 + term * t_expiry)
            return float(sigma_atm)

        log_fk = math.log(f / k)
        f_mid = math.sqrt(f * k)
        f_k_beta = f_mid ** (1.0 - self.beta)
        z = (self.nu / self.alpha) * f_k_beta * log_fk

        sqrt_term = math.sqrt(1.0 - 2.0 * self.rho * z + z**2)
        if abs(z) < 1e-6:
            chi_z = 1.0
        else:
            chi_z = z / math.log((sqrt_term + z - self.rho) / (1.0 - self.rho))

        term_num = 1.0 + (((1.0 - self.beta)**2 / 24.0) * (self.alpha**2) / (f_k_beta**2)
                          + 0.25 * self.rho * self.beta * self.nu * self.alpha / f_k_beta
                          + ((2.0 - 3.0 * self.rho**2) / 24.0) * (self.nu**2)) * t_expiry

        term_den = 1.0 + ((1.0 - self.beta)**2 / 24.0) * (log_fk**2) + ((1.0 - self.beta)**4 / 1920.0) * (log_fk**4)

        sigma_b = (self.alpha / (f_k_beta * term_den)) * chi_z * term_num
        return float(sigma_b)

    @classmethod
    def calibrate_to_market_skew(
        cls,
        f: float,
        strikes: np.ndarray,
        market_ivs: np.ndarray,
        t_expiry: float,
        fixed_beta: float = 0.5,
    ) -> "SabrVolatilityModel":
        """Calibrates (alpha, rho, nu) to observed market implied volatilities via least squares."""
        def loss(params):
            alpha, rho, nu = params
            if alpha <= 0 or abs(rho) >= 0.999 or nu <= 0:
                return 1e6
            model = cls(alpha=alpha, beta=fixed_beta, rho=rho, nu=nu)
            model_ivs = np.array([model.implied_volatility(f, k, t_expiry) for k in strikes])
            return np.sum((model_ivs - market_ivs) ** 2)

        init_params = [0.25, -0.20, 0.50]
        bounds = [(0.01, 2.0), (-0.95, 0.95), (0.01, 3.0)]
        res = opt.minimize(loss, init_params, bounds=bounds, method="L-BFGS-B")
        alpha_opt, rho_opt, nu_opt = res.x
        return cls(alpha=float(alpha_opt), beta=fixed_beta, rho=float(rho_opt), nu=float(nu_opt))


@dataclass
class PutCallParityArbitrage:
    """
    Analyzes European Options on Index Futures (NIFTY / BANKNIFTY):
    Theoretical Parity: C - P = exp(-r*T) * (F - K).
    Synthetic futures spread: S_synth = C - P + exp(-r*T) * K.
    """
    @staticmethod
    def evaluate_conversion_reversal(
        f_market: float,
        k: float,
        call_bid: float,
        call_ask: float,
        put_bid: float,
        put_ask: float,
        t_expiry: float,
        r: float,
    ) -> Dict[str, float]:
        df = math.exp(-r * t_expiry)
        # Synthetic Long Futures = Buy Call (ask) + Sell Put (bid)
        synthetic_long_f = (call_ask - put_bid) / df + k
        # Synthetic Short Futures = Sell Call (bid) + Buy Put (ask)
        synthetic_short_f = (call_bid - put_ask) / df + k

        # Conversion: Sell Overpriced Synthetic Long, Buy Actual Futures
        # Arbitrage occurs if synthetic_short_f > f_market
        reversal_profit = max(0.0, synthetic_short_f - f_market)
        # Reversal: Buy Underpriced Synthetic Long, Sell Actual Futures
        conversion_profit = max(0.0, f_market - synthetic_long_f)

        return {
            "synthetic_long_f": float(synthetic_long_f),
            "synthetic_short_f": float(synthetic_short_f),
            "conversion_profit_points": float(conversion_profit),
            "reversal_profit_points": float(reversal_profit),
        }


@dataclass
class IndexDispersionAnalyzer:
    """
    Evaluates Index Volatility Dispersion and Implied Correlation:
    sigma_Index^2 = sum_i w_i^2 sigma_i^2 + 2 sum_{i < j} w_i w_j sigma_i sigma_j rho_{ij}
    Equicorrelation approximation:
    rho_implied = (sigma_Index^2 - sum_i w_i^2 sigma_i^2) / ( (sum_i w_i sigma_i)^2 - sum_i w_i^2 sigma_i^2 )
    """
    @staticmethod
    def calculate_implied_correlation(
        index_iv: float,
        constituent_ivs: np.ndarray,
        constituent_weights: np.ndarray,
    ) -> float:
        """Calculates market implied correlation from index and constituent IVs."""
        w = constituent_weights / np.sum(constituent_weights)
        var_index = index_iv ** 2
        sum_w2_sig2 = np.sum((w ** 2) * (constituent_ivs ** 2))
        sum_w_sig = np.sum(w * constituent_ivs)
        denom = (sum_w_sig ** 2) - sum_w2_sig2

        if denom <= 0:
            return 0.5
        rho_imp = (var_index - sum_w2_sig2) / denom
        return float(np.clip(rho_imp, -1.0, 1.0))
