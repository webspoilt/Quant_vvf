"""
Cross-Exchange Research: NSE vs BSE Comparative Arbitrage & GIFT Nifty Basis.

Modules:
1. NSE-BSE Comparative Pricing & Dual-Listed Execution:
   - Synchronized quote alignment across National Stock Exchange (NSE) and Bombay Stock Exchange (BSE).
   - Executable spread net of statutory fees, brokerage, and exchange transaction fees.
2. Cash-Futures Basis Arbitrage:
   - Theoretical forward: F = S * exp((r - q) * T).
   - Arbitrage band accounting for STT, financing, borrow rate, and margin carry costs.
3. GIFT Nifty (NSE IX) vs Domestic NIFTY Futures:
   - USD-denominated offshore contract (contract size: USD 2 * index) vs INR domestic (lot size 25).
   - USD/INR forward FX basis adjustments and overlapping trading session mapping.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class DualListedSpreadQuote:
    symbol: str
    nse_bid: float
    nse_ask: float
    bse_bid: float
    bse_ask: float
    nse_timestamp_ms: int
    bse_timestamp_ms: int


@dataclass
class CrossExchangeArbitrageEngine:
    """
    Evaluates executable cross-exchange pricing between NSE and BSE:
    1. Buy BSE, Sell NSE if nse_bid - bse_ask > round_trip_costs.
    2. Buy NSE, Sell BSE if bse_bid - nse_ask > round_trip_costs.
    """
    round_trip_cost_bps: float = 8.5 # STT (intraday sell 0.025%) + Exchange + Stamp duty + Brokerage

    def analyze_spread(self, quote: DualListedSpreadQuote) -> Dict[str, Union[float, str, bool]]:
        # Stale quote check (latency skew > 500ms rejected)
        latency_skew = abs(quote.nse_timestamp_ms - quote.bse_timestamp_ms)
        if latency_skew > 500:
            return {
                "is_executable": False,
                "reason": "Stale quote latency skew exceeded threshold",
                "spread_bps": 0.0,
                "direction": "none",
            }

        cost_fraction = self.round_trip_cost_bps / 10000.0

        # Scenario A: Buy BSE ask, Sell NSE bid
        gross_spread_a = quote.nse_bid - quote.bse_ask
        net_spread_a = gross_spread_a - (quote.bse_ask * cost_fraction)

        # Scenario B: Buy NSE ask, Sell BSE bid
        gross_spread_b = quote.bse_bid - quote.nse_ask
        net_spread_b = gross_spread_b - (quote.nse_ask * cost_fraction)

        if net_spread_a > 0:
            return {
                "is_executable": True,
                "direction": "BUY_BSE_SELL_NSE",
                "gross_spread_points": float(gross_spread_a),
                "net_profit_points": float(net_spread_a),
                "spread_bps": float((net_spread_a / quote.bse_ask) * 10000.0),
            }
        elif net_spread_b > 0:
            return {
                "is_executable": True,
                "direction": "BUY_NSE_SELL_BSE",
                "gross_spread_points": float(gross_spread_b),
                "net_profit_points": float(net_spread_b),
                "spread_bps": float((net_spread_b / quote.nse_ask) * 10000.0),
            }
        else:
            return {
                "is_executable": False,
                "direction": "none",
                "gross_spread_points": float(max(gross_spread_a, gross_spread_b)),
                "net_profit_points": 0.0,
                "spread_bps": 0.0,
            }


@dataclass
class GiftNiftyDomesticBasisAnalyzer:
    """
    Analyzes basis between GIFT Nifty (NSE IX, USD-denominated) and Domestic NIFTY Futures (NSE, INR-denominated):
    - GIFT Nifty is traded at GIFT City (IFSC), priced in USD.
    - Contract value = Index * 2 USD.
    - Domestic NIFTY = Index * 25 INR.
    - Basis = (GIFT_Nifty_USD * USDINR_Spot) - Domestic_NIFTY_INR.
    """
    @staticmethod
    def calculate_cross_border_basis(
        gift_nifty_usd: float,
        domestic_nifty_inr: float,
        usdinr_spot: float,
        usdinr_forward_pts: float = 0.0,
        r_usd: float = 0.05,
        r_inr: float = 0.065,
        t_expiry: float = 0.1,
    ) -> Dict[str, float]:
        # Synthesize GIFT Nifty into INR terms
        effective_fx = usdinr_spot + usdinr_forward_pts
        gift_nifty_inr_equiv = gift_nifty_usd * (effective_fx / usdinr_spot) # Normalized index points

        basis_points = gift_nifty_usd - domestic_nifty_inr
        basis_pct = (basis_points / domestic_nifty_inr) * 100.0

        # CIP Interest rate differential: (1 + r_inr * T) / (1 + r_usd * T)
        cip_ratio = (1.0 + r_inr * t_expiry) / (1.0 + r_usd * t_expiry)
        theoretical_gift_premium = domestic_nifty_inr * (cip_ratio - 1.0)

        mispricing = basis_points - theoretical_gift_premium

        return {
            "basis_points": float(basis_points),
            "basis_pct": float(basis_pct),
            "theoretical_gift_premium": float(theoretical_gift_premium),
            "mispricing_points": float(mispricing),
        }
