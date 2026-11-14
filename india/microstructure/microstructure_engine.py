"""
Indian Market Microstructure: Multi-level OFI, Microprice, and Queue Dynamics.

Modules:
1. Multi-Level Order Flow Imbalance (OFI):
   - Level-k OFI: changes in bid/ask quantities under price increases, price decreases, or constant price.
   - Aggregate OFI vector predicting short-term tick price movements.
2. Microprice Estimator (Stoikov 2018):
   - Incorporates bid-ask queue imbalance: P_micro = P_mid + (P_spread / 2) * ((Q_bid - Q_ask) / (Q_bid + Q_ask)).
3. Option Order Flow & PCR / OI Shift Analytics:
   - Strike-level volume imbalance and Open Interest (OI) concentration walls (Max Pain).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class OrderBookLevelSnapshot:
    bid_prices: np.ndarray # Top N bid prices descending
    bid_sizes: np.ndarray # Top N bid quantities
    ask_prices: np.ndarray # Top N ask prices ascending
    ask_sizes: np.ndarray # Top N ask quantities


class IndianMarketMicrostructureEngine:
    """
    Computes High-Frequency Market Microstructure metrics from order book snapshots.
    """
    @staticmethod
    def calculate_microprice(bid: float, ask: float, bid_size: float, ask_size: float) -> float:
        """
        Stoikov (2018) order queue imbalance microprice:
        P_micro = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
        """
        total_size = bid_size + ask_size
        if total_size <= 0:
            return 0.5 * (bid + ask)
        return float((bid * ask_size + ask * bid_size) / total_size)

    @staticmethod
    def calculate_multilevel_ofi(
        prev_snap: OrderBookLevelSnapshot,
        curr_snap: OrderBookLevelSnapshot,
        n_levels: int = 5,
    ) -> float:
        """
        Multi-level Order Flow Imbalance (Cont, Kukanov, Stoikov 2014):
        Delta Q_bid = +curr if curr_p > prev_p; curr - prev if curr_p == prev_p; -prev if curr_p < prev_p.
        Delta Q_ask = -curr if curr_p < prev_p; curr - prev if curr_p == prev_p; +prev if curr_p > prev_p.
        OFI = Delta Q_bid - Delta Q_ask.
        """
        ofi_sum = 0.0
        levels = min(n_levels, len(prev_snap.bid_prices), len(curr_snap.bid_prices))

        for k in range(levels):
            # Bid side
            p_b_prev = prev_snap.bid_prices[k]
            q_b_prev = prev_snap.bid_sizes[k]
            p_b_curr = curr_snap.bid_prices[k]
            q_b_curr = curr_snap.bid_sizes[k]

            if p_b_curr > p_b_prev:
                delta_b = q_b_curr
            elif p_b_curr == p_b_prev:
                delta_b = q_b_curr - q_b_prev
            else:
                delta_b = -q_b_prev

            # Ask side
            p_a_prev = prev_snap.ask_prices[k]
            q_a_prev = prev_snap.ask_sizes[k]
            p_a_curr = curr_snap.ask_prices[k]
            q_a_curr = curr_snap.ask_sizes[k]

            if p_a_curr < p_a_prev:
                delta_a = q_a_curr
            elif p_a_curr == p_a_prev:
                delta_a = q_a_curr - q_a_prev
            else:
                delta_a = -q_a_prev

            # Decay factor for deeper levels (e.g. 1 / (1 + k))
            weight = 1.0 / (1.0 + 0.5 * k)
            ofi_sum += weight * (delta_b - delta_a)

        return float(ofi_sum)

    @staticmethod
    def calculate_max_pain_strike(
        strikes: np.ndarray,
        call_oi: np.ndarray,
        put_oi: np.ndarray,
    ) -> float:
        """
        Computes Max Pain strike where option buyers lose the maximum cumulative value:
        Total Loss(S) = sum_k [ Call_OI_k * max(S - K_k, 0) + Put_OI_k * max(K_k - S, 0) ]
        """
        losses = []
        for s in strikes:
            call_loss = np.sum(call_oi * np.maximum(s - strikes, 0.0))
            put_loss = np.sum(put_oi * np.maximum(strikes - s, 0.0))
            losses.append(call_loss + put_loss)

        min_idx = np.argmin(losses)
        return float(strikes[min_idx])
