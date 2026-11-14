"""
Quant_vvf Predictive Intelligence & Forward Validation Engine (PIFVE)
Subsystem 2: Triple-Barrier Method & Ground Truth Labeling

Formulation (Marcos López de Prado, 2018):
- Upper profit-taking barrier: S_t * (1 + pt * sigma_t)
- Lower stop-loss barrier: S_t * (1 - sl * sigma_t)
- Vertical time horizon barrier: t + horizon (e.g. 2 hours)

Outcomes:
- +1: Upper barrier touched first
- -1: Lower barrier touched first
-  0: Vertical barrier reached first (neutral / expiry return)
- Ambiguous handling: If high/low within a single bar cross both barriers simultaneously,
  marks AMBIGUOUS_BAR or applies conservative worst-case stop policy.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from research.pifve.prediction_ledger.unconditional_ledger import OutcomeBarrier


@dataclass
class BarrierSpecification:
    upper_profit_mult: float = 2.0  # Multiplier on rolling volatility
    lower_loss_mult: float = 2.0    # Multiplier on rolling volatility
    max_horizon_bars: int = 60      # e.g., 60 bars (1-min bars = 1 hr)
    min_volatility_floor: float = 0.002 # 20 bps minimum barrier width


class TripleBarrierLabeler:
    """
    Computes rigorous triple-barrier event labeling for high-frequency or OHLCV feeds.
    """
    def __init__(self, spec: Optional[BarrierSpecification] = None):
        self.spec = spec or BarrierSpecification()

    def label_event(
        self,
        prices_df: pd.DataFrame,
        start_idx: int,
        rolling_volatility: float,
    ) -> Tuple[OutcomeBarrier, int, float]:
        """
        Labels a single prediction event starting at start_idx.
        Requires prices_df with columns ['close'] or ['high', 'low', 'close'].
        Returns: (OutcomeBarrier, bars_to_touch, realized_return_pct)
        """
        vol = max(self.spec.min_volatility_floor, rolling_volatility)
        s0 = float(prices_df["close"].iloc[start_idx])
        
        upper_target = s0 * (1.0 + self.spec.upper_profit_mult * vol)
        lower_target = s0 * (1.0 - self.spec.lower_loss_mult * vol)

        end_idx = min(len(prices_df), start_idx + self.spec.max_horizon_bars + 1)
        sub_df = prices_df.iloc[start_idx + 1 : end_idx]

        has_high_low = "high" in sub_df.columns and "low" in sub_df.columns

        for offset, (_, row) in enumerate(sub_df.iterrows(), start=1):
            if has_high_low:
                high = float(row["high"])
                low = float(row["low"])
            else:
                high = low = float(row["close"])

            upper_hit = high >= upper_target
            lower_hit = low <= lower_target

            # Simultaneous crossing in same bar
            if upper_hit and lower_hit:
                # Conservative convention: Assume stop-loss hit first
                ret = (lower_target - s0) / s0
                return OutcomeBarrier.AMBIGUOUS_BAR, offset, float(ret)

            if upper_hit:
                ret = (upper_target - s0) / s0
                return OutcomeBarrier.UPPER_TOUCHED, offset, float(ret)

            if lower_hit:
                ret = (lower_target - s0) / s0
                return OutcomeBarrier.LOWER_TOUCHED, offset, float(ret)

        # Vertical barrier reached (expiry)
        terminal_price = float(prices_df["close"].iloc[min(len(prices_df) - 1, start_idx + self.spec.max_horizon_bars)])
        ret = (terminal_price - s0) / s0
        return OutcomeBarrier.VERTICAL_TOUCHED, len(sub_df), float(ret)
