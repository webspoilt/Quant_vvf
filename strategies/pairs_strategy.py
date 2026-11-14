"""
Statistical Arbitrage Pairs Trading Strategy
Trades cointegrated log-price spread deviations based on rolling Z-scores.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy, StrategySignal, SignalType


class PairsTradingStrategy(BaseStrategy):
    """Generates mean-reverting signals from cointegrated asset spreads."""

    def __init__(
        self,
        strategy_id: str,
        hedge_ratio: float,
        entry_z_threshold: float = 2.0,
        exit_z_threshold: float = 0.5,
        target_clip_btc: float = 1.0,
        config: Dict[str, Any] = None
    ):
        super().__init__(strategy_id, config or {})
        self.hedge_ratio = hedge_ratio
        self.entry_z = entry_z_threshold
        self.exit_z = exit_z_threshold
        self.target_clip = target_clip_btc

    def generate_signal(self, market_data: pd.DataFrame) -> StrategySignal:
        """Evaluates latest Z-score of spread = price_y - beta * price_x."""
        if "spread_z" not in market_data.columns:
            raise ValueError("Input market data must contain 'spread_z' column")

        current_z = market_data["spread_z"].iloc[-1]
        latest_price = market_data["price"].iloc[-1] if "price" in market_data.columns else None

        if current_z >= self.entry_z:
            # Spread is overextended: Short spread (Sell Y, Buy X)
            sig_type = SignalType.SELL
            qty = self.target_clip
            conf = min(1.0, abs(current_z) / 3.0)
        elif current_z <= -self.entry_z:
            # Spread is depressed: Long spread (Buy Y, Sell X)
            sig_type = SignalType.BUY
            qty = self.target_clip
            conf = min(1.0, abs(current_z) / 3.0)
        elif abs(current_z) <= self.exit_z:
            # Mean reverted: Exit position
            sig_type = SignalType.HOLD
            qty = 0.0
            conf = 1.0
        else:
            sig_type = SignalType.HOLD
            qty = 0.0
            conf = 0.5

        return StrategySignal(
            strategy_id=self.strategy_id,
            symbol="BTC-PERPETUAL",
            signal_type=sig_type,
            target_quantity_btc=qty,
            limit_price=latest_price,
            confidence_score=float(conf),
            metadata={"current_z": float(current_z), "hedge_ratio": self.hedge_ratio}
        )
