"""
Abstract Strategy Base Interface and Signal Contracts.
Strategies produce signals only; they do not have direct execution or risk authorization authority.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional
import pandas as pd


class SignalType(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2
    REBALANCE = 3


@dataclass
class StrategySignal:
    strategy_id: str
    symbol: str
    signal_type: SignalType
    target_quantity_btc: float
    limit_price: Optional[float]
    confidence_score: float
    metadata: Dict[str, Any]


class BaseStrategy(ABC):
    """Abstract base class for all quantitative and algorithmic strategies."""

    def __init__(self, strategy_id: str, config: Dict[str, Any]):
        self.strategy_id = strategy_id
        self.config = config

    @abstractmethod
    def generate_signal(self, market_data: pd.DataFrame) -> StrategySignal:
        """Processes point-in-time market data and returns a structured StrategySignal."""
        pass
