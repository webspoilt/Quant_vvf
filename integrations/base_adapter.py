"""
Base Broker/Exchange Adapter Interface.
Defines abstract contracts for market data, order submission, cancellation, and position reconciliation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"


@dataclass
class NormalizedOrder:
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: float
    quantity: float
    filled_quantity: float
    status: str
    created_at_ns: int


@dataclass
class AccountPosition:
    symbol: str
    size_btc: float
    entry_price_usd: float
    mark_price_usd: float
    unrealized_pnl_usd: float
    maintenance_margin_usd: float


class BaseExchangeAdapter(ABC):
    """Abstract interface for all institutional broker and exchange gateways."""

    @abstractmethod
    def connect(self) -> bool:
        """Establishes authenticated transport sessions (REST/WebSocket)."""
        pass

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        price: float,
        quantity: float,
        client_order_id: str
    ) -> NormalizedOrder:
        """Submits new order to exchange gateway."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancels an existing resting order."""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[AccountPosition]:
        """Retrieves authoritative position state from exchange ledger."""
        pass

    @abstractmethod
    def reconcile_positions(self) -> List[AccountPosition]:
        """Performs full end-of-interval position reconciliation."""
        pass
