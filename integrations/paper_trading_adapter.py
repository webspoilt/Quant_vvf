"""
Deterministic Paper Trading Adapter
Simulates exchange order matching, fills, cancellations, and position reconciliation
with zero external network dependencies.
"""

from typing import Dict, List, Optional
import time
from integrations.base_adapter import (
    BaseExchangeAdapter,
    NormalizedOrder,
    AccountPosition,
    OrderSide,
    OrderType,
)


class PaperTradingAdapter(BaseExchangeAdapter):
    """Local deterministic mock exchange matching engine for paper trading and integration tests."""

    def __init__(self, initial_cash_usd: float = 100_000.0):
        self.initial_cash = initial_cash_usd
        self.cash = initial_cash_usd
        self.orders: Dict[str, NormalizedOrder] = {}
        self.positions: Dict[str, AccountPosition] = {}
        self.connected = False
        self.order_counter = 0

    def connect(self) -> bool:
        self.connected = True
        return True

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        price: float,
        quantity: float,
        client_order_id: str
    ) -> NormalizedOrder:
        if not self.connected:
            raise ConnectionError("Adapter is not connected")
        if price <= 0 or quantity <= 0:
            raise ValueError("Price and quantity must be positive")

        self.order_counter += 1
        order_id = f"PAPER-ORD-{self.order_counter}"

        # Immediately match for market/paper simulation
        order = NormalizedOrder(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            price=price,
            quantity=quantity,
            filled_quantity=quantity,
            status="FILLED",
            created_at_ns=time.time_ns(),
        )
        self.orders[order_id] = order

        # Update position
        pos = self.positions.get(symbol, AccountPosition(symbol, 0.0, price, price, 0.0, 0.0))
        delta_size = quantity if side == OrderSide.BUY else -quantity
        new_size = pos.size_btc + delta_size
        self.positions[symbol] = AccountPosition(
            symbol=symbol,
            size_btc=new_size,
            entry_price_usd=price,
            mark_price_usd=price,
            unrealized_pnl_usd=0.0,
            maintenance_margin_usd=abs(new_size * price * 0.02)
        )

        return order

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders and self.orders[order_id].status != "FILLED":
            self.orders[order_id].status = "CANCELED"
            return True
        return False

    def get_position(self, symbol: str) -> Optional[AccountPosition]:
        return self.positions.get(symbol)

    def reconcile_positions(self) -> List[AccountPosition]:
        return list(self.positions.values())
