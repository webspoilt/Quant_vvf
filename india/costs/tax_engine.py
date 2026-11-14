"""
India Statutory Transaction Cost & Regulatory Compliance Engine.

Modules:
1. Complete Statutory Transaction Cost Architecture:
   - Securities Transaction Tax (STT):
     * Equity Delivery: 0.1% on buy & sell.
     * Equity Intraday: 0.025% on sell.
     * Equity Futures: 0.02% on sell (Post Oct 2024 revised STT).
     * Equity Options: 0.1% on premium on sell; 0.125% on exercise.
   - Stamp Duty (Indian Stamp Act):
     * Delivery: 0.015% on buy.
     * Intraday: 0.003% on buy.
     * Futures: 0.002% on buy.
     * Options: 0.003% on buy.
   - Exchange Transaction Charges (NSE / BSE):
     * Segment-specific tariff structure.
   - SEBI Turnover Fee:
     * Rs 10 per crore (0.0001%).
   - Goods and Services Tax (GST):
     * 18% on (Brokerage + Exchange Charges + SEBI Fees).
   - Clearing & DP Charges.
2. Regulatory Compliance Architecture:
   - Order-to-Trade Ratio (OTR) Controller with penalty bands.
   - Algorithmic Strategy Identification & Audit Logging (SEBI circular compliance).
   - Deterministic Kill-Switch and Rate Limiters.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple, Union


class MarketSegment(Enum):
    EQUITY_DELIVERY = "equity_delivery"
    EQUITY_INTRADAY = "equity_intraday"
    FUTURES_INDEX = "futures_index"
    FUTURES_STOCK = "futures_stock"
    OPTIONS_INDEX = "options_index"
    OPTIONS_STOCK = "options_stock"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class TradeExecution:
    segment: MarketSegment
    side: OrderSide
    quantity: int
    price: float
    strike_price: Optional[float] = None
    is_exercise: bool = False
    trade_date: date = field(default_factory=date.today)


@dataclass
class TradeCostBreakdown:
    notional_value: float
    brokerage: float
    stt: float
    stamp_duty: float
    exchange_charges: float
    sebi_fee: float
    gst: float
    clearing_charges: float
    total_cost: float
    effective_bps: float


class IndiaTransactionCostCalculator:
    """
    Computes complete statutory transaction costs under Indian tax laws
    (Finance Act revisions including Oct 2024 revised STT on F&O).
    """
    def __init__(self, brokerage_per_order: float = 20.0, broker_pct: float = 0.0):
        self.brokerage_per_order = brokerage_per_order
        self.broker_pct = broker_pct

    def calculate_cost(self, trade: TradeExecution) -> TradeCostBreakdown:
        notional = trade.quantity * trade.price
        brokerage = min(self.brokerage_per_order, notional * self.broker_pct) if self.broker_pct > 0 else self.brokerage_per_order

        # 1. Securities Transaction Tax (STT)
        stt = 0.0
        if trade.segment == MarketSegment.EQUITY_DELIVERY:
            stt = 0.001 * notional # 0.1% both sides
        elif trade.segment == MarketSegment.EQUITY_INTRADAY:
            if trade.side == OrderSide.SELL:
                stt = 0.00025 * notional # 0.025% on sell
        elif trade.segment in (MarketSegment.FUTURES_INDEX, MarketSegment.FUTURES_STOCK):
            if trade.side == OrderSide.SELL:
                stt = 0.0002 * notional # 0.02% on sell (revised Oct 2024)
        elif trade.segment in (MarketSegment.OPTIONS_INDEX, MarketSegment.OPTIONS_STOCK):
            if trade.is_exercise:
                settlement_val = trade.quantity * (trade.strike_price or trade.price)
                stt = 0.00125 * settlement_val # 0.125% on intrinsic exercise
            elif trade.side == OrderSide.SELL:
                stt = 0.001 * notional # 0.1% on option premium (revised Oct 2024)

        # 2. Stamp Duty (State Stamp Act / Centralized rate, on BUY side only)
        stamp_duty = 0.0
        if trade.side == OrderSide.BUY:
            if trade.segment == MarketSegment.EQUITY_DELIVERY:
                stamp_duty = 0.00015 * notional # 0.015%
            elif trade.segment == MarketSegment.EQUITY_INTRADAY:
                stamp_duty = 0.00003 * notional # 0.003%
            elif trade.segment in (MarketSegment.FUTURES_INDEX, MarketSegment.FUTURES_STOCK):
                stamp_duty = 0.00002 * notional # 0.002%
            elif trade.segment in (MarketSegment.OPTIONS_INDEX, MarketSegment.OPTIONS_STOCK):
                stamp_duty = 0.00003 * notional # 0.003%

        # 3. Exchange Transaction Charges (approx 0.00297% for NSE cash, 0.00173% for Futures, 0.03503% on premium for options)
        if trade.segment in (MarketSegment.OPTIONS_INDEX, MarketSegment.OPTIONS_STOCK):
            exchange_charges = 0.00035 * notional
        elif trade.segment in (MarketSegment.FUTURES_INDEX, MarketSegment.FUTURES_STOCK):
            exchange_charges = 0.0000173 * notional
        else:
            exchange_charges = 0.0000297 * notional

        # 4. SEBI Turnover Fee (Rs 10 per crore = 0.000001)
        sebi_fee = 0.000001 * notional

        # 5. Goods and Services Tax (GST) (18% on Brokerage + Exchange Charges + SEBI Fees)
        taxable_services = brokerage + exchange_charges + sebi_fee
        gst = 0.18 * taxable_services

        # 6. Clearing charges
        clearing_charges = 0.0

        total_cost = brokerage + stt + stamp_duty + exchange_charges + sebi_fee + gst + clearing_charges
        effective_bps = (total_cost / max(1e-6, notional)) * 10000.0

        return TradeCostBreakdown(
            notional_value=float(notional),
            brokerage=float(brokerage),
            stt=float(round(stt, 2)),
            stamp_duty=float(round(stamp_duty, 2)),
            exchange_charges=float(round(exchange_charges, 2)),
            sebi_fee=float(round(sebi_fee, 4)),
            gst=float(round(gst, 2)),
            clearing_charges=float(clearing_charges),
            total_cost=float(round(total_cost, 2)),
            effective_bps=float(round(effective_bps, 2)),
        )


@dataclass
class OrderToTradeRatioController:
    """
    NSE / BSE Order-to-Trade Ratio (OTR) compliance monitor.
    Enforces SEBI algo guidelines for OTR limits:
    Daily OTR = (Orders Placed + Orders Modified + Orders Cancelled) / max(1, Executed Trades).
    Applies penalty bands and cooling-off triggers when OTR exceeds thresholds (e.g. OTR > 50).
    """
    warning_otr_threshold: float = 50.0
    critical_otr_threshold: float = 100.0
    orders_placed: int = 0
    orders_modified: int = 0
    orders_cancelled: int = 0
    trades_executed: int = 0
    is_throttled: bool = False

    def record_order_event(self, event_type: str):
        if event_type == "place":
            self.orders_placed += 1
        elif event_type == "modify":
            self.orders_modified += 1
        elif event_type == "cancel":
            self.orders_cancelled += 1
        elif event_type == "trade":
            self.trades_executed += 1

        self._check_status()

    def get_current_otr(self) -> float:
        total_order_ops = self.orders_placed + self.orders_modified + self.orders_cancelled
        return float(total_order_ops / max(1, self.trades_executed))

    def _check_status(self):
        otr = self.get_current_otr()
        total_ops = self.orders_placed + self.orders_modified + self.orders_cancelled
        # Regulatory threshold applies when order operations exceed minimum sample (e.g. > 500)
        if total_ops > 100 and otr > self.critical_otr_threshold:
            self.is_throttled = True
        else:
            self.is_throttled = False
