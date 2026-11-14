"""
India Statutory Transaction Cost & Regulatory Surveillance Engine.

Priority 0 Remediation:
1. Versioned Statutory Tax Schedule with Effective-Date Boundary Selection:
   - Schedule A: Historical (Pre-October 2024 / Finance Act 2024)
   - Schedule B: Finance Act 2024 (Oct 1, 2024 - March 31, 2026)
     * Futures sale: 0.02%
     * Options premium sale: 0.10%
     * Options intrinsic exercise: 0.125%
   - Schedule C: Finance Act 2026 (Effective April 1, 2026 onward):
     * Futures sale: 0.05%
     * Options premium sale: 0.15%
     * Options exercise: 0.15% on intrinsic value (Settlement Price - Strike Price)+, NOT full notional.
2. Segment-aware Stamp Duty (Indian Stamp Act, Buy Side Only).
3. Exchange transaction charges, SEBI turnover fees, and GST on taxable services.
4. Segregated Regulatory OTR Surveillance:
   - Official NSE Surveillance: Tracks OTR >= 2,000 on >= 3 instances within 30 rolling trading days.
   - Broker Risk Gate: Distinct from regulatory threshold.
   - Strategy Internal Throttle: Configurable (e.g., 50:1).
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple, Union


class ExchangeVenue(Enum):
    NSE = "NSE"
    BSE = "BSE"
    NSE_IX = "NSE_IX" # GIFT City IFSC (STT/Stamp exempt)


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
    exchange: ExchangeVenue = ExchangeVenue.NSE
    segment: MarketSegment = MarketSegment.EQUITY_DELIVERY
    side: OrderSide = OrderSide.BUY
    quantity_units: int = 1
    quantity: Optional[int] = None # Backwards compatibility alias for quantity_units
    lot_size: int = 1
    price: float = 0.0          # Executed trade price (option premium or futures price)
    strike_price: Optional[float] = None
    settlement_price: Optional[float] = None # Applicable on expiry for intrinsic calculation
    is_exercise: bool = False
    trade_date: date = field(default_factory=date.today)

    def __post_init__(self):
        if self.quantity is not None:
            self.quantity_units = self.quantity


@dataclass
class TradeCostBreakdown:
    policy_version: str
    notional_value: float
    taxable_stt_base: float
    brokerage: float
    stt: float
    stamp_duty: float
    exchange_charges: float
    sebi_fee: float
    gst: float
    clearing_charges: float
    total_cost: float
    effective_bps: float


class IndiaStatutoryCostEngine:
    """
    Versioned Indian statutory cost engine enforcing official Finance Act and exchange schedules.
    """
    BOUNDARY_DATE_2024 = date(2024, 10, 1)
    BOUNDARY_DATE_2026 = date(2026, 4, 1)

    def __init__(self, brokerage_per_order: float = 20.0, broker_pct: float = 0.0):
        self.brokerage_per_order = brokerage_per_order
        self.broker_pct = broker_pct

    def determine_policy_version(self, trade_date: date) -> str:
        if trade_date < self.BOUNDARY_DATE_2024:
            return "PRE_OCT_2024"
        elif trade_date < self.BOUNDARY_DATE_2026:
            return "OCT_2024_TO_MAR_2026"
        else:
            return "APR_2026_ONWARD"

    def calculate_cost(self, trade: TradeExecution) -> TradeCostBreakdown:
        policy_ver = self.determine_policy_version(trade.trade_date)
        notional = trade.quantity_units * trade.price
        brokerage = min(self.brokerage_per_order, notional * self.broker_pct) if self.broker_pct > 0 else self.brokerage_per_order

        # IFSC / GIFT City exemptions (NSE IX trades are exempt from domestic STT, GST, and Stamp Duty)
        if trade.exchange == ExchangeVenue.NSE_IX:
            exchange_fee = 0.00010 * notional # 1 bps exchange charge
            total_cost = brokerage + exchange_fee
            return TradeCostBreakdown(
                policy_version="GIFT_CITY_IFSC_EXEMPT",
                notional_value=float(notional),
                taxable_stt_base=0.0,
                brokerage=float(brokerage),
                stt=0.0,
                stamp_duty=0.0,
                exchange_charges=float(round(exchange_fee, 2)),
                sebi_fee=0.0,
                gst=0.0,
                clearing_charges=0.0,
                total_cost=float(round(total_cost, 2)),
                effective_bps=float(round((total_cost / max(1e-6, notional)) * 10000.0, 2)),
            )

        # 1. Securities Transaction Tax (STT) Rates & Taxable Base
        stt = 0.0
        taxable_stt_base = 0.0

        if trade.segment in (MarketSegment.FUTURES_INDEX, MarketSegment.FUTURES_STOCK):
            if trade.side == OrderSide.SELL:
                taxable_stt_base = notional
                if policy_ver == "PRE_OCT_2024":
                    stt = 0.000125 * taxable_stt_base # 0.0125%
                elif policy_ver == "OCT_2024_TO_MAR_2026":
                    stt = 0.00020 * taxable_stt_base  # 0.02%
                else: # APR_2026_ONWARD
                    stt = 0.00050 * taxable_stt_base  # 0.05% (Finance Act 2026)

        elif trade.segment in (MarketSegment.OPTIONS_INDEX, MarketSegment.OPTIONS_STOCK):
            if trade.is_exercise:
                # Intrinsic value calculation base: (Settlement - Strike)+ for Call, (Strike - Settlement)+ for Put
                # Strict statutory requirement: STT applies to INTRINSIC VALUE, not strike price notional
                settle = trade.settlement_price or trade.price
                strike = trade.strike_price or trade.price
                intrinsic = max(0.0, settle - strike) if trade.side == OrderSide.BUY else max(0.0, strike - settle)
                taxable_stt_base = trade.quantity_units * intrinsic

                if policy_ver == "PRE_OCT_2024":
                    stt = 0.00125 * taxable_stt_base # 0.125%
                elif policy_ver == "OCT_2024_TO_MAR_2026":
                    stt = 0.00125 * taxable_stt_base # 0.125%
                else: # APR_2026_ONWARD
                    stt = 0.00150 * taxable_stt_base # 0.150% (Finance Act 2026)
            elif trade.side == OrderSide.SELL:
                taxable_stt_base = notional # Premium turnover
                if policy_ver == "PRE_OCT_2024":
                    stt = 0.000625 * taxable_stt_base # 0.0625%
                elif policy_ver == "OCT_2024_TO_MAR_2026":
                    stt = 0.00100 * taxable_stt_base  # 0.100%
                else: # APR_2026_ONWARD
                    stt = 0.00150 * taxable_stt_base  # 0.150% (Finance Act 2026)

        elif trade.segment == MarketSegment.EQUITY_DELIVERY:
            taxable_stt_base = notional
            stt = 0.001 * taxable_stt_base # 0.1% both sides
        elif trade.segment == MarketSegment.EQUITY_INTRADAY:
            if trade.side == OrderSide.SELL:
                taxable_stt_base = notional
                stt = 0.00025 * taxable_stt_base # 0.025% on sell

        # 2. Stamp Duty (Indian Stamp Act, Centralized rate, BUY side only)
        stamp_duty = 0.0
        if trade.side == OrderSide.BUY and not trade.is_exercise:
            if trade.segment == MarketSegment.EQUITY_DELIVERY:
                stamp_duty = 0.00015 * notional # 0.015%
            elif trade.segment == MarketSegment.EQUITY_INTRADAY:
                stamp_duty = 0.00003 * notional # 0.003%
            elif trade.segment in (MarketSegment.FUTURES_INDEX, MarketSegment.FUTURES_STOCK):
                stamp_duty = 0.00002 * notional # 0.002%
            elif trade.segment in (MarketSegment.OPTIONS_INDEX, MarketSegment.OPTIONS_STOCK):
                stamp_duty = 0.00003 * notional # 0.003% on premium

        # 3. Exchange Transaction Charges
        # NSE vs BSE tariff schedules
        if trade.exchange == ExchangeVenue.BSE:
            if trade.segment in (MarketSegment.OPTIONS_INDEX, MarketSegment.OPTIONS_STOCK):
                exchange_charges = 0.000325 * notional # BSE options tariff
            elif trade.segment in (MarketSegment.FUTURES_INDEX, MarketSegment.FUTURES_STOCK):
                exchange_charges = 0.0000150 * notional
            else:
                exchange_charges = 0.0000275 * notional
        else: # NSE
            if trade.segment in (MarketSegment.OPTIONS_INDEX, MarketSegment.OPTIONS_STOCK):
                exchange_charges = 0.0003503 * notional # NSE options tariff on premium
            elif trade.segment in (MarketSegment.FUTURES_INDEX, MarketSegment.FUTURES_STOCK):
                exchange_charges = 0.0000173 * notional
            else:
                exchange_charges = 0.0000297 * notional

        # 4. SEBI Turnover Fee (Rs 10 per crore = 0.000001)
        sebi_fee = 0.000001 * notional

        # 5. Goods and Services Tax (GST) (18% on Brokerage + Exchange Charges + SEBI Fees)
        taxable_services = brokerage + exchange_charges + sebi_fee
        gst = 0.18 * taxable_services

        clearing_charges = 0.0
        total_cost = brokerage + stt + stamp_duty + exchange_charges + sebi_fee + gst + clearing_charges
        effective_bps = (total_cost / max(1e-6, notional)) * 10000.0

        return TradeCostBreakdown(
            policy_version=policy_ver,
            notional_value=float(notional),
            taxable_stt_base=float(round(taxable_stt_base, 2)),
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


# Backwards compatibility aliases
IndiaTransactionCostCalculator = IndiaStatutoryCostEngine


class OrderToTradeRatioController:
    """Compatibility wrapper around NseBseSurveillanceOtrController."""
    def __init__(self, critical_otr_threshold: float = 50.0):
        self._controller = NseBseSurveillanceOtrController(internal_strategy_throttle_otr=critical_otr_threshold)

    def record_order_event(self, event_type: str):
        self._controller.record_order_event(event_type)

    def get_current_otr(self) -> float:
        sess = self._controller._get_current_session()
        return sess.otr

    @property
    def is_throttled(self) -> bool:
        return self._controller.check_internal_strategy_throttle()


@dataclass
class DailySessionOrderStats:
    session_date: date
    orders_placed: int = 0
    orders_modified: int = 0
    orders_cancelled: int = 0
    trades_executed: int = 0

    @property
    def total_order_operations(self) -> int:
        return self.orders_placed + self.orders_modified + self.orders_cancelled

    @property
    def otr(self) -> float:
        return float(self.total_order_operations / max(1, self.trades_executed))


class NseBseSurveillanceOtrController:
    """
    Segregates regulatory exchange surveillance conditions from broker/strategy gates:
    1. Official NSE Regulatory Surveillance:
       - Trigger: OTR >= 2,000 on >= 3 trading days in the preceding 30 rolling trading days.
       - Action: Exchange cooling-off penalty and surveillance notice.
    2. Conservative Internal Strategy Throttle:
       - Configurable threshold (default: 50:1).
    """
    def __init__(self, internal_strategy_throttle_otr: float = 50.0):
        self.internal_strategy_throttle_otr = internal_strategy_throttle_otr
        self.session_history: Dict[date, DailySessionOrderStats] = {}
        self.current_date: date = date.today()

    def _get_current_session(self) -> DailySessionOrderStats:
        if self.current_date not in self.session_history:
            self.session_history[self.current_date] = DailySessionOrderStats(session_date=self.current_date)
        return self.session_history[self.current_date]

    def record_order_event(self, event_type: str, event_date: Optional[date] = None):
        if event_date:
            self.current_date = event_date
        sess = self._get_current_session()

        if event_type == "place":
            sess.orders_placed += 1
        elif event_type == "modify":
            sess.orders_modified += 1
        elif event_type == "cancel":
            sess.orders_cancelled += 1
        elif event_type == "trade":
            sess.trades_executed += 1

    def check_nse_official_surveillance_breach(self) -> Tuple[bool, int, List[date]]:
        """
        Evaluates official NSE OTR Surveillance Condition:
        >= 3 instances of OTR >= 2,000 within previous 30 trading days.
        Returns: (is_breached, violation_count, dates_of_violation)
        """
        cutoff_date = self.current_date - timedelta(days=45) # Calendar span covering 30 trading days
        relevant_sessions = [
            s for d, s in self.session_history.items()
            if d >= cutoff_date and s.total_order_operations >= 500 # Minimum activity threshold
        ]

        violation_dates = [s.session_date for s in relevant_sessions if s.otr >= 2000.0]
        count = len(violation_dates)
        is_breached = count >= 3
        return is_breached, count, violation_dates

    def check_internal_strategy_throttle(self) -> bool:
        """
        Internal safety throttle (e.g. 50:1) to prevent run-away algorithmic loops.
        """
        sess = self._get_current_session()
        if sess.total_order_operations > 100 and sess.otr > self.internal_strategy_throttle_otr:
            return True # Strategy should throttle/pause
        return False
