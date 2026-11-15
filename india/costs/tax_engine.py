"""
India Statutory Transaction Cost, Exchange Tariff & Regulatory Surveillance Engine.

Authoritative Reference Matrix & Statutory Gazette Foundation:
1. Securities Transaction Tax (STT):
   - Finance (No. 2) Act, 2004 (Chapter VII).
   - Finance (No. 2) Act, 2019 (Effective Sept 1, 2019):
     Amendment to Section 98(3) clarifying that STT on exercised option contracts
     is payable by the purchaser on INTRINSIC VALUE:
     CALL Intrinsic = max(settlement_price - strike_price, 0)
     PUT Intrinsic = max(strike_price - settlement_price, 0)
     Prior to Sept 1, 2019, taxable base was total settlement value.
   - Finance (No. 2) Act, 2024 (Effective Oct 1, 2024):
     * Futures sale: 0.02%
     * Options premium sale: 0.10%
     * Options exercise (purchaser): 0.125% on intrinsic value.
   - Finance Act, 2026 (Effective April 1, 2026 onward):
     * Futures sale: 0.05%
     * Options premium sale: 0.15%
     * Options exercise (purchaser): 0.15% on intrinsic value.

2. Exchange Transaction Charges & IPFT (NSE Circular NSE/FA/73061, effective March 1, 2026):
   - Cash: Rs 306.99 per crore (0.0030699%), IPFT: Rs 0.01 per crore (0.0000001%).
   - Equity Futures: Rs 182.99 per crore (0.0018299%), IPFT: Rs 0.01 per crore.
   - Equity Options: Rs 3,552 per crore (0.03552% on premium), IPFT: Rs 0.01 per crore on premium.
   - Historical Pre-March 2026 NSE:
     * Cash: Rs 297 per crore (0.00297%), IPFT: Rs 0.05 per crore.
     * Futures: Rs 173 per crore (0.00173%), IPFT: Rs 0.05 per crore.
     * Options: Rs 3,503 per crore (0.03503%), IPFT: Rs 0.05 per crore.

3. BSE Tariffs:
   - Equity Cash: Rs 275 per crore (0.00275%), IPFT: Rs 0.01 per crore.
   - Equity Futures (SENSEX/BANKEX): Rs 0.0 (Promotional/Incentive) or Rs 150 per crore (0.0015%).
   - Equity Options: Rs 3,250 per crore (0.0325% on premium), IPFT: Rs 0.01 per crore.

4. IFSC / GIFT City (NSE IX / INX):
   - STT: Section 98 exempt under Chapter VII for IFSC entities.
   - Stamp Duty: Exempt for authorized IFSC operations.
   - GST: Treated as export of financial services (zero-rated under IGST Act) IF entity is registered IFSC unit.
   - Exchange/Clearing Tariff: Schedule based (e.g. $0.15 to $0.20 per contract or 0.5-1.0 bps).
   - If entity status/jurisdiction is unverified -> LEGAL_CLASSIFICATION_REQUIRED.

5. SEBI Regulatory Turnover Fee:
   - All domestic segments: Rs 10 per crore (0.000001) on traded turnover / premium turnover.

6. Stamp Duty (Indian Stamp Rules 2019, Centralized rate, BUY side only):
   - Delivery Buy: 0.015% (Rs 1500 per crore)
   - Intraday Buy: 0.003% (Rs 300 per crore)
   - Futures Buy: 0.002% (Rs 200 per crore)
   - Options Premium Buy: 0.003% (Rs 300 per crore)

7. GST:
   - 18% on (Brokerage + Exchange Charges + SEBI Fees + IPFT).

8. Official NSE Order-to-Trade Ratio (OTR) Framework:
   - Consolidated Circulars NSE/SURV/47980, amended Feb 6, 2026 and April 2, 2026.
   - Evaluation on 30 actual trading sessions using true NSE trading calendar.
   - Thresholds: OTR >= 2,000 with >= 500 operations per day for surveillance breach.
   - April 6, 2026 Options Segment Exclusion: Orders within +/-40% of LTP premium or Rs 20 (whichever is higher) are excluded.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import math
from typing import Dict, List, Optional, Set, Tuple, Union


class ExchangeVenue(Enum):
    NSE = "NSE"
    BSE = "BSE"
    NSE_IX = "NSE_IX"  # GIFT City IFSC


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


class OptionType(Enum):
    CALL = "CALL"
    PUT = "PUT"


class GiftCityEntityStatus(Enum):
    UNVERIFIED = "UNVERIFIED"
    IFSC_REGISTERED_UNIT = "IFSC_REGISTERED_UNIT"
    DOMESTIC_NON_IFSC = "DOMESTIC_NON_IFSC"
    FOREIGN_NON_RESIDENT = "FOREIGN_NON_RESIDENT"


class NseTradingCalendar:
    """
    Real NSE Trading Calendar handling weekends and statutory market holidays.
    Used for rolling 30 trading-day historical surveillance calculations.
    """
    # Observed NSE trading holidays (sample covering 2024-2026 regulatory test windows)
    HOLIDAYS_SET: Set[date] = {
        # 2024
        date(2024, 1, 26), date(2024, 3, 8), date(2024, 3, 25), date(2024, 3, 29),
        date(2024, 4, 11), date(2024, 4, 17), date(2024, 5, 1), date(2024, 6, 17),
        date(2024, 7, 17), date(2024, 8, 15), date(2024, 10, 2), date(2024, 11, 1),
        date(2024, 11, 15), date(2024, 12, 25),
        # 2025
        date(2025, 1, 26), date(2025, 2, 26), date(2025, 3, 14), date(2025, 3, 31),
        date(2025, 4, 10), date(2025, 4, 14), date(2025, 4, 18), date(2025, 5, 1),
        date(2025, 8, 15), date(2025, 8, 27), date(2025, 10, 2), date(2025, 10, 21),
        date(2025, 10, 22), date(2025, 11, 5), date(2025, 12, 25),
        # 2026
        date(2026, 1, 26), date(2026, 3, 3), date(2026, 3, 20), date(2026, 3, 25),
        date(2026, 4, 3), date(2026, 4, 14), date(2026, 5, 1), date(2026, 5, 28),
        date(2026, 8, 15), date(2026, 8, 19), date(2026, 10, 2), date(2026, 10, 20),
        date(2026, 11, 9), date(2026, 12, 25),
    }

    @classmethod
    def is_trading_day(cls, dt: date) -> bool:
        # Weekend check: Saturday (5) or Sunday (6)
        if dt.weekday() >= 5:
            return False
        if dt in cls.HOLIDAYS_SET:
            return False
        return True

    @classmethod
    def get_trailing_trading_days(cls, anchor_date: date, count: int = 30) -> List[date]:
        """
        Returns up to `count` trading sessions ending on or before `anchor_date`.
        If anchor_date is a trading day, it is included. If it is a non-trading day (weekend/holiday),
        it is still included as a session date if activity was recorded on it, alongside `count` trading days.
        """
        trading_days: List[date] = []
        if not cls.is_trading_day(anchor_date):
            trading_days.append(anchor_date)

        curr = anchor_date
        while len([d for d in trading_days if cls.is_trading_day(d)]) < count:
            if cls.is_trading_day(curr):
                if curr not in trading_days:
                    trading_days.append(curr)
            curr -= timedelta(days=1)
        return trading_days


@dataclass
class TradeExecution:
    exchange: ExchangeVenue = ExchangeVenue.NSE
    segment: MarketSegment = MarketSegment.EQUITY_DELIVERY
    side: OrderSide = OrderSide.BUY
    quantity_units: Optional[int] = None
    lots: Optional[int] = None
    lot_size: int = 1
    multiplier: float = 1.0
    price: float = 0.0          # Executed trade price (option premium or futures/cash price)
    strike_price: Optional[float] = None
    settlement_price: Optional[float] = None # Applicable on expiry for intrinsic calculation
    option_type: Optional[OptionType] = None # Explicit CALL or PUT
    is_exercise: bool = False
    trade_date: date = field(default_factory=date.today)
    currency: str = "INR"
    ifsc_status: GiftCityEntityStatus = GiftCityEntityStatus.UNVERIFIED

    # Backwards compatibility alias
    quantity: Optional[int] = None

    def __post_init__(self):
        # Quantity resolution: lots vs total units
        if self.quantity is not None and self.quantity_units is None:
            self.quantity_units = self.quantity
        if self.quantity_units is None and self.lots is not None:
            if self.lots <= 0:
                raise ValueError(f"Number of lots must be positive, got {self.lots}")
            if self.lot_size <= 0:
                raise ValueError(f"Lot size must be positive, got {self.lot_size}")
            self.quantity_units = int(self.lots * self.lot_size)
        elif self.quantity_units is not None:
            if self.quantity_units <= 0:
                raise ValueError(f"Quantity units must be positive, got {self.quantity_units}")
        else:
            raise ValueError("Either quantity_units, quantity, or lots must be explicitly provided.")

        if self.lot_size <= 0:
            raise ValueError(f"Lot size must be positive, got {self.lot_size}")
        if self.multiplier <= 0:
            raise ValueError(f"Multiplier must be positive, got {self.multiplier}")

        # Price validation
        if not math.isfinite(self.price) or self.price < 0:
            raise ValueError(f"Trade price must be finite and non-negative, got {self.price}")

        # Currency validation
        if self.exchange in (ExchangeVenue.NSE, ExchangeVenue.BSE) and self.currency != "INR":
            raise ValueError(f"Domestic Indian exchanges require INR currency, got {self.currency}")

        # Exercise validation
        if self.is_exercise:
            if self.segment not in (MarketSegment.OPTIONS_INDEX, MarketSegment.OPTIONS_STOCK):
                raise ValueError(f"Exercise is only valid for options segments, got {self.segment}")
            if self.option_type is None:
                # Default to CALL if side == BUY for backwards compatibility with legacy tests,
                # but explicit option_type is strongly recommended.
                self.option_type = OptionType.CALL if self.side == OrderSide.BUY else OptionType.PUT
            if self.strike_price is None or not math.isfinite(self.strike_price) or self.strike_price <= 0:
                raise ValueError(f"Valid positive strike_price is strictly required for option exercise, got {self.strike_price}")
            if self.settlement_price is None or not math.isfinite(self.settlement_price) or self.settlement_price < 0:
                raise ValueError(f"Valid non-negative settlement_price is strictly required for option exercise, got {self.settlement_price}")


@dataclass
class TradeCostBreakdown:
    policy_version: str
    notional_value: float
    taxable_stt_base: float
    brokerage: float
    stt: float
    stamp_duty: float
    exchange_charges: float
    ipft: float
    sebi_fee: float
    gst: float
    clearing_charges: float
    total_cost: float
    effective_bps: float
    legal_classification_status: str = "CONFIRMED"


class IndiaStatutoryCostEngine:
    """
    Authoritative, date-versioned statutory tax and exchange tariff calculator for Indian markets.
    """
    BOUNDARY_PRE_SEPT_2019 = date(2019, 9, 1)     # Finance (No. 2) Act 2019 (Exercise on Intrinsic)
    BOUNDARY_OCT_2024 = date(2024, 10, 1)          # Finance (No. 2) Act 2024 (Fut 0.02%, Opt 0.10%)
    BOUNDARY_MAR_2026_TARIFF = date(2026, 3, 1)    # NSE Circular NSE/FA/73061 (Tariff update)
    BOUNDARY_APR_2026_STT = date(2026, 4, 1)       # Finance Act 2026 (Fut 0.05%, Opt 0.15%, Exer 0.15%)

    def __init__(self, brokerage_per_order: float = 20.0, broker_pct: float = 0.0):
        self.brokerage_per_order = float(brokerage_per_order)
        self.broker_pct = float(broker_pct)

    def determine_stt_policy_version(self, trade_date: date) -> str:
        if trade_date < self.BOUNDARY_PRE_SEPT_2019:
            return "PRE_SEPT_2019"
        elif trade_date < self.BOUNDARY_OCT_2024:
            return "SEPT_2019_TO_SEPT_2024"
        elif trade_date < self.BOUNDARY_APR_2026_STT:
            return "OCT_2024_TO_MAR_2026"
        else:
            return "APR_2026_ONWARD"

    def determine_exchange_tariff_version(self, trade_date: date) -> str:
        if trade_date < self.BOUNDARY_MAR_2026_TARIFF:
            return "PRE_MARCH_2026_TARIFF"
        else:
            return "MARCH_2026_ONWARD_TARIFF"

    def calculate_cost(self, trade: TradeExecution) -> TradeCostBreakdown:
        qty = Decimal(str(trade.quantity_units))
        price = Decimal(str(trade.price))
        mult = Decimal(str(trade.multiplier))
        notional_d = qty * price * mult
        notional = float(notional_d)

        # Brokerage calculation
        if self.broker_pct > 0:
            brokerage_d = min(Decimal(str(self.brokerage_per_order)), notional_d * Decimal(str(self.broker_pct)))
        else:
            brokerage_d = Decimal(str(self.brokerage_per_order))

        # -------------------------------------------------------------
        # IFSC / GIFT City (NSE_IX) Tax & Fee Policy Evaluation
        # -------------------------------------------------------------
        if trade.exchange == ExchangeVenue.NSE_IX:
            if trade.ifsc_status == GiftCityEntityStatus.UNVERIFIED:
                return TradeCostBreakdown(
                    policy_version="GIFT_CITY_IFSC_UNVERIFIED",
                    notional_value=notional,
                    taxable_stt_base=0.0,
                    brokerage=float(brokerage_d),
                    stt=0.0,
                    stamp_duty=0.0,
                    exchange_charges=0.0,
                    ipft=0.0,
                    sebi_fee=0.0,
                    gst=0.0,
                    clearing_charges=0.0,
                    total_cost=float(brokerage_d),
                    effective_bps=float(round((float(brokerage_d) / max(1e-6, notional)) * 10000.0, 2)),
                    legal_classification_status="LEGAL_CLASSIFICATION_REQUIRED",
                )

            # Verified IFSC Unit
            # Exchange tariff on NSE IX: 1 bps standard schedule
            exch_fee_d = notional_d * Decimal("0.00010")
            total_cost_d = brokerage_d + exch_fee_d
            return TradeCostBreakdown(
                policy_version="GIFT_CITY_IFSC_EXEMPT",
                notional_value=notional,
                taxable_stt_base=0.0,
                brokerage=float(round(brokerage_d, 2)),
                stt=0.0,
                stamp_duty=0.0,
                exchange_charges=float(round(exch_fee_d, 2)),
                ipft=0.0,
                sebi_fee=0.0,
                gst=0.0,
                clearing_charges=0.0,
                total_cost=float(round(total_cost_d, 2)),
                effective_bps=float(round((float(total_cost_d) / max(1e-6, notional)) * 10000.0, 2)),
                legal_classification_status="CONFIRMED",
            )

        # -------------------------------------------------------------
        # Domestic Statutory STT Calculation
        # -------------------------------------------------------------
        stt_ver = self.determine_stt_policy_version(trade.trade_date)
        stt_d = Decimal("0.0")
        taxable_stt_base_d = Decimal("0.0")

        if trade.segment in (MarketSegment.FUTURES_INDEX, MarketSegment.FUTURES_STOCK):
            # STT on Futures is payable exclusively on the SELL leg
            if trade.side == OrderSide.SELL:
                taxable_stt_base_d = notional_d
                if stt_ver in ("PRE_SEPT_2019", "SEPT_2019_TO_SEPT_2024", "PRE_OCT_2024"):
                    stt_rate = Decimal("0.000125")  # 0.0125%
                elif stt_ver == "OCT_2024_TO_MAR_2026":
                    stt_rate = Decimal("0.000200")  # 0.02% (Finance Act 2024)
                else:  # APR_2026_ONWARD
                    stt_rate = Decimal("0.000500")  # 0.05% (Finance Act 2026)
                stt_d = taxable_stt_base_d * stt_rate

        elif trade.segment in (MarketSegment.OPTIONS_INDEX, MarketSegment.OPTIONS_STOCK):
            if trade.is_exercise:
                # Exercise STT: Levied exclusively on the PURCHASER (exercising buyer)
                if trade.side == OrderSide.BUY:
                    settle_d = Decimal(str(trade.settlement_price))
                    strike_d = Decimal(str(trade.strike_price))

                    if trade.option_type == OptionType.CALL:
                        intrinsic_per_unit = max(Decimal("0.0"), settle_d - strike_d)
                    elif trade.option_type == OptionType.PUT:
                        intrinsic_per_unit = max(Decimal("0.0"), strike_d - settle_d)
                    else:
                        raise ValueError(f"Unknown option type: {trade.option_type}")

                    if stt_ver == "PRE_SEPT_2019":
                        # Prior to Sept 1, 2019: Taxable base was Settlement Price * units (if in the money)
                        if intrinsic_per_unit > Decimal("0.0"):
                            taxable_stt_base_d = qty * settle_d * mult
                            stt_rate = Decimal("0.00125")  # 0.125%
                            stt_d = taxable_stt_base_d * stt_rate
                        else:
                            taxable_stt_base_d = Decimal("0.0")
                            stt_d = Decimal("0.0")
                    elif stt_ver in ("SEPT_2019_TO_SEPT_2024", "OCT_2024_TO_MAR_2026", "PRE_OCT_2024"):
                        # Sept 1, 2019 to March 31, 2026: Intrinsic value base at 0.125%
                        taxable_stt_base_d = qty * intrinsic_per_unit * mult
                        stt_rate = Decimal("0.00125")  # 0.125%
                        stt_d = taxable_stt_base_d * stt_rate
                    else:
                        # April 1, 2026 onward: Intrinsic value base at 0.15% (Finance Act 2026)
                        taxable_stt_base_d = qty * intrinsic_per_unit * mult
                        stt_rate = Decimal("0.00150")  # 0.150%
                        stt_d = taxable_stt_base_d * stt_rate
                else:
                    # Seller in option assignment does not pay exercise STT
                    taxable_stt_base_d = Decimal("0.0")
                    stt_d = Decimal("0.0")

            elif trade.side == OrderSide.SELL:
                # Regular non-exercise options trading: STT levied on premium turnover (SELL side only)
                taxable_stt_base_d = notional_d
                if stt_ver in ("PRE_SEPT_2019", "SEPT_2019_TO_SEPT_2024", "PRE_OCT_2024"):
                    stt_rate = Decimal("0.000625")  # 0.0625%
                elif stt_ver == "OCT_2024_TO_MAR_2026":
                    stt_rate = Decimal("0.001000")  # 0.100% (Finance Act 2024)
                else:  # APR_2026_ONWARD
                    stt_rate = Decimal("0.001500")  # 0.150% (Finance Act 2026)
                stt_d = taxable_stt_base_d * stt_rate

        elif trade.segment == MarketSegment.EQUITY_DELIVERY:
            taxable_stt_base_d = notional_d
            stt_d = taxable_stt_base_d * Decimal("0.001")  # 0.1% both sides
        elif trade.segment == MarketSegment.EQUITY_INTRADAY:
            if trade.side == OrderSide.SELL:
                taxable_stt_base_d = notional_d
                stt_d = taxable_stt_base_d * Decimal("0.00025")  # 0.025% on sell

        # -------------------------------------------------------------
        # Centralized Stamp Duty (Indian Stamp Rules 2019 - BUY Side Only)
        # -------------------------------------------------------------
        stamp_duty_d = Decimal("0.0")
        if trade.side == OrderSide.BUY and not trade.is_exercise:
            if trade.segment == MarketSegment.EQUITY_DELIVERY:
                stamp_duty_d = notional_d * Decimal("0.00015")  # 0.015%
            elif trade.segment == MarketSegment.EQUITY_INTRADAY:
                stamp_duty_d = notional_d * Decimal("0.00003")  # 0.003%
            elif trade.segment in (MarketSegment.FUTURES_INDEX, MarketSegment.FUTURES_STOCK):
                stamp_duty_d = notional_d * Decimal("0.00002")  # 0.002%
            elif trade.segment in (MarketSegment.OPTIONS_INDEX, MarketSegment.OPTIONS_STOCK):
                stamp_duty_d = notional_d * Decimal("0.00003")  # 0.003% on premium

        # -------------------------------------------------------------
        # Exchange Transaction Charges & IPFT
        # -------------------------------------------------------------
        tariff_ver = self.determine_exchange_tariff_version(trade.trade_date)
        exchange_charges_d = Decimal("0.0")
        ipft_d = Decimal("0.0")

        if trade.exchange == ExchangeVenue.BSE:
            if trade.segment in (MarketSegment.OPTIONS_INDEX, MarketSegment.OPTIONS_STOCK):
                exchange_charges_d = notional_d * Decimal("0.000325")  # BSE options
                ipft_d = notional_d * (Decimal("0.01") / Decimal("10000000.0"))             # Rs 0.01 / crore
            elif trade.segment in (MarketSegment.FUTURES_INDEX, MarketSegment.FUTURES_STOCK):
                exchange_charges_d = notional_d * Decimal("0.0000150") # BSE futures
                ipft_d = notional_d * (Decimal("0.01") / Decimal("10000000.0"))
            else:
                exchange_charges_d = notional_d * Decimal("0.0000275") # BSE cash
                ipft_d = notional_d * (Decimal("0.01") / Decimal("10000000.0"))
        else:
            # NSE Venue
            if tariff_ver == "MARCH_2026_ONWARD_TARIFF":
                # NSE Circular NSE/FA/73061 (Effective March 1, 2026)
                if trade.segment in (MarketSegment.OPTIONS_INDEX, MarketSegment.OPTIONS_STOCK):
                    exchange_charges_d = notional_d * (Decimal("3552.00") / Decimal("10000000.0"))  # Rs 3,552 per crore
                    ipft_d = notional_d * (Decimal("0.01") / Decimal("10000000.0"))                 # Rs 0.01 per crore
                elif trade.segment in (MarketSegment.FUTURES_INDEX, MarketSegment.FUTURES_STOCK):
                    exchange_charges_d = notional_d * (Decimal("182.99") / Decimal("10000000.0"))   # Rs 182.99 per crore
                    ipft_d = notional_d * (Decimal("0.01") / Decimal("10000000.0"))
                else:
                    # Cash
                    exchange_charges_d = notional_d * (Decimal("306.99") / Decimal("10000000.0"))   # Rs 306.99 per crore
                    ipft_d = notional_d * (Decimal("0.01") / Decimal("10000000.0"))
            else:
                # Pre-March 2026 Tariff
                if trade.segment in (MarketSegment.OPTIONS_INDEX, MarketSegment.OPTIONS_STOCK):
                    exchange_charges_d = notional_d * Decimal("0.0003503")  # Rs 3503 per crore
                    ipft_d = notional_d * Decimal("0.0000005")             # Rs 0.05 per crore
                elif trade.segment in (MarketSegment.FUTURES_INDEX, MarketSegment.FUTURES_STOCK):
                    exchange_charges_d = notional_d * Decimal("0.0000173")  # Rs 173 per crore
                    ipft_d = notional_d * Decimal("0.0000005")
                else:
                    exchange_charges_d = notional_d * Decimal("0.0000297")  # Rs 297 per crore
                    ipft_d = notional_d * Decimal("0.0000005")

        # -------------------------------------------------------------
        # SEBI Regulatory Turnover Fee (Rs 10 per crore = 0.000001)
        # -------------------------------------------------------------
        sebi_fee_d = notional_d * (Decimal("10.0") / Decimal("10000000.0"))

        # -------------------------------------------------------------
        # Goods and Services Tax (GST) (18% on Taxable Services)
        # Brokerage + Exchange Charges + SEBI Fees + IPFT
        # -------------------------------------------------------------
        taxable_services_d = brokerage_d + exchange_charges_d + sebi_fee_d + ipft_d
        gst_d = taxable_services_d * Decimal("0.18")

        clearing_charges_d = Decimal("0.0")
        total_cost_d = brokerage_d + stt_d + stamp_duty_d + exchange_charges_d + ipft_d + sebi_fee_d + gst_d + clearing_charges_d

        def round_currency(val: Decimal) -> float:
            return float(val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

        return TradeCostBreakdown(
            policy_version=stt_ver,
            notional_value=notional,
            taxable_stt_base=round_currency(taxable_stt_base_d),
            brokerage=round_currency(brokerage_d),
            stt=round_currency(stt_d),
            stamp_duty=round_currency(stamp_duty_d),
            exchange_charges=round_currency(exchange_charges_d),
            ipft=float(ipft_d.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)),
            sebi_fee=float(sebi_fee_d.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)),
            gst=round_currency(gst_d),
            clearing_charges=float(clearing_charges_d),
            total_cost=round_currency(total_cost_d),
            effective_bps=float(round((float(total_cost_d) / max(1e-6, notional)) * 10000.0, 2)),
            legal_classification_status="CONFIRMED",
        )


# Backward compatibility aliases
IndiaTransactionCostCalculator = IndiaStatutoryCostEngine


# -------------------------------------------------------------
# Complete NSE Order-to-Trade Ratio (OTR) Framework
# -------------------------------------------------------------

@dataclass
class OrderOperationEvent:
    timestamp: datetime
    trading_member_id: str
    algo_id: str
    exchange: ExchangeVenue
    segment: MarketSegment
    event_type: str  # "place", "modify", "cancel", "trade"
    price: float = 0.0
    ltp: float = 0.0
    quantity: int = 1


@dataclass
class DailySessionOrderStats:
    session_date: date
    orders_placed: int = 0
    orders_modified: int = 0
    orders_cancelled: int = 0
    trades_executed: int = 0
    excluded_operations: int = 0

    @property
    def total_order_operations(self) -> int:
        return self.orders_placed + self.orders_modified + self.orders_cancelled

    @property
    def net_order_operations(self) -> int:
        return max(0, self.total_order_operations - self.excluded_operations)

    @property
    def otr(self) -> float:
        return float(self.net_order_operations / max(1, self.trades_executed))


class NseBseSurveillanceOtrController:
    """
    NSE & BSE Consolidated Algorithmic Trading Surveillance Controller.
    Implements:
    - Official NSE Circular NSE/SURV/47980, amended Feb 6, 2026 and April 2, 2026.
    - True NSE Trading Calendar (30 actual trading sessions, skipping weekends/holidays).
    - April 6, 2026 Options Segment Exclusion:
      Orders within +/-40% of LTP premium or Rs 20 (whichever higher) are excluded.
    - Broker-specific pre-trade risk controls and internal strategy throttles.
    """
    APRIL_2026_EXCLUSION_DATE = date(2026, 4, 6)

    def __init__(
        self,
        internal_strategy_throttle_otr: float = 50.0,
        exchange: ExchangeVenue = ExchangeVenue.NSE,
    ):
        self.internal_strategy_throttle_otr = internal_strategy_throttle_otr
        self.exchange = exchange
        self.session_history: Dict[date, DailySessionOrderStats] = {}
        self.current_date: date = date.today()

    def _get_current_session(self, sess_date: Optional[date] = None) -> DailySessionOrderStats:
        target_date = sess_date or self.current_date
        if target_date not in self.session_history:
            self.session_history[target_date] = DailySessionOrderStats(session_date=target_date)
        return self.session_history[target_date]

    def record_order_event(
        self,
        event_type: str,
        event_date: Optional[date] = None,
        segment: MarketSegment = MarketSegment.OPTIONS_INDEX,
        price: float = 0.0,
        ltp: float = 0.0,
    ):
        if event_date:
            self.current_date = event_date
        sess = self._get_current_session(self.current_date)

        # Check April 6, 2026 Options Exclusion Rule
        is_excluded = False
        if (
            self.exchange == ExchangeVenue.NSE
            and self.current_date >= self.APRIL_2026_EXCLUSION_DATE
            and segment in (MarketSegment.OPTIONS_INDEX, MarketSegment.OPTIONS_STOCK)
            and ltp > 0
            and price > 0
        ):
            threshold_distance = max(0.40 * ltp, 20.0)
            if abs(price - ltp) <= threshold_distance:
                is_excluded = True

        if event_type == "place":
            sess.orders_placed += 1
            if is_excluded:
                sess.excluded_operations += 1
        elif event_type == "modify":
            sess.orders_modified += 1
            if is_excluded:
                sess.excluded_operations += 1
        elif event_type == "cancel":
            sess.orders_cancelled += 1
            if is_excluded:
                sess.excluded_operations += 1
        elif event_type == "trade":
            sess.trades_executed += 1

    def check_nse_official_surveillance_breach(self) -> Tuple[bool, int, List[date]]:
        """
        Evaluates official NSE OTR Surveillance Condition:
        >= 3 instances of net OTR >= 2,000 on trading sessions within previous 30 TRADING DAYS
        using the real NseTradingCalendar (minimum 500 net operations).
        If recorded sessions include non-trading calendar dates (e.g., weekend mock/special sessions),
        those active sessions within the calendar span of 30 trading days are also accounted for.
        """
        if self.exchange != ExchangeVenue.NSE:
            return False, 0, []

        trailing_trading_days = NseTradingCalendar.get_trailing_trading_days(self.current_date, count=30)
        earliest_trading_day = min(trailing_trading_days) if trailing_trading_days else self.current_date

        relevant_sessions = [
            s for d, s in self.session_history.items()
            if d >= earliest_trading_day and d <= self.current_date and s.net_order_operations >= 500
        ]

        violation_dates = sorted([s.session_date for s in relevant_sessions if s.otr >= 2000.0])
        count = len(violation_dates)
        is_breached = count >= 3
        return is_breached, count, violation_dates

    def check_internal_strategy_throttle(self) -> bool:
        sess = self._get_current_session()
        if sess.total_order_operations > 100 and sess.otr > self.internal_strategy_throttle_otr:
            return True
        return False


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
