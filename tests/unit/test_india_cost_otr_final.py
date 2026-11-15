"""
Comprehensive Verification and Independent Validation Suite for:
- Phase 1: Explicit Option Type (CALL/PUT) and Exercise STT.
- Phase 2: Historical Exercise Tax Schedule (Pre-Sept 2019, Sept 2019-Sept 2024, Oct 2024-Mar 2026, Apr 2026+).
- Phase 3: NSE Transaction Charges and Separate IPFT (Circular NSE/FA/73061 March 1, 2026 vs Pre-March 2026).
- Phase 4: Exchange Cost Engine Input Integrity (Lots, Multipliers, Negative Price Rejections, Decimal Precision).
- Phase 5: Complete NSE OTR Surveillance Framework (Real Trading Calendar, Weekend/Holiday Skips, April 6 2026 Option Exclusions).
- Phase 6: GIFT City / NSE IX Entity Classification & Legal Fallback.
- Phase 7: Cross-Exchange Multi-State Execution Lifecycle & Clock Skew Validation.
- Phase 8: SABR Bid/Ask Diagnostics & Non-Clipped Dispersion Feasibility.
"""

from datetime import date, datetime, timedelta
import math
import numpy as np
import pytest

from india.costs.tax_engine import (
    IndiaStatutoryCostEngine,
    TradeExecution,
    TradeCostBreakdown,
    ExchangeVenue,
    MarketSegment,
    OrderSide,
    OptionType,
    GiftCityEntityStatus,
    NseTradingCalendar,
    NseBseSurveillanceOtrController,
)
from india.cross_exchange.cross_exchange_engine import (
    CrossExchangeArbitrageEngine,
    DualListedSpreadQuote,
    ExecutionOpportunityState,
)
from india.derivatives.indian_options import (
    SabrVolatilityModel,
    IndexDispersionAnalyzer,
)


class TestPhase1AndPhase2OptionExerciseAndHistoricalTax:
    """Tests Phase 1 & Phase 2: Option Type, Exercise STT and Historical Tax Schedules."""

    def test_itm_put_exercise_april_2026(self):
        """
        NIFTY 24,000 PUT. Settlement = 23,800. Quantity = 100 units.
        Intrinsic = 200 INR. Taxable base = 20,000 INR. STT @ 0.15% = 30.00 INR.
        Must not return zero.
        """
        engine = IndiaStatutoryCostEngine(brokerage_per_order=20.0)
        trade = TradeExecution(
            exchange=ExchangeVenue.NSE,
            segment=MarketSegment.OPTIONS_INDEX,
            side=OrderSide.BUY,  # Exercising purchaser
            quantity_units=100,
            price=0.0,
            strike_price=24000.0,
            settlement_price=23800.0,
            option_type=OptionType.PUT,
            is_exercise=True,
            trade_date=date(2026, 4, 15),
        )
        cost = engine.calculate_cost(trade)
        assert cost.taxable_stt_base == 20000.00
        assert cost.stt == 30.00
        assert cost.policy_version == "APR_2026_ONWARD"

    def test_itm_call_exercise_april_2026(self):
        """
        NIFTY 24,000 CALL. Settlement = 24,200. Quantity = 100 units.
        Intrinsic = 200 INR. Taxable base = 20,000 INR. STT @ 0.15% = 30.00 INR.
        """
        engine = IndiaStatutoryCostEngine(brokerage_per_order=20.0)
        trade = TradeExecution(
            exchange=ExchangeVenue.NSE,
            segment=MarketSegment.OPTIONS_INDEX,
            side=OrderSide.BUY,
            quantity_units=100,
            price=0.0,
            strike_price=24000.0,
            settlement_price=24200.0,
            option_type=OptionType.CALL,
            is_exercise=True,
            trade_date=date(2026, 4, 15),
        )
        cost = engine.calculate_cost(trade)
        assert cost.taxable_stt_base == 20000.00
        assert cost.stt == 30.00

    def test_otm_and_atm_call_exercise(self):
        """OTM and ATM Call exercise intrinsic must be zero, STT must be zero."""
        engine = IndiaStatutoryCostEngine()
        # OTM Call
        trade_otm = TradeExecution(
            exchange=ExchangeVenue.NSE,
            segment=MarketSegment.OPTIONS_INDEX,
            side=OrderSide.BUY,
            quantity_units=100,
            price=0.0,
            strike_price=24500.0,
            settlement_price=24200.0,
            option_type=OptionType.CALL,
            is_exercise=True,
            trade_date=date(2026, 4, 15),
        )
        assert engine.calculate_cost(trade_otm).stt == 0.0

        # ATM Call
        trade_atm = TradeExecution(
            exchange=ExchangeVenue.NSE,
            segment=MarketSegment.OPTIONS_INDEX,
            side=OrderSide.BUY,
            quantity_units=100,
            price=0.0,
            strike_price=24200.0,
            settlement_price=24200.0,
            option_type=OptionType.CALL,
            is_exercise=True,
            trade_date=date(2026, 4, 15),
        )
        assert engine.calculate_cost(trade_atm).stt == 0.0

    def test_otm_and_atm_put_exercise(self):
        """OTM and ATM Put exercise intrinsic must be zero, STT must be zero."""
        engine = IndiaStatutoryCostEngine()
        # OTM Put
        trade_otm = TradeExecution(
            exchange=ExchangeVenue.NSE,
            segment=MarketSegment.OPTIONS_INDEX,
            side=OrderSide.BUY,
            quantity_units=100,
            price=0.0,
            strike_price=23500.0,
            settlement_price=23800.0,
            option_type=OptionType.PUT,
            is_exercise=True,
            trade_date=date(2026, 4, 15),
        )
        assert engine.calculate_cost(trade_otm).stt == 0.0

        # ATM Put
        trade_atm = TradeExecution(
            exchange=ExchangeVenue.NSE,
            segment=MarketSegment.OPTIONS_INDEX,
            side=OrderSide.BUY,
            quantity_units=100,
            price=0.0,
            strike_price=23800.0,
            settlement_price=23800.0,
            option_type=OptionType.PUT,
            is_exercise=True,
            trade_date=date(2026, 4, 15),
        )
        assert engine.calculate_cost(trade_atm).stt == 0.0

    def test_exercise_by_seller_assumed_zero_stt(self):
        """STT on exercise is payable exclusively by purchaser (BUY side), not writer (SELL side)."""
        engine = IndiaStatutoryCostEngine()
        trade = TradeExecution(
            exchange=ExchangeVenue.NSE,
            segment=MarketSegment.OPTIONS_INDEX,
            side=OrderSide.SELL,
            quantity_units=100,
            price=0.0,
            strike_price=24000.0,
            settlement_price=24200.0,
            option_type=OptionType.CALL,
            is_exercise=True,
            trade_date=date(2026, 4, 15),
        )
        cost = engine.calculate_cost(trade)
        assert cost.stt == 0.0
        assert cost.taxable_stt_base == 0.0

    def test_pre_september_2019_historical_settlement_base(self):
        """
        Prior to Sept 1, 2019, ITM exercised options were taxed on full settlement value:
        100 units * 24,200 settlement = 2,420,000 INR taxable base.
        STT @ 0.125% = 3,025.00 INR.
        """
        engine = IndiaStatutoryCostEngine()
        trade_pre_2019 = TradeExecution(
            exchange=ExchangeVenue.NSE,
            segment=MarketSegment.OPTIONS_INDEX,
            side=OrderSide.BUY,
            quantity_units=100,
            price=0.0,
            strike_price=24000.0,
            settlement_price=24200.0,
            option_type=OptionType.CALL,
            is_exercise=True,
            trade_date=date(2019, 8, 15),  # Prior to Sept 1, 2019
        )
        cost = engine.calculate_cost(trade_pre_2019)
        assert cost.policy_version == "PRE_SEPT_2019"
        assert cost.taxable_stt_base == 2420000.00
        assert cost.stt == 3025.00

        # Post Sept 1, 2019: Taxed on intrinsic (20,000 base) @ 0.125% = 25.00 INR
        trade_post_2019 = TradeExecution(
            exchange=ExchangeVenue.NSE,
            segment=MarketSegment.OPTIONS_INDEX,
            side=OrderSide.BUY,
            quantity_units=100,
            price=0.0,
            strike_price=24000.0,
            settlement_price=24200.0,
            option_type=OptionType.CALL,
            is_exercise=True,
            trade_date=date(2019, 9, 15),
        )
        cost_post = engine.calculate_cost(trade_post_2019)
        assert cost_post.policy_version == "SEPT_2019_TO_SEPT_2024"
        assert cost_post.taxable_stt_base == 20000.00
        assert cost_post.stt == 25.00

    def test_october_2024_to_march_2026_schedule(self):
        """Tests rates under Finance Act 2024 schedule."""
        engine = IndiaStatutoryCostEngine()
        # Futures sell leg: 0.02%
        fut_sell = TradeExecution(
            exchange=ExchangeVenue.NSE,
            segment=MarketSegment.FUTURES_INDEX,
            side=OrderSide.SELL,
            quantity_units=50,
            price=24000.0,
            trade_date=date(2025, 6, 1),
        )
        assert engine.calculate_cost(fut_sell).stt == 240.00  # 1.2M * 0.02%

        # Options premium sell leg: 0.10%
        opt_sell = TradeExecution(
            exchange=ExchangeVenue.NSE,
            segment=MarketSegment.OPTIONS_INDEX,
            side=OrderSide.SELL,
            quantity_units=100,
            price=300.0,
            trade_date=date(2025, 6, 1),
        )
        assert engine.calculate_cost(opt_sell).stt == 30.00  # 30,000 * 0.10%


class TestPhase3AndPhase4TariffsAndInputIntegrity:
    """Tests Phase 3 & Phase 4: NSE Tariff Circular NSE/FA/73061, IPFT, BSE, and Input Integrity."""

    def test_nse_march_2026_tariff_circular_73061(self):
        """
        Validates NSE schedule effective March 1, 2026:
        - Cash: Rs 306.99 / crore, IPFT: Rs 0.01 / crore
        - Equity Futures: Rs 182.99 / crore, IPFT: Rs 0.01 / crore
        - Equity Options: Rs 3,552 / crore on premium, IPFT: Rs 0.01 / crore
        """
        engine = IndiaStatutoryCostEngine(brokerage_per_order=0.0)
        # 1 Crore = 10,000,000 INR
        trade_cash = TradeExecution(
            exchange=ExchangeVenue.NSE,
            segment=MarketSegment.EQUITY_DELIVERY,
            side=OrderSide.BUY,
            quantity_units=10000,
            price=1000.0,  # 1 Crore turnover
            trade_date=date(2026, 3, 15),
        )
        cost_cash = engine.calculate_cost(trade_cash)
        assert cost_cash.exchange_charges == 306.99
        assert cost_cash.ipft == 0.01
        assert cost_cash.sebi_fee == 10.00

        trade_fut = TradeExecution(
            exchange=ExchangeVenue.NSE,
            segment=MarketSegment.FUTURES_INDEX,
            side=OrderSide.BUY,
            quantity_units=10000,
            price=1000.0,
            trade_date=date(2026, 3, 15),
        )
        cost_fut = engine.calculate_cost(trade_fut)
        assert cost_fut.exchange_charges == 182.99
        assert cost_fut.ipft == 0.01

        trade_opt = TradeExecution(
            exchange=ExchangeVenue.NSE,
            segment=MarketSegment.OPTIONS_INDEX,
            side=OrderSide.BUY,
            quantity_units=100000,
            price=100.0,  # 1 Crore premium turnover
            trade_date=date(2026, 3, 15),
        )
        cost_opt = engine.calculate_cost(trade_opt)
        assert cost_opt.exchange_charges == 3552.00
        assert cost_opt.ipft == 0.01

    def test_bse_separate_fee_schedule(self):
        """BSE options charges Rs 3,250 / crore, Futures Rs 150 / crore."""
        engine = IndiaStatutoryCostEngine(brokerage_per_order=0.0)
        trade_bse_opt = TradeExecution(
            exchange=ExchangeVenue.BSE,
            segment=MarketSegment.OPTIONS_INDEX,
            side=OrderSide.BUY,
            quantity_units=100000,
            price=100.0,
            trade_date=date(2026, 3, 15),
        )
        cost = engine.calculate_cost(trade_bse_opt)
        assert cost.exchange_charges == 3250.00
        assert cost.ipft == 0.01

    def test_input_integrity_and_rejections(self):
        """Rejects negative prices, negative quantities, missing strikes on exercise, etc."""
        with pytest.raises(ValueError, match="Quantity units must be positive"):
            TradeExecution(quantity_units=-10, price=100.0)

        with pytest.raises(ValueError, match="Trade price must be finite"):
            TradeExecution(quantity_units=10, price=-50.0)

        with pytest.raises(ValueError, match="strike_price is strictly required"):
            TradeExecution(
                segment=MarketSegment.OPTIONS_INDEX,
                quantity_units=10,
                price=10.0,
                is_exercise=True,
                option_type=OptionType.CALL,
                settlement_price=100.0,
            )


class TestPhase5AndPhase6OTRAndGiftCity:
    """Tests Phase 5 & Phase 6: Real Calendar OTR, April 6 Option Exclusions, and IFSC Policies."""

    def test_nse_trading_calendar_holidays_and_weekends(self):
        """NseTradingCalendar correctly identifies weekends and statutory holidays."""
        # 2026 Republic Day: Jan 26, 2026 (Monday) is holiday
        assert not NseTradingCalendar.is_trading_day(date(2026, 1, 26))
        # 2026 Good Friday: April 3, 2026 is holiday
        assert not NseTradingCalendar.is_trading_day(date(2026, 4, 3))
        # Sunday check
        assert not NseTradingCalendar.is_trading_day(date(2026, 4, 5))
        # Normal Tuesday
        assert NseTradingCalendar.is_trading_day(date(2026, 4, 7))

    def test_april_6_2026_options_price_exclusion_rule(self):
        """
        Orders within +/-40% of LTP premium or Rs 20 (whichever higher) are excluded from OTR.
        LTP = 100. Distance = max(40, 20) = 40. Orders between 60 and 140 are excluded.
        """
        controller = NseBseSurveillanceOtrController(exchange=ExchangeVenue.NSE)
        test_date = date(2026, 4, 15)

        # 3,000 orders placed at price 110 (within +/-40)
        for _ in range(3000):
            controller.record_order_event(
                "place",
                event_date=test_date,
                segment=MarketSegment.OPTIONS_INDEX,
                price=110.0,
                ltp=100.0,
            )
        controller.record_order_event("trade", event_date=test_date)

        sess = controller._get_current_session(test_date)
        assert sess.total_order_operations == 3000
        assert sess.excluded_operations == 3000
        assert sess.net_order_operations == 0
        assert sess.otr == 0.0

    def test_gift_city_unverified_entity_legal_classification_required(self):
        """Unverified IFSC entity must return LEGAL_CLASSIFICATION_REQUIRED."""
        engine = IndiaStatutoryCostEngine()
        trade = TradeExecution(
            exchange=ExchangeVenue.NSE_IX,
            segment=MarketSegment.FUTURES_INDEX,
            quantity_units=100,
            price=2000.0,
            currency="USD",
            ifsc_status=GiftCityEntityStatus.UNVERIFIED,
        )
        cost = engine.calculate_cost(trade)
        assert cost.legal_classification_status == "LEGAL_CLASSIFICATION_REQUIRED"

        # Verified IFSC Registered Unit
        trade_verified = TradeExecution(
            exchange=ExchangeVenue.NSE_IX,
            segment=MarketSegment.FUTURES_INDEX,
            quantity_units=100,
            price=2000.0,
            currency="USD",
            ifsc_status=GiftCityEntityStatus.IFSC_REGISTERED_UNIT,
        )
        cost_v = engine.calculate_cost(trade_verified)
        assert cost_v.legal_classification_status == "CONFIRMED"
        assert cost_v.stt == 0.0
        assert cost_v.stamp_duty == 0.0


class TestPhase7AndPhase8CrossExchangeAndSABR:
    """Tests Phase 7 & Phase 8: Cross-Exchange Lifecycle States and SABR/Dispersion."""

    def test_cross_exchange_clock_skew_and_zero_timestamps(self):
        """Zero timestamps and excessive clock skew return REJECTED state."""
        engine = CrossExchangeArbitrageEngine(max_latency_skew_ms=50)

        # Zero timestamps
        quote_zero = DualListedSpreadQuote(
            symbol="RELIANCE",
            nse_bid=2505.0,
            nse_ask=2506.0,
            bse_bid=2500.0,
            bse_ask=2501.0,
            nse_timestamp_ms=0,
            bse_timestamp_ms=0,
        )
        res = engine.analyze_spread(quote_zero)
        assert res["state"] == ExecutionOpportunityState.REJECTED.value
        assert not res["is_executable"]

        # Excessive clock skew (60ms > 50ms)
        quote_skew = DualListedSpreadQuote(
            symbol="RELIANCE",
            nse_bid=2505.0,
            nse_ask=2506.0,
            bse_bid=2500.0,
            bse_ask=2501.0,
            nse_timestamp_ms=1000000,
            bse_timestamp_ms=1000060,
        )
        res_skew = engine.analyze_spread(quote_skew)
        assert res_skew["state"] == ExecutionOpportunityState.REJECTED.value
        assert "clock skew" in res_skew["reason"].lower()

    def test_cross_exchange_net_edge_vs_execution_ready(self):
        """Crossed book with profitable edge returns EXECUTION_READY_AFTER_LIVE_CHECKS."""
        engine = CrossExchangeArbitrageEngine(max_latency_skew_ms=50)
        quote = DualListedSpreadQuote(
            symbol="TCS",
            nse_bid=4020.0,
            nse_ask=4021.0,
            nse_bid_qty=500,
            bse_bid=3990.0,
            bse_ask=3995.0,
            bse_ask_qty=500,
            nse_timestamp_ms=1000000,
            bse_timestamp_ms=1000010,
        )
        res = engine.analyze_spread(quote)
        assert res["state"] == ExecutionOpportunityState.EXECUTION_READY_AFTER_LIVE_CHECKS.value
        assert res["is_executable"]
        assert not res["atomic_guarantee"]

    def test_sabr_calibration_diagnostics(self):
        """SABR calibration returns full numerical diagnostics (RMSE, objective, residuals)."""
        strikes = np.array([21500, 22000, 22500, 23000, 23500])
        market_ivs = np.array([0.22, 0.19, 0.17, 0.165, 0.175])
        diag = SabrVolatilityModel.calibrate_with_diagnostics(
            f=22500.0,
            strikes=strikes,
            market_ivs=market_ivs,
            t_expiry=0.1,
            fixed_beta=1.0,
        )
        assert diag.success
        assert diag.rmse < 0.02
        assert len(diag.residuals) == len(strikes)

    def test_dispersion_infeasibility_reporting(self):
        """Infeasible dispersion returns ValueError when allow_clip=False."""
        constituent_ivs = np.array([0.15, 0.14, 0.16])
        constituent_weights = np.array([0.33, 0.33, 0.34])

        # Grossly excessive index IV creates infeasible implied correlation > 1.0
        with pytest.raises(ValueError, match="MODEL_INCONSISTENCY"):
            IndexDispersionAnalyzer.calculate_implied_correlation(
                index_iv=0.45,
                constituent_ivs=constituent_ivs,
                constituent_weights=constituent_weights,
                allow_clip=False,
            )
