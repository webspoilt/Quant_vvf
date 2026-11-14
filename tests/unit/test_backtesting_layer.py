"""
Unit Tests for Strategy Engine and Event-Driven Backtesting.
"""

import numpy as np
import pandas as pd
import pytest

from strategies.pairs_strategy import PairsTradingStrategy
from strategies.base_strategy import SignalType
from backtesting.engine import InstitutionalBacktestingEngine, BacktestSummary


class TestStrategyAndBacktestingLayer:

    def test_pairs_trading_strategy_signals(self):
        strat = PairsTradingStrategy(
            strategy_id="PAIRS-BTC-ETH",
            hedge_ratio=0.05,
            entry_z_threshold=2.0,
            exit_z_threshold=0.5,
            target_clip_btc=1.0
        )

        df_high_z = pd.DataFrame({"spread_z": [2.5], "price": [65000.0]})
        sig_sell = strat.generate_signal(df_high_z)
        assert sig_sell.signal_type == SignalType.SELL

        df_low_z = pd.DataFrame({"spread_z": [-2.5], "price": [65000.0]})
        sig_buy = strat.generate_signal(df_low_z)
        assert sig_buy.signal_type == SignalType.BUY

        df_mean_z = pd.DataFrame({"spread_z": [0.2], "price": [65000.0]})
        sig_hold = strat.generate_signal(df_mean_z)
        assert sig_hold.signal_type == SignalType.HOLD

    def test_event_driven_backtest_execution(self):
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        spot_prices = pd.Series(60000.0 + 1000.0 * np.sin(np.linspace(0, 10, n)), index=dates)

        # Oscillating Z-score to generate trades
        spread_z = pd.Series(2.5 * np.sin(np.linspace(0, 15, n)), index=dates)
        feature_df = pd.DataFrame({"spread_z": spread_z, "price": spot_prices})

        strat = PairsTradingStrategy(
            strategy_id="TEST-STRAT",
            hedge_ratio=0.05,
            entry_z_threshold=1.5,
            exit_z_threshold=0.3,
            target_clip_btc=0.5
        )

        engine = InstitutionalBacktestingEngine(
            initial_capital_usd=100_000.0,
            fee_rate=0.0005,
            slippage_bps=2.0
        )

        summary = engine.run_backtest(strat, spot_prices, feature_df)
        assert isinstance(summary, BacktestSummary)
        assert len(summary.equity_curve) == n
        assert summary.total_trades > 0
        assert summary.total_fees_paid_usd > 0.0
        assert np.isfinite(summary.sharpe_ratio)
        assert 0.0 <= summary.max_drawdown_pct <= 100.0
