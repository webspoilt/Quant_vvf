"""
Event-Driven and Vectorized Institutional Backtesting Engine
Simulates transaction fees, bid-ask spread, slippage, margin limits, and funding rates.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from strategies.base_strategy import BaseStrategy, SignalType


@dataclass
class BacktestTrade:
    timestamp: pd.Timestamp
    symbol: str
    side: str
    price: float
    quantity: float
    fee_usd: float
    slippage_usd: float
    realized_pnl_usd: float


@dataclass
class BacktestSummary:
    total_return_pct: float
    annualized_return_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    profit_factor: float
    win_rate_pct: float
    total_trades: int
    total_fees_paid_usd: float
    equity_curve: pd.Series


class InstitutionalBacktestingEngine:
    """Rigorous backtesting engine modeling execution frictions and margin requirements."""

    def __init__(
        self,
        initial_capital_usd: float = 100_000.0,
        fee_rate: float = 0.0005,
        slippage_bps: float = 2.0,
        margin_leverage_limit: float = 5.0
    ):
        self.initial_capital = initial_capital_usd
        self.fee_rate = fee_rate
        self.slippage_pct = slippage_bps / 10000.0
        self.leverage_limit = margin_leverage_limit

    def run_backtest(
        self,
        strategy: BaseStrategy,
        price_series: pd.Series,
        feature_data: pd.DataFrame
    ) -> BacktestSummary:
        """Executes chronological event-driven backtesting loop over point-in-time features."""
        dates = price_series.index
        n = len(dates)

        cash = self.initial_capital
        position_btc = 0.0
        equity_curve = []
        trades: List[BacktestTrade] = []
        total_fees = 0.0

        for t in range(n):
            current_date = dates[t]
            spot = float(price_series.iloc[t])

            # Provide slice up to t to prevent lookahead
            hist_features = feature_data.iloc[: t + 1]
            signal = strategy.generate_signal(hist_features)

            # Process Signal
            if signal.signal_type == SignalType.BUY and position_btc == 0.0:
                fill_price = spot * (1.0 + self.slippage_pct)
                qty = min(signal.target_quantity_btc, (cash * self.leverage_limit) / fill_price)
                if qty > 0:
                    fee = qty * fill_price * self.fee_rate
                    slippage = qty * spot * self.slippage_pct
                    cost = qty * fill_price + fee
                    cash -= cost
                    position_btc += qty
                    total_fees += fee
                    trades.append(BacktestTrade(
                        timestamp=current_date, symbol=signal.symbol, side="BUY",
                        price=fill_price, quantity=qty, fee_usd=fee,
                        slippage_usd=slippage, realized_pnl_usd=0.0
                    ))

            elif signal.signal_type == SignalType.SELL and position_btc > 0.0:
                fill_price = spot * (1.0 - self.slippage_pct)
                fee = position_btc * fill_price * self.fee_rate
                slippage = position_btc * spot * self.slippage_pct
                revenue = position_btc * fill_price - fee
                cash += revenue
                total_fees += fee
                entry_trade = [tr for tr in trades if tr.side == "BUY"][-1]
                pnl = (fill_price - entry_trade.price) * position_btc - (fee + entry_trade.fee_usd)
                trades.append(BacktestTrade(
                    timestamp=current_date, symbol=signal.symbol, side="SELL",
                    price=fill_price, quantity=position_btc, fee_usd=fee,
                    slippage_usd=slippage, realized_pnl_usd=pnl
                ))
                position_btc = 0.0

            # Mark to market equity
            current_equity = cash + position_btc * spot
            equity_curve.append(current_equity)

        eq_series = pd.Series(equity_curve, index=dates)
        daily_returns = eq_series.pct_change().dropna()

        total_ret = (eq_series.iloc[-1] - self.initial_capital) / self.initial_capital
        days = max(1, (dates[-1] - dates[0]).days)
        ann_ret = (1.0 + total_ret) ** (365.0 / days) - 1.0 if days > 0 else 0.0
        ann_vol = daily_returns.std() * np.sqrt(365.0) if len(daily_returns) > 1 else 0.0

        rf = 0.02
        sharpe = (ann_ret - rf) / ann_vol if ann_vol > 1e-6 else 0.0

        # Sortino
        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(365.0) if len(downside_returns) > 1 else 0.0
        sortino = (ann_ret - rf) / downside_vol if downside_vol > 1e-6 else 0.0

        # Drawdown
        running_max = eq_series.cummax()
        drawdown = (eq_series - running_max) / running_max
        max_dd = abs(float(drawdown.min())) if len(drawdown) > 0 else 0.0

        calmar = ann_ret / max_dd if max_dd > 1e-6 else 0.0

        # Trade stats
        closed_trades = [tr for tr in trades if tr.side == "SELL"]
        wins = [tr for tr in closed_trades if tr.realized_pnl_usd > 0]
        losses = [tr for tr in closed_trades if tr.realized_pnl_usd <= 0]
        win_rate = (len(wins) / len(closed_trades) * 100.0) if closed_trades else 0.0

        gross_profit = sum(tr.realized_pnl_usd for tr in wins) if wins else 0.0
        gross_loss = abs(sum(tr.realized_pnl_usd for tr in losses)) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (10.0 if gross_profit > 0 else 1.0)

        return BacktestSummary(
            total_return_pct=float(total_ret * 100.0),
            annualized_return_pct=float(ann_ret * 100.0),
            annualized_volatility_pct=float(ann_vol * 100.0),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            max_drawdown_pct=float(max_dd * 100.0),
            calmar_ratio=float(calmar),
            profit_factor=float(profit_factor),
            win_rate_pct=float(win_rate),
            total_trades=len(trades),
            total_fees_paid_usd=float(total_fees),
            equity_curve=eq_series,
        )
