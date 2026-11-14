"""
Integration Tests for Broker Adapters, Database Persistence, and Health Monitoring.
"""

import time
import pytest

from integrations.paper_trading_adapter import PaperTradingAdapter
from integrations.deribit_adapter import DeribitLiveAdapter
from integrations.base_adapter import OrderSide
from database.repository import DatabaseRepository
from monitoring.system_health import SystemHealthMonitor, HealthStatus


class TestPlatformIntegrationLayer:

    def test_deribit_live_adapter_safety_guards(self):
        # 1. Testnet works without explicit production authorization
        adapter_testnet = DeribitLiveAdapter(use_testnet=True)
        assert adapter_testnet.connect() is True
        ord_testnet = adapter_testnet.submit_order("BTC-PERPETUAL", OrderSide.BUY, 65000.0, 0.1, "CL-TEST")
        assert ord_testnet.status == "OPEN"

        # 2. Production mode without DERIBIT_LIVE_AUTHORIZED=TRUE must FAIL CLOSED
        adapter_prod = DeribitLiveAdapter(use_testnet=False)
        assert adapter_prod.connect() is True
        with pytest.raises(PermissionError, match="DERIBIT_LIVE_AUTHORIZED=TRUE"):
            adapter_prod.submit_order("BTC-PERPETUAL", OrderSide.BUY, 65000.0, 0.1, "CL-PROD")

    def test_paper_trading_adapter_flow(self):
        adapter = PaperTradingAdapter(initial_cash_usd=100_000.0)
        assert adapter.connect() is True

        order = adapter.submit_order(
            symbol="BTC-PERPETUAL",
            side=OrderSide.BUY,
            price=65000.0,
            quantity=0.50,
            client_order_id="CL-TEST-001"
        )
        assert order.status == "FILLED"
        assert order.filled_quantity == 0.50

        pos = adapter.get_position("BTC-PERPETUAL")
        assert pos is not None
        assert pos.size_btc == 0.50
        assert pos.entry_price_usd == 65000.0

        all_pos = adapter.reconcile_positions()
        assert len(all_pos) == 1

    def test_database_repository_persistence(self):
        repo = DatabaseRepository(":memory:")
        now = time.time_ns()

        repo.insert_order("ORD-1", "CL-1", "BTC-PERPETUAL", "BUY", 65000.0, 1.0, "PENDING", now)
        order_row = repo.get_order("ORD-1")
        assert order_row is not None
        assert order_row["symbol"] == "BTC-PERPETUAL"
        assert order_row["quantity"] == 1.0

        repo.insert_execution("EXEC-1", "ORD-1", "BTC-PERPETUAL", "BUY", 65000.0, 1.0, 32.5, now)
        repo.record_journal_entry("TRD-1", "VRP-01", "BTC-PERPETUAL", "CALM", 64000.0, 65000.0, 1000.0, 32.5, 5.0, now)

        journals = repo.get_all_journal_entries()
        assert len(journals) == 1
        assert journals[0]["realized_pnl_usd"] == 1000.0

    def test_system_health_monitor(self):
        monitor = SystemHealthMonitor(max_feed_staleness_ms=200.0)
        now_ns = time.time_ns()

        # Healthy state
        status_ok = monitor.evaluate_health(
            last_packet_timestamp_ns=now_ns - 50_000_000,  # 50ms ago
            current_queue_depth=100,
            queue_capacity=1000,
            circuit_breaker_tripped=False
        )
        assert status_ok.is_healthy is True
        assert status_ok.risk_gate_open is True

        # Stale feed state
        status_stale = monitor.evaluate_health(
            last_packet_timestamp_ns=now_ns - 300_000_000,  # 300ms ago (> 200ms limit)
            current_queue_depth=100,
            queue_capacity=1000,
            circuit_breaker_tripped=False
        )
        assert status_stale.is_healthy is False
        assert "stale" in status_stale.warning_message.lower()
