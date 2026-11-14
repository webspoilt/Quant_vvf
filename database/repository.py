"""
Relational Database Persistence Layer
Authoritative PostgreSQL schemas and Python repository interfaces for
Orders, Executions, Risk Limits, Positions, and Trade Journals.
"""

from dataclasses import dataclass
from typing import List, Optional
import sqlite3  # In-memory / local fallback for development; PostgreSQL-compatible schema


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    client_order_id TEXT UNIQUE NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    status TEXT NOT NULL,
    created_at_ns INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS executions (
    exec_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    filled_price REAL NOT NULL,
    filled_qty REAL NOT NULL,
    fee_usd REAL NOT NULL,
    executed_at_ns INTEGER NOT NULL,
    FOREIGN KEY(order_id) REFERENCES orders(order_id)
);

CREATE TABLE IF NOT EXISTS trade_journal (
    journal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE NOT NULL,
    strategy_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    regime TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    realized_pnl_usd REAL NOT NULL,
    fees_usd REAL NOT NULL,
    slippage_usd REAL NOT NULL,
    closed_at_ns INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS system_health_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ns INTEGER NOT NULL,
    feed_freshness_ms REAL NOT NULL,
    sequence_gaps_detected INTEGER NOT NULL,
    risk_gate_status TEXT NOT NULL,
    queue_depth INTEGER NOT NULL
);
"""


class DatabaseRepository:
    """Manages transactional persistence for audit and recovery."""

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.executescript(SCHEMA_SQL)

    def insert_order(self, order_id: str, client_order_id: str, symbol: str, side: str, price: float, quantity: float, status: str, created_at_ns: int):
        with self.conn:
            self.conn.execute(
                "INSERT INTO orders (order_id, client_order_id, symbol, side, price, quantity, status, created_at_ns) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (order_id, client_order_id, symbol, side, price, quantity, status, created_at_ns)
            )

    def insert_execution(self, exec_id: str, order_id: str, symbol: str, side: str, filled_price: float, filled_qty: float, fee_usd: float, executed_at_ns: int):
        with self.conn:
            self.conn.execute(
                "INSERT INTO executions (exec_id, order_id, symbol, side, filled_price, filled_qty, fee_usd, executed_at_ns) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (exec_id, order_id, symbol, side, filled_price, filled_qty, fee_usd, executed_at_ns)
            )

    def record_journal_entry(self, trade_id: str, strategy_id: str, symbol: str, regime: str, entry_price: float, exit_price: float, realized_pnl: float, fees: float, slippage: float, closed_at_ns: int):
        with self.conn:
            self.conn.execute(
                "INSERT INTO trade_journal (trade_id, strategy_id, symbol, regime, entry_price, exit_price, realized_pnl_usd, fees_usd, slippage_usd, closed_at_ns) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (trade_id, strategy_id, symbol, regime, entry_price, exit_price, realized_pnl, fees, slippage, closed_at_ns)
            )

    def get_order(self, order_id: str) -> Optional[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
        return cur.fetchone()

    def get_all_journal_entries(self) -> List[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM trade_journal ORDER BY closed_at_ns DESC")
        return cur.fetchall()
