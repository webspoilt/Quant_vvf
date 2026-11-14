"""
Real-time System Health & Feed Integrity Monitoring Service
Monitors market-data staleness, sequence gap frequencies, queue saturation, and risk gate status.
"""

from dataclasses import dataclass
import time


@dataclass
class HealthStatus:
    is_healthy: bool
    feed_freshness_ms: float
    sequence_gaps_detected: int
    queue_occupancy_pct: float
    risk_gate_open: bool
    warning_message: str


class SystemHealthMonitor:
    """Monitors live telemetry and fails closed if feed or queue invariants degrade."""

    def __init__(
        self,
        max_feed_staleness_ms: float = 500.0,
        max_queue_occupancy_pct: float = 80.0
    ):
        self.max_staleness = max_feed_staleness_ms
        self.max_queue_occ = max_queue_occupancy_pct
        self.total_gaps = 0

    def record_sequence_gap(self):
        self.total_gaps += 1

    def evaluate_health(
        self,
        last_packet_timestamp_ns: int,
        current_queue_depth: int,
        queue_capacity: int,
        circuit_breaker_tripped: bool
    ) -> HealthStatus:
        now_ns = time.time_ns()
        staleness_ms = max(0.0, (now_ns - last_packet_timestamp_ns) / 1_000_000.0)
        occ_pct = (current_queue_depth / queue_capacity * 100.0) if queue_capacity > 0 else 100.0

        is_healthy = True
        warnings = []

        if staleness_ms > self.max_staleness:
            is_healthy = False
            warnings.append(f"Market data feed is stale ({staleness_ms:.1f}ms > {self.max_staleness}ms)")

        if occ_pct > self.max_queue_occ:
            is_healthy = False
            warnings.append(f"Ring buffer queue saturation alert ({occ_pct:.1f}% occupied)")

        if circuit_breaker_tripped:
            is_healthy = False
            warnings.append("Risk circuit breaker is TRIPPED")

        return HealthStatus(
            is_healthy=is_healthy,
            feed_freshness_ms=float(staleness_ms),
            sequence_gaps_detected=self.total_gaps,
            queue_occupancy_pct=float(occ_pct),
            risk_gate_open=not circuit_breaker_tripped,
            warning_message=" | ".join(warnings) if warnings else "All Systems Operational"
        )
