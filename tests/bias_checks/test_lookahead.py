"""Bias checks: canonical_store must provably reject lookahead.

The project directive: every assertion that `canonical_store.query(t)` cannot
return a record with `event_ts_ns > t` or `received_ts_ns > t` — the store
must RAISE (AssertionError subclass) if a violating record would ever be
emitted, i.e. corruption must not pass silently. These tests are the veto
power of the data layer: any change to the filter path that leaks future
data turns this suite red before any trading code is trusted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import libs.data.canonical_store as cs
from libs.data.canonical_store import (
    CanonicalStore,
    LookaheadViolationError,
    StoreIntegrityError,
)

NS = 1  # tests use tiny synthetic nanostamps


def _mk_rows() -> pd.DataFrame:
    """Three synthetic rows covering the temporal zoo:

    r1: event 1000, received 1000 — aligned normal row.
    r2: event 2000, received 2000 — aligned normal row.
    r3: event 3000, received 2100 — late-arriving OLD-EVENT record: it exists
        in the store BEFORE its event time is admissible for queries ≥ its
        receipt time. A query at t=2000 must NOT see it (both stamps > 2000/
        received > 2000); a query at t=3500 may.
    """
    return pd.DataFrame(
        {
            "event_ts_ns": np.array([1000, 2000, 3000], dtype="int64"),
            "received_ts_ns": np.array([1000, 2000, 2100], dtype="int64"),
            "symbol": ["BTC-1AUG20-8000-P", "BTC-1AUG20-9000-C", "BTC-1AUG20-9500-P"],
            "bid_price": [0.10, 0.20, 0.15],
        }
    )


@pytest.fixture()
def store(tmp_path: Path) -> CanonicalStore:
    s = CanonicalStore(tmp_path / "processed")
    s.upsert("test_quotes", _mk_rows())
    return s


class TestPointInTimeSemantics:
    def test_query_returns_only_both_stamp_eligible_rows(self, store: CanonicalStore) -> None:
        result = store.query("test_quotes", as_of_ns=2000 * NS)
        # r1 (1000/1000) and r2 (2000/2000) are admissible; r3 is excluded
        # because received 2100 > 2000. V-28 fix: the original `or all(...)`
        # masked assertion hid the r1 omission and could not fail loudly.
        assert list(result["symbol"]) == ["BTC-1AUG20-8000-P", "BTC-1AUG20-9000-C"]
        assert result["event_ts_ns"].max() <= 2000
        assert result["received_ts_ns"].max() <= 2000

    def test_boundary_inclusive(self, store: CanonicalStore) -> None:
        result = store.query("test_quotes", as_of_ns=2000 * NS)
        assert (result["event_ts_ns"] == 2000).sum() == 1

    def test_no_rows_before_first_stamp(self, store: CanonicalStore) -> None:
        assert len(store.query("test_quotes", as_of_ns=999 * NS)) == 0

    def test_rows_grow_monotonic_with_as_of(self, store: CanonicalStore) -> None:
        once = store.query("test_quotes", as_of_ns=1500 * NS)
        later_full = store.query("test_quotes", as_of_ns=3500 * NS)
        # Symbols visible at 1500 must remain visible at 3500 (no deletion into
        # the past), and later samples are supersets.
        early = set(once["symbol"])
        late = set(later_full["symbol"])
        assert len(later_full) == 3  # all rows admissible at 3500 (r3 received 2100 <= 3500)
        assert early <= late

    def test_symbol_filter(self, store: CanonicalStore) -> None:
        result = store.query("test_quotes", as_of_ns=3500 * NS, symbols=["BTC-1AUG20-9000-C"])
        assert list(result["symbol"]) == ["BTC-1AUG20-9000-C"]

    def test_unknown_dataset_raises(self, store: CanonicalStore) -> None:
        with pytest.raises(KeyError):
            store.query("nonexistent", as_of_ns=3500 * NS)

    def test_negative_as_of_rejected(self, store: CanonicalStore) -> None:
        with pytest.raises(ValueError):
            store.query("test_quotes", as_of_ns=-1)


class TestLookaheadEnforcement:
    """The failure mode that must never pass silently."""

    def test_violating_frame_raises_assertion_subclass(self, store: CanonicalStore) -> None:
        violating = pd.DataFrame(
            {
                "event_ts_ns": np.array([2001, 3500], dtype="int64"),
                "received_ts_ns": np.array([1000, 1000], dtype="int64"),
            }
        )
        with pytest.raises(AssertionError):
            store.assert_no_lookahead(violating, as_of_ns=2000, context="unit")

    def test_late_received_raises_assertion_subclass(self, store: CanonicalStore) -> None:
        late = pd.DataFrame(
            {
                "event_ts_ns": np.array([1000], dtype="int64"),
                "received_ts_ns": np.array([1001], dtype="int64"),
            }
        )
        with pytest.raises(AssertionError):
            store.assert_no_lookahead(late, as_of_ns=1000, context="unit")

    def test_broken_filter_path_is_caught(
        self, store: CanonicalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Self-test of the enforcement layer: if the point-in-time mask is
        # regressed to admit everything (a plausible bug), query() must still
        # refuse to return future rows — the post-filter assertion fires.
        def broken_mask(df: pd.DataFrame, as_of_ns: int) -> pd.Series:
            return pd.Series(True, index=df.index)

        monkeypatch.setattr(cs, "_point_in_time_mask", broken_mask)
        with pytest.raises(LookaheadViolationError):
            store.query("test_quotes", as_of_ns=2000 * NS)

    def test_audit_raises_on_full_table_violation(self, store: CanonicalStore) -> None:
        # audit(dataset, t) is the whole-table verifier demanded by the directive:
        # a record with event > t anywhere in the dataset aborts the audit.
        with pytest.raises(AssertionError):
            store.audit("test_quotes", as_of_ns=2000 * NS)
        # At t=3500 every stored record is eligible → no violation.
        store.audit("test_quotes", as_of_ns=3500 * NS)

    def test_query_result_never_contains_violations(self, store: CanonicalStore) -> None:
        for as_of in (1000, 1500, 2000, 2100, 2500, 3500, 9000):
            result = store.query("test_quotes", as_of_ns=as_of * NS)
            assert (result["event_ts_ns"] <= as_of * NS).all()
            assert (result["received_ts_ns"] <= as_of * NS).all()


class TestIngestValidation:
    def test_missing_canonical_columns_rejected(self, tmp_path: Path) -> None:
        store = CanonicalStore(tmp_path / "p")
        bad = pd.DataFrame({"ts": [1, 2, 3]})
        with pytest.raises(StoreIntegrityError):
            store.upsert("broken", bad)

    def test_clock_skew_violation_rejected(self, tmp_path: Path) -> None:
        store = CanonicalStore(tmp_path / "p")
        # Received 2000 seconds BEFORE the event crosses the 60s skew tolerance
        # (impossible causality); a receipt after the event is legitimate.
        bad = pd.DataFrame(
            {
                "event_ts_ns": np.array([100_000_000_000_000], dtype="int64"),
                "received_ts_ns": np.array([100], dtype="int64"),
            }
        )
        with pytest.raises(StoreIntegrityError):
            store.upsert("skewy", bad)

    def test_duplicate_keys_rejected(self, tmp_path: Path) -> None:
        store = CanonicalStore(tmp_path / "p")
        dup = pd.DataFrame(
            {
                "event_ts_ns": np.array([100, 100], dtype="int64"),
                "received_ts_ns": np.array([100, 100], dtype="int64"),
                "symbol": ["A", "A"],
            }
        )
        with pytest.raises(StoreIntegrityError):
            store.upsert("dup", dup)

    def test_float_timestamps_rejected(self, tmp_path: Path) -> None:
        store = CanonicalStore(tmp_path / "p")
        bad = pd.DataFrame(
            {
                "event_ts_ns": np.array([100.5, 200.5], dtype="float64"),
                "received_ts_ns": np.array([100.5, 200.5], dtype="float64"),
            }
        )
        with pytest.raises(StoreIntegrityError):
            store.upsert("floats", bad)

    def test_persistence_roundtrip_and_determinism(self, tmp_path: Path) -> None:
        root = tmp_path / "p"
        store = CanonicalStore(root)
        store.upsert("rt", _mk_rows())
        fresh = CanonicalStore(root)
        fresh.refresh_from_disk()
        for as_of in (1000, 2000, 3500):
            a = fresh.query("rt", as_of_ns=as_of * NS)
            b = store.query("rt", as_of_ns=as_of * NS)
            pd.testing.assert_frame_equal(a, b)
            # Deterministic ordering by key:
            assert a[["event_ts_ns", "received_ts_ns"]].equals(
                a[["event_ts_ns", "received_ts_ns"]].sort_values(
                    ["event_ts_ns", "received_ts_ns"], kind="stable"
                )
            )
