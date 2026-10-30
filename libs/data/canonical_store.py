"""Point-in-time canonical data store with strict no-lookahead enforcement.

Purpose
-------
A single storage layer for every historical market-data artefact the research
and execution engines consume (option quote snapshots, trades, funding, DVOL).
Two column families are mandatory for all datasets accepted here:

* ``event_ts_ns`` — exchange-side event timestamp (nanoseconds since epoch),
* ``received_ts_ns`` — local receipt timestamp (nanoseconds since epoch).

Temporal query contract
-----------------------
``CanonicalStore.query(dataset, as_of_ns)`` returns only rows satisfying

    event_ts_ns <= as_of_ns  AND  received_ts_ns <= as_of_ns

after which it *asserts* both invariants on the result (belt-and-braces):
if a record with ``event_ts_ns > as_of_ns`` or ``received_ts_ns > as_of_ns``
would ever be returned — only possible via a corrupted store or a regression
in the filter path — a :class:`LookaheadViolationError` (an ``AssertionError``
subclass) is raised. This is the falsification harness' first-order defence
against lookahead bias: the store must be *unable* to leak future data.

Cross-check between the two clocks is bounded: ``received_ts_ns`` must be at
least ``event_ts_ns - MAX_CLOCK_SKEW_NS`` (remote exchange clocks can beat
local clocks by small skews; crossing that bound indicates data corruption).

Design notes
------------
* Storage is one parquet shard per dataset under ``data/processed``, plus a
  JSON sidecar recording row count, schema, and a content fingerprint.
* ``upsert`` validates and *replaces* the shard (the store is append-only in
  spirit; re-ingesting a dataset overwrites that dataset's shard only).
* Row ordering is deterministic: sorted by ``(event_ts_ns, received_ts_ns)``.
* The class is deliberately backend-light (pandas/pyarrow only) so the same
  code path runs in research notebooks, walk-forward sims, and live actors.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

__all__ = [
    "MAX_CLOCK_SKEW_NS",
    "CanonicalStore",
    "LookaheadViolationError",
    "StoreIntegrityError",
    "canonical_columns",
]

MAX_CLOCK_SKEW_NS: int = 60_000_000_000  # 60 seconds of tolerated clock skew

#: Columns every canonical dataset must contain (payload columns follow).
CANONICAL_KEY_COLUMNS: tuple[str, ...] = ("event_ts_ns", "received_ts_ns")


def _point_in_time_mask(df: pd.DataFrame, as_of_ns: int) -> pd.Series:
    """Boolean mask: rows admissible at query time ``as_of_ns``.

    A row is admissible iff BOTH ``event_ts_ns <= as_of_ns`` and
    ``received_ts_ns <= as_of_ns``. Kept as a module-level function so the
    bias-check suite can tripwire a broken filter path and prove the
    downstream assertion layer still refuses to emit leaking rows.
    """
    return (df["event_ts_ns"] <= as_of_ns) & (df["received_ts_ns"] <= as_of_ns)


class StoreIntegrityError(RuntimeError):
    """Raised when a dataset fails schema/integrity validation on ingest."""


class LookaheadViolationError(AssertionError):
    """Raised when a temporal query would return (or found) a lookahead-violating row.

    Subclasses :class:`AssertionError` per the project bias-check directive:
    a query whose result (or scanned store) contains a record with
    ``event_ts_ns > as_of_ns`` or ``received_ts_ns > as_of_ns`` must abort.
    """


def canonical_columns() -> list[str]:
    """Names of the mandatory canonical key columns."""
    return list(CANONICAL_KEY_COLUMNS)


def _require_timestamp_int64(df: pd.DataFrame, where: str) -> None:
    if not (df["event_ts_ns"].dtype == "int64" and df["received_ts_ns"].dtype == "int64"):
        raise StoreIntegrityError(
            f"{where}: event_ts_ns/received_ts_ns must be int64 nanoseconds, got "
            f"{df['event_ts_ns'].dtype}/{df['received_ts_ns'].dtype}"
        )


_DIGEST_JOIN: bytes = b"|"


class CanonicalStore:
    """Parquet-backed point-in-time store enforcing the no-lookahead contract.

    Parameters
    ----------
    root:
        Directory holding ``{dataset}.parquet`` shards (typically
        ``data/processed``).
    """

    def __init__(self, root: str | Path) -> None:
        self.root: Path = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._frames: dict[str, pd.DataFrame] = {}

    # ---------------------------------------------------------------- ingest
    def upsert(self, dataset: str, df: pd.DataFrame) -> int:
        """Validate ``df`` and write it as the authoritative shard of ``dataset``.

        Validates canonical columns, int64 timestamps, finiteness, the clock
        skew invariant, and duplicates on the key (event_ts_ns, received_ts_ns,
        symbol) when a symbol column is present. Rows are sorted by the key
        before persistence. Returns the persisted row count.
        """
        missing = [c for c in CANONICAL_KEY_COLUMNS if c not in df.columns]
        if missing:
            raise StoreIntegrityError(f"{dataset}: missing canonical columns {missing}")
        _require_timestamp_int64(df, dataset)
        if df["event_ts_ns"].isna().any() or df["received_ts_ns"].isna().any():
            raise StoreIntegrityError(f"{dataset}: NaN timestamps are not allowed")
        skew = df["event_ts_ns"] - df["received_ts_ns"] > MAX_CLOCK_SKEW_NS
        if bool(skew.any()):
            n = int(skew.sum())
            raise StoreIntegrityError(
                f"{dataset}: {n} rows have received_ts_ns more than {MAX_CLOCK_SKEW_NS} ns "
                "behind event_ts_ns (clock-skew/corruption)."
            )
        if "symbol" in df.columns:
            dup_cols = ["event_ts_ns", "received_ts_ns", "symbol"]
        else:
            dup_cols = ["event_ts_ns", "received_ts_ns"]
        if df.duplicated(subset=dup_cols).any():
            raise StoreIntegrityError(
                f"{dataset}: duplicate key rows on {dup_cols} are not allowed"
            )
        out = df.sort_values(dup_cols, kind="stable").reset_index(drop=True)
        self._frames[dataset] = out
        shard = self.root / f"{dataset}.parquet"
        out.to_parquet(shard, engine="pyarrow")
        sidecar = {
            "dataset": dataset,
            "rows": int(len(out)),
            "columns": list(out.columns),
            "content_sha256": self._fingerprint(out),
        }
        (self.root / f"{dataset}.meta.json").write_text(json.dumps(sidecar, indent=2))
        return int(len(out))

    def refresh_from_disk(self, datasets: Sequence[str] | None = None) -> list[str]:
        """(Re)load parquet shards from disk into the in-memory cache."""
        if datasets is None:
            datasets = sorted(p.stem.rsplit(".", 1)[0] for p in self.root.glob("*.parquet"))
        for name in datasets:
            self._frames[name] = pd.read_parquet(self.root / f"{name}.parquet")
        return list(datasets)

    # ----------------------------------------------------------------- query
    def query(
        self,
        dataset: str,
        as_of_ns: int,
        *,
        symbols: Sequence[str] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Point-in-time read: rows with both stamps ``<= as_of_ns``.

        Any query time must be non-negative. The result is asserted to contain
        no record with ``event_ts_ns > as_of_ns`` or ``received_ts_ns > as_of_ns``
        (:class:`LookaheadViolationError` otherwise). Deterministic ordering.
        """
        if as_of_ns < 0:
            raise ValueError(f"as_of_ns must be >= 0, got {as_of_ns}")
        df = self._get(dataset)
        mask = _point_in_time_mask(df, as_of_ns)
        result = df.loc[mask].copy()
        if symbols is not None and "symbol" in result.columns:
            result = result.loc[result["symbol"].isin(list(symbols))]
        if columns is not None:
            keep = [c for c in columns if c in result.columns]
            result = result.loc[:, keep]
        self.assert_no_lookahead(result, as_of_ns, context=dataset)
        return result.reset_index(drop=True)

    def audit(
        self,
        dataset: str,
        as_of_ns: int,
        *,
        subset: pd.DataFrame | None = None,
    ) -> None:
        """Raise :class:`LookaheadViolationError` if any record violates the
        ``stamp <= as_of_ns`` invariants. Default scope: full stored shard."""
        df = self._get(dataset) if subset is None else subset
        self.assert_no_lookahead(df, as_of_ns, context=f"audit:{dataset}")

    def assert_no_lookahead(
        self,
        result: pd.DataFrame,
        as_of_ns: int,
        *,
        context: str,
    ) -> None:
        """Post-filter invariant: raising here means a *result* row leaked.

        The filter above guarantees compliance for well-formed stores; firing
        this assertion indicates store corruption or a filter-path regression,
        which is exactly what must not pass silently.
        """
        if "event_ts_ns" not in result.columns or "received_ts_ns" not in result.columns:
            raise StoreIntegrityError(f"{context}: canonical timestamp columns lost in query")
        n_event = int((result["event_ts_ns"] > as_of_ns).sum())
        n_received = int((result["received_ts_ns"] > as_of_ns).sum())
        if n_event or n_received:
            raise LookaheadViolationError(
                f"Lookahead violation in {context!r}: {n_event} record(s) with "
                f"event_ts_ns > t={as_of_ns} and {n_received} record(s) with "
                f"received_ts_ns > t. Aborting query."
            )

    # ---------------------------------------------------------------- internals
    def _get(self, dataset: str) -> pd.DataFrame:
        if dataset not in self._frames:
            shard = self.root / f"{dataset}.parquet"
            if not shard.exists():
                raise KeyError(f"unknown dataset {dataset!r} (no shard at {shard})")
            self.refresh_from_disk([dataset])
        return self._frames[dataset]

    def _fingerprint(self, df: pd.DataFrame) -> str:
        payload = df.sort_values(
            list(CANONICAL_KEY_COLUMNS) + (["symbol"] if "symbol" in df.columns else []),
            kind="stable",
        )
        digests: list[bytes] = []
        for chunk in (payload[col].to_numpy().tobytes() for col in payload.columns):
            digests.append(hashlib.sha256(chunk).digest())
        blob = _DIGEST_JOIN.join(digests) + str(payload.shape).encode()
        return hashlib.sha256(blob).hexdigest()
