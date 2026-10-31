"""Deribit data loaders — Tardis free monthly samples (keyless) + normalizers.

Ingests real Deribit market data from Tardis.dev free sample datasets (no API
key): Tardis publishes complete daily datasets for **day 01 of every month**
(paid API unlocks other days). Verified URL layout::

    https://datasets.tardis.dev/v1/deribit/{data_type}/{YYYY}/{MM}/{DD}/{SYMBOL}.csv.gz

Data types used by this project
-------------------------------
* ``quotes/OPTIONS``  — top-of-book per option/vertical: ``exchange, symbol,
  timestamp, local_timestamp, ask_amount, ask_price, bid_price, bid_amount``.
  Timestamps are **microseconds** (exchange event, local receipt).
* ``options_chain/OPTIONS`` — full-chain snapshots with IVs and greeks:
  ``exchange, symbol, timestamp, local_timestamp, type, strike_price,
  expiration, open_interest, last_price, bid_price, bid_amount, bid_iv,
  ask_price, ask_amount, ask_iv, mark_price, mark_iv, underlying_index,
  underlying_price, delta, gamma, vega, theta, rho``.
* ``derivative_ticker/PERPETUALS`` — perp tickers incl. funding fields.

Both normalizers rename the clock columns to canonical
``event_ts_ns``/``received_ts_ns`` (µs → ns), keep Deribit rows whose symbol
starts with ``BTC-`` and (for chains) ``type == "option"``, and write parquet
shards consumable by :class:`~libs.data.canonical_store.CanonicalStore`.
Network downloads are cached in ``data/raw``; every ingest path is
deterministic and validated on write.
"""
from __future__ import annotations

import http.client
import os
import re
import sys
import urllib.request  # noqa: F401  (referenced in download_tardis_sample)
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "TARDIS_BASE",
    "DATA_TYPE_WHITELIST",
    "SYMBOL_PATTERN",
    "download_tardis_sample",
    "normalize_options_quotes",
    "normalize_options_chain",
    "ingest_btc_option_quotes",
    "ingest_btc_option_chain",
    "main",
]

TARDIS_BASE: str = "https://datasets.tardis.dev/v1/deribit"

# V-05: data_type and symbol are interpolated into both an download URL and a
# local filesystem path; they must be whitelisted/pattern-checked (fail-closed)
# so a caller-controlled string cannot traverse paths or redirect the fetch.
DATA_TYPE_WHITELIST: frozenset[str] = frozenset(
    {"quotes", "options_chain", "derivatives_ticker"}
)
SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,128}$")

QUOTE_COLUMNS: dict[str, str] = {
    "symbol": "symbol",
    "timestamp": "event_ts_ns",
    "local_timestamp": "received_ts_ns",
    "bid_price": "bid_price",
    "bid_amount": "bid_amount",
    "ask_price": "ask_price",
    "ask_amount": "ask_amount",
}

CHAIN_COLUMNS: dict[str, str] = {
    "symbol": "symbol",
    "timestamp": "event_ts_ns",
    "local_timestamp": "received_ts_ns",
    "type": "type",
    "strike_price": "strike_price",
    "expiration": "expiration_ts_ns",
    "bid_price": "bid_price",
    "bid_amount": "bid_amount",
    "bid_iv": "bid_iv",
    "ask_price": "ask_price",
    "ask_amount": "ask_amount",
    "ask_iv": "ask_iv",
    "mark_price": "mark_price",
    "mark_iv": "mark_iv",
    "underlying_index": "underlying_index",
    "underlying_price": "underlying_price",
}

CHAIN_KEEP_ORDER: tuple[str, ...] = (
    "event_ts_ns",
    "received_ts_ns",
    "symbol",
    "type",
    "strike_price",
    "expiration_ts_ns",
    "bid_price",
    "bid_amount",
    "bid_iv",
    "ask_price",
    "ask_amount",
    "ask_iv",
    "mark_price",
    "mark_iv",
    "underlying_index",
    "underlying_price",
)

CHUNK_ROWS: int = 2_000_000


def download_tardis_sample(
    data_type: str,
    year: int,
    month: int,
    day: int,
    symbol: str,
    raw_dir: str | Path,
) -> Path:
    """Download (or reuse) a Tardis free sample CSV in ``raw_dir``, return path.

    Raises ``RuntimeError`` with the HTTP status when the file is not
    available (non-free days return 401; wrong paths 400/404). ``data_type``
    must be in :data:`DATA_TYPE_WHITELIST` and ``symbol`` must match
    :data:`SYMBOL_PATTERN` (V-05 — both are interpolated into a URL and a
    filesystem path; unsanitized input enables traversal/URL manipulation).
    """
    if data_type not in DATA_TYPE_WHITELIST:
        raise ValueError(
            f"Invalid data_type {data_type!r}: must be one of {sorted(DATA_TYPE_WHITELIST)}"
        )
    if not (isinstance(symbol, str) and SYMBOL_PATTERN.match(symbol) and ".." not in symbol):
        raise ValueError(f"Invalid symbol {symbol!r}: must match {SYMBOL_PATTERN.pattern}")

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = f"{TARDIS_BASE}/{data_type}/{year:04d}/{month:02d}/{day:02d}/{symbol}.csv.gz"
    target = raw_dir / f"{data_type}_{year:04d}{month:02d}{day:02d}_{symbol}.csv.gz"
    if target.exists() and target.stat().st_size > 0:
        return target

    # V-18: download to a temp sibling and atomically swap in, so concurrent
    # readers never observe a truncated/partial CSV under the final name.
    tmp = target.with_name(f".{target.name}.part-{os.getpid()}-{os.urandom(4).hex()}")
    request = urllib.request.Request(url, method="GET", headers={"User-Agent": "quant-vrp/phase1"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response, tmp.open("wb") as handle:
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        os.replace(tmp, target)
    except (urllib.error.URLError, http.client.HTTPException) as exc:
        if tmp.exists():
            (tmp.unlink)()
        raise RuntimeError(f"download failed for {url}: {exc}") from exc
    return target


def normalize_options_quotes(gz_path: str | Path) -> pd.DataFrame:
    """Parse a ``quotes/OPTIONS`` sample into canonical BTC-option columns.

    Streams the gzip in chunks, keeps Deribit rows whose symbol starts with
    ``BTC-``, then:

    * ``timestamp``        -> ``event_ts_ns``    (µs -> ns, exchange event)
    * ``local_timestamp``  -> ``received_ts_ns`` (µs -> ns, local receipt)
    * prices cast to float64, timestamps to int64.

    Returns a validated :class:`pandas.DataFrame` ready for
    :meth:`CanonicalStore.upsert` (canonical key columns first).
    """
    frames: list[pd.DataFrame] = []
    reader = pd.read_csv(
        str(gz_path),
        compression="gzip",
        usecols=list(QUOTE_COLUMNS),
        dtype={"symbol": "str"},
        chunksize=CHUNK_ROWS,
    )
    for chunk in reader:
        btc = chunk.loc[chunk["symbol"].str.startswith("BTC-")]
        if btc.empty:
            continue
        out = btc.loc[:, list(QUOTE_COLUMNS)].rename(columns=QUOTE_COLUMNS)
        out["event_ts_ns"] = (out["event_ts_ns"].astype("int64") * 1000).astype("int64")
        out["received_ts_ns"] = (out["received_ts_ns"].astype("int64") * 1000).astype("int64")
        for col in ("bid_price", "ask_price", "bid_amount", "ask_amount"):
            out[col] = out[col].astype("float64")
        frames.append(
            out.loc[:, ["event_ts_ns", "received_ts_ns", "symbol", "bid_price", "bid_amount", "ask_price", "ask_amount"]]
        )
    if not frames:
        raise RuntimeError(f"no BTC- quotes rows found in {gz_path}")
    return pd.concat(frames, ignore_index=True)


def normalize_options_chain(gz_path: str | Path) -> pd.DataFrame:
    """Parse an ``options_chain/OPTIONS`` sample into canonical BTC chain rows.

    Keeps ``type == "option"`` rows for BTC (drops ETH and option/vertical
    cards), renames the µs clocks to canonical int64 ns stamps and ``strike_
    price`` / ``expiration`` accordingly, casts numeric payloads to float64.
    Columns follow :data:`CHAIN_KEEP_ORDER` (canonical keys first).
    """
    frames: list[pd.DataFrame] = []
    reader = pd.read_csv(
        str(gz_path),
        compression="gzip",
        usecols=list(CHAIN_COLUMNS),
        dtype={"symbol": "str", "type": "str"},
        chunksize=CHUNK_ROWS,
    )
    for chunk in reader:
        btc = chunk.loc[chunk["symbol"].str.startswith("BTC-") & (chunk["type"] == "option")]
        if btc.empty:
            continue
        out = btc.loc[:, list(CHAIN_COLUMNS)].rename(columns=CHAIN_COLUMNS)
        out["event_ts_ns"] = (out["event_ts_ns"].astype("int64") * 1000).astype("int64")
        out["received_ts_ns"] = (out["received_ts_ns"].astype("int64") * 1000).astype("int64")
        out["expiration_ts_ns"] = (out["expiration_ts_ns"].astype("int64") * 1000).astype("int64")
        out["strike_price"] = out["strike_price"].astype("float64")
        for col in (
            "bid_price", "bid_amount", "bid_iv", "ask_price", "ask_amount",
            "ask_iv", "mark_price", "mark_iv", "underlying_price",
        ):
            out[col] = out[col].astype("float64")
        frames.append(out.loc[:, list(CHAIN_KEEP_ORDER)])
    if not frames:
        raise RuntimeError(f"no BTC- option chain rows found in {gz_path}")
    return pd.concat(frames, ignore_index=True)


def ingest_btc_option_quotes(
    year: int,
    month: int,
    day: int,
    raw_dir: str | Path,
    processed_dir: str | Path,
    *,
    dataset_name: str | None = None,
) -> tuple[int, str]:
    """End-to-end quotes ingest: download -> normalize -> canonical shard.

    Returns ``(rows, dataset_name)``.
    """
    gz = download_tardis_sample("quotes", year, month, day, "OPTIONS", raw_dir)
    df = normalize_options_quotes(gz)
    name = dataset_name or f"deribit_options_quotes_{year:04d}{month:02d}{day:02d}"
    return _persist(df, processed_dir, name)


def ingest_btc_option_chain(
    year: int,
    month: int,
    day: int,
    raw_dir: str | Path,
    processed_dir: str | Path,
    *,
    dataset_name: str | None = None,
) -> tuple[int, str]:
    """End-to-end options-chain ingest (Phase 1 SVI calibration input).

    Returns ``(rows, dataset_name)``.
    """
    gz = download_tardis_sample("options_chain", year, month, day, "OPTIONS", raw_dir)
    df = normalize_options_chain(gz)
    name = dataset_name or f"deribit_options_chain_{year:04d}{month:02d}{day:02d}"
    return _persist(df, processed_dir, name)


def _persist(df: pd.DataFrame, processed_dir: str | Path, name: str) -> tuple[int, str]:
    from libs.data.canonical_store import CanonicalStore

    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    store = CanonicalStore(processed)
    rows = store.upsert(name, df)
    return rows, name


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m libs.data.deribit_loader chain 2022 11 1`` (quotes|chain)."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Tardis free Deribit option data (day-01 samples).")
    parser.add_argument("kind", choices=["quotes", "chain"])
    parser.add_argument("year", type=int)
    parser.add_argument("month", type=int)
    parser.add_argument("day", type=int)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    raw, processed = root / "data" / "raw", root / "data" / "processed"
    if args.kind == "quotes":
        rows, name = ingest_btc_option_quotes(args.year, args.month, args.day, raw, processed)
    else:
        rows, name = ingest_btc_option_chain(args.year, args.month, args.day, raw, processed)
    print(f"ingested {rows} rows -> {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
