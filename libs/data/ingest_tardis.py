"""Ingest real Deribit market data from Tardis.dev free monthly sample datasets.

No API key is required for the free samples: Tardis publishes complete datasets
for **day 01 of every month** (paid API unlocks other days). Verified URL
layout::

    https://datasets.tardis.dev/v1/deribit/{data_type}/{YYYY}/{MM}/{DD}/{SYMBOL}.csv.gz

Data types used in this project
-------------------------------
* ``quotes/OPTIONS``  — top-of-book quote per options/verticals instrument:
  columns include ``exchange, symbol, timestamp, local_timestamp, ask_amount,
  ask_price, bid_price, bid_amount``. ``timestamp`` is the exchange-side event
  time in **microseconds**, ``local_timestamp`` the local receipt time in µs.
* ``options_chain/OPTIONS`` — full chain snapshots (adds IV, greeks, strike,
  expiration, underlying). Larger (GB-scale); used by SVI calibration.
* ``derivative_ticker/PERPETUALS`` — perp tickers including 8h funding fields.

Phase 0 (data-feasibility gate) needs **quotes/OPTIONS**: historical bid/ask
with receipt timestamps to exercise the point-in-time store end-to-end.

This module streams the remote .csv.gz, filters BTC option rows, renames the
clock columns to canonical ``event_ts_ns``/``received_ts_ns`` (µs -> ns), and
writes a parquet shard for :class:`~libs.data.canonical_store.CanonicalStore`.
All logic is deterministic; network downloads are cached in ``data/raw``.
"""
from __future__ import annotations

import gzip
import http.client
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "normalize_options_quotes",
    "download_tardis_sample",
    "ingest_btc_option_quotes",
]

TARDIS_BASE: str = "https://datasets.tardis.dev/v1/deribit"

QUOTE_COLUMNS: dict[str, str] = {
    "symbol": "symbol",
    "timestamp": "event_ts_ns",
    "local_timestamp": "received_ts_ns",
    "bid_price": "bid_price",
    "bid_amount": "bid_amount",
    "ask_price": "ask_price",
    "ask_amount": "ask_amount",
}

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
    available (non-free days return 401; wrong paths 400/404).
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = f"{TARDIS_BASE}/{data_type}/{year:04d}/{month:02d}/{day:02d}/{symbol}.csv.gz"
    target = raw_dir / f"{data_type}_{year:04d}{month:02d}{day:02d}_{symbol}.csv.gz"
    if target.exists() and target.stat().st_size > 0:
        return target
    import urllib.request

    request = urllib.request.Request(url, method="GET", headers={"User-Agent": "quant-vrp/phase0"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response, target.open("wb") as handle:
            total = 0
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                total += len(chunk)
    except (urllib.error.URLError, http.client.HTTPException) as exc:  # type: ignore[attr-defined]
        if target.exists():
            (target.unlink)()
        raise RuntimeError(f"download failed for {url}: {exc}") from exc
    return target


def normalize_options_quotes(gz_path: str | Path) -> pd.DataFrame:
    """Parse a ``quotes/OPTIONS`` sample into canonical columns for BTC options.

    Streams the gzip in chunks, keeps Deribit rows whose symbol starts with
    ``BTC-`` (option series; verticals like ``BTC-PERPETUAL`` cards and other
    currencies are dropped), then:

    * ``timestamp``        -> ``event_ts_ns``    (µs -> ns, exchange event)
    * ``local_timestamp``  -> ``received_ts_ns`` (µs -> ns, local receipt)
    * prices cast to float64, timestamps to int64.

    Returns a validated :class:`pandas.DataFrame` ready for
    :meth:`CanonicalStore.upsert` (column order: canonical keys first).
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
        frames.append(out.loc[:, ["event_ts_ns", "received_ts_ns", "symbol", "bid_price", "bid_amount", "ask_price", "ask_amount"]])
    if not frames:
        raise RuntimeError(f"no BTC- quotes rows found in {gz_path}")
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
    """End-to-end: download -> normalize -> persist canonical shard.

    Parameters
    ----------
    year/month/day:
        Sample day (free coverage = the 1st of every month).
    raw_dir, processed_dir:
        Repository data directories.
    dataset_name:
        Overrides the default ``deribit_options_quotes_{yyyymmdd}``.

    Returns
    -------
    (rows, dataset_name)
    """
    gz = download_tardis_sample("quotes", year, month, day, "OPTIONS", raw_dir)
    df = normalize_options_quotes(gz)
    name = dataset_name or f"deribit_options_quotes_{year:04d}{month:02d}{day:02d}"
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    from libs.data.canonical_store import CanonicalStore

    store = CanonicalStore(processed_dir)
    rows = store.upsert(name, df)
    return rows, name


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``python -m libs.data.ingest_tardis 2020 11 01``."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Tardis free Deribit option quotes (day-01 samples).")
    parser.add_argument("year", type=int)
    parser.add_argument("month", type=int)
    parser.add_argument("day", type=int)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    rows, name = ingest_btc_option_quotes(args.year, args.month, args.day, root / "data" / "raw", root / "data" / "processed")
    print(f"ingested {rows} rows -> {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
