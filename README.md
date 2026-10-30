# quant-vrp — Falsification-First Volatility Risk Premium Engine

Research & execution engine for trading the **Volatility Risk Premium** on
Deribit BTC options and perpetual futures. NOT a money printer: an empirical
test harness designed to **falsify** the hypothesis

> *Deribit BTC VRP produces positive risk-adjusted returns net of costs when
> conditioned on non-parametric tail triggers.*

## Engineering tenets

1. **Hypothesis falsification over feature creep** — every phase is a kill
   gate. A failed gate halts the project; complexity may not rescue a claim.
2. **Sequential kill gates** — Phase N is built only after Phase N-1 passes.
   Current status lives in the gate table below.
3. **Instruments over weights** — no portfolio optimizers size option legs.
   The strategy layer emits *Target Greek Profiles* (Δ, Γ, Vega, Θ bounds,
   margin, cash buffer); the solver picks discrete strikes/expiries.
4. **No uncalibrated data** — mark IVs are never used raw for pricing/hedging;
   a quasi-explicit SVI calibration is fitted per expiry slice (Phase 1+).
5. **Rigorous numeraire accounting** — Deribit settles in BTC: payoffs,
   collateral, and margin drift non-linearly with the BTC price. USD
   accounting is explicit, ledger-checked, and reconciled to 1e-6.

## Repository layout

```
quant-platform/
├── libs/
│   ├── core/                  # Shared instrument math (single source for research & prod)
│   │   ├── inverse_payoffs.py # BTC numeraire, inverse perp & option settlement, collateral drift
│   │   └── svi_surface.py     # (Phase 1) Arbitrage-free quasi-explicit SVI
│   └── data/
│       ├── canonical_store.py # Point-in-time reader: receipt_ts <= T, NO lookahead
│       ├── ingest_tardis.py   # Real-data loader (Tardis free monthly samples, keyless)
│       └── realized_kernels.py# (Phase 2) 5-min subsampled RV
├── strategies/                # (Phase 4) TargetGreekProfile + strike solver
├── research/                  # Reproducible experiment notebooks
├── simulation/                # (Phase 3) Purged walk-forward engine over prod code
├── risk/                      # INDEPENDENT AUTHORITY: tail_triggers veto strategies
├── execution/                 # (Phase 5) NautilusTrader actor + variance breakers
├── monitoring/                # (Phase 5) Live greeks/PnL surface
├── tests/
│   ├── unit/                  # Hand-verified analytical math tests
│   ├── bias_checks/           # canonical_store lookahead tripwires (veto power)
│   └── regression/            # Phase kill-gate suites (0–3)
├── configs/                   # Pre-registration YAMLs land with their phase
└── data/
    ├── raw/                   # Source-of-train archives (Tardis .csv.gz samples)
    └── processed/             # Canonical parquet shards + metadata sidecars
```

Data-source note (verified, no API key needed):

* Daily **full** history of Deribit option quotes is a paid service; the free
  keyless coverage is the **Tardis sample datasets (day 01 of each month)**.
* `quotes/OPTIONS`, `options_chain/OPTIONS`, `derivative_ticker/PERPETUALS`
  are the types consumed; Phase 0 uses `quotes/OPTIONS` (bid/ask + receipt
  timestamps). Continuous funding/DVOL backfill comes from Deribit's public
  REST API (`phase 1–3`).

## Phase-0 kill gate (current)

`libs/core/inverse_payoffs.py` models the numeraire facts, all hand-verified:

* Inverse perp USD P&L is *linear* in the mark (`C*(S/P_e − 1)`), while the
  BTC delta scales as `C/S²`: a ±30% gap rescales BTC-delta hedge coverage by
  exactly `(S₀/S₁)²` — the 1/S² hedge drift. Rebalancing in BTC-delta space
  without this factor silently de-hedges the book.
* BTC-settled short-put liability `(K−S)/S` grows without bound as S→0 while
  the USD liability stays capped at K (left-tail liquidation mechanics).
* Accounting that freezes BTC flows at entry-time index mis-states terminal
  USD wealth by exactly `prem_btc × (S₁ − S₀)` (collateral convexity drift),
  verified via a two-view ledger (`marked == realized + drift`).

Run:

```bash
python -m pytest -v -s tests/unit tests/bias_checks            # fast, offline
python -m pytest -v -s -m "not realdata" tests/regression/test_phase0_numeraire.py
python -m pytest -v -s -m realdata tests/regression/test_phase0_numeraire.py  # downloads
python -m mypy --strict                                        # targeted by pyproject
```

## Kill-gate ledger

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | Numeraire P&L accounting + real-data feasibility | **PASSED** (40/40 tests; ledger residual ≤ 5.4e-13 USD across ±30% gaps; 14.0M real Deribit BTC-option quote rows loaded point-in-time, zero lookahead) |
| 1 | SVI calibration + baseline VRP regression | **next up** (gate 0 passed) |
| 2 | HAR-RV vs GARCH vs Kronos QLIKE competition | not built |
| 3 | Pre-registered tail triggers + DSR | not built |
| 4 | Greek-profile generator + execution solver | not built |
| 5 | NautilusTrader actor + variance breakers | not built |
