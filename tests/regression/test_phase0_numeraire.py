"""PHASE 0 falsification gate: numeraire reconciliation + real-data feasibility.

Kill-gate criteria (from the task spec, verbatim semantics):
  1. Historical Deribit options data with valid bid/ask quotes and timestamps
     must load point-in-time without lookahead.  Fail -> HALT PROJECT.
  2. Inverse payoff accounting must balance to within 1e-6 BTC/USD: the USD
     ledger of a delta-hedged ATM short straddle across a +/–30% instant spot
     gap must reconcile against independent closed forms, and the test must
     PROVE that the inverse-perp delta hedge is dollar-linear while the
     BTC-delta hedge ratio rescales by exactly (S0/S1)**2 (the 1/S**2
     hedge-drift) — i.e. naive USD accounting mis-states realised wealth by
     the collateral convexity drift (S1-S0)*prem_btc.  Fail -> HALT PROJECT.

Verification table is printed with `-s` (see README run instructions); every
row prints BEFORE its assertion so a failure still emits diagnostics.
"""
from __future__ import annotations

import pytest

from libs.core.inverse_payoffs import (
    GapScenarioResult,
    inverse_perp_btc_delta,
    inverse_perp_hedge_notional_for_usd_delta,
    inverse_perp_pnl_btc,
    inverse_perp_pnl_usd,
    inverse_perp_hedge_rescale_factor,
    option_payoff_btc,
    simulate_instant_gap,
    straddle_delta_usd,
    straddle_premium_usd,
)

# Scenario constants (locked for Phase 0; matches the regression spec):
SPOT_ENTRY: float = 100_000.0
SIGMA: float = 0.50
TAU_YEARS: float = 14.0 / 365.0
GAP_DIRECTIONS: tuple[float, ...] = (0.70, 1.30)  # ±30% instantaneous gap


# --------------------------------------------------------------------------- #
# Closed forms — derived independently of the library's scenario builder so a
# coding error inside `simulate_instant_gap` cannot self-confirm.
# --------------------------------------------------------------------------- #
def _closed_forms(result: GapScenarioResult) -> dict[str, float]:
    """Recompute every scenario quantity from first principles (test-only)."""
    s0 = result.spot_entry
    s1 = result.spot_after
    tau = result.tau_years
    sigma = result.sigma
    r = 0.0
    # Entry straddle (strike = forward = S0 * exp(r*T), r=0 -> K = S0):
    call0, put0 = straddle_premium_usd(s0, s0, sigma, tau, r)
    prem_btc_cf = (call0 + put0) / s0
    # USD-equivalent straddle delta = N(d1) + (N(d1)-1) = 2*N(d1)-1 per parity
    # (independently hand-verified in tests/unit; NOT derived from value math,
    #  which coincides with the delta only in the exact ATM r=0 special case).
    straddle_delta_cf = straddle_delta_usd(s0, s0, sigma, tau, r)
    hedge_notional_cf = inverse_perp_hedge_notional_for_usd_delta(-straddle_delta_cf, s0)
    # Gap-time straddle value (re-mark at same sigma/tau, no time passage):
    call1, put1 = straddle_premium_usd(s1, s0, sigma, tau, r)
    v_after_cf = call1 + put1
    # Perp leg:
    pnl_btc_cf = inverse_perp_pnl_btc(hedge_notional_cf, s0, s1)
    pnl_usd_cf = inverse_perp_pnl_usd(hedge_notional_cf, s0, s1)
    # Ledger views (closed form):
    exact_cf = s1 * (prem_btc_cf + pnl_btc_cf) - v_after_cf
    naive_cf = prem_btc_cf * s0 + pnl_usd_cf - v_after_cf  # prem frozen at S0
    drift_cf = prem_btc_cf * (s1 - s0)
    return {
        "prem_btc": prem_btc_cf,
        "straddle_value_after_usd": v_after_cf,
        "straddle_delta_entry": straddle_delta_cf,
        "hedge_notional_usd": hedge_notional_cf,
        "perp_btc_pnl": pnl_btc_cf,
        "perp_usd_pnl": pnl_usd_cf,
        "exact_marked_usd_pnl": exact_cf,
        "naive_realized_usd_pnl": naive_cf,
        "collateral_drift_usd": drift_cf,
        "hedge_rescale_factor": (s0 / s1) ** 2,
    }


# --------------------------------------------------------------------------- #
# Verification table plumbing
# --------------------------------------------------------------------------- #
_TABLE: list[str] = []


def _row(check: str, value: float, expectation: str, tol: str, ok: bool) -> None:
    _TABLE.append(
        f"{check:<58} {value:>+18.6f}  {expectation:<46} tol={tol:<10} {'PASS' if ok else 'FAIL'}"
    )


def _print_table(title: str) -> None:
    print(f"\n=== {title} ===")
    for line in _TABLE:
        # ASCII-safe for Windows cp1252 consoles; diagnostics must never crash.
        print(line.encode("ascii", "replace").decode("ascii"))
    print(f"=== {title}: checks={len(_TABLE)} ===\n")


def _reset_table() -> None:
    _TABLE.clear()


@pytest.mark.parametrize("gap_ratio", GAP_DIRECTIONS)
def test_core_numeraire_reconciliation(gap_ratio: float) -> None:
    """Ledger vs independent closed forms across ±30% instantaneous gaps."""
    _reset_table()
    scenario = simulate_instant_gap(SPOT_ENTRY, SIGMA, TAU_YEARS, gap_ratio)
    cf = _closed_forms(scenario)
    scale = max(1.0, abs(cf["exact_marked_usd_pnl"]))

    checks: list[tuple[str, float, float, float]] = []

    def check(name: str, got: float, want: float, tol: float) -> None:
        ok = abs(got - want) <= tol
        checks.append((name, got - want, tol))
        _row(name, abs(got - want), f"== {want:.6f}", tol, ok)
        assert ok, f"{name}: got {got}, expected {want} (tol {tol})"

    # 1. Ledger-marked exact USD P&L vs closed form (the 1e-6 balance gate):
    check("exact_marked_ledger_minus_closed_form", scenario.exact_marked_usd_pnl, cf["exact_marked_usd_pnl"], 1e-6)
    # 2. Naive realized accounting (prem frozen at S0) vs its closed form:
    check(
        "naive_realized_ledger_minus_closed_form",
        scenario.naive_realized_usd_pnl,
        cf["naive_realized_usd_pnl"],
        1e-6,
    )
    # 3. Collateral convexity drift == prem_btc * (S1 - S0) — the dollar-size of
    #    the mis-statement between the two accounting views:
    check("collateral_drift_minus_closed_form", scenario.collateral_drift_usd, cf["collateral_drift_usd"], 1e-6)
    # 4. Ledger invariant: marked == realized + drift exactly:
    ledger_residual = scenario.exact_marked_usd_pnl - (scenario.naive_realized_usd_pnl + scenario.collateral_drift_usd)
    _row("ledger_invariant_marked_eq_realized_plus_drift", ledger_residual, 0.0, "1e-6", abs(ledger_residual) <= 1e-6)
    assert abs(ledger_residual) <= 1e-6, f"invariant residual {ledger_residual}"
    # 5. BTC-delta hedge rescaling factor == (S0/S1)**2 exactly (the 1/S**2 drift):
    check("hedge_rescale_factor_minus_closed_form", scenario.hedge_rescale_factor, cf["hedge_rescale_factor"], 1e-12)
    pre_delta = inverse_perp_btc_delta(scenario.hedge_signed_notional_usd, scenario.spot_entry)
    post_delta = inverse_perp_btc_delta(scenario.hedge_signed_notional_usd, scenario.spot_after)
    ratio = post_delta / pre_delta
    _row("btc_delta_ratio_equals_rescale_factor", ratio - (scenario.spot_entry / scenario.spot_after) ** 2, 0.0, "1e-15", abs(ratio - (scenario.spot_entry / scenario.spot_after) ** 2) <= 1e-15)
    assert ratio == pytest.approx((scenario.spot_entry / scenario.spot_after) ** 2, abs=1e-15)
    # 6. Hedge USD-linearity: perp USD P&L == notional*(S1/S0 - 1) — the *linear
    #    dollar protection* fact; the non-linearity lives in BTC-delta space,
    #    not left tail of the option book:
    linear = scenario.hedge_signed_notional_usd * (scenario.spot_after / scenario.spot_entry - 1.0)
    _row("perp_usd_pnl_linearity_residual", scenario.perp_usd_pnl_marked - linear, 0.0, "1e-6", abs(scenario.perp_usd_pnl_marked - linear) <= 1e-6)
    assert abs(scenario.perp_usd_pnl_marked - linear) <= 1e-6
    # 7. Premium drift component is sign-consistent: exact - naive == drift:
    residual = scenario.exact_marked_usd_pnl - scenario.naive_realized_usd_pnl
    check("exact_minus_naive_equals_prem_drift", residual, cf["collateral_drift_usd"], 1e-6)
    # 8. FX separation: BTC-settled payoff identity payoff_btc*S == payoff_usd:
    call_btc = option_payoff_btc("call", scenario.strike, scenario.spot_after)
    check("btc_settled_identity_call", call_btc * scenario.spot_after, max(0.0, scenario.spot_after - scenario.strike), 1e-6)

    _print_table(f"PHASE 0 NUMERAIRE RECONCILIATION (gap_ratio={gap_ratio})")


def test_naive_accounting_vs_exact_error_magnitude() -> None:
    """Sanity on the mis-statement between the two accounting conventions.

    The naive (premium-frozen-at-S0) USD account understates/mis-states the
    terminal USD wealth by exactly |drift| = prem_btc * |S1 - S0|. Reports the
    error in USD and relative to the position scale.
    """
    _reset_table()
    scenario = simulate_instant_gap(SPOT_ENTRY, SIGMA, TAU_YEARS, 1.30)
    error_usd = abs(scenario.exact_marked_usd_pnl - scenario.naive_realized_usd_pnl)
    prem_usd0 = scenario.prem_btc * scenario.spot_entry
    scale_ref = abs(scenario.straddle_value_after_usd)
    _row("naive_vs_exact_error_usd", error_usd, "~ prem_usd0 * 0.30", "1e-6USD", abs(error_usd - prem_usd0 * 0.30) <= 1e-6)
    assert abs(error_usd - prem_usd0 * 0.30) <= 1e-6
    _row("error_relative_to_expiry_liability", error_usd / scale_ref, "print-scale", "n/a", True)
    assert 0.0 < error_usd / scale_ref < 0.5
    _print_table("PHASE 0: NAIVE vs EXACT ACCOUNTING ERROR (+30% gap)")


@pytest.mark.realdata
def test_phase0_real_data_feasibility_gate(preload_report: None = None) -> None:
    """Kill Gate 0, criterion 1: real Deribit option quotes load point-in-time.

    Downloads Tardis free monthly sample (day 01) of Deribit *quotes/OPTIONS*
    (verified keyless), normalizes to canonical timestamps, persists a parquet
    shard, and performs strict point-in-time reads + quote-sanity checks.
    """
    _reset_table()
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw"
    processed = root / "data" / "processed"

    from libs.data.canonical_store import CanonicalStore, LookaheadViolationError  # noqa: F401
    from libs.data.ingest_tardis import ingest_btc_option_quotes

    try:
        rows, dataset = ingest_btc_option_quotes(2020, 11, 1, raw_dir, processed)
    except RuntimeError as exc:
        _row("realdata_download_ingest", -1.0, str(exc)[:100], "n/a", False)
        _print_table("PHASE 0 REAL-DATA GATE (crashed)")
        pytest.fail(f"real-data ingestion failed: {exc}")
        return

    store = CanonicalStore(processed)
    store.refresh_from_disk([dataset])
    shard = store._frames[dataset]
    t_first, t_last = int(shard["event_ts_ns"].min()), int(shard["event_ts_ns"].max())
    mid_day = (t_first + t_last) // 2

    result = store.query(dataset, as_of_ns=mid_day)
    bid = result["bid_price"]
    ask = result["ask_price"]
    n = len(result)
    frac_pos_bid = float((bid > 0).mean())
    frac_pos_ask = float((ask > 0).mean())
    frac_uncrossed = float((bid <= ask).mean())
    frac_clock_ok = float(((shard["event_ts_ns"] - shard["received_ts_ns"]) <= 60_000_000_000).mean())

    _row("realdata_rows_ingested", float(rows), "> 10000", "count", rows > 10_000)
    assert rows > 10_000, f"only {rows} quotes ingested"
    _row("realdata_midday_query_rows", float(n), "> 1000", "count", n > 1_000)
    assert n > 1_000, f"mid-day point-in-time query returned only {n} rows"
    _row("realdata_fraction_positive_bid", frac_pos_bid, ">= 0.99", "frac", frac_pos_bid >= 0.99)
    assert frac_pos_bid >= 0.99
    _row("realdata_fraction_positive_ask", frac_pos_ask, ">= 0.99", "frac", frac_pos_ask >= 0.99)
    assert frac_pos_ask >= 0.99
    _row("realdata_fraction_bid_leq_ask", frac_uncrossed, ">= 0.97", "frac", frac_uncrossed >= 0.97)
    assert frac_uncrossed >= 0.97
    _row("realdata_clock_skew_ok_fraction", frac_clock_ok, "== 1.0", "frac", frac_clock_ok >= 0.9999)
    assert frac_clock_ok >= 0.9999

    # Point-in-time discipline on REAL data: query at 5%ile and 95%ile stamps;
    # early slice must be a strict subset of the late slice.
    early_t = int(shard["event_ts_ns"].quantile(0.05))
    late_t = int(shard["event_ts_ns"].quantile(0.95))
    early = store.query(dataset, as_of_ns=early_t)
    late = store.query(dataset, as_of_ns=late_t)
    subset_ok = set(early["symbol"].value_counts().index.tolist()) <= set(late["symbol"].value_counts().index.tolist())
    _row("realdata_pit_growth_subset", len(late) - len(early), "> 0", "rows", len(late) > len(early) and subset_ok)
    assert len(late) > len(early) and subset_ok

    _print_table("PHASE 0 REAL-DATA FEASIBILITY GATE (Deribit options, Tardis sample)")


def test_gate_summary_epilogue() -> None:
    """Final Phase-0 epilogue printed for the run log (no assertions)."""
    print(
        "\nPHASE 0 KILL GATE SUMMARY\n"
        "  criteria: (1) real bid/ask option history loads point-in-time  "
        "[test_phase0_real_data_feasibility_gate]\n"
        "            (2) ledger reconciles to 1e-6 across +/-30% gaps, incl. 1/S^2 "
        "hedge-drift proof  [test_core_numeraire_reconciliation]\n"
        "  all green -> proceed to PHASE 1 (SVI calibration + VRP regression)\n"
    )
