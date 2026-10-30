"""One-shot builder: research/phase0_numeraire.ipynb — executes the Phase-0
numeraire reconciliation against the SAME libs.core functions used by tests,
sim, and (later) the live actor. Generated+executed via nbclient."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells: list = []

cells.append(nbf.v4.new_markdown_cell(
    "# Phase 0 — Numeraire P&L Accounting Falsification Harness\n"
    "\n"
    "**Hypothesis under test (nothing else):** Deribit BTC-settled inverse instruments can\n"
    "be accounted with a single BTC-denominated ledger that reconciles to USD closed forms\n"
    "to 1e-6 across ±30% instantaneous spot shocks, and the point-in-time store admits zero\n"
    "lookahead. If this fails, the project HALTS.\n"
    "\n"
    "All math comes from `libs.core.inverse_payoffs` — the exact module the regression\n"
    "suite, the walk-forward simulator, and the live actor import (no research/prod drift).\n"
))

cells.append(nbf.v4.new_code_cell(
    "from libs.core.inverse_payoffs import (\n"
    "    simulate_instant_gap, straddle_premium_usd, straddle_delta_usd,\n"
    "    inverse_perp_pnl_btc, inverse_perp_pnl_usd,\n"
    "    inverse_perp_hedge_notional_for_usd_delta, inverse_perp_btc_delta,\n"
    "    inverse_perp_hedge_rescale_factor,\n"
    ")\n"
    "import pandas as pd\n"
    "S0, SIGMA, TAU = 100_000.0, 0.50, 14.0/365.0\n"
))

cells.append(nbf.v4.new_code_cell(
    "rows = []\n"
    "for gap in (0.70, 1.30):\n"
    "    sc = simulate_instant_gap(S0, SIGMA, TAU, gap)\n"
    "    call0, put0 = straddle_premium_usd(S0, S0, SIGMA, TAU)\n"
    "    prem_btc = (call0 + put0) / S0\n"
    "    delta0 = straddle_delta_usd(S0, S0, SIGMA, TAU)\n"
    "    hedge = inverse_perp_hedge_notional_for_usd_delta(-delta0, S0)\n"
    "    call1, put1 = straddle_premium_usd(sc.spot_after, S0, SIGMA, TAU)\n"
    "    v1 = call1 + put1\n"
    "    pnl_btc = inverse_perp_pnl_btc(hedge, S0, sc.spot_after)\n"
    "    exact_cf = sc.spot_after * (prem_btc + pnl_btc) - v1\n"
    "    rows.append(dict(\n"
    "        gap=round(gap, 2), premium_btc=prem_btc, straddle_delta=delta0,\n"
    "        hedge_usd=hedge, straddle_value_after_usd=v1,\n"
    "        perp_btc_pnl=pnl_btc, perp_usd_pnl=inverse_perp_pnl_usd(hedge, S0, sc.spot_after),\n"
    "        exact_usd_pnl=exact_cf, ledger_usd_pnl=sc.exact_marked_usd_pnl,\n"
    "        naive_usd_pnl=sc.naive_realized_usd_pnl,\n"
    "        drift_usd=sc.collateral_drift_usd,\n"
    "        ledger_minus_closed=abs(sc.exact_marked_usd_pnl - exact_cf),\n"
    "        hedge_rescale=(sc.spot_entry/sc.spot_after)**2,\n"
    "        rescale_delta_ratio=(\n"
    "            inverse_perp_btc_delta(hedge, sc.spot_after)\n"
    "            / inverse_perp_btc_delta(hedge, sc.spot_entry)\n"
    "        ),\n"
    "    ))\n"
    "pd.set_option('display.float_format', lambda x: f'{x:,.6f}')\n"
    "df = pd.DataFrame(rows)\n"
    "df\n"
))

cells.append(nbf.v4.new_markdown_cell(
    "Reading: `ledger_minus_closed` is the ordre-1e-6 reconciliation residual;\n"
    "`rescale_delta_ratio` reproduces the (S0/S1)^2 BTC-delta hedge drift — the\n"
    "non-linear risk the naive `naive_usd_pnl` accounting silently eats via `drift_usd`.\n"
    "With K=S (ATM): naive vs exact == prem_btc*(S1-S0): the premium converted at the *\nterminal* index differs from premium frozen at the entry index by the collateral\n"
    "convexity term.\n"
))

cells.append(nbf.v4.new_code_cell(
    "worst = max(df['ledger_minus_closed'])\n"
    "drismatch = max(abs(df['rescale_delta_ratio'] - df['hedge_rescale']))\n"
    "verdict = 'PROCEED to Phase 1' if (worst <= 1e-6 and drismatch <= 1e-12) else 'HALT'\n"
    "f'worst ledger residual: {worst:.3e} USD | worst rescale mismatch: {drismatch:.1e} | KILL GATE 0: {verdict}'\n"
))

nb.cells = cells
path = ROOT / "research" / "phase0_numeraire.ipynb"
nbf.write(nb, str(path))
print(f"written {path}")
