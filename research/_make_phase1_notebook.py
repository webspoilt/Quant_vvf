"""One-shot builder: research/phase1_vrp_regression.ipynb — executes Phase 1 SVI calibration
and VRP OLS regression against the SAME libs.core functions used by tests, sim, and live actor.
"""
from __future__ import annotations

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells: list = []

cells.append(nbf.v4.new_markdown_cell(
    "# Phase 1 — SVI Calibration & Baseline VRP Regression Falsification Harness\n"
    "\n"
    "**Hypothesis under test:** Deribit BTC options exhibit a structural Volatility Risk Premium\n"
    "$(IV^2 - RV^2) > 0$ that monetizes into statistically positive net returns after subtracting\n"
    "Deribit option taker fees (0.03%), perpetual taker fees (0.05%), and bid/ask crossing slippage.\n"
    "\n"
    "**Kill-gate criteria:** If $\\beta \\le 0$ or net-of-cost $\\alpha \\le 0$ ($p > 0.05$), the project HALTS.\n"
))

cells.append(nbf.v4.new_code_cell(
    "import numpy as np\n"
    "import pandas as pd\n"
    "import statsmodels.api as sm\n"
    "from libs.core.svi_surface import SVIParameters, calibrate_svi_quasi_explicit, check_butterfly_arbitrage, svi_implied_vol, svi_total_variance\n"
    "from tests.regression.test_phase1_vrp_regression import generate_btc_vrp_series\n"
))

cells.append(nbf.v4.new_code_cell(
    "# 1. SVI Slice Calibration Demonstration\n"
    "tau = 14.0 / 365.0\n"
    "target_iv = 0.65\n"
    "w0 = (target_iv ** 2) * tau\n"
    "true_params = SVIParameters(a=w0 - 0.1*w0*0.1, b=0.1*w0, rho=-0.20, m=0.0, sigma=0.10)\n"
    "k_grid = np.linspace(-0.3, 0.3, 15)\n"
    "w_quotes = svi_total_variance(k_grid, true_params)\n"
    "calib = calibrate_svi_quasi_explicit(k_grid, w_quotes)\n"
    "is_free, min_g = check_butterfly_arbitrage(calib)\n"
    "print(f'Calibrated SVI: {calib}')\n"
    "print(f'Arbitrage-free butterfly: {is_free} (min g = {min_g:.6f})')\n"
))

cells.append(nbf.v4.new_code_cell(
    "# 2. Rolling 14-day DTE Straddle Series Simulation\n"
    "trades = generate_btc_vrp_series(n_days=120, seed=42)\n"
    "df = pd.DataFrame([vars(t) for t in trades])\n"
    "df[['cycle_id', 'entry_spot', 'iv_atm', 'rv_realized', 'iv2_minus_rv2', 'pnl_gross_usd', 'total_costs_usd', 'pnl_net_usd']].head(10)\n"
))

cells.append(nbf.v4.new_code_cell(
    "# 3. OLS Regression with Newey-West HAC Standard Errors\n"
    "spread = df['iv2_minus_rv2'].values\n"
    "mean_spread = float(np.mean(spread))\n"
    "spread_centered = spread - mean_spread\n"
    "X = sm.add_constant(spread_centered)\n"
    "y = df['pnl_net_usd'].values\n"
    "model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 14})\n"
    "print(model.summary())\n"
))

cells.append(nbf.v4.new_code_cell(
    "# 4. Kill-Gate 1 Decision\n"
    "net_alpha = float(model.params[0])\n"
    "beta = float(model.params[1])\n"
    "alpha_p = float(model.pvalues[0])\n"
    "beta_p = float(model.pvalues[1])\n"
    "passed = (beta > 0 and beta_p < 0.05 and net_alpha > 0 and alpha_p < 0.05)\n"
    "print(f'KILL GATE 1 VERDICT: {\"PASSED -> PROCEED TO PHASE 2\" if passed else \"HALT PROJECT\"}')\n"
))

nb.cells = cells
path = ROOT / "research" / "phase1_vrp_regression.ipynb"
nbf.write(nb, str(path))
print(f"written {path}")
