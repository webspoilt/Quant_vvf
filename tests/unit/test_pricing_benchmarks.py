"""QuantLib Math Benchmark: Dual-Run Verification for Options Pricing & Greeks.

Cross-validates internal analytical Black-76, inverse numeraire settlement formulas,
and Gatheral SVI surface calculations against institutional continuous-time C++
implementations in QuantLib (ql.BlackCalculator and ql.blackFormulaImpliedStdDev).
"""
from __future__ import annotations

import math
import numpy as np
import pytest

ql = pytest.importorskip("QuantLib", reason="QuantLib not installed (optional benchmark)")

from libs.core.inverse_payoffs import (
    straddle_delta_usd,
    straddle_premium_usd,
    usd_to_btc_cash,
)
from libs.core.svi_surface import (
    SVIParameters,
    svi_implied_vol,
    svi_total_variance,
)


def test_black76_pricing_against_quantlib_benchmark() -> None:
    """Verify internal Black-76 pricing matches QuantLib to within machine precision.

    Sweeps:
    - Strikes from deep ITM (0.3x) to deep OTM (2.5x)
    - Volatilities from 25% to 130% (crypto vol regime)
    - Expiries from 1 day to 90 days
    - Rates: 0.0% and 5.0%
    """
    spot = 65000.0
    strikes = [20000.0, 40000.0, 55000.0, 65000.0, 75000.0, 90000.0, 130000.0]
    sigmas = [0.25, 0.50, 0.75, 1.10]
    maturities = [1.0 / 365.0, 7.0 / 365.0, 14.0 / 365.0, 30.0 / 365.0, 90.0 / 365.0]
    rates = [0.0, 0.05]

    for r in rates:
        for tau in maturities:
            discount = math.exp(-r * tau)
            forward = spot * math.exp(r * tau)
            for sigma in sigmas:
                std_dev = sigma * math.sqrt(tau)
                for strike in strikes:
                    # 1. Platform internal calculations
                    call_internal, put_internal = straddle_premium_usd(
                        spot=spot,
                        strike=strike,
                        sigma=sigma,
                        tau_years=tau,
                        r=r,
                    )

                    # 2. QuantLib benchmark calculations
                    call_payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
                    put_payoff = ql.PlainVanillaPayoff(ql.Option.Put, strike)

                    ql_call_calc = ql.BlackCalculator(call_payoff, forward, std_dev, discount)
                    ql_put_calc = ql.BlackCalculator(put_payoff, forward, std_dev, discount)

                    ql_call_price = ql_call_calc.value()
                    ql_put_price = ql_put_calc.value()

                    # 3. Assert relative tolerance within 1e-9 (or absolute 1e-7 for deep OTM)
                    np.testing.assert_allclose(
                        call_internal,
                        ql_call_price,
                        rtol=1e-9,
                        atol=1e-7,
                        err_msg=f"Call mismatch at K={strike}, sig={sigma}, tau={tau}, r={r}",
                    )
                    np.testing.assert_allclose(
                        put_internal,
                        ql_put_price,
                        rtol=1e-9,
                        atol=1e-7,
                        err_msg=f"Put mismatch at K={strike}, sig={sigma}, tau={tau}, r={r}",
                    )


def test_straddle_delta_against_quantlib_benchmark() -> None:
    """Verify net straddle delta against sum of QuantLib call and put spot deltas."""
    spot = 70000.0
    strikes = [50000.0, 65000.0, 70000.0, 80000.0, 100000.0]
    sigmas = [0.40, 0.65, 0.90]
    maturities = [3.0 / 365.0, 14.0 / 365.0, 45.0 / 365.0]

    for tau in maturities:
        std_dev = 0.5 * math.sqrt(tau)
        discount = 1.0  # r=0
        forward = spot
        for sigma in sigmas:
            std_dev = sigma * math.sqrt(tau)
            for strike in strikes:
                internal_delta = straddle_delta_usd(
                    spot=spot,
                    strike=strike,
                    sigma=sigma,
                    tau_years=tau,
                    r=0.0,
                )

                call_calc = ql.BlackCalculator(
                    ql.PlainVanillaPayoff(ql.Option.Call, strike), forward, std_dev, discount
                )
                put_calc = ql.BlackCalculator(
                    ql.PlainVanillaPayoff(ql.Option.Put, strike), forward, std_dev, discount
                )

                ql_call_delta = call_calc.delta(spot)
                ql_put_delta = put_calc.delta(spot)
                ql_straddle_delta = ql_call_delta + ql_put_delta

                np.testing.assert_allclose(
                    internal_delta,
                    ql_straddle_delta,
                    rtol=1e-9,
                    atol=1e-8,
                    err_msg=f"Straddle delta mismatch at K={strike}, sig={sigma}, tau={tau}",
                )


def test_svi_implied_vol_quantlib_roundtrip_inversion() -> None:
    """Verify SVI implied volatilities round-trip invert cleanly via QuantLib IV solver.

    SVI w(k) -> sigma_SVI -> Black-76 price -> QuantLib blackFormulaImpliedStdDev -> sigma_QL.
    """
    params = SVIParameters(a=0.04, b=0.12, rho=-0.35, m=0.02, sigma=0.15)
    tau = 21.0 / 365.0
    spot = 65000.0
    forward = spot
    discount = 1.0

    # Grid of strikes around the forward
    log_moneyness_grid = np.linspace(-0.35, 0.35, 15)
    strikes = spot * np.exp(log_moneyness_grid)

    for strike in strikes:
        k = math.log(strike / forward)
        sigma_svi = float(svi_implied_vol(k, tau, params))

        # Generate option price using internal Black formula
        call_price, _ = straddle_premium_usd(
            spot=spot,
            strike=strike,
            sigma=sigma_svi,
            tau_years=tau,
            r=0.0,
        )

        # Invert via QuantLib C++ solver
        std_dev_ql = ql.blackFormulaImpliedStdDev(
            ql.Option.Call,
            strike,
            forward,
            call_price,
            discount,
        )
        sigma_ql = std_dev_ql / math.sqrt(tau)

        # Verify inverted vol matches SVI vol to within QuantLib default solver tolerance (1e-6)
        np.testing.assert_allclose(
            sigma_ql,
            sigma_svi,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"QuantLib IV inversion mismatch at strike {strike}",
        )


def test_inverse_numeraire_put_call_parity_with_quantlib() -> None:
    """Verify BTC-denominated put-call parity matches QuantLib discounted payoff."""
    spot = 68000.0
    strike = 65000.0
    sigma = 0.60
    tau = 14.0 / 365.0
    r = 0.03
    discount = math.exp(-r * tau)
    forward = spot * math.exp(r * tau)

    call_usd, put_usd = straddle_premium_usd(spot, strike, sigma, tau, r=r)

    # Deribit BTC-denominated premiums
    call_btc = usd_to_btc_cash(call_usd, spot)
    put_btc = usd_to_btc_cash(put_usd, spot)

    # In BTC numeraire: Call_BTC - Put_BTC == (Forward - Strike) * discount / Spot
    parity_lhs_btc = call_btc - put_btc
    parity_rhs_btc = (forward - strike) * discount / spot

    np.testing.assert_allclose(parity_lhs_btc, parity_rhs_btc, rtol=1e-10)

    # Check parity with QuantLib values directly
    ql_call = ql.BlackCalculator(ql.PlainVanillaPayoff(ql.Option.Call, strike), forward, sigma * math.sqrt(tau), discount).value()
    ql_put = ql.BlackCalculator(ql.PlainVanillaPayoff(ql.Option.Put, strike), forward, sigma * math.sqrt(tau), discount).value()

    np.testing.assert_allclose(call_usd, ql_call, rtol=1e-10)
    np.testing.assert_allclose(put_usd, ql_put, rtol=1e-10)
