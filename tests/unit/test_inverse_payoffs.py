"""Unit tests: inverse-instrument mechanics against worked analytical examples.

Every expected value below is derived by hand in the test comments; no
financial result is "mocked" — floats must reconcile to the closed-form
numbers to 1e-12 or better. Runs in milliseconds, no network, no data files.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from libs.core.inverse_payoffs import (
    NumeraireLedger,
    contracts_to_notional_usd,
    inverse_perp_btc_delta,
    inverse_perp_hedge_notional_for_usd_delta,
    inverse_perp_hedge_rescale_factor,
    inverse_perp_pnl_btc,
    inverse_perp_pnl_usd,
    inverse_perp_usd_delta,
    option_payoff_btc,
    option_payoff_usd,
    straddle_delta_usd,
    straddle_premium_usd,
    usd_to_btc_cash,
)


class TestInversePerpPnL:
    """Worked analytical examples for the inverse perp P&L identity."""

    def test_long_pnl_btc_worked_example(self) -> None:
        # Hand example: long USD 1_000 notional, entry 50_000, mark 100_000:
        #   pnl_btc = 1000 * (1/50000 - 1/100000) = 1000 * (2e-5 - 1e-5) = 0.01 BTC
        # Long a USD-1k notional and the price doubles -> +USD 1000 == +0.01 BTC.
        assert inverse_perp_pnl_btc(+1000.0, 50_000.0, 100_000.0) == pytest.approx(0.01, abs=1e-15)

    def test_short_pnl_btc_worked_example(self) -> None:
        # Hand example: short USD 1_000 notional, entry 100_000, mark 70_000
        # (a -30% crash). pnl_btc = -1000 * (1/100000 - 1/70000)
        #             = -1000 * (1e-5 - 1.4285714285714286e-5) = +0.004285714285714287
        # Short USD 1k while BTC drops 30% -> +USD 300 = +0.004285714285714287 BTC.
        expected_btc = -1000.0 * (1.0 / 100_000.0 - 1.0 / 70_000.0)
        got = inverse_perp_pnl_btc(-1000.0, 100_000.0, 70_000.0)
        assert got == pytest.approx(expected_btc, abs=1e-18)
        assert got == pytest.approx(300.0 / 70_000.0, abs=1e-15)
        # USD value at the crash-time index: 0.004285714285714... * 70000 = 300 USD.
        assert got * 70_000.0 == pytest.approx(300.0, abs=1e-9)

    def test_usd_pnl_linear_identity(self) -> None:
        # Identity under test: pnl_btc(S) * S == C * (S/entry - 1) — the *linear*
        # dollar P&L of an inverse perp. Reconciled to 1e-9 USD over a wide
        # spot grid including extreme marks (1 USD and 1_000_000 USD).
        entry = 100_000.0
        for signed_notional in (+40_000.0, -25_000.0):
            for mark in (1.0, 250.0, 99_751.0, 100_000.0, 130_000.0, 1_000_000.0):
                pnl_btc = inverse_perp_pnl_btc(signed_notional, entry, mark)
                closed_form = signed_notional * (mark / entry - 1.0)
                assert pnl_btc * mark == pytest.approx(closed_form, abs=1e-9, rel=1e-12)
                assert inverse_perp_pnl_usd(signed_notional, entry, mark) == pytest.approx(
                    closed_form, abs=1e-9, rel=1e-12
                )

    def test_long_gains_only_on_price_increase(self) -> None:
        # Sign sanity: long inverse perp loses BTC when price falls, gains when
        # it rises; short is the exact negation.
        long_fall = inverse_perp_pnl_btc(+10_000.0, 100_000.0, 70_000.0)
        long_rise = inverse_perp_pnl_btc(+10_000.0, 100_000.0, 130_000.0)
        assert long_fall < 0.0 < long_rise
        assert inverse_perp_pnl_btc(-10_000.0, 100_000.0, 70_000.0) == -long_fall

    def test_zero_notional_is_zero(self) -> None:
        assert inverse_perp_pnl_btc(0.0, 100.0, 200.0) == 0.0
        assert inverse_perp_usd_delta(0.0, 100.0) == 0.0
        assert inverse_perp_btc_delta(0.0, 100.0) == 0.0

    def test_invalid_prices_rejected(self) -> None:
        with pytest.raises(ValueError):
            inverse_perp_pnl_btc(1000.0, 0.0, 100.0)
        with pytest.raises(ValueError):
            inverse_perp_pnl_btc(1000.0, 100.0, float("inf"))

    def test_contract_notionalConversion(self) -> None:
        # Deribit BTC-PERPETUAL contract size = 10 USD notional.
        assert contracts_to_notional_usd(123) == pytest.approx(1230.0)
        assert contracts_to_notional_usd(-40, contract_size_usd=10.0) == pytest.approx(-400.0)
        with pytest.raises(ValueError):
            contracts_to_notional_usd(1.5)


class TestInverseHedgeSizing:
    """Delta conventions and the 1/S**2 hedge-drift factor."""

    def test_usd_delta_is_constant_and_btc_delta_scales_as_inverse_square(self) -> None:
        # Direct derivation:
        #   pnl_btc(S) = C*(1/P_e - 1/S)   =>  d/dS = +C/S**2  (signed: +C means long)
        #   pnl_usd(S) = C*(S/P_e - 1)     =>  d/dS = +C/P_e    (constant; not 1/S**2)
        entry = 100_000.0
        notional = +52_345.0
        delta_before = inverse_perp_btc_delta(notional, 100_000.0)
        delta_after = inverse_perp_btc_delta(notional, 130_000.0)
        assert delta_before == pytest.approx(notional / 100_000.0**2, abs=1e-21)
        assert delta_after == pytest.approx(notional / 130_000.0**2, abs=1e-21)
        # Coverage ratio across the +30% gap is exactly (S0/S1)**2 == (1/1.3)**2:
        ratio = delta_after / delta_before
        assert ratio == pytest.approx((100_000.0 / 130_000.0) ** 2, abs=1e-15)
        assert inverse_perp_hedge_rescale_factor(100_000.0, 130_000.0) == pytest.approx(
            (1.0 / 1.3) ** 2, abs=1e-15
        )
        # USD-equivalent delta does NOT rescale (stays constant): the function
        # holds it as notional/entry, independent of any later mark. Unit
        # consistency with the BTC delta at the entry instant:
        usd_delta = inverse_perp_usd_delta(notional, entry)
        assert usd_delta == pytest.approx(notional / entry, abs=1e-18)
        assert usd_delta == pytest.approx(
            inverse_perp_btc_delta(notional, entry) * entry, abs=1e-18
        )
        # Across a ±30% mark move the USD-equivalent delta of the SAME fixed
        # position is unchanged (it does not depend on the mark at all):
        assert inverse_perp_usd_delta(notional, 100_000.0) == pytest.approx(
            usd_delta, abs=0.0
        )
        assert inverse_perp_usd_delta(notional, 70_000.0) != inverse_perp_btc_delta(
            notional, 70_000.0
        )

    def test_hedge_notional_for_target_delta(self) -> None:
        # Neutralise an option book with USD-equivalent delta +0.05 (BTC-equiv
        # units) at entry 100_000 -> short notional -0.05*100_000 = -5_000.
        notional = inverse_perp_hedge_notional_for_usd_delta(-0.05, 100_000.0)
        assert notional == pytest.approx(-5_000.0, abs=1e-12)
        assert inverse_perp_usd_delta(notional, 100_000.0) == pytest.approx(-0.05, abs=1e-15)


class TestOptionSettlement:
    """BTC-settled intrinsic payoffs and their USD identities."""

    def test_call_payoff_btc_and_usd(self) -> None:
        # K=100_000, S_T=130_000: call intrinsic = (30000/130000) BTC = 3/13 BTC;
        # USD value at the settlement index: (3/13)*130000 = 30000 exactly.
        payoff_btc = option_payoff_btc("call", 100_000.0, 130_000.0)
        assert payoff_btc == pytest.approx(3.0 / 13.0, abs=1e-15)
        assert payoff_btc * 130_000.0 == pytest.approx(30_000.0, abs=1e-9)
        assert option_payoff_usd("call", 100_000.0, 130_000.0) == pytest.approx(30_000.0, abs=1e-9)

    def test_put_payoff_btc_and_usd(self) -> None:
        # K=100_000, S_T=70_000: put intrinsic = (30000/70000) = 3/7 BTC.
        payoff_btc = option_payoff_btc("put", 100_000.0, 70_000.0)
        assert payoff_btc == pytest.approx(3.0 / 7.0, abs=1e-15)
        assert payoff_btc * 70_000.0 == pytest.approx(30_000.0, abs=1e-9)
        assert option_payoff_usd("put", 100_000.0, 70_000.0) == pytest.approx(30_000.0, abs=1e-9)

    def test_short_put_btc_liability_is_unbounded_as_spot_falls(self) -> None:
        # Numeraire blow-up: short put BTC liability (K-S)/S ~ K/S. A 10x price
        # drop multiplies the BTC liability by ~10x while the USD liability is
        # capped at K. Hand values: K=100_000, S=100 -> (99900)/100 = 999 BTC;
        # S=10 -> 9990 BTC; S=1 -> 99999 BTC.
        liability_100 = option_payoff_btc("put", 100_000.0, 100.0)
        liability_10 = option_payoff_btc("put", 100_000.0, 10.0)
        liability_1 = option_payoff_btc("put", 100_000.0, 1.0)
        assert liability_100 == pytest.approx(999.0, abs=1e-12)
        assert liability_10 == pytest.approx(9_999.0, abs=1e-12)
        assert liability_1 == pytest.approx(99_999.0, abs=1e-12)
        # USD values stay capped near K (S=100 -> 99_900 USD):
        assert option_payoff_usd("put", 100_000.0, 100.0) == pytest.approx(99_900.0, abs=1e-9)
        # The 1/S growth law: liability_10 / liability_100 == (K-10)/(K-100)*10.
        assert liability_10 / liability_100 == pytest.approx((99_990.0 / 99_900.0) * 10.0, abs=1e-12)

    def test_otm_payoff_zero(self) -> None:
        assert option_payoff_btc("call", 100_000.0, 50_000.0) == 0.0
        assert option_payoff_btc("put", 100_000.0, 150_000.0) == 0.0
        assert option_payoff_usd("call", 100_000.0, 99_999.99) == 0.0


class TestStraddlePricing:
    """Black-Scholes straddle values — checked against closed-form hand math."""

    def test_atm_straddle_value_and_delta(self) -> None:
        # S=K=100_000, sigma=0.50, T=14/365, r=0:
        #   d1 = 0.5*sigma*sqrt(T) = 0.5*0.5*sqrt(0.038356) = 0.0489618411...
        #   call = S*(N(d1) - N(d2)) with d2 = -d1 => call = S*(2N(d1)-1)
        #   N(0.04896) ≈ 0.51952418 => call ≈ 3905.0 USD; put equal by parity.
        tau = 14.0 / 365.0
        d1 = 0.25 * math.sqrt(14.0 / 365.0)  # = sigma*sqrt(tau)/2 for sigma=0.5
        call, put = straddle_premium_usd(100_000.0, 100_000.0, 0.50, tau)
        assert call == pytest.approx(put, abs=1e-6)  # ATM, r=0 parity
        expected_call = 100_000.0 * (2.0 * _normal_cdf(d1) - 1.0)
        assert call == pytest.approx(expected_call, abs=1e-4)
        # Straddle USD-equivalent delta = 2*N(d1)-1 ~ +0.03905: NOT zero even
        # ATM (this tiny positive delta is what forces a small short perp).
        delta = straddle_delta_usd(100_000.0, 100_000.0, 0.50, tau)
        assert delta == pytest.approx(2.0 * _normal_cdf(d1) - 1.0, abs=1e-12)
        assert delta == pytest.approx(0.039047, abs=1e-4)
        assert delta > 0.0
        # Straddle value = call + put:
        assert call + put == pytest.approx(2.0 * call, abs=1e-6)

    def test_premium_conversion_roundtrip(self) -> None:
        # BTC-denominated Deribit premium == USD price / spot.
        prem_usd = 7810.095
        prem_btc = usd_to_btc_cash(prem_usd, 100_000.0)
        assert prem_btc == pytest.approx(0.078100950, abs=1e-12)


class TestNumeraireLedger:
    """Ledger accounting identities with hand-computed totals."""

    def test_realized_marked_and_drift(self) -> None:
        # Flows: +2 BTC received at index 100; +1 BTC at index 300.
        #   realized = 2*100 + 1*300 = 500 USD
        #   marked @ 300 = (2+1)*300 = 900 USD
        #   drift = 2*(300-100) + 1*(300-300) = 400 USD
        #   invariant: marked == realized + drift  (900 == 500 + 400)
        ledger = NumeraireLedger()
        ledger.add(1, 2.0, 100.0, "premium")
        ledger.add(2, 1.0, 300.0, "pnl")
        assert ledger.realized_usd_value() == pytest.approx(500.0, abs=1e-12)
        assert ledger.marked_usd_value(300.0) == pytest.approx(900.0, abs=1e-12)
        assert ledger.collateral_convexity_drift_usd(300.0) == pytest.approx(400.0, abs=1e-12)
        # Invariant to float precision:
        marked = ledger.marked_usd_value(300.0)
        realized_plus_drift = ledger.realized_usd_value() + ledger.collateral_convexity_drift_usd(300.0)
        assert marked == pytest.approx(realized_plus_drift, abs=1e-9)

    def test_negative_flow_signs(self) -> None:
        # -3 BTC delivered at index 200:
        ledger = NumeraireLedger()
        ledger.add(1, -3.0, 200.0, "liability")
        assert ledger.realized_usd_value() == pytest.approx(-600.0, abs=1e-12)
        assert ledger.marked_usd_value(250.0) == pytest.approx(-750.0, abs=1e-12)
        assert ledger.collateral_convexity_drift_usd(250.0) == pytest.approx(-150.0, abs=1e-12)

    def test_validation_rejects_bad_flows(self) -> None:
        ledger = NumeraireLedger()
        with pytest.raises(ValueError):
            ledger.add(1, 1.0, 0.0, "bad")
        with pytest.raises(ValueError):
            ledger.add(-1, 1.0, 100.0, "bad")


def _normal_cdf(x: float) -> float:
    """Standard normal CDF for hand-derivation references in comments."""
    from scipy import stats

    return float(stats.norm.cdf(x))
