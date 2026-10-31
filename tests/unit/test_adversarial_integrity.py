"""Adversarial integrity harness: corrupt data must be rejected deterministically.

Companion suite for the fail-closed remediation (audit V-01..V-22). Each test
injects a plausible real-world corruption — NaN/Inf telemetry, path traversal,
insufficient budgets, cross-expiry contamination, post-test training data —
and asserts the gate REJECTS/ABORTS rather than silently passing. IEEE 754
comparison semantics (NaN > x == False) are the failure mode under test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from execution.nautilus_actor import VRPActorConfig, VRPOptionsExecutionActor
from libs.core.har_rv import qlike_loss
from libs.core.realized_measures import compute_log_returns
from libs.core.svi_surface import SVIParameters
from libs.data.canonical_store import CanonicalStore
from monitoring.greeks_monitor import GreeksMonitor
from risk.limits import PortfolioRiskMonitor, RiskLimitsConfig
from risk.risk_constants import RiskGatekeeper
from risk.tail_triggers import TailTriggerMonitor
from simulation.walk_forward_sim import PurgedWalkForwardEngine
from strategies.greek_solver import DiscreteGreekSolver, OptionChainQuote
from strategies.vrp_profile import TargetGreekProfile

BAD_NUMBERS = (float("nan"), float("inf"), float("-inf"))


def _profile(**overrides: float) -> TargetGreekProfile:
    base = dict(
        target_vega_btc=-10.0,
        min_delta_btc=-0.15,
        max_delta_btc=0.15,
        max_gamma=0.005,
        min_dte_days=7.0,
        max_dte_days=35.0,
        max_margin_utilization=0.70,
        iv_atm=0.65,
        rv_forecast=0.50,
        variance_spread=0.15,
        is_active=True,
        reason="Active",
    )
    base.update(overrides)
    return TargetGreekProfile(**base)


def _quote(instrument_id: str, strike: float, tau: float, kind: str) -> OptionChainQuote:
    return OptionChainQuote(
        instrument_id=instrument_id,
        strike=strike,
        expiry_tau=tau,
        option_type=kind,  # type: ignore[arg-type]
        bid_price_btc=0.045,
        ask_price_btc=0.046,
        mid_price_btc=0.0455,
        delta_btc=0.52 if kind == "call" else -0.48,
        gamma=0.00002,
        vega_btc=0.05,
        initial_margin_btc=0.15,
    )


# ---------------------------------------------------------------------------
# V-01: PortfolioRiskMonitor must fail closed on NaN/Inf and absurd economics
# ---------------------------------------------------------------------------
class TestRiskLimitsFailClosed:
    def setup_method(self) -> None:
        self.monitor = PortfolioRiskMonitor(RiskLimitsConfig())
        self.valid = dict(
            portfolio_delta_usd=1_000.0,
            spot_price=50_000.0,
            portfolio_vega_btc=10.0,
            portfolio_gamma=0.001,
            current_margin_used_btc=0.5,
            total_equity_btc=2.0,
        )

    @pytest.mark.parametrize("field", list(dict(
        portfolio_delta_usd=1.0, spot_price=1.0, portfolio_vega_btc=1.0,
        portfolio_gamma=1.0, current_margin_used_btc=1.0, total_equity_btc=1.0,
    )))
    def test_nan_in_any_field_is_rejected(self, field: str) -> None:
        for bad in BAD_NUMBERS:
            inputs = dict(self.valid)
            inputs[field] = bad
            result = self.monitor.check_portfolio(**inputs)
            assert not result.passed, f"{field}={bad} must fail the check"
            assert any("CORRUPT_INPUT" in b for b in result.breaches)

    def test_negative_spot_rejected(self) -> None:
        result = self.monitor.check_portfolio(**{**self.valid, "spot_price": -10.0})
        assert not result.passed
        assert any("INVALID_SPOT_PRICE" in b for b in result.breaches)

    def test_zero_equity_rejected_as_insolvency(self) -> None:
        for equity in (0.0, -0.3):
            result = self.monitor.check_portfolio(**{**self.valid, "total_equity_btc": equity})
            assert not result.passed
            assert any("INSOLVENCY_DETECTED" in b for b in result.breaches)

    def test_sanitized_inputs_still_pass(self) -> None:
        assert self.monitor.check_portfolio(**self.valid).passed

    def test_nan_config_limits_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError):
            PortfolioRiskMonitor(RiskLimitsConfig(max_portfolio_delta_abs=float("nan")))


# ---------------------------------------------------------------------------
# V-02: RiskGatekeeper must fail closed on corrupt proposals; V-24 sanitized
# ---------------------------------------------------------------------------
class TestGatekeeperFailClosed:
    def test_nan_proposals_rejected(self) -> None:
        for bad in BAD_NUMBERS:
            for field in ("proposed_vega_btc", "proposed_delta_btc", "proposed_gamma",
                          "proposed_margin_btc", "total_equity_btc"):
                kwargs = dict(
                    proposed_vega_btc=-10.0, proposed_delta_btc=0.05, proposed_gamma=0.001,
                    proposed_margin_btc=1.0, total_equity_btc=10.0,
                )
                kwargs[field] = bad
                result = RiskGatekeeper().validate_proposed_execution(**kwargs)
                assert not result.approved, f"{field}={bad} must be rejected"
                assert any("CORRUPT_INPUT" in r for r in result.reasons)

    def test_fail_closed_reason_is_sanitized(self) -> None:
        # V-24: raw exception text must not leak into reasons. Force the
        # evaluation-error branch with a hostile strike-count payload.
        result = RiskGatekeeper().validate_proposed_execution(
            proposed_vega_btc=-10.0, proposed_delta_btc=0.05, proposed_gamma=0.001,
            proposed_margin_btc=1.0, total_equity_btc=10.0,
            strike_contract_counts=["not-a-count", "also-bad"],  # type: ignore[list-item]
        )
        assert not result.approved
        assert result.reasons[0].startswith("FAIL-CLOSED triggered due to evaluation error:")
        # Sanitized: only the exception *type* name, no raw payload/message.
        assert "not-a-count" not in result.reasons[0]
        assert result.reasons[0].endswith(": TypeError")

    def test_pure_straddle_is_not_deadlocked_by_concentration(self) -> None:
        # V-14: single-strike structures are structural; share is 100% by
        # construction and must NOT veto the legitimate one-pair strategy.
        result = RiskGatekeeper().validate_proposed_execution(
            proposed_vega_btc=-10.0, proposed_delta_btc=0.04, proposed_gamma=0.00004,
            proposed_margin_btc=1.0, total_equity_btc=10.0,
            strike_contract_counts=[200],  # one strike incl. call+put legs
        )
        assert result.approved

    def test_multi_strike_concentration_still_enforced(self) -> None:
        result = RiskGatekeeper().validate_proposed_execution(
            proposed_vega_btc=-10.0, proposed_delta_btc=0.05, proposed_gamma=0.001,
            proposed_margin_btc=1.0, total_equity_btc=10.0,
            strike_contract_counts=[70, 30, 0],  # 70% at one strike > 35%
        )
        assert not result.approved
        assert any("concentration" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# V-03: TailTriggerMonitor must veto on corrupt telemetry, not emit ALL_CLEAR
# ---------------------------------------------------------------------------
class TestTailTriggerFailClosed:
    def test_nan_scalars_vetoed(self) -> None:
        monitor = TailTriggerMonitor()
        defaults = dict(
            current_24h_rv=0.40,
            baseline_30d_rv=0.30,
            current_25d_skew=0.05,
            historical_90d_skew=np.linspace(0.0, 0.1, 90),
            current_funding_rate=0.0001,
            historical_28d_funding=np.linspace(0.0, 0.0002, 112),
        )
        for field in ("current_24h_rv", "baseline_30d_rv", "current_25d_skew", "current_funding_rate"):
            for bad in BAD_NUMBERS:
                decision = monitor.evaluate(**{**defaults, field: bad})
                assert decision.veto_active, f"{field}={bad} must veto"
                assert decision.action == "HALT_NEW_ENTRIES"
                assert any("CORRUPT_INPUT" in r for r in decision.reasons)

    def test_nan_in_history_arrays_vetoed(self) -> None:
        monitor = TailTriggerMonitor()
        decision = monitor.evaluate(
            current_24h_rv=0.40,
            baseline_30d_rv=0.30,
            current_25d_skew=0.05,
            historical_90d_skew=np.concatenate([np.linspace(0.0, 0.1, 89), [np.nan]]),
            current_funding_rate=0.0001,
            historical_28d_funding=np.linspace(0.0, 0.0002, 112),
        )
        assert decision.veto_active
        assert any("historical_90d_skew" in r for r in decision.reasons)

    def test_corrupt_telemetry_does_not_arm_shock_cooldown(self) -> None:
        # Cooldown encodes a live tail event; data corruption is a different
        # state and must not silently consume cooldown book-keeping.
        monitor = TailTriggerMonitor()
        before = monitor.cooldown_remaining
        monitor.evaluate(
            current_24h_rv=float("nan"), baseline_30d_rv=0.30, current_25d_skew=0.05,
            historical_90d_skew=np.linspace(0.0, 0.1, 90),
            current_funding_rate=0.0001,
            historical_28d_funding=np.linspace(0.0, 0.0002, 112),
        )
        assert monitor.cooldown_remaining == before

    def test_negative_baseline_vetoed_not_clamped(self) -> None:
        monitor = TailTriggerMonitor()
        decision = monitor.evaluate(
            current_24h_rv=0.40, baseline_30d_rv=-0.30, current_25d_skew=0.05,
            historical_90d_skew=np.linspace(0.0, 0.1, 90),
            current_funding_rate=0.0001,
            historical_28d_funding=np.linspace(0.0, 0.0002, 112),
        )
        assert decision.veto_active
        assert any("Baseline Invalid" in r for r in decision.reasons)


# ---------------------------------------------------------------------------
# V-08: NaN tick must trip the breaker and NOT poison _last_tick_price
# ---------------------------------------------------------------------------
class TestTickDataIntegrity:
    def test_nan_tick_trips_corrupt_data_breaker(self) -> None:
        actor = VRPOptionsExecutionActor(VRPActorConfig())
        assert actor.on_tick(1_000_000_000, 40_000.0)
        assert not actor.on_tick(1_000_000_001, float("nan"))
        assert actor.circuit_breaker_tripped
        event = actor.circuit_breaker_events[-1]
        assert event.action == "HALT_CORRUPT_TICK_DATA"

    def test_nan_tick_does_not_poison_last_price(self) -> None:
        actor = VRPOptionsExecutionActor(VRPActorConfig())
        actor.on_tick(1_000_000_000, 40_000.0)
        actor.on_tick(1_000_000_001, float("nan"))
        assert actor._last_tick_price == 40_000.0  # corrupted tick NOT stored

    def test_zero_and_inf_ticks_trip_breaker(self) -> None:
        for bad in (0.0, -1.0, float("inf"), -float("inf")):
            actor = VRPOptionsExecutionActor(VRPActorConfig())
            assert actor.on_tick(1, bad) is False
            assert actor.circuit_breaker_tripped

    def test_reset_clears_corrupt_last_tick_state(self) -> None:
        # Checklist item 27: stale pre-halt ticks are corrupt by definition.
        actor = VRPOptionsExecutionActor(VRPActorConfig())
        actor.on_tick(1, 40_000.0)
        actor.reset_circuit_breaker()
        assert actor._last_tick_price is None
        assert actor._last_tick_ts_ns == 0


# ---------------------------------------------------------------------------
# V-04: CanonicalStore dataset names must prevent path traversal (V-19 atomic)
# ---------------------------------------------------------------------------
class TestCanonicalStoreTraversal:
    def test_traversal_names_rejected(self, tmp_path: Path) -> None:
        store = CanonicalStore(tmp_path / "processed")
        for hostile in (
            "../../etc/passwd",
            "..\\..\\secrets",
            "nested/path",
            "dataset/subpath",
            ".hidden",
            "a..b",
            "",
            "date;drop",
        ):
            with pytest.raises(ValueError, match="dataset name"):
                store.upsert(hostile, pd.DataFrame({"event_ts_ns": [1], "received_ts_ns": [1]}))

    def test_safe_dataset_roundtrip_still_works(self, tmp_path: Path) -> None:
        store = CanonicalStore(tmp_path / "processed")
        df = pd.DataFrame(
            {
                "event_ts_ns": np.array([1000, 2000], dtype="int64"),
                "received_ts_ns": np.array([1000, 2000], dtype="int64"),
            }
        )
        rows = store.upsert("safe__name-1.0", df)
        assert rows == 2
        assert (tmp_path / "processed" / "safe__name-1.0.parquet").exists()
        # No temp residue (V-19 atomic write)
        assert not any(tmp.name.startswith(".") for tmp in (tmp_path / "processed").iterdir())


# ---------------------------------------------------------------------------
# V-06: solver must emit ZERO orders when budget/gamma capacity is insufficient
# ---------------------------------------------------------------------------
class TestSolverBudgetEnforcement:
    chain = [
        _quote("BTC-14D-40000-C", 40000.0, 14.0 / 365.0, "call"),
        _quote("BTC-14D-40000-P", 40000.0, 14.0 / 365.0, "put"),
    ]

    def test_insufficient_margin_produces_zero_orders(self) -> None:
        # Budget 70% * 0.1 BTC < pair margin 0.3 BTC -> must refuse, not force
        result = DiscreteGreekSolver().solve(
            _profile(target_vega_btc=-10.0),
            self.chain,
            spot_price=40_000.0,
            available_margin_btc=0.1,
        )
        assert not result.success
        assert result.orders == ()
        assert "Insufficient margin" in result.message

    def test_gamma_ceiling_produces_zero_orders(self) -> None:
        # pair_gamma 0.00004 > max_gamma 0.000001 -> must refuse
        result = DiscreteGreekSolver().solve(
            _profile(max_gamma=0.000001),
            self.chain,
            spot_price=40_000.0,
            available_margin_btc=100.0,
        )
        assert not result.success
        assert result.orders == ()
        assert "Gamma ceiling exceeded" in result.message

    def test_viable_budget_still_solves(self) -> None:
        result = DiscreteGreekSolver().solve(
            _profile(target_vega_btc=-10.0),
            self.chain,
            spot_price=40_000.0,
            available_margin_btc=100.0,
        )
        assert result.success
        assert len(result.orders) == 2
        assert result.achieved_vega_btc < 0.0

    def test_non_finite_market_inputs_refused(self) -> None:
        for bad in BAD_NUMBERS:
            result = DiscreteGreekSolver().solve(
                _profile(), self.chain, spot_price=bad, available_margin_btc=100.0,
            )
            assert not result.success and result.orders == ()


# ---------------------------------------------------------------------------
# V-10: cross-expiry diagonal pairing must be impossible
# ---------------------------------------------------------------------------
class TestExpiryGrouping:
    def test_no_diagonal_pairs_across_expiries(self) -> None:
        tau_near = 14.0 / 365.0
        tau_far = 28.0 / 365.0
        chain = [
            _quote("BTC-NEAR-40000-C", 40000.0, tau_near, "call"),
            _quote("BTC-FAR-40000-P", 40000.0, tau_far, "put"),
        ]
        result = DiscreteGreekSolver().solve(
            _profile(), chain, spot_price=40_000.0, available_margin_btc=100.0,
        )
        assert not result.success
        assert "within any single expiry" in result.message

    def test_pairs_stay_within_winner_expiry(self) -> None:
        tau_near = 14.0 / 365.0
        tau_far = 28.0 / 365.0
        near_pair = [
            _quote("BTC-NEAR-40000-C", 40000.0, tau_near, "call"),
            _quote("BTC-NEAR-40000-P", 40000.0, tau_near, "put"),
        ]
        far_pair = [
            _quote("BTC-FAR-39000-C", 39000.0, tau_far, "call"),
            _quote("BTC-FAR-39000-P", 39000.0, tau_far, "put"),
        ]
        # Both expiries offer complete pairs; whatever strike/expiry wins,
        # the two legs may never mix expiries (V-10 diagonal spread ban).
        chain = [*far_pair, *near_pair]
        result = DiscreteGreekSolver().solve(
            _profile(), chain, spot_price=39_500.0, available_margin_btc=100.0,
        )
        assert result.success
        chosen_ids = {order.instrument_id for order in result.orders}
        chosen_taus = {q.expiry_tau for q in chain if q.instrument_id in chosen_ids}
        assert len(chosen_taus) == 1  # both legs share ONE expiry


# ---------------------------------------------------------------------------
# V-11: walk-forward training data must strictly precede the test window
# ---------------------------------------------------------------------------
class TestWalkForwardNoFutureLeak:
    def test_training_never_includes_post_test_data(self) -> None:
        engine = PurgedWalkForwardEngine(holding_period_steps=14)
        folds = list(engine.generate_purged_walk_forward_splits(n_samples=200, n_splits=4, min_train_size=60))
        assert folds
        for fold in folds:
            assert len(set(fold.train_indices) & set(fold.test_indices)) == 0
            assert (fold.train_indices < fold.test_indices[0]).all(), (
                "walk-forward training leaked post-test observations (V-11)"
            )
            assert fold.test_indices[0] - fold.train_indices[-1] >= 14

    def test_growth_is_anchored_expanding_window(self) -> None:
        engine = PurgedWalkForwardEngine(holding_period_steps=0)
        folds = list(engine.generate_purged_walk_forward_splits(n_samples=200, n_splits=4, min_train_size=60))
        sizes = [len(f.train_indices) for f in folds]
        assert sizes[0] == 60 and sizes == sorted(sizes)  # anchored, growing


# ---------------------------------------------------------------------------
# V-12: profile generator must deactivate on corrupt RV/IV rather than fire
# ---------------------------------------------------------------------------
class TestProfileGeneratorFailClosed:
    def test_nan_rv_forecast_deactivates_profile(self) -> None:
        from strategies.vrp_profile import VRPProfileGenerator
        from libs.core.svi_surface import SVISurface

        surface = SVISurface()
        tau = 14.0 / 365.0
        w0 = (0.65 ** 2) * tau
        params = SVIParameters(a=w0 - 0.1 * w0 * 0.1, b=0.1 * w0, rho=-0.2, m=0.0, sigma=0.10)
        surface.add_slice(tau, params, 40_000.0)

        class CorruptHAR:
            def predict_one_step(self, history: np.ndarray) -> float:
                return float("nan")

        generator = VRPProfileGenerator()
        profile = generator.generate_profile(surface, CorruptHAR(), np.linspace(1e-3, 2e-3, 50))
        assert not profile.is_active
        assert "Fail-closed" in profile.reason

        class NegativeVarHAR:
            def predict_one_step(self, history: np.ndarray) -> float:
                return -0.01

        profile_neg = generator.generate_profile(surface, NegativeVarHAR(), np.linspace(1e-3, 2e-3, 50))
        assert not profile_neg.is_active


# ---------------------------------------------------------------------------
# V-20/V-21/V-22: numerical library guards
# ---------------------------------------------------------------------------
class TestNumericalLibraryGuards:
    def test_qlike_loss_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            qlike_loss(0.01, float("nan"))

    def test_log_returns_reject_nan_prices(self) -> None:
        with pytest.raises(ValueError, match="finite and strictly positive"):
            compute_log_returns(np.array([100.0, float("nan"), 102.0]))

    def test_svi_parameters_reject_nan(self) -> None:
        for field in ("a", "b", "rho", "m", "sigma"):
            kwargs = dict(a=0.01, b=0.1, rho=-0.2, m=0.0, sigma=0.1)
            kwargs[field] = float("nan")
            with pytest.raises(ValueError, match="finite"):
                SVIParameters(**kwargs)


# ---------------------------------------------------------------------------
# V-15/V-16/V-17: state-consistency fixes
# ---------------------------------------------------------------------------
class TestStateConsistency:
    def test_hedge_state_advances_only_on_fill(self) -> None:
        actor = VRPOptionsExecutionActor(VRPActorConfig())
        order = actor.check_and_rebalance_delta(portfolio_delta_btc=0.25, spot_price=50_000.0, timestamp_ns=1)
        assert order is not None
        assert actor.current_perp_hedge_usd == 0.0  # NOT optimistically advanced (V-15)
        assert len(actor.pending_hedge_orders) == 1

        actor.on_fill("F", instrument_id=actor._cfg.perp_instrument_id, quantity=25,
                      side="SELL", fill_price_btc=50_000.0, timestamp_ns=2)
        assert actor.current_perp_hedge_usd == pytest.approx(-12_500.0)
        assert not actor.pending_hedge_orders

    def test_greeks_monitor_rejects_insolvent_equity(self) -> None:
        monitor = GreeksMonitor()
        from monitoring.greeks_monitor import PositionLeg
        leg = PositionLeg("X", quantity=-1.0, strike=50_000.0, is_option=True,
                          option_type="call", delta_usd=25_000.0, gamma=2e-5,
                          vega_btc=0.05, theta_usd=-200.0, margin_btc=0.15)
        with pytest.raises(ValueError, match="insolvent"):
            monitor.update_state({"X": leg}, 0.0, spot_price=50_000.0,
                                 total_equity_btc=0.0, entry_spot=50_000.0)

    def test_entry_spot_refreshes_on_explicit_anchor(self) -> None:
        monitor = GreeksMonitor()
        from monitoring.greeks_monitor import PositionLeg
        leg = PositionLeg("X", quantity=-1.0, strike=50_000.0, is_option=True,
                          option_type="call", delta_usd=0.0, gamma=0.0,
                          vega_btc=0.0, theta_usd=0.0, margin_btc=0.1)
        monitor.update_state({"X": leg}, 0.0, spot_price=50_000.0, total_equity_btc=10.0)
        assert monitor._entry_spot == 50_000.0
        monitor.update_state({"X": leg}, 0.0, spot_price=65_000.0, total_equity_btc=10.0,
                             entry_spot=52_000.0)
        assert monitor._entry_spot == 52_000.0  # explicit anchors refresh (V-16)
