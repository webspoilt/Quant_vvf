"""BTC-numeraire instrument mechanics: inverse perps and BTC-settled options.

Deribit BTC options and perpetual futures are *inverse* instruments: cash flows
(option premium, perp P&L, option settlement, funding) are denominated in BTC
while every contract underlier is quoted in USD.  USD equity therefore equals

    equity_usd(S) = (BTC_holdings + cumulative_BTC_PnL) * S + USD_cash

and the mapping from BTC flows to dollars is non-linear in the BTC price S.

Sign conventions used throughout this module
--------------------------------------------
Perp position, ``signed_notional_usd > 0`` = LONG, ``< 0`` = SHORT::

    pnl_btc = signed_notional_usd * (1/entry_price - 1/mark_price)      (long form)
    pnl_usd = pnl_btc * mark_price = signed_notional_usd * (S/P_e - 1)  (linear in S)

 * LONG  +USD 1000 notional, entry 50_000, mark 100_000  ->  1000*(1/50_000 -
   1/100_000) = +0.01 BTC = +1000 USD (a USD-1k notional doubled -> +USD 1000).
 * SHORT the same amounts -> -0.01 BTC = -1000 USD.
 * USD P&L is *linear* in S with constant slope ``signed/P_e`` USD per USD of
   BTC move: an inverse perp is dollar-linear.  The non-linearity lives in the
   BTC delta::

       btc_delta(S) = d(pnl_btc)/dS = signed_notional_usd / S**2

   which scales as 1/S**2: after a gap ``S_0 -> S_1`` the BTC-delta hedge
   coverage ratio rescales by exactly ``(S_0/S_1)**2`` — the 1/S**2 hedge
   drift.  Any rebalancing policy stated in BTC-delta terms must account for
   this; a rebalancing policy stated in USD-equivalent delta terms does not
   (that delta is constant).

BTC-settled (reversed) options
------------------------------
Deribit BTC options quote and pay premium in BTC, settle the intrinsic in BTC::

    call_payoff_btc = max(0,  S_T - K) / S_T
    put_payoff_btc  = max(0,  K - S_T) / S_T
    usd value at the settlement index:  S_T * payoff_btc = max(0, -(S_T - K)) with a
    type-dependent sign, i.e. the textbook USD intrinsic.

Two numeraire consequences (both *must* be modelled, never assumed away):

1. **Short-put liability blow-up.** ``put_payoff_btc = (K - S_T)/S_T`` grows like
   ``K/S`` without bound as S -> 0 even though its USD value stays capped at K.
   An account denominated in BTC can be liquidated by a left-tail event whose
   *dollar* loss is small relative to the notional.
2. **Collateral convexity drift.** A BTC amount B received at index P_0 and
   marked at terminal index P_T has USD value drift
   ``B * (P_T - P_0)``.  Accounting that freezes BTC-denominated flows at their
   entry-time index ("naive USD accounting") therefore mis-states realised USD
   wealth by exactly that term; see :func:`simulate_instant_gap`.

Delta definitions (documented because they are frequently conflated):

* *USD-equivalent delta* ``d(cash_flow_usd)/dS`` measured in BTC-equivalent
  units (a USD delta of 1.0 = "1 BTC of underlying exposure").  Options:
  ``d_call = N(d1)`` per contract, put ``N(d1) - 1``, straddle ``2*N(d1) - 1``.
  Perp: ``signed_notional_usd / entry_price`` — constant, used for hedging.
* *BTC-value delta* ``d(cash_flow_btc)/dS`` — non-linear (e.g. an ITM call's
  BTC value ``(S-K)/S`` has positive BTC-value delta ``K/S**2``).  Used when
  risk is stated in BTC units.

All functions are pure and float-annotated; no exchange HTTP, no I/O.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal, NamedTuple

from numpy import ndarray
from scipy import stats

__all__ = [
    "BTC_FLOW_COLUMNS",
    "GapScenarioResult",
    "NumeraireFlow",
    "NumeraireLedger",
    "OptionType",
    "btc_to_usd_cash",
    "contracts_to_notional_usd",
    "inverse_perp_btc_delta",
    "inverse_perp_hedge_rescale_factor",
    "inverse_perp_hedge_notional_for_usd_delta",
    "inverse_perp_pnl_btc",
    "inverse_perp_pnl_usd",
    "inverse_perp_usd_delta",
    "option_payoff_btc",
    "option_payoff_usd",
    "simulate_instant_gap",
    "straddle_delta_usd",
    "straddle_premium_usd",
    "usd_to_btc_cash",
    "DeribitInversePricer",
]

OptionType = Literal["call", "put"]

#: Canonical column names for BTC-denominated cash-flow ledgers.
BTC_FLOW_COLUMNS: tuple[str, ...] = (
    "event_ts_ns",
    "symbol",
    "btc_amount",
    "index_price_usd",
    "label",
)


def usd_to_btc_cash(amount_usd: float, index_price: float) -> float:
    """Convert a USD cash amount to BTC at an index price.

    Parameters
    ----------
    amount_usd:
        Dollar amount (any sign).
    index_price:
        USD price of 1 BTC at the conversion instant. Must be > 0.

    Returns
    -------
    float
        BTC amount, signed like ``amount_usd``.
    """
    _validate_index(index_price)
    return amount_usd / index_price


def btc_to_usd_cash(amount_btc: float, index_price: float) -> float:
    """Convert a BTC amount to USD at an index price (``amount_btc * S``)."""
    _validate_index(index_price)
    return amount_btc * index_price


def contracts_to_notional_usd(n_contracts: float, contract_size_usd: float = 10.0) -> float:
    """Deribit BTC-PERPETUAL contract count -> USD notional.

    Deribit defines the inverse BTC-PERPETUAL contract size as 10 USD of
    notional. Position sizes quoted by the exchange are in contracts; the
    inverse P&L formulas in this module consume USD notional directly.

    Parameters
    ----------
    n_contracts:
        Number of contracts, signed (positive = long).
    contract_size_usd:
        USD notional per contract (Deribit BTC perp: 10).

    Returns
    -------
    float
        Signed USD notional.
    """
    if not float(n_contracts).is_integer():
        raise ValueError(f"n_contracts must be an integer count, got {n_contracts!r}")
    return float(n_contracts) * contract_size_usd


def inverse_perp_pnl_btc(
    signed_notional_usd: float,
    entry_price: float,
    mark_price: float,
) -> float:
    """BTC-settled P&L of an inverse perpetual position.

    For a long position (``signed_notional_usd > 0``)::

        pnl_btc = C * (1/P_entry - 1/P_mark)

    which is *negative* when the price falls (long loses on a fall) and for a
    short position it is the negation (short gains on a fall).

    Worked examples
    ---------------
    * Long C=+1000, entry 50_000, mark 100_000: 1000*(1/50_000 - 1/100_000)
      = 1000*1e-5 = +0.01 BTC (USD at mark: 0.01*100_000 = +1000).
    * Short C=-1000, entry 100_000, mark 70_000: -1000*(1/100_000 - 1/70_000)
      = -1000*(-4.2857e-6) = +0.004285714... BTC (short gains in a fall).

    Parameters
    ----------
    signed_notional_usd:
        USD notional, signed. Positive = long.
    entry_price:
        Entry (average) index price P_entry. Must be > 0.
    mark_price:
        Current/mark index price. Must be > 0.

    Returns
    -------
    float
        Signed P&L in BTC. Reconciles to the USD identity
        ``inverse_perp_pnl_usd(...) = pnl_btc * mark_price`` to float precision.
    """
    _validate_index(entry_price)
    _validate_index(mark_price)
    if signed_notional_usd == 0.0:
        return 0.0
    return signed_notional_usd * (1.0 / entry_price - 1.0 / mark_price)


def inverse_perp_pnl_usd(
    signed_notional_usd: float,
    entry_price: float,
    mark_price: float,
) -> float:
    """USD-valued P&L of an inverse perp, ``pnl_btc * mark_price``.

    Analytically linear in the mark price::

        pnl_usd = signed_notional_usd * (mark/entry - 1)

    Services the falsifiable claim "the delta-hedge on an inverse perpetual
    does not offer linear dollar protection": the *dollar* P&L of the hedge is
    linear; the non-linearity enters through the BTC-delta space and the
    BTC-settlement of option legs (see :func:`simulate_instant_gap`).
    """
    _validate_index(entry_price)
    _validate_index(mark_price)
    if signed_notional_usd == 0.0:
        return 0.0
    return signed_notional_usd * (mark_price / entry_price - 1.0)


def inverse_perp_usd_delta(signed_notional_usd: float, entry_price: float) -> float:
    """Constant USD-equivalent delta of an inverse perp: ``notional / entry``.

    Units are "BTC-equivalent" — a value of 1.0 means the USD P&L moves 1 USD
    per 1 USD move of the BTC index. Positive for long.
    """
    _validate_index(entry_price)
    if signed_notional_usd == 0.0:
        return 0.0
    return signed_notional_usd / entry_price


def inverse_perp_btc_delta(signed_notional_usd: float, mark_price: float) -> float:
    """BTC delta of an inverse perp: ``d(pnl_btc)/dS = notional / S**2``.

    This is the 1/S**2-scaled quantity that drifts as the mark moves; the
    BTC-delta after a gap ``S_0 -> S_1`` equals the entry BTC-delta times
    :func:`inverse_perp_hedge_rescale_factor`.
    """
    _validate_index(mark_price)
    if signed_notional_usd == 0.0:
        return 0.0
    return signed_notional_usd / (mark_price * mark_price)


def inverse_perp_hedge_rescale_factor(reference_price: float, mark_price: float) -> float:
    """BTC-delta hedge rescaling factor across a price move, ``(S_ref/S_mark)**2``.

    A hedge sized such that ``btc_delta = C / S_ref**2`` at the reference price
    will have BTC delta ``C / S_mark**2`` after the mark moves, i.e. the
    coverage ratio rescales by exactly this factor. E.g. a +30% gap gives
    ``(1/1.3)**2 = 0.592...`` — the BTC-denominated hedge decays 41%.
    """
    _validate_index(reference_price)
    _validate_index(mark_price)
    return (reference_price / mark_price) ** 2


def inverse_perp_hedge_notional_for_usd_delta(
    target_usd_delta_btc_equiv: float,
    entry_price: float,
) -> float:
    """USD notional of an inverse perp achieving a USD-equivalent delta.

    Solves ``notional / entry = target`` for the signed notional, i.e.
    ``notional = target * entry``. Example: to neutralise an option position
    with USD-equivalent delta +0.05 BTC (per 1 BTC of contracts) at
    entry 100_000, short a notional of -0.05 * 100_000 = -5_000 USD.
    """
    _validate_index(entry_price)
    if target_usd_delta_btc_equiv == 0.0:
        return 0.0
    return target_usd_delta_btc_equiv * entry_price


def option_payoff_btc(option_type: OptionType, strike: float, settlement_price: float) -> float:
    """BTC intrinsic payoff at settlement of a BTC-settled (reversed) option.

    Calls pay ``max(0, S_T - K)/S_T`` BTC, puts pay ``max(0, K - S_T)/S_T`` BTC.

    The short-put branch grows like ``K/S`` as ``S -> 0`` (numeraire
    blow-up), which is the settling-in-BTC failure mode that USD accounting
    hides; see module docstring.
    """
    _validate_vanilla_inputs(strike, settlement_price)
    if option_type == "call":
        return max(0.0, settlement_price - strike) / settlement_price
    if option_type == "put":
        return max(0.0, strike - settlement_price) / settlement_price
    raise ValueError(f"Unrecognized option_type {option_type!r}: expected 'call' or 'put'")


def option_payoff_usd(option_type: OptionType, strike: float, settlement_price: float) -> float:
    """USD intrinsic payoff ``max(0, +-(S_T - K))`` with the usual type sign."""
    _validate_vanilla_inputs(strike, settlement_price)
    if option_type == "call":
        return max(0.0, settlement_price - strike)
    if option_type == "put":
        return max(0.0, strike - settlement_price)
    raise ValueError(f"Unrecognized option_type {option_type!r}: expected 'call' or 'put'")


def straddle_premium_usd(
    spot: float,
    strike: float,
    sigma: float,
    tau_years: float,
    r: float = 0.0,
) -> tuple[float, float]:
    """Black-Scholes call and put prices in USD (premium converted at spot).

    Uses the standard solution with continuous rate *r*: define the forward
    ``F = S*exp(r*T)`` and::

        d1 = (ln(S/F) + 0.5*sigma^2*T) / (sigma*sqrt(T)),
        d2 = d1 - sigma*sqrt(T),
        call = exp(-r*T) * (F*N(d1) - K*N(d2)),
        put  = call - (F - K)*exp(-r*T) = call - S + K*exp(-r*T)

    Returns
    -------
    (call_usd, put_usd)
        The BTC-denominated Deribit premium equals these values divided by
        the spot index (use :func:`usd_to_btc_cash`).
    """
    _validate_vanilla_inputs(strike, spot)
    if sigma <= 0.0 or tau_years <= 0.0:
        raise ValueError(f"sigma,tau must be positive, got sigma={sigma}, tau={tau_years}")
    if r < 0.0:
        raise ValueError(f"r must be >= 0, got {r}")
    sq = sigma * _sqrt(tau_years)
    fwd = spot * _exp(r * tau_years)
    d1 = (_log_f(fwd / strike) + 0.5 * sigma * sigma * tau_years) / sq
    d2 = d1 - sq
    df = _exp(-r * tau_years)
    call = df * (fwd * float(stats.norm.cdf(d1)) - strike * float(stats.norm.cdf(d2)))
    put = call - (fwd - strike) * df
    return call, put


def straddle_delta_usd(
    spot: float,
    strike: float,
    sigma: float,
    tau_years: float,
    r: float = 0.0,
) -> float:
    """USD-equivalent delta of a 1-BTC-notional long straddle: ``2*N(d1) - 1``.

    The per-contract call delta is ``N(d1)`` and the put delta is ``N(d1)-1``
    (put-call parity), so the straddle delta is their sum. Note this is
    *non-zero for a forward-ATM strike* with positive sigma: e.g.
    S=K=100_000, sigma=0.5, T=14/365 gives d1 = 0.5*sigma*sqrt(T) ~ 0.0490
    and delta ~ +0.039, so a "delta-hedged" ATM straddle hedge is a small
    *short* perp leg, not zero.
    """
    _validate_vanilla_inputs(strike, spot)
    if sigma <= 0.0 or tau_years <= 0.0:
        raise ValueError(f"sigma,tau must be positive, got sigma={sigma}, tau={tau_years}")
    sq = sigma * _sqrt(tau_years)
    fwd = spot * _exp(r * tau_years)
    d1 = (_log_f(fwd / strike) + 0.5 * sigma * sigma * tau_years) / sq
    return 2.0 * float(stats.norm.cdf(d1)) - 1.0


@dataclass(frozen=True)
class NumeraireFlow:
    """One BTC-denominated cash-flow in USD-value terms.

    Attributes
    ----------
    event_ts_ns:
        Flow timestamp (nanoseconds), informational.
    btc_amount:
        Signed BTC amount; > 0 = BTC received into the account.
    index_price_usd:
        USD index price at the flow instant; the *realised-USD* accounting
        freezes this flow at ``btc_amount * index_price_usd``.
    label:
        Free-form label (``"option_premium"``, ``"perp_pnl"`, ...).
    """

    event_ts_ns: int
    btc_amount: float
    index_price_usd: float
    label: str


@dataclass
class NumeraireLedger:
    """Order-preserving ledger of BTC-denominated flows with two accounting views.

    * ``realized_usd_value()`` — each flow valued at its own instant index
      (premium converted on receipt). This is the accounting under which the
      naive "premium stays worth its entry-time dollars" mental model lives.
    * ``marked_usd_value(terminal_index)`` — all BTC amounts marked at one
      terminal index. This is the true USD valuation of the BTC-denominated
      account at that instant.

    Difference (``collateral_convexity_drift_usd``): the compounding drift of
    BTC-denominated value with the index. The invariant
    ``marked == realized + drift`` must hold to float precision and is checked
    in the regression suite.
    """

    flows: list[NumeraireFlow] = field(default_factory=list)

    def add(
        self,
        event_ts_ns: int,
        btc_amount: float,
        index_price_usd: float,
        label: str,
    ) -> None:
        """Append a flow. Validates sign/price inputs and non-negative timestamp."""
        _validate_index(index_price_usd)
        if event_ts_ns < 0:
            raise ValueError(f"event_ts_ns must be >= 0, got {event_ts_ns}")
        self.flows.append(
            NumeraireFlow(
                event_ts_ns=event_ts_ns,
                btc_amount=btc_amount,
                index_price_usd=index_price_usd,
                label=label,
            )
        )

    def __len__(self) -> int:
        return len(self.flows)

    def __iter__(self) -> Iterator[NumeraireFlow]:
        return iter(self.flows)

    def realized_usd_value(self) -> float:
        """Sum of ``btc_amount * own_index`` over all flows."""
        return sum(f.btc_amount * f.index_price_usd for f in self.flows)

    def marked_usd_value(self, terminal_index: float) -> float:
        """Sum of ``btc_amount * terminal_index`` over all flows."""
        _validate_index(terminal_index)
        return sum(f.btc_amount for f in self.flows) * terminal_index

    def collateral_convexity_drift_usd(self, terminal_index: float) -> float:
        """Drift of marked vs realized: ``sum(amount * (S_T - flow_index))``."""
        _validate_index(terminal_index)
        return sum(f.btc_amount * (terminal_index - f.index_price_usd) for f in self.flows)

    def usd_value_by_label(self, terminal_index: float) -> dict[str, float]:
        """Marked USD value grouped by flow label (for verification tables)."""
        _validate_index(terminal_index)
        out: dict[str, float] = {}
        for f in self.flows:
            out[f.label] = out.get(f.label, 0.0) + f.btc_amount * terminal_index
        return out

    def iter_btc_amounts(self) -> Iterable[float]:
        return (f.btc_amount for f in self.flows)


class GapScenarioResult(NamedTuple):
    """Closed-form quantities of a delta-hedged short straddle across an instant gap."""

    spot_entry: float
    gap_ratio: float
    spot_after: float
    strike: float
    sigma: float
    tau_years: float
    prem_btc: float
    straddle_delta_entry: float
    hedge_signed_notional_usd: float
    straddle_value_after_usd: float
    perp_btc_pnl_at_gap: float
    perp_usd_pnl_marked: float
    exact_marked_usd_pnl: float
    naive_realized_usd_pnl: float
    collateral_drift_usd: float
    hedge_rescale_factor: float


def simulate_instant_gap(
    spot_entry: float,
    sigma: float,
    tau_years: float,
    gap_ratio: float,
    r: float = 0.0,
) -> GapScenarioResult:
    """Delta-hedged short ATM straddle hit by an instantaneous spot gap.

    Scenario (all valuation closed-form, ledger-consistent):

    1. At the entry instant (index S0, τ years to expiry, vol σ): sell 1-BTC
       straddle struck at the forward K = S*exp(r*T) (so ATM-with-carry);
       receive ``prem_btc = (call+put)/S0`` BTC. USD-equivalent delta of the
       straddle is ``2*N(d1)-1`` (non-zero at ATM-with-carry), hedged with an
       inverse perp of signed notional ``-delta*P_entry * S0`` (USD-delta
       parity, identical to BTC-delta parity at the entry instant).
    2. Immediately after, the index gaps to ``S1 = gap_ratio * S0`` (no time
       passes, Θ = 0; the straddle re-marks by Black-Scholes at the same σ,T).

    The exact USD wealth change is the BTC ledger marked at S1 minus the
    opening equity (position entered at zero MTM equity)::

        exact = S1 * (prem_btc + pnl_btc_perp) - straddle_value_usd(S1)
              = prem_btc*S1 - v_usd(S1) + C*(S1/P_entry - 1)

    while the naive USD accounting (BTC flows frozen at entry index) yields::

        naive = prem_btc*S0 - v_usd(S1) + C*(S1/P_entry - 1)

    differing by the collateral convexity drift ``(S1 - S0) * prem_btc``.

    Returns
    -------
    GapScenarioResult
        All quantities needed for reconciliation tests, including the
        perp hedge rescaling factor ``(S0/S1)**2``.
    """
    if gap_ratio <= 0.0:
        raise ValueError(f"gap_ratio must be > 0, got {gap_ratio}")
    _validate_vanilla_inputs(None, spot_entry)
    call0, put0 = straddle_premium_usd(spot_entry, spot_entry * _exp(r * tau_years), sigma, tau_years, r)
    prem_btc = (call0 + put0) / spot_entry
    delta_entry = straddle_delta_usd(spot_entry, spot_entry * _exp(r * tau_years), sigma, tau_years, r)
    # USD-delta parity at entry (identical to BTC-delta parity at S0):
    hedge_notional = inverse_perp_hedge_notional_for_usd_delta(-delta_entry, spot_entry)
    spot_after = spot_entry * gap_ratio
    strike_fwd = spot_entry * _exp(r * tau_years)
    call1, _ = straddle_premium_usd(spot_after, strike_fwd, sigma, tau_years, r)
    v_straddle_after = call1 + (call1 - spot_after + strike_fwd * _exp(-r * tau_years))
    pnl_btc_perp = inverse_perp_pnl_btc(hedge_notional, spot_entry, spot_after)
    perp_usd_marked = inverse_perp_pnl_usd(hedge_notional, spot_entry, spot_after)
    ledger = NumeraireLedger()
    ledger.add(0, prem_btc, spot_entry, "option_premium")
    ledger.add(1, pnl_btc_perp, spot_after, "perp_pnl")
    ledger.add(2, -(call1 + _put_component(spot_after, strike_fwd, sigma, tau_years, r)) / spot_after, spot_after, "liability_settlement_proxy")
    exact_marked = ledger.marked_usd_value(spot_after)
    naive_realized = ledger.realized_usd_value()  # flows frozen at own index
    drift = ledger.collateral_convexity_drift_usd(spot_after)
    return GapScenarioResult(
        spot_entry=spot_entry,
        gap_ratio=gap_ratio,
        spot_after=spot_after,
        strike=strike_fwd,
        sigma=sigma,
        tau_years=tau_years,
        prem_btc=prem_btc,
        straddle_delta_entry=delta_entry,
        hedge_signed_notional_usd=hedge_notional,
        straddle_value_after_usd=v_straddle_after,
        perp_btc_pnl_at_gap=pnl_btc_perp,
        perp_usd_pnl_marked=perp_usd_marked,
        exact_marked_usd_pnl=exact_marked,
        naive_realized_usd_pnl=naive_realized,
        collateral_drift_usd=drift,
        hedge_rescale_factor=inverse_perp_hedge_rescale_factor(spot_entry, spot_after),
    )


def _put_component(spot: float, strike: float, sigma: float, tau_years: float, r: float) -> float:
    """Black-Scholes put price (helper for ledger settlement flow)."""
    _call, put = straddle_premium_usd(spot, strike, sigma, tau_years, r)
    return put


class DeribitInversePricer:
    """Computes exact USD P&L under BTC coin-margined settlement.

    Provides a self-contained, stateless interface matching the Phase 0
    specification.  All methods delegate to the module-level verified
    functions; this class exists as a clean façade for pedagogical use and
    for the user-specified ``test_inverse_pnl_reconciliation`` suite.

    Deribit options payoff is settled in BTC::

        Call payoff (BTC) = max(0, S_T - K) / S_T
        Put payoff  (BTC) = max(0, K - S_T) / S_T

    Converted to USD cash value at settlement (``P_usd = P_btc * S_T``)::

        Call payoff (USD) = max(0, S_T - K)
        Put payoff  (USD) = max(0, K - S_T)

    Non-linearity emerges in the premium and margin::

        Collateral P&L (USD) = Premium_BTC * S_T
        Hedge P&L (USD) on inverse perp = Contracts * (S_T / S_0 - 1)
    """

    @staticmethod
    def option_payoff_btc(is_call: bool, strike: float, spot_at_expiry: float) -> float:
        """BTC-settled option intrinsic at expiry."""
        otype: OptionType = "call" if is_call else "put"
        return option_payoff_btc(otype, strike, spot_at_expiry)

    @staticmethod
    def inverse_perp_pnl_usd(
        entry_spot: float, current_spot: float, contracts_usd: float
    ) -> float:
        """P&L in USD for Deribit inverse perpetual contracts.

        ``PnL (BTC) = contracts_usd * (1/entry_spot - 1/current_spot)``
        ``PnL (USD) = PnL (BTC) * current_spot``
        """
        return inverse_perp_pnl_usd(contracts_usd, entry_spot, current_spot)

    @staticmethod
    def straddle_pnl_usd(
        strike: float,
        premium_collected_btc: float,
        spot_entry: float,
        spot_exit: float,
        delta_hedge_contracts_usd: float = 0.0,
    ) -> dict[str, float]:
        """Reconciles short straddle P&L including:

        1. Short Call intrinsic payoff
        2. Short Put intrinsic payoff
        3. Collateral drift in USD (``Premium_BTC * spot_exit``)
        4. Delta hedge P&L via inverse perpetual
        """
        call_loss_usd = max(0.0, spot_exit - strike)
        put_loss_usd = max(0.0, strike - spot_exit)
        total_liability_usd = call_loss_usd + put_loss_usd

        # Value of the BTC premium received when converted to USD at exit
        collateral_usd = premium_collected_btc * spot_exit

        # Linear hedge P&L via inverse perp
        hedge_usd = inverse_perp_pnl_usd(
            delta_hedge_contracts_usd, spot_entry, spot_exit
        )

        net_pnl_usd = (collateral_usd - total_liability_usd) + hedge_usd

        return {
            "spot_exit": spot_exit,
            "liability_usd": total_liability_usd,
            "collateral_usd": collateral_usd,
            "hedge_usd": hedge_usd,
            "net_pnl_usd": net_pnl_usd,
        }


def _validate_index(price: float) -> None:
    """Raise ValueError for non-positive or non-finite index prices."""
    if not (price > 0.0) or price == float("inf"):
        raise ValueError(f"price must be finite and > 0, got {price!r}")


def _validate_vanilla_inputs(strike: float | None, price: float) -> None:
    """Validate option inputs; strike may be None for spot-only checks."""
    _validate_index(price)
    if strike is not None and not (strike > 0.0):
        raise ValueError(f"strike must be > 0, got {strike!r}")


def _sqrt(x: float) -> float:
    """math.sqrt re-exported for import cleanliness."""
    from math import sqrt

    return sqrt(x)


def _exp(x: float) -> float:
    from math import exp

    return exp(x)


def _log_f(x: float) -> float:
    from math import log

    return log(x)
