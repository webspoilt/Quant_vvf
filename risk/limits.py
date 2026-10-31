"""Hard Margin, Delta, and Vega Portfolio Risk Limits.

Enforces portfolio-level risk ceilings:
- Global portfolio vega limit (BTC units)
- Portfolio absolute delta neutral boundary
- Max portfolio gamma limit
- Initial and maintenance margin utilization ceilings
"""
from __future__ import annotations

from dataclasses import dataclass

from libs.core.validation import non_finite_fields


@dataclass(frozen=True)
class RiskLimitsConfig:
    max_portfolio_delta_abs: float = 0.15
    max_portfolio_vega_btc: float = 50.0
    max_portfolio_gamma: float = 0.005
    max_margin_utilization: float = 0.70
    min_collateral_buffer_btc: float = 0.50
    local_variance_circuit_breaker: float = 0.0025


@dataclass(frozen=True)
class LimitCheckResult:
    passed: bool
    breaches: tuple[str, ...]
    delta_utilization: float
    vega_utilization: float
    gamma_utilization: float
    margin_utilization: float


class PortfolioRiskMonitor:
    """Monitors live portfolio exposure against hard limits."""

    def __init__(self, config: RiskLimitsConfig | None = None) -> None:
        self.config = config or RiskLimitsConfig()
        # A non-finite limit constant would re-open the NaN comparison hole
        # from the config side; verify the boundary set itself is sound.
        limit_fields = non_finite_fields(
            {
                "max_portfolio_delta_abs": self.config.max_portfolio_delta_abs,
                "max_portfolio_vega_btc": self.config.max_portfolio_vega_btc,
                "max_portfolio_gamma": self.config.max_portfolio_gamma,
                "max_margin_utilization": self.config.max_margin_utilization,
                "min_collateral_buffer_btc": self.config.min_collateral_buffer_btc,
                "local_variance_circuit_breaker": self.config.local_variance_circuit_breaker,
            }
        )
        if limit_fields:
            raise ValueError(f"RiskLimitsConfig contains non-finite limits: {limit_fields}")

    def check_portfolio(
        self,
        portfolio_delta_usd: float,
        spot_price: float,
        portfolio_vega_btc: float,
        portfolio_gamma: float,
        current_margin_used_btc: float,
        total_equity_btc: float,
    ) -> LimitCheckResult:
        """Validate live Greek and margin utilization against hard ceilings.

        Fail-closed: any non-finite (NaN/±Inf) input rejects the check with a
        ``CORRUPT_INPUT`` breach. IEEE 754 defines ``NaN > x`` as False, so an
        unguarded comparison would *pass* on corrupted feeds — the exact data
        corruption mode of a market shock.

        Returns:
            LimitCheckResult with detailed breach diagnostics
        """
        # 1. Fail-closed numerical integrity gate (before any comparison logic).
        corrupt = non_finite_fields(
            {
                "portfolio_delta_usd": portfolio_delta_usd,
                "spot_price": spot_price,
                "portfolio_vega_btc": portfolio_vega_btc,
                "portfolio_gamma": portfolio_gamma,
                "current_margin_used_btc": current_margin_used_btc,
                "total_equity_btc": total_equity_btc,
            }
        )
        if corrupt:
            return LimitCheckResult(
                passed=False,
                breaches=tuple(f"CORRUPT_INPUT: {field}" for field in corrupt),
                delta_utilization=float("nan"),
                vega_utilization=float("nan"),
                gamma_utilization=float("nan"),
                margin_utilization=float("nan"),
            )

        # 2. Economic boundary sanity: zero/negative catastrophe inputs must
        # abort here, not be masked by clamps like `max(1.0, spot_price)`.
        nan_utils = dict(
            delta_utilization=float("nan"),
            vega_utilization=float("nan"),
            gamma_utilization=float("nan"),
            margin_utilization=float("nan"),
        )
        if spot_price <= 0.0:
            return LimitCheckResult(
                passed=False,
                breaches=(f"INVALID_SPOT_PRICE: spot_price={spot_price!r} <= 0",),
                **nan_utils,
            )
        if total_equity_btc <= 0.0:
            return LimitCheckResult(
                passed=False,
                breaches=(f"INSOLVENCY_DETECTED: total_equity_btc={total_equity_btc!r} <= 0",),
                **nan_utils,
            )

        breaches: list[str] = []

        # Convert USD delta to normalized BTC equivalent delta
        equiv_btc_delta = abs(portfolio_delta_usd / spot_price)
        delta_util = float(equiv_btc_delta / self.config.max_portfolio_delta_abs)
        if equiv_btc_delta > self.config.max_portfolio_delta_abs:
            breaches.append(f"Delta breach: |Delta|={equiv_btc_delta:.3f} > {self.config.max_portfolio_delta_abs:.3f}")

        # Vega cap
        vega_util = float(abs(portfolio_vega_btc) / self.config.max_portfolio_vega_btc)
        if abs(portfolio_vega_btc) > self.config.max_portfolio_vega_btc:
            breaches.append(f"Vega breach: {portfolio_vega_btc:.2f} BTC > {self.config.max_portfolio_vega_btc:.2f} BTC")

        # Gamma limit
        gamma_util = float(abs(portfolio_gamma) / self.config.max_portfolio_gamma)
        if abs(portfolio_gamma) > self.config.max_portfolio_gamma:
            breaches.append(f"Gamma breach: {portfolio_gamma:.5f} > {self.config.max_portfolio_gamma:.5f}")

        # Margin utilization (total_equity_btc > 0 guaranteed above)
        margin_util = float(current_margin_used_btc / total_equity_btc)
        if margin_util > self.config.max_margin_utilization:
            breaches.append(f"Margin breach: {margin_util * 100:.1f}% > {self.config.max_margin_utilization * 100:.1f}%")

        # Free collateral buffer
        free_collateral = total_equity_btc - current_margin_used_btc
        if free_collateral < self.config.min_collateral_buffer_btc:
            breaches.append(f"Collateral buffer breach: {free_collateral:.3f} BTC < {self.config.min_collateral_buffer_btc:.3f} BTC")

        return LimitCheckResult(
            passed=len(breaches) == 0,
            breaches=tuple(breaches),
            delta_utilization=delta_util,
            vega_utilization=vega_util,
            gamma_utilization=gamma_util,
            margin_utilization=margin_util,
        )
