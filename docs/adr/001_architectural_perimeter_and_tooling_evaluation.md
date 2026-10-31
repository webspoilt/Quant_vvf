# ADR 001: Architectural Perimeter and Tooling Evaluation (Retail Backtesters vs. Institutional Math vs. Multi-Agent LLMs)

- **Status:** Accepted
- **Date:** 2026-10-31
- **Deciders:** Quantitative Research & Systems Architecture

---

## 1. Context & Problem Statement

The platform is a systematic, falsification-first research and execution engine designed to trade the **Volatility Risk Premium (VRP)** on Deribit BTC options and perpetual futures. Its core mandate is to test and execute under the hypothesis:

> *Deribit BTC VRP produces positive risk-adjusted returns net of costs when conditioned on non-parametric tail triggers.*

Given the proliferation of open-source quantitative trading frameworks—spanning retail spot/perp backtesters, quantitative finance libraries, and multi-agent Large Language Model (LLM) trading systems—an architectural evaluation was conducted to benchmark this platform against the broader ecosystem:
- **Retail linear backtesters:** `backtrader`, `freqtrade`, `zipline`, `binance-trade-bot`, `QUANTAXIS`, `Superalgos`
- **Institutional pricing & factor libraries:** `QuantLib`, `QuantsPlaybook`, `awesome-quant`
- **Interactive retail terminals:** `OpenBB`
- **Multi-agent LLM trading frameworks:** `TradingAgents-CN`, `QuantDinger`
- **Institutional event-driven engines:** `NautilusTrader`

We must establish clear architectural boundaries to prevent architectural drift, maintain reproducible falsification-first research, and ensure sub-millisecond execution safety.

---

## 2. Decision Matrix & Evaluation

### A. Rejected Frameworks (Redundant & Harmful)

#### 1. Retail Linear Backtesters (`backtrader`, `zipline`, `freqtrade`, `Superalgos`, etc.)
* **The Fatal Flaw:** These systems model portfolio returns as linear combinations of asset weights ($w^T \mu$) or treat instruments as discrete cash/spot quantities.
* **Why Rejected for Crypto Options VRP:**
  1. **Non-Linear Inverse Numeraire Settlement:** Deribit options and perpetual contracts settle in BTC, not USD. In `libs/core/inverse_payoffs.py`, we proved that hedging a short straddle with an inverse perpetual incurs an **$(S_0 / S_1)^2$ delta drift factor**. Over a $\pm 30\%$ spot move, hedging in linear contract or USD space without this factor creates up to $49\%$ under- or over-hedging, silently de-hedging the book.
  2. **Asymmetric BTC-Settled Short Put Liabilities:** The BTC short put liability $\max(K - S, 0) / S = K(1/S - 1/K)$ grows towards $+\infty$ as $S \to 0$, whereas the USD liability is capped at $K$. Linear backtesters cannot track coin-margined liquidation mechanics or maintenance margin drift under portfolio margin rules.
  3. **Non-Linear Greek Surface Dynamics:** Options exhibit non-linear continuous decay ($\Theta dt + \frac{1}{2}\Gamma dS^2 + \mathcal{V} d\sigma$). Strategies on this platform emit **Target Greek Profiles** (bounding Delta, Gamma, Vega, Theta, and margin utilization) rather than capital allocation weights. Hacking retail backtesters to support multi-leg Greek surface decay destroys event loop performance and creates false fills.

#### 2. Multi-Agent LLM Frameworks (`TradingAgents-CN`, `QuantDinger`)
* **The Fatal Flaw:** Delegating strategy generation, Greek sizing, or regime classification to qualitative natural-language LLM debates (e.g., "Bull Agent" vs. "Bear Agent").
* **Why Rejected for Crypto Options VRP:**
  1. **Execution Latency Mismatch:** The variance circuit breaker in `execution/nautilus_actor.py` executes in sub-millisecond timeframes when local tick return variance $r^2 > 0.0025$. LLM inference latency (500 ms to 5,000 ms) is several orders of magnitude too slow to prevent flash-crash liquidations.
  2. **Violation of Falsification & Pre-Registration:** The platform enforces the Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio (DSR) with a pre-registered trial budget ($N \le 3$) and strict point-in-time receipts ($T_{\text{receipt}} \le T$). Qualitative LLM debates introduce non-deterministic prompt drift, hallucinated rules, and uncontrolled degrees of freedom that invalidate statistical rigor.
  3. **Mathematical Incompetence in Continuous Geometry:** LLMs cannot reliably fit Gatheral quasi-explicit SVI surfaces, evaluate Durrleman non-arbitrage density factors $g(k) \ge 0$, or solve integer-constrained mixed-Greek quadratic programs.

#### 3. Interactive Retail Terminals (`OpenBB`)
* **Why Rejected for Core Execution:** While valuable for exploratory manual research, monolithic terminals add bloated dependencies and cannot act as a deterministic microsecond risk-veto authority or high-frequency Greek rebalancer.

---

### B. Accepted Core Stack & Borrowed Components

| System / Tool | Status | Architectural Role | Justification |
| :--- | :--- | :--- | :--- |
| **`NautilusTrader`** | **Core Execution Engine** | Production execution, event-driven tick replay, sub-second order lifecycle | Institutional-grade event engine natively supporting multi-leg crypto options, asynchronous actor concurrency, and sub-millisecond execution. |
| **`QuantLib`** | **Math Benchmark Oracle** | Unit testing & analytical cross-verification in `tests/unit/` | Gold standard continuous-time pricing library. Used to audit Black-76 option pricing, IV solvers, and Greeks against custom NumPy implementations down to $10^{-10}$ relative tolerance. |
| **`QuantsPlaybook`** | **Research Methodology Reference** | Factor construction & non-parametric regime modeling | High-quality institutional sell-side replication. Informs rolling quantile normalization of volatility risk premia and microstructure tail indicators. |
| **`awesome-quant`** | **Reference Directory** | Fast lookup of specialized C/C++ mathematical bindings | Reference index for high-performance numerical algorithms and volatility surface parameterizations. |

---

## 3. Structural Tenets Enshrined

1. **Deterministic Mathematical Kernels Only:** All pricing, surface fitting (Gatheral SVI), volatility forecasting (Corsi Log-HAR-RV), and payoff accounting reside in `libs/core/` and execute as pure, deterministic numerical code.
2. **Independent Fail-Closed Risk Authority:** Strategy modules have zero direct order routing authority. The `RiskGatekeeper` and `tail_triggers.py` exercise unilateral, programmatic veto authority (`HALT_NEW_ENTRIES`, `FLATTEN_SHORT_GAMMA`).
3. **Discrete Greek Solvers over Capital Weights:** Portfolio allocation strictly solves for integer contract counts matching multi-dimensional Greek envelopes subject to Deribit margin bounds (`strategies/greek_solver.py`).
4. **Immediate Exchange-Native Cloud Brackets:** On fill confirmation, exchange-native conditional stops are placed synchronously (`execution/nautilus_actor.py`), guaranteeing venue-side liquidation protection during local disconnects.

---

## 4. Consequences

* **Phase Roadmap Integrity:** The Phase 0–5 implementation roadmap remains strictly intact without deviations or framework rewrites.
* **Continuous Verification:** `tests/unit/test_pricing_benchmarks.py` validates internal mathematical formulas against `QuantLib` (or high-precision closed-form analytical standards when running in lean environments).
* **Research Quality:** Future factor extensions (Phase 3+) will adopt institutional sell-side statistical methods (e.g., rolling percentile normalization, order-flow skew metrics) rather than qualitative heuristics.
