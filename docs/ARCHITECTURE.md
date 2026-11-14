# COMPLETE PLATFORM ARCHITECTURE SPECIFICATION — QUANT_VVF

## 1. System Mission & Core Directives

The **Quant_vvf** trading and quantitative research platform is engineered to combine deep statistical volatility research, analytical options pricing, multi-agent AI research assistance, institutional portfolio optimization, and sub-microsecond execution infrastructure into a coherent, testable, and maintainable framework.

The platform prioritizes **Deribit Bitcoin (BTC) options, coin-margined inverse perpetuals, volatility risk premium (VRP) harvesting, and dynamic discrete delta hedging**, while maintaining clear extensibility to multi-asset cross-sectional portfolios.

---

## 2. Unidirectional Governance Architecture

To eliminate the catastrophic financial risks inherent in autonomous black-box trading systems, `Quant_vvf` enforces an inviolable unidirectional workflow:

```text
[ Market Data & SBE Streams ]
             ↓
[ Feature Engineering & Market Context ]
             ↓
[ AI Research Proposal (JARVIS) ]
             ↓
[ Quantitative Research & Hypothesis Testing ]
             ↓
[ Event-Driven Backtesting & Stress Validation ]
             ↓
[ Portfolio Allocation & Sizing Engine ]
             ↓
[ Deterministic Inline Pre-Trade Risk Engine ] ──(Fails Invariant)──> REJECT / HALT
             ↓ (Approved)
[ Low-Latency C++20 Execution Engine ]
             ↓
[ SBE TCP Order Gateway & Exchange Acks ]
             ↓
[ Durable WAL & Post-Trade Reconciliation ]
             ↓
[ Monitoring & Relational Audit Journal ]
```

### Deterministic Risk Sovereignty
* **Zero Autonomous Live Authority:** LLM agent confidence scores, generative prompts, or predictive model outputs are strictly advisory. No AI component possesses execution gateway access or the authority to approve an order.
* **Fail-Closed Gatekeeping:** Any non-finite value (`NaN`, $\infty$), non-positive fill price, invalid clip size, stale feed ($> 500$ ms), or sequence discontinuity instantly trips safety interlocks.
* **Separation of Concerns:** Python operates the research, strategy backtesting, portfolio construction, and advisory intelligence layers; C++20 and Rust own low-latency execution, pre-trade risk ledgers, and zero-copy binary network parsing.

---

## 3. Subsystem Breakdown

### 3.1 Market Data Ingestion & Protocol Normalization (`rust_hft/`)
* **Zero-Copy SBE Parser:** Implements direct wire-mapped parsing conforming to Deribit's Starbase Binary protocol (24-byte UDP packet headers, 16-byte SBE message headers, 68-byte packed TCP order requests).
* **Multi-Channel Sequence Arbitrator:** Tracks 16 concurrent channels independently. Requests for a 17th channel trigger an explicit `ChannelCapacityExhausted` error, preventing memory aliasing and state corruption.
* **Verified Retransmit Pipeline:** Sequence gaps trigger stale-book flags that can only be cleared when verified contiguous retransmission packets close the exact missing interval (`missing_gap_start` to `missing_gap_end`).
* **Snapshot Fallback:** If gap retransmission fails, the arbitrator ingests an authoritative full-depth snapshot to restore sequence continuity.

### 3.2 Low-Latency Execution & Pre-Trade Risk Engine (`hft/`)
* **Reserved ID Rejection:** Explicitly rejects execution reports carrying IDs `0` (`EMPTY_SLOT`) or `UINT64_MAX` (`TOMBSTONE`), returning `REJECT_RESERVED_EXEC_ID` before any state or risk ledger mutation.
* **Atomic Pre-Validation:** Validates order IDs, trade sides, prices, quantities, and risk limits prior to committing IDs into the deduplication hash table, preventing poisoning by malformed reports.
* **Tombstone-Safe Open Addressing:** Hash-table deletions insert tombstones (`UINT64_MAX`) rather than resetting to zero, preserving linear probing chains across FIFO evictions.
* **Durable Write-Ahead Log (WAL):** Synchronously appends committed execution records to disk (`execution_journal.hpp`), ensuring full crash recovery and restart reconciliation.
* **Dual-Domain Clock Segregation:** Enforces clock domain compatibility (`HOST_MONOTONIC`, `PTP_HARDWARE_EPOCH`, `SIMULATION_EPOCH`). Explicitly reports hardware PHC as unsupported on Windows hosts.

### 3.3 Quantitative Research Suite (`research/`)
* **P1 Volatility:** Maximum-likelihood GARCH(1,1) parameter estimation, rolling realized volatility measures, and walk-forward out-of-sample benchmarking.
* **P2 Pairs Cointegration:** OLS hedge ratio estimation, Engle-Granger two-step cointegration, Augmented Dickey-Fuller testing, and Ornstein-Uhlenbeck mean-reversion half-life diagnostics.
* **P3 Market Regimes:** Unsupervised Gaussian Mixture Models clustering Calm vs. Crisis volatility states with empirical transition probability matrices.
* **P4 Latent Factors:** Spectral decomposition (`eigh`) of standardized asset returns, factor loadings, and explained variance ratios.
* **P5 Directional Significance:** Chronological past/future machine learning evaluation with exact binomial tests against random 50% baselines.

### 3.4 Options Analytics & Derivatives Modeling (`options/`)
* **Analytical Black-Scholes/Black-76:** Closed-form Greeks ($\Delta, \Gamma, \mathcal{V}, \Theta, \rho$) adhering to Call-Put parity.
* **Implied Volatility Inversion:** Robust root-finding using Brent's method.
* **VRP Term Structure Analytics:** Empirical forward $IV - RV$ distributions and 25-delta risk reversal skew metrics.
* **Dynamic Delta Hedging:** Discrete pathwise rebalancing of coin-margined BTC options hedged with inverse perpetual futures, accounting for non-linear $(S_0/S_1)^2$ drift, transaction fees, and slippage.

### 3.5 Institutional Portfolio Management (`portfolio/`)
* **Ledoit-Wolf Covariance Shrinkage:** Optimal analytical shrinkage towards constant correlation targets.
* **Equal Risk Contribution (Risk Parity):** Nonlinear SQP optimization equalizing marginal risk contributions.
* **Hierarchical Risk Parity (HRP):** Single-linkage clustering on correlation distance metrics, quasi-diagonalization, and recursive bisection.

### 3.6 Event-Driven Backtesting Engine (`backtesting/`)
* **Execution Realism:** Simulates transaction fees, bid-ask spread, linear/quadratic slippage, margin leverage limits, Sharpe, Sortino, Calmar, and maximum drawdown curves.

### 3.7 JARVIS Multi-Agent Copilot (`ai_orchestrator/`)
* **Specialist Hierarchy:** Central Supervisor coordinating Market Analyst, Research Agent, Strategy Agent, Risk Analyst, Portfolio Analyst, and Execution Analyst.
* **Mandatory Safety Interlock:** All proposals enforce `required_human_approval = True` and `deterministic_risk_approved = False`.

### 3.8 Persistence, Monitoring & Operations (`database/`, `monitoring/`, `integrations/`)
* **Relational Persistence:** SQLite/PostgreSQL schema for orders, executions, auditable trade journals, and system health logs.
* **Health Monitor:** Real-time monitoring of feed staleness, sequence gap frequencies, queue saturation, and circuit breakers.
* **Exchange Adapters:** High-level base interfaces, local deterministic paper-trading matching engines, and authenticated Deribit live transport with strict environment-token execution gates (`DERIBIT_LIVE_AUTHORIZED=TRUE`).
* **Terminal UI:** Responsive trading terminal dashboard displaying real-time market tickers, risk exposure meters, strategy status, and execution audit tables.
