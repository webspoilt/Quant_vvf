# Quant_vvf Architecture & Source Code Inventory Audit

**Audit Date:** 2026-11-14  
**Auditor:** Principal Quantitative Engineering Team  
**Scope:** Complete inspection of files, modules, dependencies, tests, mathematical foundations, and execution pipelines in `Quant_vvf`.

---

## 1. Directory Structure & Module Responsibilities

The existing repository is structured into distinct, highly specialized layers spanning quantitative research, low-latency C++ execution, Rust binary market-data ingestion, and regression testing:

```text
Quant_vvf/
├── configs/                   # Static limits and pre-registration parameters (YAML)
│   ├── execution_limits.yaml  # Max clip sizes, dollar caps, daily drawdown limits
│   └── pre_registration.yaml  # Hypothesis pre-registration parameters
├── docs/                      # Architectural decision records & specifications
│   ├── adr/                   # Architectural Decision Records (ADR 001)
│   ├── BASELINE_TEST_REPORT.md# Verified baseline execution logs (161 tests)
│   └── REPOSITORY_AUDIT.md    # This comprehensive inventory and audit
├── execution/                 # NautilusTrader integration & execution adapter
│   └── nautilus_actor.py      # Microsecond variance circuit breaker & delta rebalancing
├── hft/                       # C++20 low-latency execution, risk, and FPGA acceleration
│   ├── fpga/                  # Hardware RTL (Verilog) for BBO tracking & inline risk
│   ├── include/
│   │   ├── core/              # Memory arena, lock-free SPSC queues, CPU core affinity
│   │   ├── engine/            # High-frequency order lifecycle & execution deduplication
│   │   ├── math/              # Inverse derivatives pricing & (S0/S1)^2 delta rescaling
│   │   ├── network/           # Kernel-bypass interfaces (DPDK/EF_VI) and wire types
│   │   ├── risk/              # Open-addressing pre-trade risk ledger & exposure checks
│   │   └── sync/              # Dual-domain clock architecture (PTP hardware vs monotonic)
│   └── tests/                 # C++ test programs (clang++ -std=c++20)
├── libs/                      # Core quantitative Python library
│   ├── core/                  # HAR-RV, Fourier/COS pricing, SVI calibration, inverse math
│   └── utils/                 # Strict point-in-time time-series store (PITStore)
├── monitoring/                # Live risk and tail trigger monitoring
│   └── tail_risk_monitor.py   # Real-time RV shock, skew spike, and funding rate triggers
├── research/                  # Quantitative research papers, experiments, and notebooks
│   ├── math_papers/           # Rigorous mathematical foundations (Malliavin, EVT, SVD)
│   └── ...                    # Empirical notebooks and pipeline scripts
├── risk/                      # Python risk management & invariant checks
│   ├── limits.py              # Portfolio delta, vega, and concentration checks
│   ├── risk_constants.py      # Authoritative fail-closed risk thresholds
│   └── tail_triggers.py       # Cooldown-aware tail shock monitors
├── rust_hft/                  # Rust 2024 low-latency market-data ingestion & SBE gateway
│   ├── src/
│   │   ├── deribit_sbe.rs     # Starbase SBE zero-copy parser & multi-channel sequencer
│   │   ├── lib.rs             # Crate exports and unit test module
│   │   ├── main.rs            # Production-grade dual-core ingestion pipeline
│   │   ├── protocol.rs        # Wire ITCH and Starbase binary struct layouts
│   │   ├── ring_buffer.rs     # Safe lock-free SPSC ring buffer (split Producer/Consumer)
│   │   ├── tests.rs           # 13 verified unit tests covering gaps, SBE, capacity
│   │   └── threading.rs       # Core-affinity pinned worker threads
│   └── Cargo.toml             # Zero-copy, core-affinity dependencies
├── simulation/                # Chronological backtesting & walk-forward simulation
│   └── walk_forward_sim.py    # Purged, embargoed walk-forward engine
├── strategies/                # Volatility and Greek rebalancing strategies
│   ├── greek_solver.py        # Discrete Greek optimization under transaction costs
│   └── vrp_profile.py         # Empirical VRP profile calculation and regime filtering
└── tests/                     # Comprehensive Python test suite (128 passing tests)
    ├── bias_checks/           # Point-in-time lookahead bias enforcement
    ├── regression/            # Phase 0–3 quantitative regression gates
    └── unit/                  # Unit tests for SVI, math, risk, pricing, and execution
```

---

## 2. Programming Languages & Dependencies

### Python 3.11 Foundation
* **Core Libraries:** `numpy (>=1.26)`, `scipy (>=1.12)`, `pandas (>=2.2)`, `pyarrow (>=15.0)`
* **Financial & ML:** `scikit-learn (>=1.4)`, `statsmodels (>=0.14)`, `nautilus_trader`
* **Testing:** `pytest (>=8.0)`, `anyio`, `platformdirs`

### C++20 Low-Latency Execution Layer
* **Standard:** ISO C++20 (`-std=c++20`)
* **Compilers Tested:** Clang 19.1.2 (llvm-mingw UCRT)
* **Optimization:** `-O2`, `-march=native`, zero-allocation hot path, no heap allocations after startup
* **Key Components:**
  - Header-only low-latency libraries under `hft/include/`
  - Linear-probing open-addressing table for execution ID deduplication (`TOMBSTONE = UINT64_MAX`)
  - Open-addressed risk reservation ledger (`MAX_TRACKED_ORDERS = 1024`)

### Rust 2024 Market-Data Ingestion Layer
* **Edition:** Rust 2024 (`rustc 1.100.0`, target `x86_64-pc-windows-gnu`)
* **Dependencies:**
  - `zerocopy = { version = "0.8", features = ["derive"] }`: Zero-overhead byte slicing
  - `core_affinity = "0.8"`: Strict CPU pinning
* **Key Components:**
  - `StarbaseFeedArbitrator`: 16-channel fixed tracking; 17th channel triggers explicit `ChannelCapacityExhausted`
  - `StarbaseZeroCopyParser`: Validates SBE Schema Version 1 and Template IDs (100, 101, 102)
  - `InnerRing`: Lock-free SPSC buffer with cache-line-padded indices (64 bytes) to prevent false sharing

---

## 3. Existing Functionality & Strengths

1. **Inverse Payoff & Numeraire Rescaling:** Complete $(S_0/S_1)^2$ delta drift correction for coin-margined BTC options and perpetual contracts. Verified across all price regimes.
2. **Quasi-Explicit SVI Volatility Calibration:** Fast Gatheral SVI calibration with static arbitrage diagnostics (butterfly and calendar spread checks).
3. **Execution ID Integrity & Dedup:** Tombstone-safe open addressing prevents probing chain breaks during deduplication FIFO evictions.
4. **Pre-Trade Risk Gatekeeping:** Authoritative exposure calculations evaluate both directional risk and gross reservations atomically.
5. **Multi-Channel Starbase Sequencer:** Rust-based zero-copy SBE parser with gap detection and verified retransmission recovery.

---

## 4. Integration Gaps & Technical Debt

1. **Modular AI Orchestration (JARVIS):** Needs structured multi-agent architecture (Market Analyst, Research Agent, Strategy Agent, Risk Analyst, Portfolio Analyst) operating under strict deterministic risk gates.
2. **Quantitative Research Suite (P1–P5):** While foundational modules exist in `libs/core`, standalone, cleanly structured research drivers for P1 (Volatility), P2 (Pairs Cointegration), P3 (Regimes), P4 (PCA), and P5 (Directional Significance) need to be formalized in `research/`.
3. **Portfolio Construction Engine:** Modern optimization methods (Hierarchical Risk Parity, Covariance Shrinkage, Black-Litterman with view uncertainty, Risk Parity) need dedicated modules in `portfolio/`.
4. **Unified Strategy Framework:** Clear separation between research hypotheses, formal strategy rules, and backtesting adapters.
5. **Database & Persistence:** Relational schemas for trade journaling, risk audits, orders, and execution reports.
6. **Trading Terminal Frontend:** Web-based operational dashboard for real-time monitoring, options analytics, and human approval workflows.
7. **Simulation vs. Production Boundaries:** Hardware PTP grandmaster synchronization, physical Solarflare kernel bypass (EF_VI), and live exchange sockets must remain strictly segregated from software-simulated backtests.
