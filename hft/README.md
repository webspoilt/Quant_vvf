# Quant_vvf High-Frequency Trading (HFT) Engine

This directory provides the production-grade, zero-garbage collection, lock-free C++20 and FPGA hardware execution framework designed to transform `Quant_vvf` from a Python test harness into an elite HFT engine.

---

## Architecture Overview

```
                          10/25 GbE Ethernet (PTP IEEE 1588 Sync)
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ FPGA Hardware Pipeline (Alveo/Xilinx) │
                     │  - In-Hardware BBO Order Book         │
                     │  - Single-Cycle Pre-Trade Risk Gate   │
                     └───────────────────┬───────────────────┘
                                         │ Direct DMA / PCIe
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ Kernel-Bypass Network Layer           │
                     │  - Zero-Copy Wire Frame Ingestion     │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ Pinned CPU Core (C++20 Zero-GC)       │
                     │  - Lock-Free SPSC Ring Buffer         │
                     │  - Pre-allocated Memory Arena         │
                     │  - Inverse Coin-Margined Greek Math   │
                     │  - Microsecond Variance Breaker       │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                       Outbound Binary Orders to Exchange Wire
```

---

## Directory Structure

```
hft/
├── include/
│   ├── network/
│   │   ├── types.hpp              # Zero-copy binary structs (packed, IEEE 1588 PTP timestamps)
│   │   └── kernel_bypass.hpp      # Zero-copy memory map parser / serializer (DPDK/EF_VI interface)
│   ├── core/
│   │   ├── lock_free_queue.hpp    # Cache-line aligned SPSC lock-free ring buffer
│   │   ├── memory_arena.hpp       # Pre-allocated memory pool (eliminating heap malloc/free)
│   │   └── cpu_pinning.hpp        # Thread-to-core affinity controller (NUMA isolation)
│   ├── risk/
│   │   └── pre_trade_risk.hpp     # Deterministic inline pre-trade risk filter & variance circuit breaker
│   ├── math/
│   │   └── inverse_math.hpp       # Ultra-fast (S0/S1)^2 rescale & analytical Gatheral SVI evaluations
│   ├── sync/
│   │   └── ptp_clock.hpp          # RDTSC cycle counter and PTP hardware time synchronizer
│   └── engine/
│       └── execution_engine.hpp   # End-to-end nanosecond tick-to-trade execution pipeline
├── fpga/
│   └── hft_bbo_risk_engine.v      # Verilog RTL for FPGA BBO cache & inline pre-trade risk gating
└── tests/
    ├── test_hft_suite.cpp         # C++20 comprehensive benchmark test suite
    └── verify_hft_pipeline.py     # End-to-end validation driver
```

---

## Verification & Execution

To run the verification suite:
```bash
python Quant_vvf/hft/tests/verify_hft_pipeline.py
```
