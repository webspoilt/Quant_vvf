# QUANT_VVF — BENCHMARK HARDWARE & EXECUTION CONFIGURATION

**Audit Date:** 2026-11-14  
**Benchmark Target:** C++20 HFT Execution Engine, Risk Ledger, SPSC Ring Queue, and Durable WAL  
**Branch:** `feature/evidence-based-release-audit`  

---

## 1. Physical Host System Specifications

All benchmarks recorded in this repository were physically executed on bare-metal host hardware (not simulated or virtualized in cloud containers).

| Component | Specification |
| :--- | :--- |
| **CPU Model** | 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz (Max Boost 4.50GHz) |
| **Microarchitecture** | Intel Tiger Lake (10nm SuperFin), AVX-512 & SHA instruction set support |
| **Cores / Threads** | 6 Physical Cores / 12 Logical Hyperthreads |
| **Base / Max Clock** | 2.688 GHz Base / ~4.5 GHz Boost |
| **L1 / L2 / L3 Cache** | 48KB L1d + 32KB L1i per core / 1.25MB L2 per core / 12MB Shared L3 |
| **Installed RAM** | 16.00 GB DDR4-3200 (Dual Channel) |
| **Storage Device** | NVMe SSD: Micron 2400 MTFDKBA512QFM (512 GB PCIe 4.0 NVMe M.2) |
| **Operating System** | Microsoft Windows 11 Home Single Language (64-bit, Build 26300) |
| **Filesystem** | NTFS (4KB Sector Size) |
| **High-Precision Clock**| Windows QPC (`QueryPerformanceCounter`) / `std::chrono::steady_clock` (100ns precision) |

---

## 2. Toolchain and Build Configuration

### C++20 Native Engine
- **Compiler:** MinGW-w64 LLVM Clang++ version 23.0.0 (Martin Störsjö UCRT distribution)
- **C++ Standard:** `-std=c++20`
- **Optimization Flags:** `-O3 -DNDEBUG`
- **Include Paths:** `-I../include`
- **Memory Model:** Zero dynamic allocations on hot path; stack/arena-allocated structures with cacheline alignment (`alignas(64)`).

### Rust 2024 SBE Gateway
- **Compiler:** `rustc 1.91.0-nightly` (x86_64-pc-windows-gnu)
- **Profile:** Release (`opt-level = 3`, `lto = "thin"`, `codegen-units = 1`)
- **Memory Safety:** `#![deny(unsafe_op_in_unsafe_fn)]`, zero copy casting via `zerocopy 0.8`.

---

## 3. Storage Hierarchy & Durability Distinctions

A critical principle of this engineering audit is distinguishing in-memory execution from physical storage synchronization:

1. **In-Memory Buffering (Userspace `fwrite`):** Data copied from CPU registers to userspace C stdio `FILE*` buffer. Latency: **sub-microsecond** (Median ~0 ns to 100 ns). *Does not survive OS crashes or power failures.*
2. **OS Kernel Page Cache:** Data flushed from stdio buffer to OS kernel memory via `std::fflush()`. *Survives application crashes, but lost on kernel panic or hardware power failure.*
3. **Physical Hardware Durability (`_commit` / `fsync`):** OS issues hardware barrier command (`IOCTL_STORAGE_PERSIST` / NVMe Flush command `0x00`) to force SSD onboard DRAM controller cache to persist bits into non-volatile NAND flash. Latency: **~740 µs to 850 µs** on consumer/client PCIe 4.0 NVMe SSD.
