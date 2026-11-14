# QUANT_VVF — LATENCY BENCHMARK RESULTS & EMPIRICAL EVIDENCE

**Execution Date:** 2026-11-14  
**Hardware Host:** 11th Gen Intel Core i5-11400H @ 2.70GHz, 16GB RAM, Micron 2400 NVMe SSD, Windows 11  
**Compiler:** MinGW-w64 Clang++ 23.0.0 (`-std=c++20 -O3 -DNDEBUG`)  
**Harness Binary:** `hft/benchmarks/bench_engine.exe`  

---

## 1. Measured Latency Table (Real Physical Benchmarks)

| Benchmark Stage | Sample Count ($N$) | Min | Median (P50) | P90 | P95 | P99 | P99.9 | Max | StdDev | Execution Domain |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. SPSC Queue Push** | 100,000 | 0.0 ns | **0.0 ns** | 100.0 ns | 100.0 ns | 100.0 ns | 100.0 ns | 1,200.0 ns | 42.4 ns | In-Memory (L1d / Registers) |
| **2. SPSC Queue Pop** | 100,000 | 0.0 ns | **0.0 ns** | 100.0 ns | 100.0 ns | 100.0 ns | 100.0 ns | 25,800.0 ns | 95.6 ns | In-Memory (L1d / Registers) |
| **3. Pre-Trade Risk Gate** | 99,500 | 500.0 ns | **700.0 ns** | 700.0 ns | 800.0 ns | 800.0 ns | 2,100.0 ns | 39,600.0 ns | 207.9 ns | In-Memory (Open-Addr Table) |
| **4. Execution Engine Dedup** | 20,000 | 0.0 ns | **2,800.0 ns**| 2,900.0 ns | 3,100.0 ns | 5,100.0 ns | 12,100.0 ns| 22,000.0 ns | 1,121.9 ns | In-Memory (Two-Phase Hash) |
| **5. WAL Buffer Append** | 100,000 | 0.0 ns | **0.0 ns** | 100.0 ns | 200.0 ns | 2,600.0 ns | 14,600.0 ns| 117,600.0 ns| 840.6 ns | Userspace Buffer (`fwrite`) |
| **6. WAL Hardware Sync** | 500 | 743.8 µs | **850.2 µs** | 961.8 µs | 1,120.4 µs | 2,232.5 µs | 6,836.6 µs | 6,836.6 µs | 334.2 µs | Physical Micron NVMe SSD |

*(Note: 0.0 ns readings represent operations completing within clock quantization intervals below 100ns).*

---

## 2. In-Memory vs. Durable Storage Latency Analysis

### A. The Microsecond vs. Sub-Microsecond Reality
- In-memory lock-free SPSC transfers, pre-trade risk evaluation, and hash deduplication execute within **sub-microsecond** to **low single-digit microsecond** durations (Median 0.7 µs to 2.8 µs).
- Userspace WAL appends (`std::fwrite`) buffer instantaneously in memory (Median 0.0 ns to 100 ns).

### B. Physical Durability Overhead
- True crash-resilient hardware writes (`_commit` on Windows / `fsync` on POSIX) require the NVMe controller to flush volatile DRAM cache lines to physical NAND flash blocks.
- On client-grade NVMe hardware (Micron 2400 PCIe 4.0), median sync latency is **850.2 microseconds** (~0.85 ms), with P99 tail reaching **2.23 milliseconds**.
- **Public Claim Rectification:** Any claim of "sub-microsecond durable WAL persistence to disk" is technically false unless backed by specialized battery-backed NVDIMM / persistent memory (PMEM) or battery-backed write cache controllers in Equinix LD4. The public documentation has been corrected to explicitly separate in-memory buffer append (<100 ns) from physical storage durability (~850 µs).
