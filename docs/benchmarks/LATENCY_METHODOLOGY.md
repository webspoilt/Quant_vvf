# QUANT_VVF — LATENCY BENCHMARKING METHODOLOGY

**Specification:** Institutional Benchmark Rigor & Measurement Protocol  
**Benchmark Binary:** `hft/benchmarks/bench_engine.exe`  
**Source Code:** `hft/benchmarks/bench_engine_latency.cpp`  

---

## 1. Measurement Principles & Disclosures

### A. Clock Source & Serialization
- Measurements utilize `Calibrated hardware RDTSCP instruction with memory serialization barriers and std::chrono::steady_clock` calibrated against the platform hardware invariant TSC / QueryPerformanceCounter.
- To prevent out-of-order execution reordering across measurement boundaries, assembly memory fences and atomic releases (`std::memory_order_release` / `std::memory_order_acquire`) serialize the measured sections.
- CPU cycles are never presented as nanoseconds without documented nominal frequency conversion.

### B. Sample Count & Warmup Protocol
- **Warmup:** Every benchmark loop executes 10,000 unmeasured iterations to ensure CPU instruction cache (L1i) and data cache (L1d) lines are populated, branch predictors are trained, and memory pages are faulted in.
- **Sample Count:** 100,000 samples for in-memory operations (SPSC queue, Risk check, Execution dedup, WAL buffer append); 500 samples for physical disk durability commits (`_commit` / `fsync`) to prevent thermal throttling and NVMe write endurance degradation.

### C. Distribution Metrics
For every stage, the following percentiles are computed on the sorted distribution:
- **Min:** Best-case hardware execution time.
- **Median (P50):** 50th percentile nominal operating latency.
- **P90 / P95:** Tail performance under typical cache interference.
- **P99 / P99.9:** Extreme tail latency under OS thread preemption or context switching.
- **Max:** Maximum outlier recorded in sample window.
- **StdDev:** Sample standard deviation.

---

## 2. Benchmark Stage Definitions

1. **Stage 1: SPSC Queue Enqueue (`hft::core::SpscLockFreeQueue::push`):**
   - Ingests 64-bit integer payload.
   - Lock-free ring buffer tail index atomic update with acquire/release barriers.
2. **Stage 2: SPSC Queue Dequeue (`hft::core::SpscLockFreeQueue::pop`):**
   - Pop 64-bit item, increments atomic head counter.
3. **Stage 3: Pre-Trade Risk Check (`hft::risk::InlinePreTradeRiskGate::evaluate_order`):**
   - Checks clip size, worst-case gross exposure, price collar against best bid/ask, variance circuit breaker, and scans reservation table (open-addressing collision scan).
4. **Stage 4: Execution Engine Dedup & Mutate (`hft::engine::HftExecutionEngine::on_execution_report`):**
   - Rejects reserved IDs (0, `UINT64_MAX`), verifies atomic two-phase deduplication table, checks terminal status immutability, validates price/quantity, and updates position ledger.
5. **Stage 5: WAL Buffer Append (`std::fwrite` to userspace buffer):**
   - Appends 96-byte packed binary (compile-time asserted) `JournalRecord` with magic header (`0x57414C31`), standard IEEE 802.3 CRC32 checksum, and timestamp.
6. **Stage 6: True Hardware Durability Commit (`_commit` / `fsync`):**
   - Flushes userspace buffer and issues NVMe hardware flush command forcing non-volatile NAND persistence.
