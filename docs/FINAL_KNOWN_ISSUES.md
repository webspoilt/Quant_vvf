# FINAL KNOWN ISSUES, DISCLOSURES & OPERATIONAL BOUNDARIES — QUANT_VVF

**Document Version:** 1.0.0  
**Status:** Authoritative Operational Disclosure  

In compliance with our commitment to absolute engineering honesty, this document explicitly details every known limitation, operational boundary, and unverified infrastructure component in `Quant_vvf`.

---

## 1. Simulation vs. Production Boundaries

### 1.1 Physical Kernel Bypass (DPDK / Solarflare EF_VI)
* **Status:** SIMULATED IN SOFTWARE
* **Disclosure:** While zero-copy ring buffers and packet structure layouts are fully implemented in `hft/include/network/kernel_bypass.hpp`, actual direct memory access (DMA) kernel bypass requires physical Solarflare or Intel high-speed NICs running Onload/EF_VI or DPDK drivers in a bare-metal Linux environment. In Windows and development environments, packet reception is simulated in userspace.

### 1.2 Hardware IEEE 1588 PTP Grandmaster Clock
* **Status:** SIMULATED / DOMAIN SEGREGATED
* **Disclosure:** Sub-100ns precision timestamps synchronized via an atomic PTP grandmaster cross-connect in Equinix LD4 (Slough) require a dedicated NIC hardware clock (`/dev/ptpX`). On Windows hosts, `sync::DualDomainClock::get_hardware_ptp_epoch_ns()` explicitly reports `is_supported = false`. Clock domain segregation strictly enforces `d1 == d2`, preventing invalid cross-domain comparisons between software monotonic time and hardware PTP epochs.

### 1.3 Live Exchange TCP/UDP Sockets
* **Status:** TESTNET & PAPER-TRADING VERIFIED / PRODUCTION COLOCATION UNVERIFIED
* **Disclosure:** The binary Simple Binary Encoding (SBE) parser, 16-channel feed arbitrator, sequence gap detection, and verified retransmission recovery pipeline have been verified using authoritative protocol test packets. However, sustained institutional live sessions over Deribit's private cross-connect in Equinix LD4 require institutional cross-connect provisioning. Production order dispatch is physically gated behind `DERIBIT_LIVE_AUTHORIZED=TRUE` to prevent accidental execution.

### 1.4 FPGA Hardware RTL Synthesis
* **Status:** SOURCE INSPECTED / HARDWARE UNVERIFIED
* **Disclosure:** Verilog RTL for line-rate BBO extraction and inline price/clip collars is drafted under `hft/fpga/hft_bbo_risk_engine.v`. Physical FPGA deployment requires Xilinx Vivado bitstream synthesis, place-and-route, and timing closure on a target acceleration card (e.g., Alveo U50).

---

## 2. In-Memory Execution Deduplication Window

* **Status:** BOUNDED FIFO DEDUPLICATION WITH DURABLE WAL BACKUP
* **Disclosure:** The C++ in-memory deduplication table utilizes an open-addressed hash table with a FIFO eviction ring of 1024 entries. Evictions insert tombstones (`UINT64_MAX`) to preserve linear probing chains. While durable restart reconciliation is now backed by a binary append-only Write-Ahead Log (`execution_journal.hpp`), the in-memory deduplication window cannot detect duplicate execution IDs arriving more than 1024 unique executions later if the log has been pruned.

---

## 3. Python Statistical Dependencies

* **Warning Disclosures:**
  - `statsmodels.tsa.stattools.adfuller`: Emits a `FutureWarning` in statsmodels 0.15 regarding return object tuples in future versions (0.16+). This warning does not affect numerical calculation accuracy.
  - `scipy.cluster.hierarchy.linkage`: In `portfolio/optimizer.py`, computing HRP distance on correlation matrices emits a `ClusterWarning` regarding condensing redundant symmetric distance matrices. The resulting single-linkage hierarchy remains mathematically identical.
