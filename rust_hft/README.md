# Production HFT Ingestion & Order Gateway Skeleton (Rust Edition 2024)

An ultra-low-latency, zero-allocation, lock-free market data ingestion and order gateway engine built strictly to HFT performance constraints.

---

## Performance & Micro-Architectural Blueprint

| Constraint | Implementation | Micro-Architectural Jitter Mitigation |
| :--- | :--- | :--- |
| **1. Threading & Pinning** | `core_affinity::set_for_current` | Threads are bound to isolated physical CPU cores (`isolcpus`), eliminating OS context-switch overheads, thread migration penalties, and TLB flushes. |
| **2. Zero Allocations** | `[u8; 1024]` stack buffers & pre-allocated structs | Zero heap allocations (`malloc`/`free`, `Box`, `Vec`, `String`) on the hot path. Eliminates heap fragmentation and dynamic memory locks. |
| **3. Lock-Free IPC** | Bounded SPSC Ring Buffer + `#[repr(align(64))]` | Reader and writer pointers reside on separate 64-byte cache lines, preventing MESI/MOESI cache invalidation storms (false sharing) between producer and consumer cores. |
| **4. Zero-Copy Parsing** | `zerocopy` + byte slicing (`&[u8]`) | Directly parses wire frames in-place from NIC DMA memory buffers without copying bytes into intermediate representations. |
| **5. Busy-Poll Loop** | `std::hint::spin_loop()` | Replaces `tokio`/epoll sleeping syscalls with busy polling, ensuring the CPU core never down-clocks into lower C-states (C1/C6). |

---

## Project Structure

```
rust_hft/
├── Cargo.toml               # Configured for Edition 2024, LTO=fat, panic=abort, opt-level=3
├── README.md                # Architecture & micro-architectural documentation
└── src/
    ├── lib.rs               # Library root export
    ├── protocol.rs          # Packed ITCH wire packets & zerocopy parser
    ├── ring_buffer.rs       # Cache-aligned (64-byte) SPSC lock-free ring buffer
    ├── threading.rs         # CPU core affinity and thread pinning builder
    └── main.rs              # End-to-end hot path simulation (Ingestion -> SPSC -> Strategy)
```

---

## Building and Running

Ensure Rust (Edition 2024 / 1.85+) is available:

```bash
cd Quant_vvf/rust_hft

# Run optimized release build
cargo run --release
```
