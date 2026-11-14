#include <chrono>
#include <vector>
#include <iostream>
#include <cstdio>
#include <thread>
#include <atomic>
#include <cmath>
#include <iomanip>

#include "latency_stats.hpp"
#include "core/lock_free_queue.hpp"
#include "risk/pre_trade_risk.hpp"
#include "engine/execution_engine.hpp"
#include "engine/execution_journal.hpp"
#include "sync/dual_domain_clock.hpp"

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#endif

using namespace hft;

// RDTSCP cycle counter helper with serialization barrier
static inline uint64_t rdtscp_cycles() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int aux;
    return __rdtscp(&aux);
#else
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
#endif
}

// Calibrate RDTSCP to nanoseconds ratio
static double calibrate_cycles_per_ns() {
    auto start_time = std::chrono::steady_clock::now();
    uint64_t start_cycles = rdtscp_cycles();
    
    // Spin for ~50ms
    while (std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - start_time).count() < 50) {
        // burn cycles
    }
    
    auto end_time = std::chrono::steady_clock::now();
    uint64_t end_cycles = rdtscp_cycles();
    
    double ns = std::chrono::duration<double, std::nano>(end_time - start_time).count();
    double cycles = static_cast<double>(end_cycles - start_cycles);
    return cycles / ns;
}

int main() {
    std::cout << "==========================================================================================" << std::endl;
    std::cout << "  Quant_vvf Rigorous Multi-Threaded Latency & Durability Benchmark Harness                 " << std::endl;
    std::cout << "==========================================================================================" << std::endl;

    double cycles_per_ns = calibrate_cycles_per_ns();
    std::cout << "[CALIBRATION] Calibrated RDTSCP Frequency: " << std::fixed << std::setprecision(3)
              << cycles_per_ns << " cycles/ns (" << cycles_per_ns * 1000.0 << " MHz)\n\n";

    constexpr size_t BATCH_SIZE = 10000;
    constexpr size_t BATCH_ITERS = 100;
    constexpr size_t TOTAL_OPS = BATCH_SIZE * BATCH_ITERS; // 1,000,000 ops

    // -------------------------------------------------------------------------
    // 1. Batched Queue Push & Pop Microbenchmark (Amortized ns/op & Throughput)
    // -------------------------------------------------------------------------
    {
        core::SpscLockFreeQueue<uint64_t, 65536> queue;
        std::vector<double> push_batch_ns;
        std::vector<double> pop_batch_ns;
        push_batch_ns.reserve(BATCH_ITERS);
        pop_batch_ns.reserve(BATCH_ITERS);

        for (size_t b = 0; b < BATCH_ITERS; ++b) {
            auto t0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < BATCH_SIZE; ++i) {
                queue.push(i);
            }
            auto t1 = std::chrono::steady_clock::now();
            double dur_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
            push_batch_ns.push_back(dur_ns / BATCH_SIZE);

            uint64_t val;
            auto t2 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < BATCH_SIZE; ++i) {
                queue.pop(val);
            }
            auto t3 = std::chrono::steady_clock::now();
            double dur_pop_ns = std::chrono::duration<double, std::nano>(t3 - t2).count();
            pop_batch_ns.push_back(dur_pop_ns / BATCH_SIZE);
        }

        auto push_stats = bench::compute_stats(push_batch_ns);
        auto pop_stats = bench::compute_stats(pop_batch_ns);
        bench::print_stats("1. SPSC Queue Push (Batched)", push_stats);
        bench::print_stats("2. SPSC Queue Pop  (Batched)", pop_stats);
        std::cout << "   -> Push Throughput: " << std::fixed << std::setprecision(1)
                  << (1000.0 / push_stats.median_ns) << " Million ops/sec\n";
        std::cout << "   -> Pop  Throughput: " << std::fixed << std::setprecision(1)
                  << (1000.0 / pop_stats.median_ns) << " Million ops/sec\n\n";
    }

    // -------------------------------------------------------------------------
    // 2. Genuine Multi-Threaded SPSC Concurrency Benchmark (2 Independent Threads)
    // -------------------------------------------------------------------------
    {
        constexpr size_t CONCURRENT_SAMPLES = 200000;
        core::SpscLockFreeQueue<uint64_t, 65536> queue;
        std::vector<double> e2e_latencies;
        e2e_latencies.resize(CONCURRENT_SAMPLES, 0.0);

        std::atomic<bool> start_flag{false};
        std::atomic<bool> producer_done{false};

        // Consumer thread
        std::thread consumer([&]() {
            while (!start_flag.load(std::memory_order_acquire)) {
                // spin wait
            }
            size_t received = 0;
            while (received < CONCURRENT_SAMPLES) {
                uint64_t payload = 0;
                if (queue.pop(payload)) {
                    uint64_t rx_cycles = rdtscp_cycles();
                    uint64_t tx_cycles = payload;
                    if (rx_cycles > tx_cycles) {
                        double delta_ns = static_cast<double>(rx_cycles - tx_cycles) / cycles_per_ns;
                        e2e_latencies[received] = delta_ns;
                    } else {
                        e2e_latencies[received] = 1.0; // Bounded minimal interval
                    }
                    received++;
                }
            }
        });

        // Producer thread
        std::thread producer([&]() {
            while (!start_flag.load(std::memory_order_acquire)) {
                // spin wait
            }
            for (size_t i = 0; i < CONCURRENT_SAMPLES; ++i) {
                uint64_t tx_cycles = rdtscp_cycles();
                while (!queue.push(tx_cycles)) {
                    // backoff / spin wait for consumer
                }
            }
            producer_done.store(true, std::memory_order_release);
        });

        start_flag.store(true, std::memory_order_release);
        producer.join();
        consumer.join();

        auto e2e_stats = bench::compute_stats(e2e_latencies);
        bench::print_stats("3. SPSC 2-Thread E2E Concurrency", e2e_stats);
        std::cout << "\n";
    }

    // -------------------------------------------------------------------------
    // 3. Pre-Trade Risk Ledger Check (In-Memory Open-Addressing Slot Scan)
    // -------------------------------------------------------------------------
    {
        constexpr size_t SAMPLES = 100000;
        risk::RiskConfig r_cfg;
        r_cfg.max_order_clip_btc = 5.0;
        r_cfg.max_gross_exposure_btc = 500.0;
        r_cfg.max_collar_deviation_pct = 0.05;
        r_cfg.max_orders_per_second = 100000;
        r_cfg.max_market_data_age_ns = 500'000'000ULL;

        risk::InlinePreTradeRiskGate risk_gate(r_cfg);
        std::vector<double> risk_lats;
        risk_lats.reserve(SAMPLES);

        net::OrderRequest req{};
        req.order_id = 1;
        req.symbol = net::InstrumentSymbol("BTC-PERPETUAL");
        req.side = net::Side::BUY;
        req.quantity = 1.0;
        req.price = 50000.0;

        uint64_t now_ns = sync::DualDomainClock::get_host_monotonic_ns();

        // Warmup
        for (uint64_t i = 1; i <= 500; ++i) {
            req.order_id = i;
            risk_gate.evaluate_order(req, 49999.0, 50001.0, 50000.0, now_ns, false);
            risk_gate.release_reservation(i);
        }

        for (uint64_t i = 1; i <= SAMPLES; ++i) {
            uint64_t oid = (i % 800) + 1;
            req.order_id = oid;

            uint64_t c0 = rdtscp_cycles();
            auto res = risk_gate.evaluate_order(req, 49999.0, 50001.0, 50000.0, now_ns, false);
            uint64_t c1 = rdtscp_cycles();

            if (res == risk::RiskCheckResult::APPROVED) {
                double ns = static_cast<double>(c1 - c0) / cycles_per_ns;
                risk_lats.push_back(ns);
            }
            risk_gate.release_reservation(oid);
        }

        auto risk_stats = bench::compute_stats(risk_lats);
        bench::print_stats("4. Pre-Trade Risk Check (In-Mem)", risk_stats);
    }

    // -------------------------------------------------------------------------
    // 4. Execution Dedup & State Mutate (In-Memory 2-Phase Commit)
    // -------------------------------------------------------------------------
    {
        constexpr size_t SAMPLES = 100000;
        risk::RiskConfig r_cfg;
        r_cfg.max_order_clip_btc = 5.0;
        r_cfg.max_gross_exposure_btc = 500.0;
        r_cfg.max_collar_deviation_pct = 0.05;
        r_cfg.max_orders_per_second = 100000;
        r_cfg.max_market_data_age_ns = 500'000'000ULL;

        engine::HftExecutionEngine exec_engine(r_cfg);
        std::vector<double> exec_lats;
        exec_lats.reserve(SAMPLES);

        exec_engine.set_current_delta_btc(0.20, 50000.0);
        net::MarketDepthPacket depth{};
        depth.symbol = net::InstrumentSymbol("BTC-PERPETUAL");
        depth.spot_index = 50000.0;
        depth.best_bid_price = 49999.0;
        depth.best_ask_price = 50001.0;
        depth.mark_price = 50000.0;
        depth.nic_rx_time_ns = sync::DualDomainClock::get_simulation_epoch_ns();

        exec_engine.process_market_tick(depth);

        for (uint64_t i = 1; i <= SAMPLES; ++i) {
            uint64_t eid = 10000000ULL + i;
            uint64_t c0 = rdtscp_cycles();
            auto res = exec_engine.on_execution_report(eid, 1, net::Side::SELL, 0.00001, 49999.0);
            uint64_t c1 = rdtscp_cycles();

            if (res == engine::ExecutionResult::SUCCESS_APPLIED) {
                double ns = static_cast<double>(c1 - c0) / cycles_per_ns;
                exec_lats.push_back(ns);
            }
        }

        auto exec_stats = bench::compute_stats(exec_lats);
        bench::print_stats("5. Execution Dedup & State Mutate", exec_stats);
        std::cout << "\n";
    }

    // -------------------------------------------------------------------------
    // 5. WAL Durability Benchmarks: fwrite, fflush, and Physical _commit/fsync
    // -------------------------------------------------------------------------
    {
        const char* wal_bench_file = "test_bench_durability.wal";
        std::remove(wal_bench_file);

        constexpr size_t SAMPLES = 50000;
        std::vector<double> fwrite_lats;
        std::vector<double> fflush_lats;
        fwrite_lats.reserve(SAMPLES);
        fflush_lats.reserve(1000);

        FILE* f = std::fopen(wal_bench_file, "wb");
        if (f) {
            engine::JournalRecord rec{};
            rec.magic = engine::WAL_MAGIC_V2;
            rec.version = engine::WAL_SCHEMA_VERSION_V2;

            for (size_t i = 0; i < SAMPLES; ++i) {
                rec.exec_id = 500000ULL + i;
                uint64_t c0 = rdtscp_cycles();
                size_t w = std::fwrite(&rec, sizeof(engine::JournalRecord), 1, f);
                uint64_t c1 = rdtscp_cycles();
                if (w == 1) {
                    fwrite_lats.push_back(static_cast<double>(c1 - c0) / cycles_per_ns);
                }

                if (i % 50 == 0) {
                    uint64_t f0 = rdtscp_cycles();
                    int fl = std::fflush(f);
                    uint64_t f1 = rdtscp_cycles();
                    if (fl == 0) {
                        fflush_lats.push_back(static_cast<double>(f1 - f0) / cycles_per_ns);
                    }
                }
            }
            std::fclose(f);
            std::remove(wal_bench_file);
        }

        auto fwrite_stats = bench::compute_stats(fwrite_lats);
        auto fflush_stats = bench::compute_stats(fflush_lats);
        bench::print_stats("6. WAL Buffer Write (std::fwrite)", fwrite_stats);
        bench::print_stats("7. WAL OS Page Flush (std::fflush)", fflush_stats);

        // Hardware Durable _commit / fsync (500 samples on physical flash)
        constexpr size_t SYNC_SAMPLES = 500;
        std::vector<double> sync_lats;
        sync_lats.reserve(SYNC_SAMPLES);
        size_t success_commits = 0;
        size_t failed_commits = 0;

        engine::DurableExecutionJournal durable_journal(
            wal_bench_file, engine::WalDurabilityPolicy::SYNCHRONOUSLY_DURABLE
        );

        for (size_t i = 0; i < SYNC_SAMPLES; ++i) {
            uint64_t eid = 9000000ULL + i;
            auto t0 = std::chrono::steady_clock::now();
            auto st = durable_journal.append_record(
                engine::JournalRecordType::ORDER_FILLED,
                eid, 1, "BTC-PERPETUAL", 0, 0.1, 50000.0, 1700000000ULL, i
            );
            auto t1 = std::chrono::steady_clock::now();

            if (st == engine::WalStatus::SUCCESS) {
                success_commits++;
                sync_lats.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
            } else {
                failed_commits++;
            }
        }

        auto sync_stats = bench::compute_stats(sync_lats);
        bench::print_stats("8. WAL Hardware Sync (_commit/fsync)", sync_stats);
        std::cout << "   -> Confirmed Physical Commits: " << success_commits
                  << " / " << SYNC_SAMPLES << " (Failures: " << failed_commits << ")\n";

        std::remove(wal_bench_file);
    }

    std::cout << "==========================================================================================" << std::endl;
    return 0;
}
