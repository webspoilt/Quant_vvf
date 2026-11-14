#include <chrono>
#include <vector>
#include <iostream>
#include <cstdio>
#include "latency_stats.hpp"
#include "core/lock_free_queue.hpp"
#include "risk/pre_trade_risk.hpp"
#include "engine/execution_engine.hpp"
#include "engine/execution_journal.hpp"
#include "sync/dual_domain_clock.hpp"

using namespace hft;

int main() {
    std::cout << "==========================================================================================" << std::endl;
    std::cout << "  Quant_vvf Genuine Hardware & In-Memory Latency Benchmark Harness                        " << std::endl;
    std::cout << "==========================================================================================" << std::endl;

    constexpr size_t WARMUP = 10000;
    constexpr size_t SAMPLES = 100000;

    // -------------------------------------------------------------------------
    // 1. SPSC Queue Push / Pop
    // -------------------------------------------------------------------------
    {
        core::SpscLockFreeQueue<uint64_t, 1024> queue;
        std::vector<double> push_lats;
        std::vector<double> pop_lats;
        push_lats.reserve(SAMPLES);
        pop_lats.reserve(SAMPLES);

        // Warmup
        for (size_t i = 0; i < WARMUP; ++i) {
            queue.push(i);
            uint64_t val;
            queue.pop(val);
        }

        // Measure push
        for (size_t i = 0; i < SAMPLES; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            bool ok = queue.push(i);
            auto t1 = std::chrono::steady_clock::now();
            if (ok) {
                push_lats.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
            }

            // Pop immediately
            uint64_t val;
            auto t2 = std::chrono::steady_clock::now();
            bool ok_pop = queue.pop(val);
            auto t3 = std::chrono::steady_clock::now();
            if (ok_pop) {
                pop_lats.push_back(std::chrono::duration<double, std::nano>(t3 - t2).count());
            }
        }

        auto push_stats = bench::compute_stats(push_lats);
        auto pop_stats = bench::compute_stats(pop_lats);
        bench::print_stats("1. SPSC Queue Push (In-Memory)", push_stats);
        bench::print_stats("2. SPSC Queue Pop (In-Memory)", pop_stats);
    }

    // -------------------------------------------------------------------------
    // 2. Pre-Trade Risk Ledger Check (In-Memory Open-Addressing Slot Scan)
    // -------------------------------------------------------------------------
    {
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

            auto t0 = std::chrono::steady_clock::now();
            auto res = risk_gate.evaluate_order(req, 49999.0, 50001.0, 50000.0, now_ns, false);
            auto t1 = std::chrono::steady_clock::now();
            if (res == risk::RiskCheckResult::APPROVED) {
                risk_lats.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
            }
            risk_gate.release_reservation(oid);
        }

        auto risk_stats = bench::compute_stats(risk_lats);
        bench::print_stats("3. Pre-Trade Risk Check (In-Mem)", risk_stats);
    }

    // -------------------------------------------------------------------------
    // 3. Execution Dedup & Order State Update (In-Memory)
    // -------------------------------------------------------------------------
    {
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
            auto t0 = std::chrono::steady_clock::now();
            auto res = exec_engine.on_execution_report(eid, 1, net::Side::SELL, 0.00001, 49999.0);
            auto t1 = std::chrono::steady_clock::now();
            if (res == engine::ExecutionResult::SUCCESS_APPLIED) {
                exec_lats.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
            }
        }

        auto exec_stats = bench::compute_stats(exec_lats);
        bench::print_stats("4. Execution Dedup & State (In-Mem)", exec_stats);
    }

    // -------------------------------------------------------------------------
    // 4. WAL Appends: Userspace fwrite vs Physical _commit/fsync
    // -------------------------------------------------------------------------
    {
        const char* wal_file = "test_bench_durability.wal";
        std::remove(wal_file);

        // Sub-test A: Userspace fwrite buffering without fsync
        std::vector<double> append_lats;
        append_lats.reserve(SAMPLES);
        FILE* raw_file = std::fopen("test_bench_raw.wal", "wb");
        if (raw_file) {
            engine::JournalRecord rec{};
            rec.magic = 0x57414C31;
            rec.version = 1;
            for (size_t i = 0; i < SAMPLES; ++i) {
                rec.exec_id = 100000ULL + i;
                auto t0 = std::chrono::steady_clock::now();
                std::fwrite(&rec, sizeof(engine::JournalRecord), 1, raw_file);
                auto t1 = std::chrono::steady_clock::now();
                append_lats.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
            }
            std::fclose(raw_file);
            std::remove("test_bench_raw.wal");
        }
        auto append_stats = bench::compute_stats(append_lats);
        bench::print_stats("5. WAL Buffer Append (fwrite)", append_stats);

        // Sub-test B: Hardware Durable _commit / fsync
        // True physical NVMe disk sync takes microseconds to milliseconds
        constexpr size_t SYNC_SAMPLES = 500;
        std::vector<double> sync_lats;
        sync_lats.reserve(SYNC_SAMPLES);

        engine::DurableExecutionJournal journal(wal_file);
        for (size_t i = 0; i < SYNC_SAMPLES; ++i) {
            uint64_t eid = 9000000ULL + i;
            auto t0 = std::chrono::steady_clock::now();
            journal.append_execution(eid, 1, 0, 0.1, 50000.0, 1700000000ULL);
            auto t1 = std::chrono::steady_clock::now();
            sync_lats.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
        }

        auto sync_stats = bench::compute_stats(sync_lats);
        bench::print_stats("6. WAL Disk Sync (_commit/fsync)", sync_stats);

        std::remove(wal_file);
    }

    std::cout << "==========================================================================================" << std::endl;
    return 0;
}
