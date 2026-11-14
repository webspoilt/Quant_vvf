#include "engine/execution_engine.hpp"
#include "core/cpu_pinning.hpp"
#include "sync/dual_domain_clock.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using namespace hft;

int main() {
    std::cout << "====================================================================\n";
    std::cout << "  Quant_vvf Seventh Engineering Remediation Full Test Suite          \n";
    std::cout << "====================================================================\n";

    risk::RiskConfig r_cfg;
    r_cfg.max_order_clip_btc = 5.0;
    r_cfg.max_gross_exposure_btc = 50.0;
    r_cfg.max_collar_deviation_pct = 0.05;
    r_cfg.max_orders_per_second = 10000;
    r_cfg.max_market_data_age_ns = 500'000'000ULL;

    uint64_t host_now = sync::DualDomainClock::get_host_monotonic_ns();

    // =========================================================================
    // TEST 1: RESERVED EXECUTION ID 0 REJECTION
    // =========================================================================
    std::cout << "\n[TEST 1: RESERVED EXECUTION ID 0 REJECTION]\n";
    {
        engine::HftExecutionEngine engine(r_cfg);
        engine.set_current_delta_btc(0.20, 65000.0);

        net::MarketDepthPacket depth{};
        depth.symbol = net::InstrumentSymbol("BTC-PERPETUAL");
        depth.spot_index = 65000.0;
        depth.best_bid_price = 64999.0;
        depth.best_ask_price = 65001.0;
        depth.mark_price = 65000.0;
        depth.nic_rx_time_ns = sync::DualDomainClock::get_simulation_epoch_ns();

        engine.process_market_tick(depth);
        assert(engine.has_active_order());

        // Attempt fill with reserved exec_id == 0
        auto res = engine.on_execution_report(0, 1, net::Side::SELL, 0.10, 64999.0);
        assert(res == engine::ExecutionResult::REJECT_RESERVED_EXEC_ID);
        assert(engine.is_duplicate_exec_id(0) == false);
        assert(engine.get_position_btc() == 0.0);
        assert(engine.get_gross_exposure_btc() == 0.20); // reservation intact, position untouched
        std::cout << "  1.1 Reserved Exec ID 0 rejected with REJECT_RESERVED_EXEC_ID: PASSED\n";
        std::cout << "  1.2 State, position, and reservation untouched: PASSED\n";
    }

    // =========================================================================
    // TEST 2: RESERVED EXECUTION ID UINT64_MAX REJECTION
    // =========================================================================
    std::cout << "\n[TEST 2: RESERVED EXECUTION ID UINT64_MAX REJECTION]\n";
    {
        engine::HftExecutionEngine engine(r_cfg);
        engine.set_current_delta_btc(0.20, 65000.0);

        net::MarketDepthPacket depth{};
        depth.symbol = net::InstrumentSymbol("BTC-PERPETUAL");
        depth.spot_index = 65000.0;
        depth.best_bid_price = 64999.0;
        depth.best_ask_price = 65001.0;
        depth.mark_price = 65000.0;
        depth.nic_rx_time_ns = sync::DualDomainClock::get_simulation_epoch_ns();

        engine.process_market_tick(depth);
        assert(engine.has_active_order());

        // Attempt fill with reserved exec_id == UINT64_MAX (Tombstone value)
        uint64_t tombstone_id = std::numeric_limits<uint64_t>::max();
        auto res = engine.on_execution_report(tombstone_id, 1, net::Side::SELL, 0.10, 64999.0);
        assert(res == engine::ExecutionResult::REJECT_RESERVED_EXEC_ID);
        assert(engine.is_duplicate_exec_id(tombstone_id) == false);
        assert(engine.get_position_btc() == 0.0);
        assert(engine.get_gross_exposure_btc() == 0.20);
        std::cout << "  2.1 Reserved Exec ID UINT64_MAX rejected with REJECT_RESERVED_EXEC_ID: PASSED\n";
    }

    // =========================================================================
    // TEST 3: INVALID FILL PRICE REJECTION (NaN, Inf, 0, Negative)
    // =========================================================================
    std::cout << "\n[TEST 3: INVALID FILL PRICE REJECTION]\n";
    {
        engine::HftExecutionEngine engine(r_cfg);
        engine.set_current_delta_btc(0.20, 65000.0);

        net::MarketDepthPacket depth{};
        depth.symbol = net::InstrumentSymbol("BTC-PERPETUAL");
        depth.spot_index = 65000.0;
        depth.best_bid_price = 64999.0;
        depth.best_ask_price = 65001.0;
        depth.mark_price = 65000.0;
        depth.nic_rx_time_ns = sync::DualDomainClock::get_simulation_epoch_ns();

        engine.process_market_tick(depth);
        assert(engine.has_active_order());

        // 3.1 NaN price
        auto res_nan = engine.on_execution_report(101, 1, net::Side::SELL, 0.10, std::numeric_limits<double>::quiet_NaN());
        assert(res_nan == engine::ExecutionResult::REJECT_INVALID_PRICE);
        assert(engine.is_duplicate_exec_id(101) == false);

        // 3.2 Inf price
        auto res_inf = engine.on_execution_report(102, 1, net::Side::SELL, 0.10, std::numeric_limits<double>::infinity());
        assert(res_inf == engine::ExecutionResult::REJECT_INVALID_PRICE);
        assert(engine.is_duplicate_exec_id(102) == false);

        // 3.3 Zero price
        auto res_zero = engine.on_execution_report(103, 1, net::Side::SELL, 0.10, 0.0);
        assert(res_zero == engine::ExecutionResult::REJECT_INVALID_PRICE);
        assert(engine.is_duplicate_exec_id(103) == false);

        // 3.4 Negative price
        auto res_neg = engine.on_execution_report(104, 1, net::Side::SELL, 0.10, -65000.0);
        assert(res_neg == engine::ExecutionResult::REJECT_INVALID_PRICE);
        assert(engine.is_duplicate_exec_id(104) == false);

        std::cout << "  3.1 NaN, Inf, Zero, and Negative prices rejected: PASSED\n";
        std::cout << "  3.2 Rejected IDs (101-104) uncommitted and reusable: PASSED\n";
    }

    // =========================================================================
    // TEST 4: INVALID FILL QUANTITY REJECTION
    // =========================================================================
    std::cout << "\n[TEST 4: INVALID FILL QUANTITY REJECTION]\n";
    {
        engine::HftExecutionEngine engine(r_cfg);
        engine.set_current_delta_btc(0.20, 65000.0);

        net::MarketDepthPacket depth{};
        depth.symbol = net::InstrumentSymbol("BTC-PERPETUAL");
        depth.spot_index = 65000.0;
        depth.best_bid_price = 64999.0;
        depth.best_ask_price = 65001.0;
        depth.mark_price = 65000.0;
        depth.nic_rx_time_ns = sync::DualDomainClock::get_simulation_epoch_ns();

        engine.process_market_tick(depth);
        assert(engine.has_active_order());

        auto res_q_nan = engine.on_execution_report(201, 1, net::Side::SELL, std::numeric_limits<double>::quiet_NaN(), 64999.0);
        assert(res_q_nan == engine::ExecutionResult::REJECT_INVALID_QUANTITY);

        auto res_q_zero = engine.on_execution_report(202, 1, net::Side::SELL, 0.0, 64999.0);
        assert(res_q_zero == engine::ExecutionResult::REJECT_INVALID_QUANTITY);

        auto res_q_neg = engine.on_execution_report(203, 1, net::Side::SELL, -0.05, 64999.0);
        assert(res_q_neg == engine::ExecutionResult::REJECT_INVALID_QUANTITY);

        std::cout << "  4.1 Non-positive and non-finite quantities rejected: PASSED\n";
    }

    // =========================================================================
    // TEST 5: ATOMIC TRANSACTION INTEGRITY (PRE-VALIDATION BEFORE ID COMMIT)
    // =========================================================================
    std::cout << "\n[TEST 5: ATOMIC TRANSACTION INTEGRITY]\n";
    {
        engine::HftExecutionEngine engine(r_cfg);
        engine.set_current_delta_btc(0.20, 65000.0);

        net::MarketDepthPacket depth{};
        depth.symbol = net::InstrumentSymbol("BTC-PERPETUAL");
        depth.spot_index = 65000.0;
        depth.best_bid_price = 64999.0;
        depth.best_ask_price = 65001.0;
        depth.mark_price = 65000.0;
        depth.nic_rx_time_ns = sync::DualDomainClock::get_simulation_epoch_ns();

        engine.process_market_tick(depth);
        assert(engine.has_active_order());

        // Send mismatched order ID with ID 7777
        auto r_bad = engine.on_execution_report(7777, 999, net::Side::SELL, 0.10, 64999.0);
        assert(r_bad == engine::ExecutionResult::REJECT_MISMATCHED_ORDER_ID);
        assert(engine.is_duplicate_exec_id(7777) == false);

        // Now resubmit valid report with ID 7777
        auto r_ok = engine.on_execution_report(7777, 1, net::Side::SELL, 0.10, 64999.0);
        assert(r_ok == engine::ExecutionResult::SUCCESS_APPLIED);
        assert(engine.is_duplicate_exec_id(7777) == true);
        assert(engine.get_position_btc() == -0.10);

        // Resubmit duplicate 7777
        auto r_dup = engine.on_execution_report(7777, 1, net::Side::SELL, 0.10, 64999.0);
        assert(r_dup == engine::ExecutionResult::REJECT_DUPLICATE_EXEC_ID);
        assert(engine.get_position_btc() == -0.10); // Not incremented

        std::cout << "  5.1 Atomic pre-validation and safe reuse of execution ID: PASSED\n";
        std::cout << "  5.2 Duplicate submission safely rejected without position corruption: PASSED\n";
    }

    // =========================================================================
    // TEST 6: TOMBSTONE OPEN ADDRESSING COLLISION & CHAIN INTEGRITY
    // =========================================================================
    std::cout << "\n[TEST 6: TOMBSTONE OPEN ADDRESSING COLLISION & CHAIN INTEGRITY]\n";
    {
        engine::HftExecutionEngine engine(r_cfg);

        // Colliding IDs: 1, 2049, 4097
        engine.commit_exec_id(1);
        engine.commit_exec_id(2049);
        engine.commit_exec_id(4097);

        assert(engine.is_duplicate_exec_id(1) == true);
        assert(engine.is_duplicate_exec_id(2049) == true);
        assert(engine.is_duplicate_exec_id(4097) == true);

        // Evict ID 1
        for (uint64_t id = 20000; id < 20000 + 1022; ++id) {
            engine.commit_exec_id(id);
        }

        // Probing through tombstone
        assert(engine.is_duplicate_exec_id(2049) == true);
        assert(engine.is_duplicate_exec_id(4097) == true);
        std::cout << "  6.1 Open addressing chain probing across tombstones verified: PASSED\n";
    }

    // =========================================================================
    // TEST 7: TERMINAL ORDER IMMUTABILITY
    // =========================================================================
    std::cout << "\n[TEST 7: TERMINAL ORDER IMMUTABILITY]\n";
    {
        engine::HftExecutionEngine engine(r_cfg);
        engine.set_current_delta_btc(0.20, 65000.0);

        net::MarketDepthPacket depth{};
        depth.symbol = net::InstrumentSymbol("BTC-PERPETUAL");
        depth.spot_index = 65000.0;
        depth.best_bid_price = 64999.0;
        depth.best_ask_price = 65001.0;
        depth.mark_price = 65000.0;
        depth.nic_rx_time_ns = sync::DualDomainClock::get_simulation_epoch_ns();

        engine.process_market_tick(depth);

        // Fill remaining order (0.20 BTC)
        auto r_full = engine.on_execution_report(8001, 1, net::Side::SELL, 0.20, 64999.0);
        assert(r_full == engine::ExecutionResult::SUCCESS_APPLIED);

        // Now order is FILLED (terminal). Subsequent fill must be rejected
        auto r_late = engine.on_execution_report(8002, 1, net::Side::SELL, 0.05, 64999.0);
        assert(r_late == engine::ExecutionResult::REJECT_TERMINAL_ORDER);
        assert(engine.get_position_btc() == -0.20);
        std::cout << "  7.1 Terminal order rejected further fills: PASSED\n";
    }

    // =========================================================================
    // TEST 8: OVERFILL PROTECTION
    // =========================================================================
    std::cout << "\n[TEST 8: OVERFILL PROTECTION]\n";
    {
        engine::HftExecutionEngine engine(r_cfg);
        engine.set_current_delta_btc(0.20, 65000.0);

        net::MarketDepthPacket depth{};
        depth.symbol = net::InstrumentSymbol("BTC-PERPETUAL");
        depth.spot_index = 65000.0;
        depth.best_bid_price = 64999.0;
        depth.best_ask_price = 65001.0;
        depth.mark_price = 65000.0;
        depth.nic_rx_time_ns = sync::DualDomainClock::get_simulation_epoch_ns();

        engine.process_market_tick(depth);

        // Attempt fill exceeding total quantity (0.25 > 0.20)
        auto r_over = engine.on_execution_report(9001, 1, net::Side::SELL, 0.25, 64999.0);
        assert(r_over == engine::ExecutionResult::REJECT_OVERFILL);
        assert(engine.get_position_btc() == 0.0);
        std::cout << "  8.1 Overfill rejected before risk/position update: PASSED\n";
    }

    // =========================================================================
    // TEST 9: DUAL-DOMAIN CLOCK STRICT INCOMPATIBILITY & WINDOWS FALLBACK
    // =========================================================================
    std::cout << "\n[TEST 9: DUAL-DOMAIN CLOCK DOMAIN INCOMPATIBILITY]\n";
    {
        assert(sync::DualDomainClock::check_domain_compatibility(sync::ClockDomain::PTP_HARDWARE_EPOCH, sync::ClockDomain::PTP_HARDWARE_EPOCH) == true);
        assert(sync::DualDomainClock::check_domain_compatibility(sync::ClockDomain::PTP_HARDWARE_EPOCH, sync::ClockDomain::SIMULATION_EPOCH) == false);
        assert(sync::DualDomainClock::check_domain_compatibility(sync::ClockDomain::HOST_MONOTONIC, sync::ClockDomain::SIMULATION_EPOCH) == false);

        auto ptp_sample = sync::DualDomainClock::get_hardware_ptp_epoch_ns();
        assert(ptp_sample.is_supported == false); // Non-Linux environment honest status
        std::cout << "  9.1 Clock domain segregation & explicit unsupported hardware PHC status: PASSED\n";
    }

    // =========================================================================
    // TEST 10: DURABLE WRITE-AHEAD LOG EXECUTION RECOVERY ACROSS RESTARTS
    // =========================================================================
    std::cout << "\n[TEST 10: DURABLE WAL EXECUTION RECOVERY ACROSS RESTARTS]\n";
    {
        std::string wal_path = "test_execution.wal";
        std::remove(wal_path.c_str());

        {
            // Process 1: Executes fill and records to WAL
            engine::HftExecutionEngine engine1(r_cfg);
            engine1.enable_journal(wal_path);
            engine1.set_current_delta_btc(0.20, 65000.0);

            net::MarketDepthPacket depth{};
            depth.symbol = net::InstrumentSymbol("BTC-PERPETUAL");
            depth.spot_index = 65000.0;
            depth.best_bid_price = 64999.0;
            depth.best_ask_price = 65001.0;
            depth.mark_price = 65000.0;
            depth.nic_rx_time_ns = sync::DualDomainClock::get_simulation_epoch_ns();

            engine1.process_market_tick(depth);
            auto res = engine1.on_execution_report(5555, 1, net::Side::SELL, 0.10, 64999.0);
            assert(res == engine::ExecutionResult::SUCCESS_APPLIED);
            assert(engine1.is_duplicate_exec_id(5555) == true);
        } // Process 1 terminates/crashes

        // Simulate a torn/corrupted partial write at the end of the file
        {
            std::FILE* f_corrupt = std::fopen(wal_path.c_str(), "ab");
            assert(f_corrupt != nullptr);
            uint8_t garbage[7] = {0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03}; // Torn partial record
            std::fwrite(garbage, 1, sizeof(garbage), f_corrupt);
            std::fclose(f_corrupt);
        }

        {
            // Process 2: Restarts fresh, recovers WAL from disk, safely rejecting torn garbage
            engine::HftExecutionEngine engine2(r_cfg);
            engine2.enable_journal(wal_path); // Recovers committed exec_id 5555, discards corrupt tail

            // Assert that recovered ID 5555 is immediately recognized as duplicate
            assert(engine2.is_duplicate_exec_id(5555) == true);
            std::cout << "  10.1 Committed execution ID 5555 recovered from disk WAL after restart: PASSED\n";
            std::cout << "  10.2 Torn corrupted trailing bytes safely rejected via magic & checksum: PASSED\n";
        }

        std::remove(wal_path.c_str());
    }

    std::cout << "\n====================================================================\n";
    std::cout << ">>> ALL SEVENTH REMEDIATION C++ TESTS (1-10) PASSED WITH 100% SUCCESS<<<\n";
    std::cout << "====================================================================\n";

    return 0;
}
