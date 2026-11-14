#include "engine/execution_engine.hpp"
#include "core/cpu_pinning.hpp"
#include "sync/dual_domain_clock.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace hft;

int main() {
    std::cout << "====================================================================\n";
    std::cout << "  Quant_vvf Sixth Engineering Remediation Regression Test Suite     \n";
    std::cout << "====================================================================\n";

    risk::RiskConfig r_cfg;
    r_cfg.max_order_clip_btc = 5.0;
    r_cfg.max_gross_exposure_btc = 50.0;
    r_cfg.max_collar_deviation_pct = 0.05;
    r_cfg.max_orders_per_second = 10000;
    r_cfg.max_market_data_age_ns = 500'000'000ULL;

    uint64_t host_now = sync::DualDomainClock::get_host_monotonic_ns();

    // =========================================================================
    // 1. CRITICAL DEFECT REPRODUCTION & VERIFICATION: TOMBSTONE-SAFE DEDUP
    // =========================================================================
    std::cout << "\n[TEST 1: TOMBSTONE-SAFE EXECUTION DEDUPLICATION]\n";
    engine::HftExecutionEngine engine(r_cfg);

    // Reproduce Audit Scenario:
    // 1. Insert colliding IDs: 1, 2049, 4097 (1 % 2048 == 1, 2049 % 2048 == 1, 4097 % 2048 == 1)
    engine.commit_exec_id(1);
    engine.commit_exec_id(2049);
    engine.commit_exec_id(4097);

    assert(engine.is_duplicate_exec_id(1) == true);
    assert(engine.is_duplicate_exec_id(2049) == true);
    assert(engine.is_duplicate_exec_id(4097) == true);
    std::cout << "  1.1 Colliding execution IDs (1, 2049, 4097) inserted & detected as duplicates: PASSED\n";

    // 2. Insert exactly 1022 unique execution IDs so FIFO reaches capacity and evicts ID 1
    // (Total items in FIFO was 3: [1, 2049, 4097]. 3 + 1022 = 1025 insertions.
    // Insertion 1025 evicts FIFO[0] which is ID 1! IDs 2049 and 4097 remain active in FIFO!)
    for (uint64_t id = 10000; id < 10000 + 1022; ++id) {
        engine.commit_exec_id(id);
    }

    // ID 1 was evicted and marked with TOMBSTONE (UINT64_MAX)
    // 3. Submit 2049 again: MUST PROBE THROUGH TOMBSTONE AND BE REJECTED!
    bool dup_2049 = engine.is_duplicate_exec_id(2049);
    assert(dup_2049 == true); // The audit defect is eliminated!
    std::cout << "  1.2 Duplicate ID 2049 after eviction of colliding ID 1: REJECTED (Tombstone probe verified): PASSED\n";

    // 4. Submit 4097 again: MUST ALSO PROBE THROUGH AND BE REJECTED!
    bool dup_4097 = engine.is_duplicate_exec_id(4097);
    assert(dup_4097 == true);
    std::cout << "  1.3 Duplicate ID 4097 after eviction: REJECTED (Chaining intact): PASSED\n";

    // =========================================================================
    // 2. ATOMIC PRE-VALIDATION BEFORE ID CONSUMPTION
    // =========================================================================
    std::cout << "\n[TEST 2: ATOMIC PRE-VALIDATION BEFORE ID CONSUMPTION]\n";
    engine::HftExecutionEngine eng2(r_cfg);

    eng2.set_current_delta_btc(0.20, 65000.0);
    net::MarketDepthPacket depth{};
    depth.symbol = net::InstrumentSymbol("BTC-PERPETUAL");
    depth.spot_index = 65000.0;
    depth.best_bid_price = 64999.0;
    depth.best_ask_price = 65001.0;
    depth.mark_price = 65000.0;
    depth.nic_rx_time_ns = sync::DualDomainClock::get_simulation_epoch_ns();

    eng2.process_market_tick(depth); // Emits order ID 1
    assert(eng2.has_active_order());

    // Scenario: An execution report arrives with invalid order ID (Order 999 instead of 1)
    // using execution ID 8888.
    auto r_bad = eng2.on_execution_report(8888, 999, net::Side::SELL, 0.10, 64999.0);
    assert(r_bad == engine::ExecutionResult::REJECT_MISMATCHED_ORDER_ID);

    // CRITICAL CHECK: Failed report must NOT poison execution ID 8888!
    assert(eng2.is_duplicate_exec_id(8888) == false);
    std::cout << "  2.1 Failed execution report does NOT poison execution ID history: PASSED\n";

    // Now submit the valid report with execution ID 8888: MUST SUCCEED!
    auto r_good = eng2.on_execution_report(8888, 1, net::Side::SELL, 0.10, 64999.0);
    assert(r_good == engine::ExecutionResult::SUCCESS_APPLIED);
    assert(eng2.is_duplicate_exec_id(8888) == true); // Now committed!
    std::cout << "  2.2 Valid report with ID 8888 succeeds and commits atomically: PASSED\n";

    // Duplicate submission of 8888 must now be rejected
    auto r_dup = eng2.on_execution_report(8888, 1, net::Side::SELL, 0.10, 64999.0);
    assert(r_dup == engine::ExecutionResult::REJECT_DUPLICATE_EXEC_ID);
    std::cout << "  2.3 Repeated execution ID 8888 rejected: PASSED\n";

    // =========================================================================
    // 3. STRICT CLOCK DOMAIN INCOMPATIBILITY
    // =========================================================================
    std::cout << "\n[TEST 3: STRICT CLOCK DOMAIN COMPATIBILITY]\n";
    assert(sync::DualDomainClock::check_domain_compatibility(sync::ClockDomain::PTP_HARDWARE_EPOCH, sync::ClockDomain::PTP_HARDWARE_EPOCH) == true);
    assert(sync::DualDomainClock::check_domain_compatibility(sync::ClockDomain::HOST_MONOTONIC, sync::ClockDomain::HOST_MONOTONIC) == true);
    assert(sync::DualDomainClock::check_domain_compatibility(sync::ClockDomain::SIMULATION_EPOCH, sync::ClockDomain::SIMULATION_EPOCH) == true);

    // PTP Hardware Epoch and Simulation Epoch are strictly INCOMPATIBLE without explicit conversion
    assert(sync::DualDomainClock::check_domain_compatibility(sync::ClockDomain::PTP_HARDWARE_EPOCH, sync::ClockDomain::SIMULATION_EPOCH) == false);
    assert(sync::DualDomainClock::check_domain_compatibility(sync::ClockDomain::HOST_MONOTONIC, sync::ClockDomain::SIMULATION_EPOCH) == false);
    std::cout << "  3.1 Cross-domain compatibility strictly false (Hardware PTP vs Sim Epoch): PASSED\n";

    std::cout << "\n====================================================================\n";
    std::cout << ">>> ALL SIXTH REMEDIATION C++ TESTS PASSED WITH 100% SUCCESS    <<<\n";
    std::cout << "====================================================================\n";

    return 0;
}
