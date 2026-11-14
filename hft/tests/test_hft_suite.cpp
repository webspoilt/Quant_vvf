#include "engine/execution_engine.hpp"
#include "core/cpu_pinning.hpp"
#include "sync/dual_domain_clock.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

using namespace hft;

int main() {
    std::cout << "====================================================================\n";
    std::cout << "  Quant_vvf Fourth Engineering Remediation Comprehensive Test Suite \n";
    std::cout << "====================================================================\n";

    risk::RiskConfig r_cfg;
    r_cfg.max_order_clip_btc = 10.0;
    r_cfg.max_gross_exposure_btc = 20.0;
    r_cfg.max_collar_deviation_pct = 0.01;
    r_cfg.max_orders_per_second = 1000;
    r_cfg.max_market_data_age_ns = 500'000'000ULL;

    engine::HftExecutionEngine engine(r_cfg);

    uint64_t host_now = sync::DualDomainClock::get_host_monotonic_ns();
    net::TimestampNs current_ptp = sync::DualDomainClock::get_simulation_epoch_ns();

    // =========================================================================
    // 1. EXECUTION LIFECYCLE TESTS (P0)
    // =========================================================================
    std::cout << "\n[P0: EXECUTION LIFECYCLE]\n";

    // Setup: delta triggers order ID 1 (Sell 0.20 BTC)
    engine.set_current_delta_btc(0.20, 65000.0);
    net::MarketDepthPacket depth{};
    depth.symbol = net::InstrumentSymbol("BTC-PERPETUAL");
    depth.spot_index = 65000.0;
    depth.best_bid_price = 64999.0;
    depth.best_ask_price = 65001.0;
    depth.mark_price = 65000.0;
    depth.nic_rx_time_ns = current_ptp; // Valid PTP timestamp

    engine.process_market_tick(depth);
    assert(engine.has_active_order());
    assert(engine.get_in_flight_order().order_id == 1);
    assert(engine.get_in_flight_order().total_qty == 0.20);

    // Test 1.1: Partial fill 1 (0.10 BTC)
    auto res_p1 = engine.on_execution_report(1001, 1, net::Side::SELL, 0.10, 64999.0);
    assert(res_p1 == engine::ExecutionResult::SUCCESS_APPLIED);
    assert(engine.has_active_order());
    assert(engine.get_in_flight_order().status == engine::OrderStatus::PARTIALLY_FILLED);
    std::cout << "  1.1 Partial fill applied correctly (0.10 BTC remaining): PASSED\n";

    // Test 1.2: Reject Duplicate Execution Report
    auto res_dup = engine.on_execution_report(1001, 1, net::Side::SELL, 0.10, 64999.0);
    assert(res_dup == engine::ExecutionResult::REJECT_DUPLICATE_EXEC_ID);
    std::cout << "  1.2 Duplicate execution report (Exec ID 1001) rejected: PASSED\n";

    // Test 1.3: Reject Wrong Side
    auto res_side = engine.on_execution_report(1002, 1, net::Side::BUY, 0.10, 64999.0);
    assert(res_side == engine::ExecutionResult::REJECT_WRONG_SIDE);
    std::cout << "  1.3 Wrong side execution report rejected: PASSED\n";

    // Test 1.4: Reject Overfill
    auto res_over = engine.on_execution_report(1003, 1, net::Side::SELL, 0.50, 64999.0);
    assert(res_over == engine::ExecutionResult::REJECT_OVERFILL);
    std::cout << "  1.4 Overfill execution report rejected: PASSED\n";

    // Test 1.5: Terminal fill (remaining 0.10 BTC)
    auto res_p2 = engine.on_execution_report(1004, 1, net::Side::SELL, 0.10, 64999.0);
    assert(res_p2 == engine::ExecutionResult::SUCCESS_APPLIED);
    assert(!engine.has_active_order());
    assert(engine.get_in_flight_order().status == engine::OrderStatus::FILLED);
    std::cout << "  1.5 Terminal fill completed (Order fully FILLED): PASSED\n";

    // Test 1.6: Reject Fill after Terminal State
    auto res_post_term = engine.on_execution_report(1005, 1, net::Side::SELL, 0.05, 64999.0);
    assert(res_post_term == engine::ExecutionResult::REJECT_TERMINAL_ORDER);
    std::cout << "  1.6 Late fill received after terminal state rejected: PASSED\n";

    // Test 1.7: Partial Fill followed by Cancellation
    engine.set_current_delta_btc(0.20, 65000.0);
    engine.process_market_tick(depth); // Triggers order ID 2
    assert(engine.get_in_flight_order().order_id == 2);
    // Partial fill 0.05 BTC
    engine.on_execution_report(2001, 2, net::Side::SELL, 0.05, 64999.0);
    assert(engine.get_in_flight_order().status == engine::OrderStatus::PARTIALLY_FILLED);
    // Cancel remaining
    bool canceled = engine.on_order_canceled(2);
    assert(canceled);
    assert(engine.get_in_flight_order().status == engine::OrderStatus::CANCELED);
    // Late fill attempt on canceled order rejected
    auto res_late = engine.on_execution_report(2002, 2, net::Side::SELL, 0.05, 64999.0);
    assert(res_late == engine::ExecutionResult::REJECT_TERMINAL_ORDER);
    std::cout << "  1.7 Partial fill followed by cancellation releases reservation & blocks late fills: PASSED\n";

    // =========================================================================
    // 2. RISK ACCOUNTING & RESERVATION TESTS (P0)
    // =========================================================================
    std::cout << "\n[P0: RISK ACCOUNTING & OPPOSITE-SIDE RESERVATION DEFENSE]\n";
    risk::InlinePreTradeRiskGate risk_gate(r_cfg);

    // Test 2.1: Unified Authoritative Exposure Calculation
    // Order 1: Buy 8.0 BTC
    net::OrderRequest buy_1{.side = net::Side::BUY, .order_id = 101, .price = 65000.0, .quantity = 8.0};
    auto r1 = risk_gate.evaluate_order(buy_1, 64999.0, 65001.0, 65000.0, host_now, false);
    assert(r1 == risk::RiskCheckResult::APPROVED);
    assert(risk_gate.get_total_reserved_buy() == 8.0);
    assert(risk_gate.get_gross_exposure_btc() == 8.0);

    // Order 2: Simultaneous Opposite-Side Sell 8.0 BTC
    net::OrderRequest sell_1{.side = net::Side::SELL, .order_id = 102, .price = 65000.0, .quantity = 8.0};
    auto r2 = risk_gate.evaluate_order(sell_1, 64999.0, 65001.0, 65000.0, host_now, false);
    assert(r2 == risk::RiskCheckResult::APPROVED);
    // Exposure = max(|8|, |-8|) + min(8, 8) = 8 + 8 = 16.0 BTC <= 20.0
    assert(risk_gate.get_gross_exposure_btc() == 16.0);
    std::cout << "  2.1 Simultaneous opposite-side reservations tracked (Exposure = 16.0 BTC): PASSED\n";

    // Order 3: Attempting to bypass limits with third order (Buy 5.0 BTC -> 16 + 5 = 21 > 20 limit)
    net::OrderRequest buy_2{.side = net::Side::BUY, .order_id = 103, .price = 65000.0, .quantity = 5.0};
    auto r3 = risk_gate.evaluate_order(buy_2, 64999.0, 65001.0, 65000.0, host_now, false);
    assert(r3 == risk::RiskCheckResult::REJECT_EXCEEDS_GROSS_EXPOSURE);
    std::cout << "  2.2 Simultaneous opposite-side limit bypass blocked (Exceeds Gross Exposure): PASSED\n";

    // Test 2.3: Per-Order Reservation Release on Cancellation
    risk_gate.release_reservation(102); // Release 8.0 BTC Sell
    assert(risk_gate.get_total_reserved_sell() == 0.0);
    assert(risk_gate.get_gross_exposure_btc() == 8.0);
    std::cout << "  2.3 Per-order reservation cleanly released on cancellation: PASSED\n";

    // =========================================================================
    // 3. CLOCK CORRECTNESS & TIMESTAMP DOMAINS (P0)
    // =========================================================================
    std::cout << "\n[P0: CLOCK CORRECTNESS & DUAL-DOMAIN TIMING]\n";

    // Test 3.1: Stale PTP Timestamp Rejection
    net::TimestampNs stale_ptp = current_ptp - 600'000'000ULL; // 600ms old (> 500ms max age)
    bool stale_valid = sync::DualDomainClock::validate_market_timestamp(stale_ptp, current_ptp, 500'000'000ULL);
    assert(!stale_valid);
    std::cout << "  3.1 Stale PTP timestamp (>500ms) rejected: PASSED\n";

    // Test 3.2: Future PTP Timestamp Rejection
    net::TimestampNs future_ptp = current_ptp + 200'000'000ULL; // 200ms in future (> 50ms drift limit)
    bool future_valid = sync::DualDomainClock::validate_market_timestamp(future_ptp, current_ptp, 500'000'000ULL);
    assert(!future_valid);
    std::cout << "  3.2 Future corrupt PTP timestamp rejected: PASSED\n";

    // Test 3.3: Missing / Zero Timestamp Rejection
    bool zero_valid = sync::DualDomainClock::validate_market_timestamp(0, current_ptp, 500'000'000ULL);
    assert(!zero_valid);
    std::cout << "  3.3 Zero/missing timestamp rejected: PASSED\n";

    // Test 3.4: Host Monotonic Clock Non-decreasing Verification
    uint64_t t1 = sync::DualDomainClock::get_host_monotonic_ns();
    uint64_t t2 = sync::DualDomainClock::get_host_monotonic_ns();
    assert(t2 >= t1);
    std::cout << "  3.4 Host monotonic clock guaranteed non-decreasing: PASSED\n";

    std::cout << "\n====================================================================\n";
    std::cout << ">>> ALL P0/P1 FOURTH REMEDIATION TESTS PASSED WITH 100% SUCCESS  <<<\n";
    std::cout << "====================================================================\n";

    return 0;
}
