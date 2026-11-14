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
    std::cout << "  Quant_vvf Fifth Engineering Remediation Regression Test Suite     \n";
    std::cout << "====================================================================\n";

    risk::RiskConfig r_cfg;
    r_cfg.max_order_clip_btc = 5.0;
    r_cfg.max_gross_exposure_btc = 50.0;
    r_cfg.max_collar_deviation_pct = 0.05;
    r_cfg.max_orders_per_second = 10000;
    r_cfg.max_market_data_age_ns = 500'000'000ULL;

    uint64_t host_now = sync::DualDomainClock::get_host_monotonic_ns();

    // =========================================================================
    // TEST GROUP 1: RESERVATION COLLISION & LEDGER INVARIANTS (P0)
    // =========================================================================
    std::cout << "\n[TEST GROUP 1: RESERVATION COLLISION & LEDGER INVARIANTS]\n";
    risk::InlinePreTradeRiskGate risk_gate(r_cfg);

    // 1.1 Colliding Order IDs: 1, 1025, 2049 (1025 % 1024 == 1, 2049 % 1024 == 1)
    net::OrderRequest o1{.side = net::Side::BUY, .order_id = 1, .price = 65000.0, .quantity = 1.0};
    net::OrderRequest o1025{.side = net::Side::BUY, .order_id = 1025, .price = 65000.0, .quantity = 2.0};
    net::OrderRequest o2049{.side = net::Side::BUY, .order_id = 2049, .price = 65000.0, .quantity = 3.0};

    assert(risk_gate.evaluate_order(o1, 64999.0, 65001.0, 65000.0, host_now, false) == risk::RiskCheckResult::APPROVED);
    assert(risk_gate.evaluate_order(o1025, 64999.0, 65001.0, 65000.0, host_now, false) == risk::RiskCheckResult::APPROVED);
    assert(risk_gate.evaluate_order(o2049, 64999.0, 65001.0, 65000.0, host_now, false) == risk::RiskCheckResult::APPROVED);

    assert(risk_gate.get_active_reservation_count() == 3);
    assert(risk_gate.get_total_reserved_buy() == 6.0);
    assert(risk_gate.verify_invariants());
    std::cout << "  1.1 Colliding order IDs (1, 1025, 2049) safely placed without overwrite: PASSED\n";

    // 1.2 Reject Duplicate Order ID While Active
    assert(risk_gate.evaluate_order(o1, 64999.0, 65001.0, 65000.0, host_now, false) == risk::RiskCheckResult::REJECT_DUPLICATE_ORDER_ID);
    assert(risk_gate.evaluate_order(o1025, 64999.0, 65001.0, 65000.0, host_now, false) == risk::RiskCheckResult::REJECT_DUPLICATE_ORDER_ID);
    std::cout << "  1.2 Duplicate order IDs (1, 1025) rejected while active: PASSED\n";

    // 1.3 Partial Fill Decrements Only That Order's Reservation
    bool fill_ok = risk_gate.record_fill(1025, net::Side::BUY, 1.0); // 1.0 filled out of 2.0
    assert(fill_ok);
    assert(risk_gate.get_active_reservation_count() == 3);
    assert(risk_gate.get_total_reserved_buy() == 5.0); // 6.0 - 1.0 = 5.0
    assert(risk_gate.verify_invariants());
    std::cout << "  1.3 Partial fill decrements exactly that order's reservation: PASSED\n";

    // 1.4 Overfill Rejection on Order 1025 (Attempting to fill 2.0 when only 1.0 remains)
    bool overfill_ok = risk_gate.record_fill(1025, net::Side::BUY, 2.0);
    assert(!overfill_ok);
    assert(risk_gate.get_total_reserved_buy() == 5.0);
    assert(risk_gate.verify_invariants());
    std::cout << "  1.4 Overfill rejected by risk ledger: PASSED\n";

    // 1.5 Cancellation Releases Exactly the Remaining Reservation
    bool cancel_ok = risk_gate.release_reservation(1025);
    assert(cancel_ok);
    assert(risk_gate.get_active_reservation_count() == 2);
    assert(risk_gate.get_total_reserved_buy() == 4.0); // 5.0 - 1.0 remaining = 4.0
    assert(risk_gate.verify_invariants());
    std::cout << "  1.5 Cancellation releases remaining reservation exactly once: PASSED\n";

    // Clean up remaining
    risk_gate.release_reservation(1);
    risk_gate.release_reservation(2049);
    assert(risk_gate.get_active_reservation_count() == 0);
    assert(risk_gate.get_total_reserved_buy() == 0.0);
    assert(risk_gate.verify_invariants());

    // 1.6 Reservation Table Exhaustion
    risk::RiskConfig r_cfg_huge = r_cfg;
    r_cfg_huge.max_gross_exposure_btc = 100000.0;
    r_cfg_huge.max_orders_per_second = 100000;
    risk::InlinePreTradeRiskGate full_gate(r_cfg_huge);

    for (size_t i = 0; i < risk::InlinePreTradeRiskGate::MAX_TRACKED_ORDERS; ++i) {
        net::OrderRequest req{.side = net::Side::BUY, .order_id = i + 1, .price = 65000.0, .quantity = 0.01};
        assert(full_gate.evaluate_order(req, 64999.0, 65001.0, 65000.0, host_now, false) == risk::RiskCheckResult::APPROVED);
    }
    assert(full_gate.get_active_reservation_count() == risk::InlinePreTradeRiskGate::MAX_TRACKED_ORDERS);
    assert(full_gate.verify_invariants());

    // 1025th order must be rejected with REJECT_RESERVATION_FULL
    net::OrderRequest req_overflow{.side = net::Side::BUY, .order_id = 99999, .price = 65000.0, .quantity = 0.01};
    assert(full_gate.evaluate_order(req_overflow, 64999.0, 65001.0, 65000.0, host_now, false) == risk::RiskCheckResult::REJECT_RESERVATION_FULL);
    std::cout << "  1.6 Reservation capacity exhaustion (1024 slots) returns REJECT_RESERVATION_FULL: PASSED\n";

    // =========================================================================
    // TEST GROUP 2: BOUNDED EXECUTION DEDUPLICATION & TERMINAL IMMUTABILITY (P0)
    // =========================================================================
    std::cout << "\n[TEST GROUP 2: EXECUTION DEDUPLICATION & TERMINAL IMMUTABILITY]\n";
    engine::HftExecutionEngine engine(r_cfg);

    // 2.1 Deduplication of over 1024 unique execution reports
    for (uint64_t id = 1; id <= 1024; ++id) {
        assert(engine.is_duplicate_exec_id(id) == false);
        engine.commit_exec_id(id);
        assert(engine.is_duplicate_exec_id(id) == true);
    }
    // Execution ID 1025 inserted into full FIFO window (evicts oldest ID 1)
    assert(engine.is_duplicate_exec_id(1025) == false);
    engine.commit_exec_id(1025);
    // Duplicate of 1025 must be rejected
    assert(engine.is_duplicate_exec_id(1025) == true);
    // Duplicate of recent active ID 500 must be rejected
    assert(engine.is_duplicate_exec_id(500) == true);
    std::cout << "  2.1 Bounded execution deduplication (>1024 IDs, evictions & duplicate rejections): PASSED\n";

    // 2.2 Terminal Immutability: FILLED order followed by Rejection
    engine.set_current_delta_btc(0.20, 65000.0);
    net::MarketDepthPacket depth{};
    depth.symbol = net::InstrumentSymbol("BTC-PERPETUAL");
    depth.spot_index = 65000.0;
    depth.best_bid_price = 64999.0;
    depth.best_ask_price = 65001.0;
    depth.mark_price = 65000.0;
    depth.nic_rx_time_ns = sync::DualDomainClock::get_simulation_epoch_ns();

    engine.process_market_tick(depth); // Emits order ID 1
    assert(engine.has_active_order());
    assert(engine.get_in_flight_order().order_id == 1);

    // Terminal fill
    auto r_fill = engine.on_execution_report(50001, 1, net::Side::SELL, 0.20, 64999.0);
    assert(r_fill == engine::ExecutionResult::SUCCESS_APPLIED);
    assert(engine.get_in_flight_order().status == engine::OrderStatus::FILLED);

    // Late rejection attempt: must NOT overwrite FILLED state
    bool rej_ok = engine.on_order_rejected(1);
    assert(!rej_ok);
    assert(engine.get_in_flight_order().status == engine::OrderStatus::FILLED);
    std::cout << "  2.2 Terminal state FILLED immutable against late order rejection: PASSED\n";

    // 2.3 Terminal Immutability: CANCELED order followed by Rejection or Fill
    engine.set_current_delta_btc(0.20, 65000.0);
    engine.process_market_tick(depth); // Emits order ID 2
    assert(engine.get_in_flight_order().order_id == 2);

    bool c_ok = engine.on_order_canceled(2);
    assert(c_ok);
    assert(engine.get_in_flight_order().status == engine::OrderStatus::CANCELED);

    // Rejection on canceled order fails
    assert(!engine.on_order_rejected(2));
    assert(engine.get_in_flight_order().status == engine::OrderStatus::CANCELED);

    // Late fill on canceled order rejected
    auto r_late = engine.on_execution_report(50002, 2, net::Side::SELL, 0.20, 64999.0);
    assert(r_late == engine::ExecutionResult::REJECT_TERMINAL_ORDER);
    std::cout << "  2.3 Terminal state CANCELED immutable against late rejections and late fills: PASSED\n";

    // =========================================================================
    // TEST GROUP 3: CLOCK ARCHITECTURE & COMPATIBILITY (P0)
    // =========================================================================
    std::cout << "\n[TEST GROUP 3: CLOCK ARCHITECTURE & COMPATIBILITY]\n";

    // 3.1 Domain Incompatibility (Hardware PTP vs Simulation Epoch are strictly incompatible)
    assert(sync::DualDomainClock::check_domain_compatibility(sync::ClockDomain::HOST_MONOTONIC, sync::ClockDomain::HOST_MONOTONIC) == true);
    assert(sync::DualDomainClock::check_domain_compatibility(sync::ClockDomain::PTP_HARDWARE_EPOCH, sync::ClockDomain::PTP_HARDWARE_EPOCH) == true);
    assert(sync::DualDomainClock::check_domain_compatibility(sync::ClockDomain::PTP_HARDWARE_EPOCH, sync::ClockDomain::SIMULATION_EPOCH) == false);
    assert(sync::DualDomainClock::check_domain_compatibility(sync::ClockDomain::HOST_MONOTONIC, sync::ClockDomain::PTP_HARDWARE_EPOCH) == false);
    std::cout << "  3.1 Clock domain compatibility enforcement: PASSED\n";

    // 3.2 Monotonicity
    uint64_t t_mono1 = sync::DualDomainClock::get_host_monotonic_ns();
    uint64_t t_mono2 = sync::DualDomainClock::get_host_monotonic_ns();
    assert(t_mono2 >= t_mono1);
    std::cout << "  3.2 Host monotonic non-decreasing clock verified: PASSED\n";

    std::cout << "\n====================================================================\n";
    std::cout << ">>> ALL FIFTH REMEDIATION TESTS COMPLETED SUCCESSFULLY          <<<\n";
    std::cout << "====================================================================\n";

    return 0;
}
