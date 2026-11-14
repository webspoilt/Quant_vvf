#pragma once

#include "network/types.hpp"
#include "network/kernel_bypass.hpp"
#include "core/lock_free_queue.hpp"
#include "risk/pre_trade_risk.hpp"
#include "sync/dual_domain_clock.hpp"
#include "math/inverse_math.hpp"
#include "engine/execution_journal.hpp"

#include <cstdint>
#include <cstring>
#include <cmath>
#include <memory>
#include <string>
#include <iostream>

namespace hft::engine {

enum class OrderStatus : uint8_t {
    NONE = 0,
    PENDING_NEW = 1,
    ACKNOWLEDGED = 2,
    PARTIALLY_FILLED = 3,
    FILLED = 4,
    CANCELED = 5,
    REJECTED = 6
};

enum class ExecutionResult : uint8_t {
    SUCCESS_APPLIED = 0,
    REJECT_RESERVED_EXEC_ID = 1,      // Rejects ID 0 or UINT64_MAX
    REJECT_DUPLICATE_EXEC_ID = 2,
    REJECT_TERMINAL_ORDER = 3,
    REJECT_MISMATCHED_ORDER_ID = 4,
    REJECT_WRONG_SIDE = 5,
    REJECT_INVALID_QUANTITY = 6,
    REJECT_INVALID_PRICE = 7,         // Rejects NaN, Inf, zero, and negative prices
    REJECT_OVERFILL = 8,
    REJECT_RISK_BREACH = 9,
    REJECT_WAL_IO_ERROR = 10,         // WAL file unopen, write error, or flush error
    REJECT_WAL_DURABILITY_BREACH = 11 // fsync / _commit failure under durable policy
};

struct InFlightOrder {
    uint64_t order_id{0};
    net::Side side{net::Side::BUY};
    double total_qty{0.0};
    double cumulative_filled_qty{0.0};
    double price{0.0};
    OrderStatus status{OrderStatus::NONE};

    [[nodiscard]] double remaining_qty() const noexcept {
        return total_qty > cumulative_filled_qty ? (total_qty - cumulative_filled_qty) : 0.0;
    }
};

/**
 * High-Frequency Order Lifecycle & Execution Manager
 * 
 * Strict Transactional Execution Flow:
 * 1. Validate report fields (Execution ID != 0/MAX, finite positive price & qty).
 * 2. Check execution ID duplication in tombstone-safe hash table.
 * 3. Validate order identity and active state (must match in-flight, not terminal).
 * 4. Validate side and quantity constraints.
 * 5. Validate overfill constraints against cumulative filled quantity.
 * 6. Calculate prospective position & risk ledger limits.
 * 7. Verify and tentatively record risk reservation fill.
 * 8. Construct structured WAL journal event.
 * 9. Append event to write-ahead log.
 * 10. Check file write result.
 * 11. Check userspace flush result.
 * 12. Check durable synchronization result (_commit/fsync) under active policy.
 * 13. Only upon confirmed write/durability success:
 *     - Commit execution ID to deduplication ring and hash table.
 *     - Mutate in-flight order cumulative filled quantity & status.
 *     - Mutate portfolio delta and position tracking.
 * 14. If WAL append fails:
 *     - Roll back tentatively applied risk reservation.
 *     - Do NOT commit execution ID (allowing safe retry).
 *     - Return explicit REJECT_WAL_IO_ERROR or REJECT_WAL_DURABILITY_BREACH.
 */
class HftExecutionEngine {
public:
    static constexpr size_t RING_CAPACITY = 65536;
    static constexpr size_t EXEC_DEDUP_CAPACITY = 2048;
    static constexpr size_t EXEC_FIFO_CAPACITY = 1024;
    static constexpr uint64_t EMPTY_SLOT = 0ULL;
    static constexpr uint64_t TOMBSTONE = UINT64_MAX;

    explicit HftExecutionEngine(const risk::RiskConfig& risk_cfg)
        : risk_gate_(risk_cfg), current_spot_(0.0), current_delta_btc_(0.0),
          last_processed_spot_(0.0), next_order_id_(0), fifo_head_(0), fifo_count_(0),
          reconciliation_required_(false) {
        for (size_t i = 0; i < EXEC_DEDUP_CAPACITY; ++i) {
            dedup_table_[i] = EMPTY_SLOT;
        }
        for (size_t i = 0; i < EXEC_FIFO_CAPACITY; ++i) {
            fifo_ring_[i] = 0;
        }
    }

    [[nodiscard]] inline bool is_duplicate_exec_id(uint64_t exec_id) const noexcept {
        if (exec_id == EMPTY_SLOT || exec_id == TOMBSTONE) return false;

        size_t initial_slot = exec_id % EXEC_DEDUP_CAPACITY;
        for (size_t i = 0; i < EXEC_DEDUP_CAPACITY; ++i) {
            size_t slot = (initial_slot + i) % EXEC_DEDUP_CAPACITY;
            uint64_t val = dedup_table_[slot];
            if (val == exec_id) {
                return true;
            }
            if (val == EMPTY_SLOT) {
                return false;
            }
        }
        return false;
    }

    inline void commit_exec_id(uint64_t exec_id) noexcept {
        if (exec_id == EMPTY_SLOT || exec_id == TOMBSTONE) return;

        if (fifo_count_ >= EXEC_FIFO_CAPACITY) {
            uint64_t oldest_id = fifo_ring_[fifo_head_];
            if (oldest_id != EMPTY_SLOT && oldest_id != TOMBSTONE) {
                size_t old_slot = oldest_id % EXEC_DEDUP_CAPACITY;
                for (size_t i = 0; i < EXEC_DEDUP_CAPACITY; ++i) {
                    size_t slot = (old_slot + i) % EXEC_DEDUP_CAPACITY;
                    if (dedup_table_[slot] == oldest_id) {
                        dedup_table_[slot] = TOMBSTONE;
                        break;
                    }
                    if (dedup_table_[slot] == EMPTY_SLOT) {
                        break;
                    }
                }
            }
            fifo_ring_[fifo_head_] = exec_id;
            fifo_head_ = (fifo_head_ + 1) % EXEC_FIFO_CAPACITY;
        } else {
            fifo_ring_[(fifo_head_ + fifo_count_) % EXEC_FIFO_CAPACITY] = exec_id;
            fifo_count_++;
        }

        size_t initial_slot = exec_id % EXEC_DEDUP_CAPACITY;
        size_t target_slot = EXEC_DEDUP_CAPACITY;

        for (size_t i = 0; i < EXEC_DEDUP_CAPACITY; ++i) {
            size_t slot = (initial_slot + i) % EXEC_DEDUP_CAPACITY;
            if (dedup_table_[slot] == exec_id) {
                return;
            }
            if ((dedup_table_[slot] == EMPTY_SLOT || dedup_table_[slot] == TOMBSTONE) && target_slot == EXEC_DEDUP_CAPACITY) {
                target_slot = slot;
            }
            if (dedup_table_[slot] == EMPTY_SLOT) {
                break;
            }
        }

        if (target_slot != EXEC_DEDUP_CAPACITY) {
            dedup_table_[target_slot] = exec_id;
        }
    }

    inline void process_market_tick(const net::MarketDepthPacket& depth) noexcept {
        uint64_t host_now_ns = sync::DualDomainClock::get_host_monotonic_ns();
        net::TimestampNs current_epoch_ns = sync::DualDomainClock::get_simulation_epoch_ns();

        bool timestamp_valid = sync::DualDomainClock::validate_market_timestamp(
            depth.nic_rx_time_ns, current_epoch_ns, 500'000'000ULL
        );
        if (!timestamp_valid) [[unlikely]] return;

        if (!risk_gate_.check_and_update_variance(depth.spot_index)) [[unlikely]] return;

        if (last_processed_spot_ > 0.0 && depth.spot_index > 0.0) {
            double rescale = math::InverseDerivativesMath::inverse_delta_rescale(last_processed_spot_, depth.spot_index);
            current_delta_btc_ *= rescale;
        }
        last_processed_spot_ = depth.spot_index;
        current_spot_ = depth.spot_index;

        if (std::abs(current_delta_btc_) >= 0.15 && !has_active_order() && !reconciliation_required_) {
            net::OrderRequest rebalance_order;
            rebalance_order.msg_type = net::MessageType::ORDER_SUBMIT;
            rebalance_order.side = current_delta_btc_ > 0 ? net::Side::SELL : net::Side::BUY;
            rebalance_order.symbol = net::InstrumentSymbol("BTC-PERPETUAL");
            rebalance_order.price = (rebalance_order.side == net::Side::SELL) ? depth.best_bid_price : depth.best_ask_price;
            rebalance_order.quantity = std::abs(current_delta_btc_);
            rebalance_order.order_id = ++next_order_id_;
            rebalance_order.client_sent_ns = current_epoch_ns;

            auto risk_res = risk_gate_.evaluate_order(
                rebalance_order,
                depth.best_bid_price,
                depth.best_ask_price,
                depth.mark_price,
                host_now_ns,
                false
            );

            if (risk_res == risk::RiskCheckResult::APPROVED) [[likely]] {
                in_flight_.order_id = rebalance_order.order_id;
                in_flight_.side = rebalance_order.side;
                in_flight_.total_qty = rebalance_order.quantity;
                in_flight_.cumulative_filled_qty = 0.0;
                in_flight_.price = rebalance_order.price;
                in_flight_.status = OrderStatus::PENDING_NEW;

                if (journal_) {
                    (void)journal_->append_record(
                        JournalRecordType::ORDER_SUBMITTED,
                        0,
                        rebalance_order.order_id,
                        "BTC-PERPETUAL",
                        static_cast<uint8_t>(rebalance_order.side),
                        rebalance_order.quantity,
                        rebalance_order.price,
                        host_now_ns,
                        0
                    );
                }
            }
        }
    }

    /**
     * Fully Validated Atomic 2-Phase Execution Report Handler
     * Transaction Flow with Rollback on WAL Append Failure
     */
    inline ExecutionResult on_execution_report(
        uint64_t exec_id,
        uint64_t order_id,
        net::Side side,
        double filled_qty,
        double fill_price,
        uint64_t exch_seq = 0
    ) noexcept {
        // Halt check: If un-reconciled state exists, refuse execution
        if (reconciliation_required_) [[unlikely]] {
            return ExecutionResult::REJECT_RISK_BREACH;
        }

        // 1. Reserved Execution ID Rejection (0 or UINT64_MAX)
        if (exec_id == EMPTY_SLOT || exec_id == TOMBSTONE) [[unlikely]] {
            return ExecutionResult::REJECT_RESERVED_EXEC_ID;
        }

        // 2. Fill Price Validation (NaN, Inf, <= 0)
        if (!std::isfinite(fill_price) || fill_price <= 0.0) [[unlikely]] {
            return ExecutionResult::REJECT_INVALID_PRICE;
        }

        // 3. Fill Quantity Validation
        if (!std::isfinite(filled_qty) || filled_qty <= 0.0) [[unlikely]] {
            return ExecutionResult::REJECT_INVALID_QUANTITY;
        }

        // 4. Duplicate ID probe
        if (is_duplicate_exec_id(exec_id)) [[unlikely]] {
            return ExecutionResult::REJECT_DUPLICATE_EXEC_ID;
        }

        // 5. Order ID match
        if (in_flight_.order_id != order_id) [[unlikely]] {
            return ExecutionResult::REJECT_MISMATCHED_ORDER_ID;
        }

        // 6. Terminal order check
        if (in_flight_.status == OrderStatus::FILLED ||
            in_flight_.status == OrderStatus::CANCELED ||
            in_flight_.status == OrderStatus::REJECTED) [[unlikely]] {
            return ExecutionResult::REJECT_TERMINAL_ORDER;
        }

        // 7. Side match
        if (in_flight_.side != side) [[unlikely]] {
            return ExecutionResult::REJECT_WRONG_SIDE;
        }

        // 8. Overfill check against order total quantity
        if ((in_flight_.cumulative_filled_qty + filled_qty) > (in_flight_.total_qty + 1e-9)) [[unlikely]] {
            return ExecutionResult::REJECT_OVERFILL;
        }

        // 9. Risk ledger reservation fill validation (phase 1: tentative reservation update)
        bool risk_updated = risk_gate_.record_fill(order_id, side, filled_qty);
        if (!risk_updated) [[unlikely]] {
            return ExecutionResult::REJECT_RISK_BREACH;
        }

        // 10. WAL Persistence Check (Phase 2: Durable Logging)
        if (journal_) {
            uint64_t now_ns = sync::DualDomainClock::get_host_monotonic_ns();
            WalStatus wal_res = journal_->append_record(
                JournalRecordType::ORDER_FILLED,
                exec_id,
                order_id,
                "BTC-PERPETUAL",
                static_cast<uint8_t>(side),
                filled_qty,
                fill_price,
                now_ns,
                exch_seq
            );

            if (wal_res != WalStatus::SUCCESS) [[unlikely]] {
                // TRANSACTION ROLLBACK:
                // Revert risk reservation & position mutation
                // Reverse the fill applied in step 9
                if (side == net::Side::BUY) {
                    risk_gate_.adjust_position_for_recovery(-filled_qty);
                } else {
                    risk_gate_.adjust_position_for_recovery(filled_qty);
                }
                // Do NOT commit execution ID; do NOT update in_flight status
                if (wal_res == WalStatus::FSYNC_FAILED) {
                    return ExecutionResult::REJECT_WAL_DURABILITY_BREACH;
                }
                return ExecutionResult::REJECT_WAL_IO_ERROR;
            }
        }

        // 11. COMMIT PHASE: All validations & durable logging succeeded
        // Commit execution ID to deduplication ring
        commit_exec_id(exec_id);

        in_flight_.cumulative_filled_qty += filled_qty;

        if (side == net::Side::SELL) {
            current_delta_btc_ -= filled_qty;
        } else {
            current_delta_btc_ += filled_qty;
        }

        if (in_flight_.cumulative_filled_qty >= (in_flight_.total_qty - 1e-9)) {
            in_flight_.status = OrderStatus::FILLED;
        } else {
            in_flight_.status = OrderStatus::PARTIALLY_FILLED;
        }

        return ExecutionResult::SUCCESS_APPLIED;
    }

    inline bool on_order_canceled(uint64_t order_id) noexcept {
        if (in_flight_.order_id == order_id) {
            if (in_flight_.status == OrderStatus::FILLED ||
                in_flight_.status == OrderStatus::CANCELED ||
                in_flight_.status == OrderStatus::REJECTED) {
                return false;
            }
            risk_gate_.release_reservation(order_id);
            in_flight_.status = OrderStatus::CANCELED;

            if (journal_) {
                (void)journal_->append_record(
                    JournalRecordType::ORDER_CANCELLED,
                    0,
                    order_id,
                    "BTC-PERPETUAL",
                    static_cast<uint8_t>(in_flight_.side),
                    in_flight_.remaining_qty(),
                    in_flight_.price,
                    sync::DualDomainClock::get_host_monotonic_ns(),
                    0
                );
            }
            return true;
        }
        return false;
    }

    inline bool on_order_rejected(uint64_t order_id) noexcept {
        if (in_flight_.order_id == order_id) {
            if (in_flight_.status == OrderStatus::FILLED ||
                in_flight_.status == OrderStatus::CANCELED ||
                in_flight_.status == OrderStatus::REJECTED) {
                return false;
            }
            risk_gate_.release_reservation(order_id);
            in_flight_.status = OrderStatus::REJECTED;

            if (journal_) {
                (void)journal_->append_record(
                    JournalRecordType::ORDER_REJECTED,
                    0,
                    order_id,
                    "BTC-PERPETUAL",
                    static_cast<uint8_t>(in_flight_.side),
                    in_flight_.remaining_qty(),
                    in_flight_.price,
                    sync::DualDomainClock::get_host_monotonic_ns(),
                    0
                );
            }
            return true;
        }
        return false;
    }

    void set_current_delta_btc(double delta, double spot) noexcept {
        current_delta_btc_ = delta;
        last_processed_spot_ = spot;
    }

    [[nodiscard]] double get_current_delta_btc() const noexcept { return current_delta_btc_; }
    [[nodiscard]] bool has_active_order() const noexcept {
        return in_flight_.status == OrderStatus::PENDING_NEW ||
               in_flight_.status == OrderStatus::ACKNOWLEDGED ||
               in_flight_.status == OrderStatus::PARTIALLY_FILLED;
    }

    [[nodiscard]] const InFlightOrder& get_in_flight_order() const noexcept { return in_flight_; }
    [[nodiscard]] double get_position_btc() const noexcept { return risk_gate_.get_net_position_btc(); }
    [[nodiscard]] double get_gross_exposure_btc() const noexcept { return risk_gate_.get_gross_exposure_btc(); }
    [[nodiscard]] bool is_reconciliation_required() const noexcept { return reconciliation_required_; }
    risk::InlinePreTradeRiskGate& get_risk_gate() noexcept { return risk_gate_; }
    DurableExecutionJournal* get_journal() noexcept { return journal_.get(); }

    /**
     * Initializes journal with durability policy and executes complete state crash recovery.
     * Reconstructs deduplication cache, filled orders, positions, and detects corruption.
     */
    bool enable_journal(
        const std::string& path,
        WalDurabilityPolicy policy = WalDurabilityPolicy::SYNCHRONOUSLY_DURABLE
    ) {
        journal_ = std::make_unique<DurableExecutionJournal>(path, policy);
        if (!journal_->is_open()) {
            reconciliation_required_ = true;
            return false;
        }

        // Full State Crash Recovery Scan
        auto recovery = journal_->scan_and_recover();
        if (!recovery.is_clean) {
            // Log corruption/torn record warning and require exchange reconciliation
            std::cerr << "[WAL RECOVERY] Warning: Journal non-clean (corrupted records: "
                      << recovery.corrupted_records_count << ", torn tail: "
                      << recovery.torn_tail_detected << "). Reconciliation required.\n";
            reconciliation_required_ = true;
        }

        // Replay valid committed records into state machine
        for (const auto& rec : recovery.valid_records) {
            auto rtype = static_cast<JournalRecordType>(rec.record_type);
            if (rtype == JournalRecordType::ORDER_FILLED) {
                commit_exec_id(rec.exec_id);
                // Accumulate recovered position
                if (rec.side == 0) { // BUY
                    risk_gate_.adjust_position_for_recovery(rec.filled_qty);
                } else { // SELL
                    risk_gate_.adjust_position_for_recovery(-rec.filled_qty);
                }
            } else if (rtype == JournalRecordType::ORDER_SUBMITTED) {
                if (rec.order_id > next_order_id_) {
                    next_order_id_ = rec.order_id;
                }
            }
        }

        return recovery.is_clean;
    }

    /**
     * Authoritative Exchange Snapshot Reconciliation Gateway
     * Verifies recovered local state against exchange authoritative account positions.
     */
    bool reconcile_with_exchange(
        double authoritative_position_btc,
        uint64_t authoritative_next_order_id,
        double tolerance_btc = 1e-6
    ) noexcept {
        double current_pos = get_position_btc();
        if (std::abs(current_pos - authoritative_position_btc) > tolerance_btc) {
            // Discrepancy beyond tolerance: remain halted
            reconciliation_required_ = true;
            return false;
        }

        if (authoritative_next_order_id > next_order_id_) {
            next_order_id_ = authoritative_next_order_id;
        }

        reconciliation_required_ = false;
        return true;
    }

private:
    risk::InlinePreTradeRiskGate risk_gate_;
    std::unique_ptr<DurableExecutionJournal> journal_{nullptr};
    double current_spot_{0.0};
    double current_delta_btc_{0.0};
    double last_processed_spot_{0.0};
    uint64_t next_order_id_{0};
    InFlightOrder in_flight_{};
    uint64_t dedup_table_[EXEC_DEDUP_CAPACITY];
    uint64_t fifo_ring_[EXEC_FIFO_CAPACITY];
    size_t fifo_head_{0};
    size_t fifo_count_{0};
    bool reconciliation_required_{false};
};

} // namespace hft::engine
