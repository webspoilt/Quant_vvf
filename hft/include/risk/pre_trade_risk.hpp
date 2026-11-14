#pragma once

#include "network/types.hpp"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace hft::risk {

enum class RiskCheckResult : uint8_t {
    APPROVED = 0,
    REJECT_INVALID_NUMERICAL = 1,
    REJECT_EXCEEDS_MAX_CLIP = 2,
    REJECT_EXCEEDS_GROSS_EXPOSURE = 3,
    REJECT_PRICE_COLLAR_VIOLATION = 4,
    REJECT_RATE_LIMIT_EXCEEDED = 5,
    REJECT_CIRCUIT_BREAKER_ACTIVE = 6,
    REJECT_STALE_MARKET_DATA = 7,
    REJECT_DUPLICATE_ORDER_ID = 8,
    REJECT_RESERVATION_FULL = 9,
    REJECT_UNKNOWN_ORDER = 10
};

struct OrderReservation {
    uint64_t order_id{0};
    net::Side side{net::Side::BUY};
    double requested_qty{0.0};
    double remaining_reserved_qty{0.0};
    double executed_qty{0.0};
    bool is_active{false};
};

struct RiskConfig {
    double max_order_clip_btc{10.0};
    double max_gross_exposure_btc{100.0};
    double max_collar_deviation_pct{0.02}; // 2%
    uint32_t max_orders_per_second{2000};
    double local_variance_limit{0.0025};   // r^2 > 0.0025
    uint64_t max_market_data_age_ns{500'000'000ULL}; // 500ms
};

/**
 * Deterministic Fail-Closed Pre-Trade Risk Gate
 * 
 * Remediation Features (Critical Defect Elimination):
 * 1. Open Addressing with Linear Probing: Prevents slot collision overwrites.
 *    Order IDs 1, 1025, 2049 map cleanly to independent slots.
 * 2. Duplicate Order ID Rejection: Scans table to ensure no active reservation shares the same order ID.
 * 3. Exact Reservation Exhaustion: Returns REJECT_RESERVATION_FULL if all slots are occupied.
 * 4. Per-Order Fill Validation: Rejects fills exceeding remaining reservation, non-finite, zero, or negative fills.
 * 5. Aggregate Invariant Guarantee: Total reserved BUY and SELL quantities strictly match the sum of active slots.
 * 6. Unified Authoritative Exposure: Same formula across evaluation and reporting.
 */
class InlinePreTradeRiskGate {
public:
    static constexpr size_t MAX_TRACKED_ORDERS = 1024;

    explicit InlinePreTradeRiskGate(const RiskConfig& config) noexcept
        : config_(config), current_net_position_btc_(0.0), total_reserved_buy_btc_(0.0),
          total_reserved_sell_btc_(0.0), active_reservation_count_(0),
          circuit_breaker_tripped_(false), last_order_sec_(0), orders_in_current_sec_(0),
          last_spot_price_(0.0) {
        for (size_t i = 0; i < MAX_TRACKED_ORDERS; ++i) {
            reservations_[i] = OrderReservation{};
        }
    }

    inline bool check_and_update_variance(double current_spot) noexcept {
        if (circuit_breaker_tripped_.load(std::memory_order_relaxed)) [[unlikely]] return false;
        if (!std::isfinite(current_spot) || current_spot <= 0.0) [[unlikely]] {
            trip_circuit_breaker();
            return false;
        }

        if (last_spot_price_ > 0.0) {
            double log_ret = std::log(current_spot / last_spot_price_);
            double tick_var = log_ret * log_ret;
            if (tick_var > config_.local_variance_limit) [[unlikely]] {
                circuit_breaker_tripped_.store(true, std::memory_order_release);
                return false;
            }
        }
        last_spot_price_ = current_spot;
        return true;
    }

    // Authoritative worst-case exposure calculation
    inline double calculate_prospective_worst_case_exposure(net::Side order_side, double order_qty) const noexcept {
        double prospective_buy_res = total_reserved_buy_btc_ + (order_side == net::Side::BUY ? order_qty : 0.0);
        double prospective_sell_res = total_reserved_sell_btc_ + (order_side == net::Side::SELL ? order_qty : 0.0);

        double worst_case_long = current_net_position_btc_ + prospective_buy_res;
        double worst_case_short = current_net_position_btc_ - prospective_sell_res;

        double max_directional = std::max(std::abs(worst_case_long), std::abs(worst_case_short));
        return max_directional + std::min(prospective_buy_res, prospective_sell_res);
    }

    inline RiskCheckResult evaluate_order(
        const net::OrderRequest& req,
        double best_bid,
        double best_ask,
        double mark_price,
        uint64_t host_monotonic_now_ns,
        bool is_feed_stale
    ) noexcept {
        if (circuit_breaker_tripped_.load(std::memory_order_relaxed)) [[unlikely]] {
            return RiskCheckResult::REJECT_CIRCUIT_BREAKER_ACTIVE;
        }

        if (is_feed_stale) [[unlikely]] {
            return RiskCheckResult::REJECT_STALE_MARKET_DATA;
        }

        // Strict numerical check
        if (!std::isfinite(req.quantity) || !std::isfinite(req.price) ||
            req.quantity <= 0.0 || req.price <= 0.0) [[unlikely]] {
            return RiskCheckResult::REJECT_INVALID_NUMERICAL;
        }

        // Clip check
        if (req.quantity > config_.max_order_clip_btc) [[unlikely]] {
            return RiskCheckResult::REJECT_EXCEEDS_MAX_CLIP;
        }

        // Gross exposure check
        double worst_exposure = calculate_prospective_worst_case_exposure(req.side, req.quantity);
        if (worst_exposure > config_.max_gross_exposure_btc) [[unlikely]] {
            return RiskCheckResult::REJECT_EXCEEDS_GROSS_EXPOSURE;
        }

        // Price collar
        double ref_price = (req.side == net::Side::BUY) ? best_ask : best_bid;
        if (!std::isfinite(ref_price) || ref_price <= 0.0) {
            ref_price = mark_price;
        }
        if (!std::isfinite(ref_price) || ref_price <= 0.0) [[unlikely]] {
            return RiskCheckResult::REJECT_INVALID_NUMERICAL;
        }

        double deviation = std::abs(req.price - ref_price) / ref_price;
        if (deviation > config_.max_collar_deviation_pct) [[unlikely]] {
            return RiskCheckResult::REJECT_PRICE_COLLAR_VIOLATION;
        }

        // Rate limiter
        uint64_t current_sec = host_monotonic_now_ns / 1'000'000'000ULL;
        if (current_sec != last_order_sec_) {
            last_order_sec_ = current_sec;
            orders_in_current_sec_ = 1;
        } else {
            if (++orders_in_current_sec_ > config_.max_orders_per_second) [[unlikely]] {
                return RiskCheckResult::REJECT_RATE_LIMIT_EXCEEDED;
            }
        }

        // Check reservation table capacity
        if (active_reservation_count_ >= MAX_TRACKED_ORDERS) [[unlikely]] {
            return RiskCheckResult::REJECT_RESERVATION_FULL;
        }

        // Linear probing open addressing: find free slot and ensure no duplicate order ID
        size_t initial_slot = req.order_id % MAX_TRACKED_ORDERS;
        size_t target_slot = MAX_TRACKED_ORDERS;

        for (size_t i = 0; i < MAX_TRACKED_ORDERS; ++i) {
            size_t slot = (initial_slot + i) % MAX_TRACKED_ORDERS;
            if (reservations_[slot].is_active) {
                if (reservations_[slot].order_id == req.order_id) [[unlikely]] {
                    return RiskCheckResult::REJECT_DUPLICATE_ORDER_ID;
                }
            } else if (target_slot == MAX_TRACKED_ORDERS) {
                target_slot = slot; // Found first available free slot
            }
        }

        if (target_slot == MAX_TRACKED_ORDERS) [[unlikely]] {
            return RiskCheckResult::REJECT_RESERVATION_FULL;
        }

        // Reserve slot
        reservations_[target_slot] = OrderReservation{
            .order_id = req.order_id,
            .side = req.side,
            .requested_qty = req.quantity,
            .remaining_reserved_qty = req.quantity,
            .executed_qty = 0.0,
            .is_active = true
        };
        active_reservation_count_++;

        if (req.side == net::Side::BUY) {
            total_reserved_buy_btc_ += req.quantity;
        } else {
            total_reserved_sell_btc_ += req.quantity;
        }

        return RiskCheckResult::APPROVED;
    }

    // Record verified fill: decrements per-order reservation and applies to position
    inline bool record_fill(uint64_t order_id, net::Side side, double filled_qty) noexcept {
        if (!std::isfinite(filled_qty) || filled_qty <= 0.0) return false;

        // Locate reservation via linear probing
        size_t initial_slot = order_id % MAX_TRACKED_ORDERS;
        size_t found_slot = MAX_TRACKED_ORDERS;

        for (size_t i = 0; i < MAX_TRACKED_ORDERS; ++i) {
            size_t slot = (initial_slot + i) % MAX_TRACKED_ORDERS;
            if (reservations_[slot].is_active && reservations_[slot].order_id == order_id) {
                found_slot = slot;
                break;
            }
        }

        if (found_slot == MAX_TRACKED_ORDERS) return false; // Order not found or not active
        auto& res = reservations_[found_slot];
        if (res.side != side) return false;

        // Reject overfills beyond remaining reservation
        if (filled_qty > (res.remaining_reserved_qty + 1e-9)) return false;

        double qty_to_release = std::min(res.remaining_reserved_qty, filled_qty);
        res.remaining_reserved_qty -= qty_to_release;
        res.executed_qty += filled_qty;

        if (side == net::Side::BUY) {
            current_net_position_btc_ += filled_qty;
            total_reserved_buy_btc_ = std::max(0.0, total_reserved_buy_btc_ - qty_to_release);
        } else {
            current_net_position_btc_ -= filled_qty;
            total_reserved_sell_btc_ = std::max(0.0, total_reserved_sell_btc_ - qty_to_release);
        }

        if (res.remaining_reserved_qty <= 1e-9) {
            res.is_active = false;
            active_reservation_count_--;
        }
        return true;
    }

    // Release remaining reservation upon order cancellation or rejection
    inline bool release_reservation(uint64_t order_id) noexcept {
        size_t initial_slot = order_id % MAX_TRACKED_ORDERS;
        size_t found_slot = MAX_TRACKED_ORDERS;

        for (size_t i = 0; i < MAX_TRACKED_ORDERS; ++i) {
            size_t slot = (initial_slot + i) % MAX_TRACKED_ORDERS;
            if (reservations_[slot].is_active && reservations_[slot].order_id == order_id) {
                found_slot = slot;
                break;
            }
        }

        if (found_slot == MAX_TRACKED_ORDERS) return false;
        auto& res = reservations_[found_slot];

        if (res.side == net::Side::BUY) {
            total_reserved_buy_btc_ = std::max(0.0, total_reserved_buy_btc_ - res.remaining_reserved_qty);
        } else {
            total_reserved_sell_btc_ = std::max(0.0, total_reserved_sell_btc_ - res.remaining_reserved_qty);
        }

        res.is_active = false;
        res.remaining_reserved_qty = 0.0;
        active_reservation_count_--;
        return true;
    }

    void trip_circuit_breaker() noexcept { circuit_breaker_tripped_.store(true, std::memory_order_release); }
    void reset_circuit_breaker() noexcept { circuit_breaker_tripped_.store(false, std::memory_order_release); }
    [[nodiscard]] bool is_halted() const noexcept { return circuit_breaker_tripped_.load(std::memory_order_relaxed); }

    inline void adjust_position_for_recovery(double delta_btc) noexcept {
        current_net_position_btc_ += delta_btc;
    }

    [[nodiscard]] double get_net_position_btc() const noexcept { return current_net_position_btc_; }
    [[nodiscard]] double get_total_reserved_buy() const noexcept { return total_reserved_buy_btc_; }
    [[nodiscard]] double get_total_reserved_sell() const noexcept { return total_reserved_sell_btc_; }
    [[nodiscard]] size_t get_active_reservation_count() const noexcept { return active_reservation_count_; }

    [[nodiscard]] double get_gross_exposure_btc() const noexcept {
        double worst_case_long = current_net_position_btc_ + total_reserved_buy_btc_;
        double worst_case_short = current_net_position_btc_ - total_reserved_sell_btc_;
        double max_directional = std::max(std::abs(worst_case_long), std::abs(worst_case_short));
        return max_directional + std::min(total_reserved_buy_btc_, total_reserved_sell_btc_);
    }

    // Invariant verification check: aggregate totals must exactly equal sum of slots
    [[nodiscard]] bool verify_invariants() const noexcept {
        double calc_buy = 0.0;
        double calc_sell = 0.0;
        size_t count = 0;

        for (size_t i = 0; i < MAX_TRACKED_ORDERS; ++i) {
            if (reservations_[i].is_active) {
                count++;
                if (reservations_[i].side == net::Side::BUY) {
                    calc_buy += reservations_[i].remaining_reserved_qty;
                } else {
                    calc_sell += reservations_[i].remaining_reserved_qty;
                }
            }
        }

        bool match_count = (count == active_reservation_count_);
        bool match_buy = std::abs(calc_buy - total_reserved_buy_btc_) < 1e-9;
        bool match_sell = std::abs(calc_sell - total_reserved_sell_btc_) < 1e-9;
        return match_count && match_buy && match_sell;
    }

private:
    RiskConfig config_;
    double current_net_position_btc_{0.0};
    double total_reserved_buy_btc_{0.0};
    double total_reserved_sell_btc_{0.0};
    size_t active_reservation_count_{0};
    OrderReservation reservations_[MAX_TRACKED_ORDERS];
    std::atomic<bool> circuit_breaker_tripped_{false};
    uint64_t last_order_sec_{0};
    uint32_t orders_in_current_sec_{0};
    double last_spot_price_{0.0};
};

} // namespace hft::risk
