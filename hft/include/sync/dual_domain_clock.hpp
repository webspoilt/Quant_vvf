#pragma once

#include "network/types.hpp"
#include <cstdint>
#include <chrono>

namespace hft::sync {

/**
 * Clock Architecture: Strict Clock Domains (P1 Fix)
 * 
 * Domains:
 * 1. HOST_MONOTONIC: Steady, non-decreasing clock (std::chrono::steady_clock),
 *    guaranteed immune to system clock adjustments, NTP steps, or leap seconds.
 *    Used strictly for local rate limiting, timeouts, and execution latencies.
 * 2. PTP_HARDWARE_EPOCH: Real hardware PTP clock from network interface controller (PHC).
 * 3. SIMULATION_EPOCH: Software wall-clock simulation (std::chrono::system_clock).
 * 
 * Rules:
 * - Cross-domain comparisons between PTP_HARDWARE_EPOCH and SIMULATION_EPOCH are INCOMPATIBLE.
 * - Hardware PTP access returns an explicit error when hardware PHC is unavailable.
 */
enum class ClockDomain : uint8_t {
    HOST_MONOTONIC = 0,
    PTP_HARDWARE_EPOCH = 1,
    SIMULATION_EPOCH = 2
};

struct PtpClockResult {
    net::TimestampNs timestamp_ns{0};
    bool is_supported{false};
};

class DualDomainClock {
public:
    static constexpr uint64_t MAX_ALLOWABLE_SKEW_NS = 500'000'000ULL; // 500ms max staleness

    // Host monotonic clock (strictly non-decreasing)
    static inline uint64_t get_host_monotonic_ns() noexcept {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()
            ).count()
        );
    }

    // Simulation Epoch Provider (Clearly labelled; NOT claimed as hardware PTP)
    static inline net::TimestampNs get_simulation_epoch_ns() noexcept {
        return static_cast<net::TimestampNs>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count()
        );
    }

    // Hardware PTP provider: Explicitly reports unsupported when PHC hardware is unavailable
    static inline PtpClockResult get_hardware_ptp_epoch_ns() noexcept {
#if defined(__linux__) && defined(CLOCK_HWSYS)
        struct timespec ts;
        if (clock_gettime(CLOCK_HWSYS, &ts) == 0) {
            return {
                .timestamp_ns = static_cast<net::TimestampNs>(ts.tv_sec) * 1'000'000'000ULL + ts.tv_nsec,
                .is_supported = true
            };
        }
#endif
        return {.timestamp_ns = 0, .is_supported = false};
    }

    // Validate market timestamps within compatible epoch domain
    static inline bool validate_market_timestamp(
        net::TimestampNs packet_time_ns,
        net::TimestampNs current_reference_epoch_ns,
        uint64_t max_age_ns
    ) noexcept {
        if (packet_time_ns == 0) [[unlikely]] return false;
        if (packet_time_ns > (current_reference_epoch_ns + 50'000'000ULL)) [[unlikely]] return false;
        if (current_reference_epoch_ns > packet_time_ns &&
            (current_reference_epoch_ns - packet_time_ns) > max_age_ns) [[unlikely]] return false;
        return true;
    }

    // Strict domain compatibility: Domains must match exactly
    static inline bool check_domain_compatibility(ClockDomain d1, ClockDomain d2) noexcept {
        return d1 == d2;
    }
};

} // namespace hft::sync
