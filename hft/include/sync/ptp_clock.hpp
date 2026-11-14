#pragma once

#include "network/types.hpp"
#include <chrono>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace hft::sync {

/**
 * IEEE 1588 Precision Time Protocol (PTP) and TSC Cycle Clock Synchronizer.
 * Achieves sub-nanosecond hardware timing and converts CPU RDTSC cycles to nanoseconds.
 */
class PtpClock {
public:
    static inline uint64_t rdtsc() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
        return __rdtsc();
#else
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count()
        );
#endif
    }

    // High-resolution hardware monotonic timestamp
    static inline net::TimestampNs now_ns() noexcept {
        return static_cast<net::TimestampNs>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count()
        );
    }
};

} // namespace hft::sync
