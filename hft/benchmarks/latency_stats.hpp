#pragma once
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace hft::bench {

struct LatencyStats {
    size_t samples;
    double min_ns;
    double median_ns;
    double p90_ns;
    double p95_ns;
    double p99_ns;
    double p99_9_ns;
    double max_ns;
    double stddev_ns;
};

inline LatencyStats compute_stats(std::vector<double>& latencies) {
    if (latencies.empty()) return {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::sort(latencies.begin(), latencies.end());
    size_t n = latencies.size();

    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double mean = sum / n;
    double sq_sum = 0.0;
    for (double v : latencies) sq_sum += (v - mean) * (v - mean);
    double stddev = std::sqrt(sq_sum / n);

    auto pct = [&](double p) {
        size_t idx = static_cast<size_t>(p * n);
        if (idx >= n) idx = n - 1;
        return latencies[idx];
    };

    return LatencyStats{
        n,
        latencies.front(),
        pct(0.50),
        pct(0.90),
        pct(0.95),
        pct(0.99),
        pct(0.999),
        latencies.back(),
        stddev
    };
}

inline void print_stats(const char* name, const LatencyStats& s) {
    std::cout << std::left << std::setw(32) << name << " | "
              << "N=" << std::setw(7) << s.samples << " | "
              << "Min: " << std::setw(8) << std::fixed << std::setprecision(1) << s.min_ns << " ns | "
              << "Med: " << std::setw(8) << s.median_ns << " ns | "
              << "P90: " << std::setw(8) << s.p90_ns << " ns | "
              << "P99: " << std::setw(8) << s.p99_ns << " ns | "
              << "P99.9: " << std::setw(8) << s.p99_9_ns << " ns | "
              << "Max: " << std::setw(8) << s.max_ns << " ns | "
              << "StdDev: " << std::setw(7) << s.stddev_ns << " ns"
              << std::endl;
}

} // namespace hft::bench
