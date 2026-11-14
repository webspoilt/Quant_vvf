#pragma once

#include <cmath>

namespace hft::math {

/**
 * High-speed vectorized / analytical mathematical utilities for coin-margined derivatives.
 * Fast approximation without external heavy libraries (e.g., QuantLib) in the hot path.
 */
class InverseDerivativesMath {
public:
    // Deribit Coin-Margined Delta Rescaling Factor: (S0 / S1)^2
    static inline double inverse_delta_rescale(double s0, double s1) noexcept {
        if (s1 <= 0.0) [[unlikely]] return 0.0;
        double ratio = s0 / s1;
        return ratio * ratio;
    }

    // BTC-Denominated Inverse Delta: C / S^2 (Where C is USD Notional)
    static inline double btc_delta(double usd_notional, double spot) noexcept {
        if (spot <= 0.0) [[unlikely]] return 0.0;
        return usd_notional / (spot * spot);
    }

    // High-speed Gatheral Raw SVI Total Implied Variance Evaluation
    // w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    static inline double svi_implied_variance(double log_moneyness_k, double a, double b, double rho, double m, double sigma) noexcept {
        double diff = log_moneyness_k - m;
        return a + b * (rho * diff + std::sqrt(diff * diff + sigma * sigma));
    }
};

} // namespace hft::math
