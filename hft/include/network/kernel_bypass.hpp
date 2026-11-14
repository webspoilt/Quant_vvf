#pragma once

#include "types.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace hft::net {

/**
 * Kernel-Bypass Zero-Copy Parser
 * Parses wire frames directly from the NIC ring buffer without allocating dynamic memory.
 * Emulates DPDK / Solarflare EF_VI zero-copy ring processing.
 */
class ZeroCopyFrameParser {
public:
    static constexpr size_t MIN_HEADER_SIZE = 4;

    static inline bool parse_market_depth(const uint8_t* raw_buffer, size_t len, MarketDepthPacket& out_packet) noexcept {
        if (len < sizeof(MarketDepthPacket)) [[unlikely]] {
            return false;
        }

        // Direct memory map / struct overlay from DMA ring buffer
        std::memcpy(&out_packet, raw_buffer, sizeof(MarketDepthPacket));
        return out_packet.msg_type == MessageType::BOOK_UPDATE;
    }

    static inline bool serialize_order(const OrderRequest& req, uint8_t* tx_ring_buffer, size_t buffer_capacity, size_t& bytes_written) noexcept {
        if (buffer_capacity < sizeof(OrderRequest)) [[unlikely]] {
            return false;
        }

        std::memcpy(tx_ring_buffer, &req, sizeof(OrderRequest));
        bytes_written = sizeof(OrderRequest);
        return true;
    }
};

} // namespace hft::net
