#pragma once

#include <cstdint>
#include <cstring>
#include <array>
#include <string_view>

namespace hft::net {

#pragma pack(push, 1)

// Precision timestamp conforming to IEEE 1588 PTP (Nanoseconds since UNIX epoch)
using TimestampNs = uint64_t;

enum class Side : uint8_t {
    BUY = 0,
    SELL = 1
};

enum class MessageType : uint8_t {
    BOOK_UPDATE = 1,
    TRADE_TICK = 2,
    ORDER_SUBMIT = 3,
    ORDER_CANCEL = 4,
    ORDER_ACK = 5,
    ORDER_REJECT = 6,
    CIRCUIT_BREAKER_HALT = 99
};

// Fixed length instrument identifier to avoid heap allocation (e.g. "BTC-29MAR24-70000-C\0")
struct InstrumentSymbol {
    char data[32]{};

    constexpr InstrumentSymbol() = default;
    constexpr explicit InstrumentSymbol(std::string_view sv) {
        size_t len = sv.size() < 31 ? sv.size() : 31;
        for (size_t i = 0; i < len; ++i) data[i] = sv[i];
        data[len] = '\0';
    }

    constexpr bool operator==(const InstrumentSymbol& other) const noexcept {
        for (size_t i = 0; i < 32; ++i) {
            if (data[i] != other.data[i]) return false;
            if (data[i] == '\0') break;
        }
        return true;
    }
};

// Zero-copy binary raw market depth tick
struct MarketDepthPacket {
    MessageType msg_type{MessageType::BOOK_UPDATE};
    uint8_t flags{0};
    uint16_t depth_levels{0};
    TimestampNs exchange_time_ns{0};
    TimestampNs nic_rx_time_ns{0}; // PTP hardware timestamp captured at NIC PHY
    InstrumentSymbol symbol;
    double best_bid_price{0.0};
    double best_bid_qty{0.0};
    double best_ask_price{0.0};
    double best_ask_qty{0.0};
    double mark_price{0.0};
    double spot_index{0.0};
};

// Ultra-fast order entry request sent directly to network ring buffer
struct OrderRequest {
    MessageType msg_type{MessageType::ORDER_SUBMIT};
    Side side{Side::BUY};
    uint8_t time_in_force{0}; // 0 = IOC, 1 = POST_ONLY, 2 = GTC
    uint8_t reserved{0};
    uint64_t order_id{0};
    TimestampNs client_sent_ns{0};
    InstrumentSymbol symbol;
    double price{0.0};
    double quantity{0.0};      // In coin (BTC) or USD notional contracts
    double max_slippage{0.0};
};

#pragma pack(pop)

} // namespace hft::net
