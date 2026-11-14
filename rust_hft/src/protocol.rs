//! Network Packet Definitions & Zero-Copy Protocol Parser
//!
//! Uses primitive integer representations for zero-copy binary wire compatibility.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

pub type TimestampNs = u64;

pub mod side {
    pub const BUY: u8 = 0;
    pub const SELL: u8 = 1;
}

pub mod message_type {
    pub const ADD_ORDER: u8 = 0;
    pub const ORDER_EXECUTED: u8 = 1;
    pub const ORDER_CANCEL: u8 = 2;
    pub const TRADE_TICK: u8 = 3;
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy, IntoBytes, FromBytes, Immutable, KnownLayout)]
pub struct WireItchPacket {
    pub msg_type: u8,
    pub locate_code: u16,
    pub tracking_number: u16,
    pub timestamp_ns: TimestampNs,
    pub order_reference: u64,
    pub side: u8,
    pub stock: [u8; 8],
    pub shares: u32,
    pub price: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NormalizedMarketEvent {
    pub msg_type: u8,
    pub timestamp_ns: TimestampNs,
    pub order_id: u64,
    pub is_buy: bool,
    pub symbol: [u8; 8],
    pub qty: u32,
    pub price_fixed: u32,
}

pub struct ZeroCopyParser;

impl ZeroCopyParser {
    #[inline(always)]
    pub fn parse_wire_frame(raw_bytes: &[u8]) -> Option<&WireItchPacket> {
        if raw_bytes.len() < size_of::<WireItchPacket>() {
            return None;
        }
        WireItchPacket::ref_from_prefix(raw_bytes)
            .ok()
            .map(|(packet, _rest)| packet)
    }

    #[inline(always)]
    pub fn normalize(packet: &WireItchPacket, event: &mut NormalizedMarketEvent) {
        event.msg_type = packet.msg_type;
        event.timestamp_ns = u64::from_be(packet.timestamp_ns);
        event.order_id = u64::from_be(packet.order_reference);
        event.is_buy = packet.side == side::BUY;
        event.symbol = packet.stock;
        event.qty = u32::from_be(packet.shares);
        event.price_fixed = u32::from_be(packet.price);
    }
}
