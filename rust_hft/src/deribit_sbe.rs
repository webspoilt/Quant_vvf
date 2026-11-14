//! Deribit Starbase SBE (Simple Binary Encoding) Wire Layout & Resilient Gateway
//!
//! Conforming to Deribit's Official Starbase Binary API Reference (Equinix LD4, Slough, UK):
//! - UDP Packet Header: 24 bytes (sendTime: i64, sequenceNum: i64, channelId: i32, type: u16, messageCount: u16)
//! - UDP Packet Types: 0x01 Incremental, 0x02 Snapshot, 0x04 Retransmit, 0x05 Inc-Retransmit, 0x00 Reject
//! - Sequence Progression: Next Expected = SequenceNum + MessageCount
//! - UDP SBE Message Header: 16 bytes (messageLength: u16, messageTypeId: u16, version: u16, flags: u16, transactTime: i64)
//! - Template & Schema Validation: Validates type_id (100 Book, 101 Trade, 102 Def) AND version == 1
//! - Per-Channel State & Capacity Exhaustion Protection: Up to 16 channels tracked independently.
//!   Requests for a 17th channel fail explicitly with `ChannelCapacityExhausted` (no overwrite or fallback).
//! - Verified Retransmission Pipeline: Enforces continuous gap closure before clearing stale state.

use core::mem::size_of;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

// ============================================================================
// 1. STARBASE PROTOCOL HEADERS & CONSTANTS
// ============================================================================

pub mod starbase_types {
    pub const PROTOCOL_ID_DERIBIT: u8 = 0xDB;
    pub const EXPECTED_SCHEMA_VERSION: u16 = 1;

    // UDP Packet Types
    #[allow(dead_code)]
    pub const PACKET_TYPE_RETRANSMIT_REJECT: u16 = 0x00;
    pub const PACKET_TYPE_INCREMENTAL: u16 = 0x01;
    pub const PACKET_TYPE_SNAPSHOT: u16 = 0x02;
    pub const PACKET_TYPE_RETRANSMIT: u16 = 0x04;
    pub const PACKET_TYPE_INC_RETRANSMIT: u16 = 0x05;

    // SBE Message Types
    pub const MSG_BOOK_ORDER_UPDATE: u16 = 100;
    pub const MSG_TRADE_TICK: u16 = 101;
    pub const MSG_INSTRUMENT_DEF: u16 = 102;

    // TCP Order Entry Types
    pub const MSG_NEW_ORDER_REQ: u16 = 100;
    #[allow(dead_code)]
    pub const MSG_CANCEL_ORDER_REQ: u16 = 101;
    #[allow(dead_code)]
    pub const MSG_NEW_ORDER_RESP: u16 = 200;
    #[allow(dead_code)]
    pub const MSG_ORDER_FILLED: u16 = 300;
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy, IntoBytes, FromBytes, Immutable, KnownLayout)]
pub struct StarbaseUdpPacketHeader {
    pub send_time: i64,     // 8 bytes
    pub sequence_num: i64,  // 8 bytes
    pub channel_id: i32,    // 4 bytes
    pub packet_type: u16,   // 2 bytes
    pub message_count: u16, // 2 bytes
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy, IntoBytes, FromBytes, Immutable, KnownLayout)]
pub struct StarbaseSbeMessageHeader {
    pub message_length: u16,  // 2 bytes
    pub message_type_id: u16, // 2 bytes
    pub version: u16,         // 2 bytes
    pub flags: u16,           // 2 bytes
    pub transact_time: i64,   // 8 bytes
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy, IntoBytes, FromBytes, Immutable, KnownLayout)]
pub struct StarbaseBookEvent {
    pub instrument_id: i64, // 8 bytes
    pub price: i64,         // 8 bytes
    pub amount: i64,        // 8 bytes
    pub side: u8,           // 1 byte
    pub action: u8,         // 1 byte
    pub reserved: u16,      // 2 bytes
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy, IntoBytes, FromBytes, Immutable, KnownLayout)]
pub struct StarbaseTcpOrderHeader {
    pub protocol_id: u8,             // 1 byte (0xDB)
    pub flags: u8,                   // 1 byte
    pub message_length: u16,         // 2 bytes (68)
    pub message_type_id: u16,        // 2 bytes (100)
    pub version: u16,                // 2 bytes
    pub sequence_num: i64,           // 8 bytes
    pub last_processed_seq_num: i64, // 8 bytes
    pub send_time: i64,              // 8 bytes
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy, IntoBytes, FromBytes, Immutable, KnownLayout)]
pub struct StarbaseNewOrderRequestBody {
    pub client_order_id: i64, // 8 bytes
    pub instrument_id: i64,   // 8 bytes
    pub limit_price: i64,     // 8 bytes
    pub amount: i64,          // 8 bytes
    pub side: u8,             // 1 byte
    pub time_in_force: u8,    // 1 byte
    pub order_type: u8,       // 1 byte
    pub flags: u8,            // 1 byte
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy, IntoBytes, FromBytes, Immutable, KnownLayout)]
pub struct StarbaseNewOrderWireMessage {
    pub header: StarbaseTcpOrderHeader,
    pub body: StarbaseNewOrderRequestBody,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NormalizedMarketEvent {
    pub sequence_num: i64,
    pub channel_id: i32,
    pub instrument_id: i64,
    pub transact_time_ns: i64,
    pub price: i64,
    pub amount: i64,
    pub is_buy: bool,
    pub is_stale_book: bool,
}

// ============================================================================
// 2. PER-CHANNEL RESILIENT ARBITRATOR & STRICT CAPACITY DEFENSE
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StarbasePacketStatus {
    Ok { message_count: u16 },
    HeartbeatOk,
    SequenceGapDetected { expected: i64, received: i64 },
    DuplicatePacket,
    InvalidPacketType,
    RetransmitPacket { sequence: i64, count: u16 },
    ChannelCapacityExhausted, // P0 FIX: Explicit capacity error on 17th channel
}

#[derive(Debug, Clone, Copy)]
pub struct ChannelSequenceState {
    pub channel_id: i32,
    pub expected_sequence_num: i64,
    pub is_book_stale: bool,
    pub missing_gap_start: i64,
    pub missing_gap_end: i64,
    pub active: bool,
}

pub struct StarbaseFeedArbitrator {
    channels: [ChannelSequenceState; 16],
}

impl StarbaseFeedArbitrator {
    pub fn new() -> Self {
        Self {
            channels: [ChannelSequenceState {
                channel_id: 0,
                expected_sequence_num: 0,
                is_book_stale: false,
                missing_gap_start: 0,
                missing_gap_end: 0,
                active: false,
            }; 16],
        }
    }

    // P0 FIX: Finds existing channel or allocates free slot.
    // If all 16 slots are occupied, returns None (NO overwrite, NO fallback corruption!)
    fn find_or_allocate_channel_idx(&mut self, channel_id: i32) -> Option<usize> {
        let mut first_free = None;
        for (i, ch) in self.channels.iter().enumerate() {
            if ch.active && ch.channel_id == channel_id {
                return Some(i);
            }
            if !ch.active && first_free.is_none() {
                first_free = Some(i);
            }
        }
        if let Some(idx) = first_free {
            self.channels[idx].channel_id = channel_id;
            self.channels[idx].expected_sequence_num = 0;
            self.channels[idx].is_book_stale = false;
            self.channels[idx].missing_gap_start = 0;
            self.channels[idx].missing_gap_end = 0;
            self.channels[idx].active = true;
            return Some(idx);
        }
        None // 17th channel rejected!
    }

    pub fn arbitrate_packet_header(
        &mut self,
        hdr: &StarbaseUdpPacketHeader,
    ) -> StarbasePacketStatus {
        let seq = i64::from_le(hdr.sequence_num);
        let msg_count = u16::from_le(hdr.message_count) as i64;
        let pkt_type = u16::from_le(hdr.packet_type);
        let chan_id = i32::from_le(hdr.channel_id);

        let ch_idx = match self.find_or_allocate_channel_idx(chan_id) {
            Some(idx) => idx,
            None => return StarbasePacketStatus::ChannelCapacityExhausted, // P0 Fix
        };

        let ch = &mut self.channels[ch_idx];

        if pkt_type == starbase_types::PACKET_TYPE_RETRANSMIT
            || pkt_type == starbase_types::PACKET_TYPE_INC_RETRANSMIT
        {
            return StarbasePacketStatus::RetransmitPacket {
                sequence: seq,
                count: msg_count as u16,
            };
        }

        if pkt_type != starbase_types::PACKET_TYPE_INCREMENTAL
            && pkt_type != starbase_types::PACKET_TYPE_SNAPSHOT
        {
            return StarbasePacketStatus::InvalidPacketType;
        }

        if ch.expected_sequence_num == 0 {
            ch.expected_sequence_num = seq.wrapping_add(if msg_count > 0 { msg_count } else { 0 });
            return if msg_count == 0 {
                StarbasePacketStatus::HeartbeatOk
            } else {
                StarbasePacketStatus::Ok {
                    message_count: msg_count as u16,
                }
            };
        }

        if msg_count == 0 {
            if seq == ch.expected_sequence_num {
                return StarbasePacketStatus::HeartbeatOk;
            } else if seq < ch.expected_sequence_num {
                return StarbasePacketStatus::DuplicatePacket;
            } else {
                let expected = ch.expected_sequence_num;
                ch.is_book_stale = true;
                ch.missing_gap_start = expected;
                ch.missing_gap_end = seq;
                return StarbasePacketStatus::SequenceGapDetected {
                    expected,
                    received: seq,
                };
            }
        }

        if seq < ch.expected_sequence_num {
            return StarbasePacketStatus::DuplicatePacket;
        }

        if seq > ch.expected_sequence_num {
            let expected = ch.expected_sequence_num;
            ch.is_book_stale = true;
            ch.missing_gap_start = expected;
            ch.missing_gap_end = seq;
            ch.expected_sequence_num = seq.wrapping_add(msg_count);
            return StarbasePacketStatus::SequenceGapDetected {
                expected,
                received: seq,
            };
        }

        ch.expected_sequence_num = seq.wrapping_add(msg_count);
        StarbasePacketStatus::Ok {
            message_count: msg_count as u16,
        }
    }

    /// Verified Retransmission Pipeline:
    /// Ingests validated retransmission packet, validates sequence continuity,
    /// and clears stale state ONLY when the verified messages close the missing gap.
    pub fn process_verified_retransmit(
        &mut self,
        channel_id: i32,
        retransmit_seq: i64,
        valid_message_count: u16,
    ) -> bool {
        if let Some(idx) = self.find_or_allocate_channel_idx(channel_id) {
            let ch = &mut self.channels[idx];
            if ch.is_book_stale
                && retransmit_seq == ch.missing_gap_start
                && retransmit_seq + (valid_message_count as i64) >= ch.missing_gap_end
            {
                ch.is_book_stale = false; // Verified gap closure
                ch.missing_gap_start = 0;
                ch.missing_gap_end = 0;
                return true;
            }
        }
        false
    }

    /// Snapshot Fallback Synchronization:
    /// When gap retransmission is unavailable or fails, an authoritative snapshot packet
    /// resynchronizes channel state, updates the expected sequence, and restores feed health.
    #[allow(dead_code)]
    pub fn apply_authoritative_snapshot(
        &mut self,
        channel_id: i32,
        snapshot_seq: i64,
        message_count: u16,
    ) -> bool {
        if let Some(idx) = self.find_or_allocate_channel_idx(channel_id) {
            let ch = &mut self.channels[idx];
            ch.expected_sequence_num = snapshot_seq.wrapping_add(message_count as i64);
            ch.is_book_stale = false;
            ch.missing_gap_start = 0;
            ch.missing_gap_end = 0;
            return true;
        }
        false
    }

    #[inline(always)]
    pub fn is_channel_stale(&mut self, channel_id: i32) -> bool {
        self.find_or_allocate_channel_idx(channel_id)
            .map(|idx| self.channels[idx].is_book_stale)
            .unwrap_or(true)
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub fn get_channel_expected_seq(&mut self, channel_id: i32) -> Option<i64> {
        self.find_or_allocate_channel_idx(channel_id)
            .map(|idx| self.channels[idx].expected_sequence_num)
    }
}

impl Default for StarbaseFeedArbitrator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// 3. ZERO-COPY ITERATIVE SBE STREAM PARSER WITH VERSION VALIDATION
// ============================================================================

pub struct StarbaseZeroCopyParser;

impl StarbaseZeroCopyParser {
    #[inline(always)]
    pub fn parse_udp_header(raw_bytes: &[u8]) -> Option<(&StarbaseUdpPacketHeader, &[u8])> {
        if raw_bytes.len() < size_of::<StarbaseUdpPacketHeader>() {
            return None;
        }
        let (hdr, payload) = StarbaseUdpPacketHeader::ref_from_prefix(raw_bytes).ok()?;
        Some((hdr, payload))
    }

    #[inline(always)]
    pub fn parse_next_sbe_message(
        sbe_slice: &[u8],
    ) -> Option<(&StarbaseSbeMessageHeader, &StarbaseBookEvent, &[u8])> {
        let min_len = size_of::<StarbaseSbeMessageHeader>() + size_of::<StarbaseBookEvent>();
        if sbe_slice.len() < min_len {
            return None;
        }

        let (hdr, rest) = StarbaseSbeMessageHeader::ref_from_prefix(sbe_slice).ok()?;
        let msg_len = u16::from_le(hdr.message_length) as usize;

        if msg_len < min_len || sbe_slice.len() < msg_len {
            return None;
        }

        let version = u16::from_le(hdr.version);
        if version != starbase_types::EXPECTED_SCHEMA_VERSION {
            return None;
        }

        let type_id = u16::from_le(hdr.message_type_id);
        if type_id != starbase_types::MSG_BOOK_ORDER_UPDATE
            && type_id != starbase_types::MSG_TRADE_TICK
            && type_id != starbase_types::MSG_INSTRUMENT_DEF
        {
            return None;
        }

        let (body, _) = StarbaseBookEvent::ref_from_prefix(rest).ok()?;
        let remaining = &sbe_slice[msg_len..];
        Some((hdr, body, remaining))
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn serialize_tcp_new_order(
        tcp_client_seq: i64,
        last_processed_seq: i64,
        now_ns: i64,
        client_order_id: i64,
        instrument_id: i64,
        limit_price: i64,
        amount: i64,
        side: u8,
        out_buf: &mut [u8; 128],
    ) -> usize {
        let total_len = size_of::<StarbaseNewOrderWireMessage>() as u16;
        let wire_msg = StarbaseNewOrderWireMessage {
            header: StarbaseTcpOrderHeader {
                protocol_id: starbase_types::PROTOCOL_ID_DERIBIT,
                flags: 0,
                message_length: total_len.to_le(),
                message_type_id: starbase_types::MSG_NEW_ORDER_REQ.to_le(),
                version: 1u16.to_le(),
                sequence_num: tcp_client_seq.to_le(),
                last_processed_seq_num: last_processed_seq.to_le(),
                send_time: now_ns.to_le(),
            },
            body: StarbaseNewOrderRequestBody {
                client_order_id: client_order_id.to_le(),
                instrument_id: instrument_id.to_le(),
                limit_price: limit_price.to_le(),
                amount: amount.to_le(),
                side,
                time_in_force: 1,
                order_type: 0,
                flags: 0,
            },
        };

        let bytes = wire_msg.as_bytes();
        out_buf[..bytes.len()].copy_from_slice(bytes);
        bytes.len()
    }
}
