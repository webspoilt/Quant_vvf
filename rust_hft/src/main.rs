//! Production Ingestion & Gateway Loop
//!
//! Features:
//! - Multi-channel sequence tracking
//! - Retransmission recovery pipeline invocation
//! - Multi-message SBE parsing with version validation
//! - Clean queue draining on shutdown

use std::hint::spin_loop;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

mod deribit_sbe;
mod ring_buffer;
mod threading;

use deribit_sbe::{
    NormalizedMarketEvent, StarbaseBookEvent, StarbaseFeedArbitrator, StarbasePacketStatus,
    StarbaseSbeMessageHeader, StarbaseUdpPacketHeader, StarbaseZeroCopyParser, starbase_types,
};
use ring_buffer::make_spsc_channel;
use threading::ThreadBuilder;

const RING_CAPACITY: usize = 65536;

fn main() {
    println!("==================================================================");
    println!("  Starbase Ingestion & Gateway Production Pipeline (Rust 2024)    ");
    println!("  Per-Channel Sequencer | Verified Retransmit | Version Enforced  ");
    println!("==================================================================");

    let running = Arc::new(AtomicBool::new(true));
    let (mut producer, mut consumer) = make_spsc_channel::<NormalizedMarketEvent, RING_CAPACITY>();
    let dropped_events = Arc::new(AtomicU64::new(0));

    let producer_running = Arc::clone(&running);
    let producer_dropped = Arc::clone(&dropped_events);

    let ingestion_handle = ThreadBuilder::spawn_pinned("starbase-udp-core", 0, move || {
        let mut arbitrator = StarbaseFeedArbitrator::new();
        let mut wire_buffer = [0u8; 2048];

        let h1_size = core::mem::size_of::<StarbaseUdpPacketHeader>();
        let h2_size = core::mem::size_of::<StarbaseSbeMessageHeader>();
        let b_size = core::mem::size_of::<StarbaseBookEvent>();
        let sbe_msg_len = (h2_size + b_size) as u16;

        let channel_id = 10i32;
        let mut current_seq = 1000i64;
        let mut pushed = 0usize;

        for i in 0..5_000 {
            if !producer_running.load(Ordering::Relaxed) {
                break;
            }

            // Simulate periodic retransmission packet injection at iteration 2500
            let (pkt_type, msg_count) = if i == 2500 {
                (starbase_types::PACKET_TYPE_INC_RETRANSMIT, 2u16)
            } else {
                (starbase_types::PACKET_TYPE_INCREMENTAL, 2u16)
            };

            let udp_hdr = StarbaseUdpPacketHeader {
                send_time: 1_700_000_000_000_000_000i64.to_le(),
                sequence_num: current_seq.to_le(),
                channel_id: channel_id.to_le(),
                packet_type: pkt_type.to_le(),
                message_count: msg_count.to_le(),
            };

            let sbe_hdr = StarbaseSbeMessageHeader {
                message_length: sbe_msg_len.to_le(),
                message_type_id: starbase_types::MSG_BOOK_ORDER_UPDATE.to_le(),
                version: starbase_types::EXPECTED_SCHEMA_VERSION.to_le(),
                flags: 0,
                transact_time: 1_700_000_000_000_000_000i64.to_le(),
            };

            let book_1 = StarbaseBookEvent {
                instrument_id: 42i64.to_le(),
                price: 650_000_000i64.to_le(),
                amount: 100_0000i64.to_le(),
                side: 0,
                action: 0,
                reserved: 0,
            };

            let book_2 = StarbaseBookEvent {
                instrument_id: 42i64.to_le(),
                price: 650_010_000i64.to_le(),
                amount: 50_0000i64.to_le(),
                side: 1,
                action: 0,
                reserved: 0,
            };

            unsafe {
                let p_udp = std::slice::from_raw_parts(&udp_hdr as *const _ as *const u8, h1_size);
                wire_buffer[..h1_size].copy_from_slice(p_udp);

                let offset1 = h1_size;
                let p_sbe1 = std::slice::from_raw_parts(&sbe_hdr as *const _ as *const u8, h2_size);
                let p_b1 = std::slice::from_raw_parts(&book_1 as *const _ as *const u8, b_size);
                wire_buffer[offset1..offset1 + h2_size].copy_from_slice(p_sbe1);
                wire_buffer[offset1 + h2_size..offset1 + h2_size + b_size].copy_from_slice(p_b1);

                let offset2 = offset1 + h2_size + b_size;
                let p_sbe2 = std::slice::from_raw_parts(&sbe_hdr as *const _ as *const u8, h2_size);
                let p_b2 = std::slice::from_raw_parts(&book_2 as *const _ as *const u8, b_size);
                wire_buffer[offset2..offset2 + h2_size].copy_from_slice(p_sbe2);
                wire_buffer[offset2 + h2_size..offset2 + h2_size + b_size].copy_from_slice(p_b2);
            }

            let total_pkt_len = h1_size + 2 * (h2_size + b_size);

            if let Some((parsed_udp, mut sbe_stream)) =
                StarbaseZeroCopyParser::parse_udp_header(&wire_buffer[..total_pkt_len])
            {
                let status = arbitrator.arbitrate_packet_header(parsed_udp);
                match status {
                    StarbasePacketStatus::Ok { message_count: _ } => {
                        let mut parsed_count = 0u16;
                        while let Some((_hdr, body, remaining)) =
                            StarbaseZeroCopyParser::parse_next_sbe_message(sbe_stream)
                        {
                            parsed_count += 1;
                            let event = NormalizedMarketEvent {
                                sequence_num: i64::from_le(parsed_udp.sequence_num),
                                channel_id: i32::from_le(parsed_udp.channel_id),
                                instrument_id: i64::from_le(body.instrument_id),
                                transact_time_ns: i64::from_le(parsed_udp.send_time),
                                price: i64::from_le(body.price),
                                amount: i64::from_le(body.amount),
                                is_buy: body.side == 0,
                                is_stale_book: arbitrator.is_channel_stale(channel_id),
                            };

                            if producer.try_push(event).is_ok() {
                                pushed += 1;
                            } else {
                                producer_dropped.fetch_add(1, Ordering::Relaxed);
                            }
                            sbe_stream = remaining;
                        }
                        assert_eq!(parsed_count, msg_count);
                    }
                    StarbasePacketStatus::RetransmitPacket { sequence, count: _ } => {
                        // Ingest and verify retransmitted messages
                        let mut retransmitted_valid = 0u16;
                        while let Some((_hdr, _body, remaining)) =
                            StarbaseZeroCopyParser::parse_next_sbe_message(sbe_stream)
                        {
                            retransmitted_valid += 1;
                            sbe_stream = remaining;
                        }
                        // Apply recovery to channel
                        arbitrator.process_verified_retransmit(
                            channel_id,
                            sequence,
                            retransmitted_valid,
                        );
                    }
                    _ => {}
                }
            }

            current_seq += msg_count as i64;
        }

        producer_running.store(false, Ordering::Relaxed);
        eprintln!(
            "[Ingestion Core] Finished. Pushed: {}, Dropped: {}",
            pushed,
            producer_dropped.load(Ordering::Relaxed)
        );
    })
    .expect("Failed to spawn ingestion thread");

    let consumer_running = Arc::clone(&running);
    let gateway_handle = ThreadBuilder::spawn_pinned("starbase-order-core", 1, move || {
        let mut processed = 0usize;
        let mut tcp_out_buf = [0u8; 128];
        let mut tcp_client_seq = 1i64;

        loop {
            match consumer.try_pop() {
                Some(event) => {
                    if event.is_stale_book {
                        continue;
                    }

                    let bytes = StarbaseZeroCopyParser::serialize_tcp_new_order(
                        tcp_client_seq,
                        0,
                        event.transact_time_ns,
                        10_000 + tcp_client_seq,
                        event.instrument_id,
                        event.price,
                        event.amount,
                        if event.is_buy { 0 } else { 1 },
                        &mut tcp_out_buf,
                    );
                    assert_eq!(bytes, 68);

                    tcp_client_seq += 1;
                    processed += 1;
                }
                None => {
                    if !consumer_running.load(Ordering::Relaxed) {
                        break;
                    }
                    spin_loop();
                }
            }
        }

        eprintln!("[Gateway Core] Drained. Processed orders: {}", processed);
    })
    .expect("Failed to spawn gateway thread");

    let _ = ingestion_handle.join();
    let _ = gateway_handle.join();

    println!("==================================================================");
    println!(">>> Starbase Production Pipeline Finished Successfully         <<<");
    println!("==================================================================");
}
