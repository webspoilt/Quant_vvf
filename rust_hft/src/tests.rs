use crate::deribit_sbe::{
    StarbaseBookEvent, StarbaseFeedArbitrator, StarbasePacketStatus, StarbaseSbeMessageHeader,
    StarbaseUdpPacketHeader, StarbaseZeroCopyParser, starbase_types,
};
use crate::ring_buffer::make_spsc_channel;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

// Test 10: 16 channels allocate successfully
#[test]
fn test_16_channel_capacity_allocation() {
    let mut arbitrator = StarbaseFeedArbitrator::new();
    for ch in 1i32..=16i32 {
        let hdr = StarbaseUdpPacketHeader {
            send_time: (1000i64).to_le(),
            sequence_num: (1i64).to_le(),
            channel_id: ch.to_le(),
            packet_type: starbase_types::PACKET_TYPE_INCREMENTAL.to_le(),
            message_count: (1u16).to_le(),
        };
        let status = arbitrator.arbitrate_packet_header(&hdr);
        match status {
            StarbasePacketStatus::Ok { message_count } => assert_eq!(message_count, 1),
            _ => panic!("Expected Ok for channel {}, got {:?}", ch, status),
        }
    }
}

// Test 11: 17th channel triggers explicit capacity exhaustion
#[test]
fn test_17th_channel_capacity_exhausted() {
    let mut arbitrator = StarbaseFeedArbitrator::new();
    for ch in 1i32..=16i32 {
        let hdr = StarbaseUdpPacketHeader {
            send_time: (1000i64).to_le(),
            sequence_num: (1i64).to_le(),
            channel_id: ch.to_le(),
            packet_type: starbase_types::PACKET_TYPE_INCREMENTAL.to_le(),
            message_count: (1u16).to_le(),
        };
        let _ = arbitrator.arbitrate_packet_header(&hdr);
    }

    // 17th distinct channel must be rejected explicitly
    let hdr17 = StarbaseUdpPacketHeader {
        send_time: (1000i64).to_le(),
        sequence_num: (1i64).to_le(),
        channel_id: 17i32.to_le(),
        packet_type: starbase_types::PACKET_TYPE_INCREMENTAL.to_le(),
        message_count: (1u16).to_le(),
    };
    let status = arbitrator.arbitrate_packet_header(&hdr17);
    match status {
        StarbasePacketStatus::ChannelCapacityExhausted => {}
        _ => panic!(
            "Expected ChannelCapacityExhausted for 17th channel, got {:?}",
            status
        ),
    }
}

// Test 12: Independent per-channel sequence numbers
#[test]
fn test_independent_per_channel_sequences() {
    let mut arbitrator = StarbaseFeedArbitrator::new();

    // Feed channel 1
    let hdr1 = StarbaseUdpPacketHeader {
        send_time: (1000i64).to_le(),
        sequence_num: (500i64).to_le(),
        channel_id: (1i32).to_le(),
        packet_type: starbase_types::PACKET_TYPE_INCREMENTAL.to_le(),
        message_count: (10u16).to_le(),
    };
    let s1 = arbitrator.arbitrate_packet_header(&hdr1);
    assert!(matches!(s1, StarbasePacketStatus::Ok { message_count: 10 }));
    assert_eq!(arbitrator.get_channel_expected_seq(1), Some(510));

    // Feed channel 2 with independent sequence
    let hdr2 = StarbaseUdpPacketHeader {
        send_time: (1000i64).to_le(),
        sequence_num: (9000i64).to_le(),
        channel_id: (2i32).to_le(),
        packet_type: starbase_types::PACKET_TYPE_INCREMENTAL.to_le(),
        message_count: (5u16).to_le(),
    };
    let s2 = arbitrator.arbitrate_packet_header(&hdr2);
    assert!(matches!(s2, StarbasePacketStatus::Ok { message_count: 5 }));
    assert_eq!(arbitrator.get_channel_expected_seq(2), Some(9005));

    // Verify channel 1 is still at 510
    assert_eq!(arbitrator.get_channel_expected_seq(1), Some(510));
}

// Test 13: Sequence gap detection and book marked stale
#[test]
fn test_sequence_gap_detection_and_stale_flag() {
    let mut arbitrator = StarbaseFeedArbitrator::new();
    let ch = 5i32;

    let hdr1 = StarbaseUdpPacketHeader {
        send_time: (1000i64).to_le(),
        sequence_num: (100i64).to_le(),
        channel_id: (ch).to_le(),
        packet_type: starbase_types::PACKET_TYPE_INCREMENTAL.to_le(),
        message_count: (5u16).to_le(),
    };
    let _ = arbitrator.arbitrate_packet_header(&hdr1);
    assert!(!arbitrator.is_channel_stale(ch));

    // Jump ahead: expected 105, received 110 (missing 105..110)
    let hdr_gap = StarbaseUdpPacketHeader {
        send_time: (2000i64).to_le(),
        sequence_num: (110i64).to_le(),
        channel_id: (ch).to_le(),
        packet_type: starbase_types::PACKET_TYPE_INCREMENTAL.to_le(),
        message_count: (2u16).to_le(),
    };
    let s = arbitrator.arbitrate_packet_header(&hdr_gap);
    match s {
        StarbasePacketStatus::SequenceGapDetected { expected, received } => {
            assert_eq!(expected, 105);
            assert_eq!(received, 110);
        }
        _ => panic!("Expected SequenceGapDetected, got {:?}", s),
    }
    assert!(arbitrator.is_channel_stale(ch));
}

// Test 14: Duplicate packet rejection
#[test]
fn test_duplicate_packet_rejection() {
    let mut arbitrator = StarbaseFeedArbitrator::new();
    let ch = 8i32;

    let hdr1 = StarbaseUdpPacketHeader {
        send_time: (1000i64).to_le(),
        sequence_num: (100i64).to_le(),
        channel_id: (ch).to_le(),
        packet_type: starbase_types::PACKET_TYPE_INCREMENTAL.to_le(),
        message_count: (10u16).to_le(),
    };
    let _ = arbitrator.arbitrate_packet_header(&hdr1);

    // Send packet with sequence < 110
    let hdr_dup = StarbaseUdpPacketHeader {
        send_time: (1050i64).to_le(),
        sequence_num: (105i64).to_le(),
        channel_id: (ch).to_le(),
        packet_type: starbase_types::PACKET_TYPE_INCREMENTAL.to_le(),
        message_count: (2u16).to_le(),
    };
    let s = arbitrator.arbitrate_packet_header(&hdr_dup);
    assert!(matches!(s, StarbasePacketStatus::DuplicatePacket));
}

// Test 15: Verified retransmission recovers stale book
#[test]
fn test_verified_retransmission_recovery() {
    let mut arbitrator = StarbaseFeedArbitrator::new();
    let ch = 3i32;

    let hdr1 = StarbaseUdpPacketHeader {
        send_time: (1000i64).to_le(),
        sequence_num: (50i64).to_le(),
        channel_id: (ch).to_le(),
        packet_type: starbase_types::PACKET_TYPE_INCREMENTAL.to_le(),
        message_count: (10u16).to_le(),
    };
    let _ = arbitrator.arbitrate_packet_header(&hdr1);

    // Gap: expected 60, receives 70
    let hdr_gap = StarbaseUdpPacketHeader {
        send_time: (2000i64).to_le(),
        sequence_num: (70i64).to_le(),
        channel_id: (ch).to_le(),
        packet_type: starbase_types::PACKET_TYPE_INCREMENTAL.to_le(),
        message_count: (5u16).to_le(),
    };
    let _ = arbitrator.arbitrate_packet_header(&hdr_gap);
    assert!(arbitrator.is_channel_stale(ch));

    // Insufficient retransmit (only covers 60..65)
    let recovered_partial = arbitrator.process_verified_retransmit(ch, 60, 5);
    assert!(!recovered_partial);
    assert!(arbitrator.is_channel_stale(ch));

    // Full retransmission covering 60..70 (count 10)
    let recovered_full = arbitrator.process_verified_retransmit(ch, 60, 10);
    assert!(recovered_full);
    assert!(!arbitrator.is_channel_stale(ch));
}

// Test 16: Zero-copy SBE parsing with valid schema version 1
#[test]
fn test_sbe_parsing_valid_version_1() {
    let h2_size = size_of::<StarbaseSbeMessageHeader>();
    let b_size = size_of::<StarbaseBookEvent>();
    let total_len = (h2_size + b_size) as u16;

    let hdr = StarbaseSbeMessageHeader {
        message_length: total_len.to_le(),
        message_type_id: starbase_types::MSG_BOOK_ORDER_UPDATE.to_le(),
        version: starbase_types::EXPECTED_SCHEMA_VERSION.to_le(),
        flags: 0,
        transact_time: (1_700_000_000i64).to_le(),
    };
    let book = StarbaseBookEvent {
        instrument_id: (99i64).to_le(),
        price: (500_000_000i64).to_le(),
        amount: (10_0000i64).to_le(),
        side: 0,
        action: 1,
        reserved: 0,
    };

    let mut buf = vec![0u8; total_len as usize];
    unsafe {
        let p_hdr = std::slice::from_raw_parts(&hdr as *const _ as *const u8, h2_size);
        let p_book = std::slice::from_raw_parts(&book as *const _ as *const u8, b_size);
        buf[..h2_size].copy_from_slice(p_hdr);
        buf[h2_size..h2_size + b_size].copy_from_slice(p_book);
    }

    let parsed = StarbaseZeroCopyParser::parse_next_sbe_message(&buf);
    assert!(parsed.is_some());
    let (p_hdr, p_book, remaining) = parsed.unwrap();
    assert_eq!(u16::from_le(p_hdr.version), 1);
    assert_eq!(i64::from_le(p_book.instrument_id), 99);
    assert_eq!(remaining.len(), 0);
}

// Test 17: SBE parsing rejects unsupported schema version
#[test]
fn test_sbe_parsing_rejects_invalid_schema_version() {
    let h2_size = size_of::<StarbaseSbeMessageHeader>();
    let b_size = size_of::<StarbaseBookEvent>();
    let total_len = (h2_size + b_size) as u16;

    let hdr = StarbaseSbeMessageHeader {
        message_length: total_len.to_le(),
        message_type_id: starbase_types::MSG_BOOK_ORDER_UPDATE.to_le(),
        version: (2u16).to_le(), // Unsupported version 2!
        flags: 0,
        transact_time: (1_700_000_000i64).to_le(),
    };
    let book = StarbaseBookEvent {
        instrument_id: (99i64).to_le(),
        price: (500_000_000i64).to_le(),
        amount: (10_0000i64).to_le(),
        side: 0,
        action: 1,
        reserved: 0,
    };

    let mut buf = vec![0u8; total_len as usize];
    unsafe {
        let p_hdr = std::slice::from_raw_parts(&hdr as *const _ as *const u8, h2_size);
        let p_book = std::slice::from_raw_parts(&book as *const _ as *const u8, b_size);
        buf[..h2_size].copy_from_slice(p_hdr);
        buf[h2_size..h2_size + b_size].copy_from_slice(p_book);
    }

    let parsed = StarbaseZeroCopyParser::parse_next_sbe_message(&buf);
    assert!(parsed.is_none(), "Must reject schema version != 1");
}

// Test 18: SBE parsing rejects unknown template ID
#[test]
fn test_sbe_parsing_rejects_unknown_template_id() {
    let h2_size = size_of::<StarbaseSbeMessageHeader>();
    let b_size = size_of::<StarbaseBookEvent>();
    let total_len = (h2_size + b_size) as u16;

    let hdr = StarbaseSbeMessageHeader {
        message_length: total_len.to_le(),
        message_type_id: (999u16).to_le(), // Invalid template ID
        version: (1u16).to_le(),
        flags: 0,
        transact_time: (1_700_000_000i64).to_le(),
    };
    let book = StarbaseBookEvent {
        instrument_id: (99i64).to_le(),
        price: (500_000_000i64).to_le(),
        amount: (10_0000i64).to_le(),
        side: 0,
        action: 1,
        reserved: 0,
    };

    let mut buf = vec![0u8; total_len as usize];
    unsafe {
        let p_hdr = std::slice::from_raw_parts(&hdr as *const _ as *const u8, h2_size);
        let p_book = std::slice::from_raw_parts(&book as *const _ as *const u8, b_size);
        buf[..h2_size].copy_from_slice(p_hdr);
        buf[h2_size..h2_size + b_size].copy_from_slice(p_book);
    }

    let parsed = StarbaseZeroCopyParser::parse_next_sbe_message(&buf);
    assert!(parsed.is_none(), "Must reject unknown template ID");
}

// Test 19: Lock-free SPSC queue concurrent transfer
#[test]
fn test_spsc_concurrent_transfer() {
    const ITEMS: usize = 100_000;
    let (mut producer, mut consumer) = make_spsc_channel::<u64, 1024>();
    let running = Arc::new(AtomicBool::new(true));

    let r_flag = Arc::clone(&running);
    let prod_thread = thread::spawn(move || {
        for i in 0..ITEMS as u64 {
            while producer.try_push(i).is_err() {
                std::hint::spin_loop();
            }
        }
        r_flag.store(false, Ordering::Release);
    });

    let cons_thread = thread::spawn(move || {
        let mut received = Vec::with_capacity(ITEMS);
        loop {
            match consumer.try_pop() {
                Some(val) => {
                    received.push(val);
                    if received.len() == ITEMS {
                        break;
                    }
                }
                None => {
                    if !running.load(Ordering::Acquire) && received.len() == ITEMS {
                        break;
                    }
                    std::hint::spin_loop();
                }
            }
        }
        received
    });

    prod_thread.join().unwrap();
    let received = cons_thread.join().unwrap();
    assert_eq!(received.len(), ITEMS);
    for (i, val) in received.iter().enumerate() {
        assert_eq!(*val, i as u64);
    }
}

// Test 20: SPSC queue full behavior returns item
#[test]
fn test_spsc_queue_full_behavior() {
    let (mut producer, mut consumer) = make_spsc_channel::<u32, 4>();
    assert!(producer.try_push(1).is_ok());
    assert!(producer.try_push(2).is_ok());
    assert!(producer.try_push(3).is_ok());
    assert!(producer.try_push(4).is_ok());

    // Queue is full (capacity 4)
    let push_result = producer.try_push(5);
    assert!(push_result.is_err());
    assert_eq!(push_result.err(), Some(5));

    // Pop one and push again
    assert_eq!(consumer.try_pop(), Some(1));
    assert!(producer.try_push(5).is_ok());
}

// Test 21: SPSC queue shutdown and complete draining
#[test]
fn test_spsc_queue_shutdown_draining() {
    let (mut producer, mut consumer) = make_spsc_channel::<u32, 16>();
    for i in 1..=10 {
        assert!(producer.try_push(i).is_ok());
    }

    // Simulate producer shutdown; consumer drains remaining items
    let mut drained = Vec::new();
    while let Some(item) = consumer.try_pop() {
        drained.push(item);
    }
    assert_eq!(drained, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    assert_eq!(consumer.try_pop(), None);
}

// Test 22: Starbase TCP new order serialization matches specification
#[test]
fn test_starbase_tcp_new_order_serialization() {
    let mut out_buf = [0u8; 128];
    let bytes_written = StarbaseZeroCopyParser::serialize_tcp_new_order(
        1001,
        500,
        1_700_000_000_000_000_000,
        9999,
        42,
        650_000_000,
        100_0000,
        0,
        &mut out_buf,
    );

    assert_eq!(bytes_written, 68);
    assert_eq!(out_buf[0], starbase_types::PROTOCOL_ID_DERIBIT);
}

// Test 23: Snapshot fallback resets stale book and resynchronizes sequence
#[test]
fn test_snapshot_fallback_resynchronization() {
    let mut arbitrator = StarbaseFeedArbitrator::new();
    let ch = 4i32;

    let hdr1 = StarbaseUdpPacketHeader {
        send_time: (1000i64).to_le(),
        sequence_num: (100i64).to_le(),
        channel_id: (ch).to_le(),
        packet_type: starbase_types::PACKET_TYPE_INCREMENTAL.to_le(),
        message_count: (10u16).to_le(),
    };
    let _ = arbitrator.arbitrate_packet_header(&hdr1);

    // Unrecoverable gap occurs
    let hdr_gap = StarbaseUdpPacketHeader {
        send_time: (2000i64).to_le(),
        sequence_num: (250i64).to_le(),
        channel_id: (ch).to_le(),
        packet_type: starbase_types::PACKET_TYPE_INCREMENTAL.to_le(),
        message_count: (5u16).to_le(),
    };
    let _ = arbitrator.arbitrate_packet_header(&hdr_gap);
    assert!(arbitrator.is_channel_stale(ch));

    // Authoritative snapshot applied (seq 250 with 50 messages)
    let applied = arbitrator.apply_authoritative_snapshot(ch, 250, 50);
    assert!(applied);
    assert!(!arbitrator.is_channel_stale(ch));
    assert_eq!(arbitrator.get_channel_expected_seq(ch), Some(300));
}
