//! Safe Lock-Free Single-Producer Single-Consumer (SPSC) Ring Buffer
//!
//! Corrected Safety Contract:
//! - Implements split `Producer` and `Consumer` endpoint handles.
//! - The internal ring buffer cannot be cloned or shared arbitrarily via `Arc`.
//! - Only the single `Producer` handle can call `try_push`.
//! - Only the single `Consumer` handle can call `try_pop`.
//! - Reader and writer indices are cache-line aligned (`#[repr(align(64))]`)
//!   preventing false sharing and MOESI/MESI invalidation storms.

use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub const CACHE_LINE_BYTES: usize = 64;

#[repr(align(64))]
struct PaddedAtomicUsize {
    value: AtomicUsize,
    _pad: [u8; CACHE_LINE_BYTES - size_of::<AtomicUsize>()],
}

impl PaddedAtomicUsize {
    const fn new(val: usize) -> Self {
        Self {
            value: AtomicUsize::new(val),
            _pad: [0u8; CACHE_LINE_BYTES - size_of::<AtomicUsize>()],
        }
    }
}

struct InnerRing<T: Copy, const CAPACITY: usize> {
    buffer: Box<[UnsafeCell<T>; CAPACITY]>,
    head: PaddedAtomicUsize, // Written only by Producer
    tail: PaddedAtomicUsize, // Written only by Consumer
}

unsafe impl<T: Copy + Send, const CAPACITY: usize> Send for InnerRing<T, CAPACITY> {}
unsafe impl<T: Copy + Send, const CAPACITY: usize> Sync for InnerRing<T, CAPACITY> {}

/// Producer endpoint: exclusively owns push capability
pub struct Producer<T: Copy + Send + 'static, const CAPACITY: usize> {
    inner: Arc<InnerRing<T, CAPACITY>>,
}

unsafe impl<T: Copy + Send, const CAPACITY: usize> Send for Producer<T, CAPACITY> {}

impl<T: Copy + Send + 'static, const CAPACITY: usize> Producer<T, CAPACITY> {
    const MASK: usize = CAPACITY - 1;

    #[inline(always)]
    pub fn try_push(&mut self, item: T) -> Result<(), T> {
        let current_head = self.inner.head.value.load(Ordering::Relaxed);
        let current_tail = self.inner.tail.value.load(Ordering::Acquire);

        if current_head.wrapping_sub(current_tail) >= CAPACITY {
            return Err(item); // Buffer full
        }

        let slot_idx = current_head & Self::MASK;
        unsafe {
            let slot = self.inner.buffer[slot_idx].get();
            *slot = item;
        }

        self.inner
            .head
            .value
            .store(current_head.wrapping_add(1), Ordering::Release);
        Ok(())
    }
}

/// Consumer endpoint: exclusively owns pop capability
pub struct Consumer<T: Copy + Send + 'static, const CAPACITY: usize> {
    inner: Arc<InnerRing<T, CAPACITY>>,
}

unsafe impl<T: Copy + Send, const CAPACITY: usize> Send for Consumer<T, CAPACITY> {}

impl<T: Copy + Send + 'static, const CAPACITY: usize> Consumer<T, CAPACITY> {
    const MASK: usize = CAPACITY - 1;

    #[inline(always)]
    pub fn try_pop(&mut self) -> Option<T> {
        let current_tail = self.inner.tail.value.load(Ordering::Relaxed);
        let current_head = self.inner.head.value.load(Ordering::Acquire);

        if current_head == current_tail {
            return None; // Buffer empty
        }

        let slot_idx = current_tail & Self::MASK;
        let item = unsafe { *self.inner.buffer[slot_idx].get() };

        self.inner
            .tail
            .value
            .store(current_tail.wrapping_add(1), Ordering::Release);
        Some(item)
    }
}

/// Creates a split Producer and Consumer pair guaranteeing SPSC concurrency safety.
pub fn make_spsc_channel<T: Copy + Send + Default + 'static, const CAPACITY: usize>()
-> (Producer<T, CAPACITY>, Consumer<T, CAPACITY>) {
    assert!(
        CAPACITY.is_power_of_two(),
        "Capacity must be a power of two for bitwise index masking"
    );

    let storage: Box<[UnsafeCell<T>; CAPACITY]> =
        Box::new(std::array::from_fn(|_| UnsafeCell::new(T::default())));

    let inner = Arc::new(InnerRing {
        buffer: storage,
        head: PaddedAtomicUsize::new(0),
        tail: PaddedAtomicUsize::new(0),
    });

    (
        Producer {
            inner: Arc::clone(&inner),
        },
        Consumer { inner },
    )
}
