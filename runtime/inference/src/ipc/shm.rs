//! Shared memory ring buffer for zero-copy token streaming.
//!
//! This module implements a lock-free SPSC (single-producer, single-consumer)
//! ring buffer backed by `/dev/shm`. Each inference session gets its own
//! ring buffer for token transfer.
//!
//! # Layout
//!
//! ```text
//! +──────────────────────────────────────────+
//! │  Header (64 bytes, cache-line aligned)   │
//! │  ├─ write_head: AtomicU64 (producer)     │
//! │  ├─ read_head:  AtomicU64 (consumer)     │
//! │  ├─ capacity:   u64                      │
//! │  └─ _padding                             │
//! +──────────────────────────────────────────+
//! │  TokenEntry[0]  (16 bytes)               │
//! │  TokenEntry[1]  (16 bytes)               │
//! │  ...                                     │
//! │  TokenEntry[capacity-1]                  │
//! +──────────────────────────────────────────+
//! ```
//!
//! # Zero-allocation hot path
//!
//! The producer writes `TokenEntry` structs directly into the shared memory
//! region. The consumer reads them without any copying or allocation.
//! Synchronization uses `Acquire`/`Release` ordering on the head pointers.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::types::TokenEntry;

/// Size of the ring buffer header in bytes.
/// Aligned to 64 bytes (cache line) to prevent false sharing.
pub const HEADER_SIZE: usize = 64;

/// Header of the shared memory ring buffer.
///
/// Must be exactly 64 bytes (one cache line) to prevent false sharing
/// between the producer's write_head and the consumer's read_head.
#[repr(C, align(64))]
pub struct RingHeader {
    /// Write position (updated by producer).
    pub write_head: AtomicU64,
    /// Read position (updated by consumer).
    pub read_head: AtomicU64,
    /// Number of token slots (must be power of 2).
    pub capacity: u64,
    /// Padding to fill the cache line.
    _pad: [u8; 40],
}

const _: () = assert!(std::mem::size_of::<RingHeader>() == HEADER_SIZE);

/// Producer side of the ring buffer (writes tokens).
///
/// Used by the inference coordinator to push generated tokens into
/// shared memory for zero-copy consumption.
pub struct RingProducer {
    /// Pointer to the shared memory header.
    header: *mut RingHeader,
    /// Pointer to the token data region.
    data: *mut TokenEntry,
    /// Capacity (cached from header for performance).
    capacity: u64,
    /// Bitmask for fast modulo (capacity - 1).
    mask: u64,
}

// SAFETY: The shared memory region is process-local and accessed via atomics.
unsafe impl Send for RingProducer {}

impl RingProducer {
    /// Create a producer from a raw shared memory pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `ptr` points to a valid shared memory region of at least
    ///   `HEADER_SIZE + capacity * size_of::<TokenEntry>()` bytes.
    /// - `capacity` is a power of 2.
    /// - The header has been initialized (capacity field set).
    /// - Only one producer exists for this buffer.
    pub unsafe fn from_raw(ptr: *mut u8, capacity: u64) -> Self {
        debug_assert!(capacity.is_power_of_two());
        let header = ptr.cast::<RingHeader>();
        let data = ptr.add(HEADER_SIZE).cast::<TokenEntry>();

        // Initialize the header
        (*header).write_head = AtomicU64::new(0);
        (*header).read_head = AtomicU64::new(0);
        (*header).capacity = capacity;

        Self {
            header,
            data,
            capacity,
            mask: capacity - 1,
        }
    }

    /// Try to write a token entry into the ring buffer.
    ///
    /// Returns `true` if the write succeeded, `false` if the buffer is full.
    /// This is a non-blocking operation — never allocates.
    pub fn try_push(&self, entry: &TokenEntry) -> bool {
        unsafe {
            let write = (*self.header).write_head.load(Ordering::Relaxed);
            let read = (*self.header).read_head.load(Ordering::Acquire);

            // Buffer is full when write is capacity ahead of read
            if write.wrapping_sub(read) >= self.capacity {
                return false;
            }

            let idx = (write & self.mask) as usize;
            std::ptr::write(self.data.add(idx), *entry);

            // Release: ensures the data write is visible before head update
            (*self.header)
                .write_head
                .store(write.wrapping_add(1), Ordering::Release);
            true
        }
    }

    /// Number of entries currently in the buffer.
    #[must_use]
    pub fn len(&self) -> u64 {
        unsafe {
            let write = (*self.header).write_head.load(Ordering::Relaxed);
            let read = (*self.header).read_head.load(Ordering::Relaxed);
            write.wrapping_sub(read)
        }
    }

    /// Check if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Consumer side of the ring buffer (reads tokens).
///
/// Used by local clients to read generated tokens from shared memory
/// without any copying or system calls.
pub struct RingConsumer {
    /// Pointer to the shared memory header.
    header: *const RingHeader,
    /// Pointer to the token data region.
    data: *const TokenEntry,
    /// Bitmask for fast modulo.
    mask: u64,
}

// SAFETY: Accessed via atomics, read-only except for read_head.
unsafe impl Send for RingConsumer {}

impl RingConsumer {
    /// Create a consumer from a raw shared memory pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `ptr` points to a valid, initialized shared memory region.
    /// - `capacity` matches the producer's capacity.
    /// - Only one consumer exists for this buffer.
    pub unsafe fn from_raw(ptr: *const u8, capacity: u64) -> Self {
        debug_assert!(capacity.is_power_of_two());
        let header = ptr.cast::<RingHeader>();
        let data = ptr.add(HEADER_SIZE).cast::<TokenEntry>();

        Self {
            header,
            data,
            mask: capacity - 1,
        }
    }

    /// Try to read a token entry from the ring buffer.
    ///
    /// Returns `Some(entry)` if data is available, `None` if empty.
    /// This is a non-blocking operation — never allocates.
    pub fn try_pop(&self) -> Option<TokenEntry> {
        unsafe {
            let read = (*(self.header as *const RingHeader))
                .read_head
                .load(Ordering::Relaxed);
            let write = (*(self.header as *const RingHeader))
                .write_head
                .load(Ordering::Acquire);

            if read == write {
                return None;
            }

            let idx = (read & self.mask) as usize;
            let entry = std::ptr::read(self.data.add(idx));

            // Release: ensures we've read the data before advancing
            (*(self.header as *mut RingHeader))
                .read_head
                .store(read.wrapping_add(1), Ordering::Release);
            Some(entry)
        }
    }

    /// Number of entries available to read.
    #[must_use]
    pub fn available(&self) -> u64 {
        unsafe {
            let write = (*self.header).write_head.load(Ordering::Acquire);
            let read = (*self.header).read_head.load(Ordering::Relaxed);
            write.wrapping_sub(read)
        }
    }

    /// Check if there are entries to read.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.available() == 0
    }
}

/// Calculate the total shared memory size needed for a ring buffer.
#[must_use]
pub fn shm_size(capacity: usize) -> usize {
    HEADER_SIZE + capacity * std::mem::size_of::<TokenEntry>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TokenFlag;

    /// Allocate a 64-byte-aligned buffer for ring buffer tests.
    /// `RingHeader` is `#[repr(C, align(64))]` so the backing memory
    /// must satisfy that alignment.
    fn alloc_test_buffer(capacity: u64) -> (*mut u8, std::alloc::Layout) {
        let size = shm_size(capacity as usize);
        let layout = std::alloc::Layout::from_size_align(size, 64).expect("valid layout");
        // SAFETY: layout has non-zero size.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "allocation failed");
        (ptr, layout)
    }

    /// Helper: deallocate a test buffer.
    unsafe fn dealloc_test_buffer(ptr: *mut u8, layout: std::alloc::Layout) {
        std::alloc::dealloc(ptr, layout);
    }

    #[test]
    fn test_ring_buffer_push_pop() {
        let (ptr, layout) = alloc_test_buffer(8);

        unsafe {
            let producer = RingProducer::from_raw(ptr, 8);
            let consumer = RingConsumer::from_raw(ptr, 8);

            let entry = TokenEntry {
                token_id: 42,
                logprob: -0.5,
                flags: TokenFlag::Normal as u32,
                _pad: 0,
            };

            assert!(producer.try_push(&entry));
            assert_eq!(producer.len(), 1);

            let read = consumer.try_pop().unwrap();
            assert_eq!(read.token_id, 42);
            assert!(consumer.is_empty());

            dealloc_test_buffer(ptr, layout);
        }
    }

    #[test]
    fn test_ring_buffer_full() {
        let (ptr, layout) = alloc_test_buffer(4);

        unsafe {
            let producer = RingProducer::from_raw(ptr, 4);

            let entry = TokenEntry {
                token_id: 1,
                logprob: 0.0,
                flags: 0,
                _pad: 0,
            };

            for _ in 0..4 {
                assert!(producer.try_push(&entry));
            }

            assert!(!producer.try_push(&entry));

            dealloc_test_buffer(ptr, layout);
        }
    }

    #[test]
    fn test_ring_buffer_empty_pop() {
        let (ptr, layout) = alloc_test_buffer(4);

        unsafe {
            let _producer = RingProducer::from_raw(ptr, 4);
            let consumer = RingConsumer::from_raw(ptr, 4);

            assert!(consumer.try_pop().is_none());
            assert!(consumer.is_empty());

            dealloc_test_buffer(ptr, layout);
        }
    }

    #[test]
    fn test_ring_buffer_wraparound() {
        let (ptr, layout) = alloc_test_buffer(4);

        unsafe {
            let producer = RingProducer::from_raw(ptr, 4);
            let consumer = RingConsumer::from_raw(ptr, 4);

            for i in 0..10u32 {
                let entry = TokenEntry {
                    token_id: i,
                    logprob: 0.0,
                    flags: 0,
                    _pad: 0,
                };
                assert!(producer.try_push(&entry));
                let read = consumer.try_pop().unwrap();
                assert_eq!(read.token_id, i);
            }

            dealloc_test_buffer(ptr, layout);
        }
    }

    #[test]
    fn test_ring_buffer_batch() {
        let (ptr, layout) = alloc_test_buffer(8);

        unsafe {
            let producer = RingProducer::from_raw(ptr, 8);
            let consumer = RingConsumer::from_raw(ptr, 8);

            for i in 0..5u32 {
                let entry = TokenEntry {
                    token_id: i,
                    logprob: -(i as f32),
                    flags: if i == 4 {
                        TokenFlag::Eos as u32
                    } else {
                        TokenFlag::Normal as u32
                    },
                    _pad: 0,
                };
                assert!(producer.try_push(&entry));
            }

            assert_eq!(consumer.available(), 5);

            for i in 0..5u32 {
                let entry = consumer.try_pop().unwrap();
                assert_eq!(entry.token_id, i);
            }

            assert!(consumer.is_empty());

            dealloc_test_buffer(ptr, layout);
        }
    }

    #[test]
    fn test_shm_size_calculation() {
        let size = shm_size(1024);
        assert_eq!(size, HEADER_SIZE + 1024 * std::mem::size_of::<TokenEntry>());
    }

    #[test]
    fn test_header_size_is_cache_line() {
        assert_eq!(HEADER_SIZE, 64);
        assert_eq!(std::mem::size_of::<RingHeader>(), 64);
    }
}
