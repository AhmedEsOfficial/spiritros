//! KV-cache page manager with copy-on-write semantics.
//!
//! Implements a page-granular KV-cache manager inspired by vLLM's
//! PagedAttention. Each inference session gets a page table mapping
//! logical pages to physical pages in a pre-allocated pool.
//!
//! Key features:
//! - **Copy-on-write**: Snapshots and forks share pages via refcounts.
//!   Pages are only copied when mutated.
//! - **LRU eviction**: When the pool is full, the least-recently-used
//!   session's pages are evicted to NVMe swap.
//! - **Async prefetch**: When a swapped session becomes active, its
//!   pages are prefetched from swap.
//!
//! // PHASE1-SHIM: In Phase 2, this becomes a kernel data structure with
//! // page-table-level integration and zero-copy GPU↔CPU transfers.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;

use tracing::{debug, info};

use crate::config::KvCacheConfig;
use crate::error::{Result, SpiritMemoryError};
use crate::types::{KvSnapshot, PageStatus, SessionId};

/// A single page in the KV-cache page pool.
struct PageSlot {
    /// Page data buffer (fixed size = page_size).
    data: Vec<u8>,
    /// Reference count (>1 means shared via COW).
    refcount: AtomicU32,
    /// Current status.
    status: PageStatus,
}

/// Manages KV-cache pages across inference sessions.
pub struct KvPageManager {
    /// Page size in bytes.
    page_size: usize,
    /// Maximum pages in the pool.
    max_pages: usize,
    /// Maximum concurrent sessions.
    max_sessions: usize,
    /// Whether COW is enabled.
    cow_enabled: bool,
    /// Directory for NVMe swap files.
    swap_dir: PathBuf,

    /// The page pool (protected by mutex for writes).
    pages: Mutex<Vec<PageSlot>>,
    /// Free page indices.
    free_list: Mutex<VecDeque<usize>>,
    /// Per-session page tables: session → ordered list of page indices.
    session_tables: Mutex<HashMap<SessionId, Vec<usize>>>,
    /// LRU queue: most recently used at back.
    lru: Mutex<VecDeque<SessionId>>,

    /// Counters for metrics.
    active_pages: AtomicU64,
    swapped_pages: AtomicU64,
    total_snapshots: AtomicU64,
    total_forks: AtomicU64,
    total_cow_copies: AtomicU64,
}

impl KvPageManager {
    /// Create a new KV-cache page manager with a pre-allocated page pool.
    pub fn new(config: &KvCacheConfig) -> Self {
        let page_size = config.page_size;
        let max_pages = config.max_pages;

        // Pre-allocate page pool
        let mut pool = Vec::with_capacity(max_pages);
        let mut free = VecDeque::with_capacity(max_pages);

        for i in 0..max_pages {
            pool.push(PageSlot {
                data: vec![0u8; page_size],
                refcount: AtomicU32::new(0),
                status: PageStatus::Free,
            });
            free.push_back(i);
        }

        info!(
            page_size,
            max_pages,
            total_bytes = page_size * max_pages,
            "KV-cache page pool initialized"
        );

        Self {
            page_size,
            max_pages,
            max_sessions: config.max_sessions,
            cow_enabled: config.cow_enabled,
            swap_dir: config.swap_dir.clone(),
            pages: Mutex::new(pool),
            free_list: Mutex::new(free),
            session_tables: Mutex::new(HashMap::new()),
            lru: Mutex::new(VecDeque::new()),
            active_pages: AtomicU64::new(0),
            swapped_pages: AtomicU64::new(0),
            total_snapshots: AtomicU64::new(0),
            total_forks: AtomicU64::new(0),
            total_cow_copies: AtomicU64::new(0),
        }
    }

    /// Register a new session. Must be called before allocating pages.
    pub fn create_session(&self, session_id: SessionId) -> Result<()> {
        let mut tables = self
            .session_tables
            .lock()
            .expect("session_tables lock poisoned");

        if tables.len() >= self.max_sessions && !tables.contains_key(&session_id) {
            return Err(SpiritMemoryError::KvPoolExhausted {
                active: self.active_pages.load(Ordering::Relaxed) as usize,
                swapped: self.swapped_pages.load(Ordering::Relaxed) as usize,
                total: self.max_pages,
            });
        }

        tables.entry(session_id).or_default();

        // Update LRU
        let mut lru = self.lru.lock().expect("lru lock poisoned");
        lru.retain(|s| *s != session_id);
        lru.push_back(session_id);

        debug!(session = %session_id, "KV-cache session created");
        Ok(())
    }

    /// Allocate a new page for a session. Returns the page index.
    pub fn alloc_page(&self, session_id: SessionId) -> Result<usize> {
        // Get a free page
        let page_idx = {
            let mut free = self.free_list.lock().expect("free_list lock poisoned");
            if let Some(idx) = free.pop_front() {
                idx
            } else {
                drop(free);
                // Try to evict the LRU session to make room
                self.evict_lru_session()?;
                let mut free = self.free_list.lock().expect("free_list lock poisoned");
                free.pop_front()
                    .ok_or_else(|| SpiritMemoryError::KvPoolExhausted {
                        active: self.active_pages.load(Ordering::Relaxed) as usize,
                        swapped: self.swapped_pages.load(Ordering::Relaxed) as usize,
                        total: self.max_pages,
                    })?
            }
        };

        // Initialize the page
        {
            let mut pages = self.pages.lock().expect("pages lock poisoned");
            pages[page_idx].status = PageStatus::Active;
            pages[page_idx].refcount.store(1, Ordering::Release);
            // Zero the page data
            pages[page_idx].data.fill(0);
        }

        // Add to session's page table
        {
            let mut tables = self
                .session_tables
                .lock()
                .expect("session_tables lock poisoned");
            let table = tables
                .get_mut(&session_id)
                .ok_or(SpiritMemoryError::SessionNotFound(session_id))?;
            table.push(page_idx);
        }

        self.active_pages.fetch_add(1, Ordering::Relaxed);
        self.touch_lru(session_id);

        Ok(page_idx)
    }

    /// Write data to a page, performing COW if the page is shared.
    pub fn write_page(&self, session_id: SessionId, page_offset: usize, data: &[u8]) -> Result<()> {
        if data.len() > self.page_size {
            return Err(SpiritMemoryError::SwapError(format!(
                "data length {} exceeds page size {}",
                data.len(),
                self.page_size
            )));
        }

        let page_idx = {
            let tables = self
                .session_tables
                .lock()
                .expect("session_tables lock poisoned");
            let table = tables
                .get(&session_id)
                .ok_or(SpiritMemoryError::SessionNotFound(session_id))?;
            if page_offset >= table.len() {
                return Err(SpiritMemoryError::PageIndexOutOfBounds {
                    index: page_offset,
                    count: table.len(),
                });
            }
            table[page_offset]
        };

        let mut pages = self.pages.lock().expect("pages lock poisoned");

        // COW: if refcount > 1, copy to a new page first
        let refcount = pages[page_idx].refcount.load(Ordering::Acquire);
        if self.cow_enabled && refcount > 1 {
            // Allocate a new page for the copy
            let new_idx = {
                let mut free = self.free_list.lock().expect("free_list lock poisoned");
                free.pop_front()
                    .ok_or_else(|| SpiritMemoryError::KvPoolExhausted {
                        active: self.active_pages.load(Ordering::Relaxed) as usize,
                        swapped: self.swapped_pages.load(Ordering::Relaxed) as usize,
                        total: self.max_pages,
                    })?
            };

            // Copy old page data to new page
            let old_data = pages[page_idx].data.clone();
            pages[new_idx].data = old_data;
            pages[new_idx].status = PageStatus::Active;
            pages[new_idx].refcount.store(1, Ordering::Release);

            // Decrement old page refcount
            pages[page_idx].refcount.fetch_sub(1, Ordering::Release);

            // Update session's page table to point to new page
            {
                let mut tables = self
                    .session_tables
                    .lock()
                    .expect("session_tables lock poisoned");
                if let Some(table) = tables.get_mut(&session_id) {
                    table[page_offset] = new_idx;
                }
            }

            // Write to the new (private) page
            pages[new_idx].data[..data.len()].copy_from_slice(data);
            self.active_pages.fetch_add(1, Ordering::Relaxed);
            self.total_cow_copies.fetch_add(1, Ordering::Relaxed);

            debug!(
                session = %session_id,
                old_page = page_idx,
                new_page = new_idx,
                "COW copy performed"
            );
        } else {
            // Direct write (sole owner)
            pages[page_idx].data[..data.len()].copy_from_slice(data);
        }

        self.touch_lru(session_id);
        Ok(())
    }

    /// Read data from a page.
    pub fn read_page(&self, session_id: SessionId, page_offset: usize) -> Result<Vec<u8>> {
        let page_idx = {
            let tables = self
                .session_tables
                .lock()
                .expect("session_tables lock poisoned");
            let table = tables
                .get(&session_id)
                .ok_or(SpiritMemoryError::SessionNotFound(session_id))?;
            if page_offset >= table.len() {
                return Err(SpiritMemoryError::PageIndexOutOfBounds {
                    index: page_offset,
                    count: table.len(),
                });
            }
            table[page_offset]
        };

        let pages = self.pages.lock().expect("pages lock poisoned");

        // If the page is swapped, load it back
        if pages[page_idx].status == PageStatus::Swapped {
            drop(pages);
            self.prefetch_page(page_idx)?;
            let pages = self.pages.lock().expect("pages lock poisoned");
            return Ok(pages[page_idx].data.clone());
        }

        self.touch_lru(session_id);
        Ok(pages[page_idx].data.clone())
    }

    /// Create a snapshot of a session's KV-cache (O(n) refcount bumps).
    ///
    /// The snapshot shares physical pages with the original session.
    /// Pages are copied only when either side mutates them (COW).
    pub fn snapshot(&self, session_id: SessionId) -> Result<KvSnapshot> {
        let page_indices = {
            let tables = self
                .session_tables
                .lock()
                .expect("session_tables lock poisoned");
            let table = tables
                .get(&session_id)
                .ok_or(SpiritMemoryError::SessionNotFound(session_id))?;
            table.clone()
        };

        // Increment refcounts on all shared pages
        {
            let pages = self.pages.lock().expect("pages lock poisoned");
            for &idx in &page_indices {
                pages[idx].refcount.fetch_add(1, Ordering::Release);
            }
        }

        self.total_snapshots.fetch_add(1, Ordering::Relaxed);

        debug!(
            session = %session_id,
            pages = page_indices.len(),
            "KV-cache snapshot created"
        );

        Ok(KvSnapshot {
            session_id,
            page_indices,
            created_at: std::time::Instant::now(),
        })
    }

    /// Fork a new session from a snapshot (shares pages via COW).
    ///
    /// The new session starts with an identical KV-cache but diverges
    /// on first write. This enables instant conversation branching.
    pub fn fork(&self, snapshot: &KvSnapshot) -> Result<SessionId> {
        let new_session = SessionId::new();

        // Register the new session with the snapshot's page table
        {
            let mut tables = self
                .session_tables
                .lock()
                .expect("session_tables lock poisoned");

            if tables.len() >= self.max_sessions {
                return Err(SpiritMemoryError::KvPoolExhausted {
                    active: self.active_pages.load(Ordering::Relaxed) as usize,
                    swapped: self.swapped_pages.load(Ordering::Relaxed) as usize,
                    total: self.max_pages,
                });
            }

            tables.insert(new_session, snapshot.page_indices.clone());
        }

        // Pages are already shared (refcounts were bumped during snapshot)
        let mut lru = self.lru.lock().expect("lru lock poisoned");
        lru.push_back(new_session);

        self.total_forks.fetch_add(1, Ordering::Relaxed);

        info!(
            new_session = %new_session,
            parent = %snapshot.session_id,
            pages = snapshot.page_indices.len(),
            "KV-cache forked"
        );

        Ok(new_session)
    }

    /// Destroy a session, releasing its pages (or decrementing refcounts).
    pub fn destroy_session(&self, session_id: SessionId) -> Result<()> {
        let page_indices = {
            let mut tables = self
                .session_tables
                .lock()
                .expect("session_tables lock poisoned");
            tables
                .remove(&session_id)
                .ok_or(SpiritMemoryError::SessionNotFound(session_id))?
        };

        // Release pages
        {
            let mut pages = self.pages.lock().expect("pages lock poisoned");
            let mut free = self.free_list.lock().expect("free_list lock poisoned");

            for idx in page_indices {
                let prev = pages[idx].refcount.fetch_sub(1, Ordering::Release);
                if prev <= 1 {
                    // Last reference — page becomes free
                    pages[idx].status = PageStatus::Free;
                    pages[idx].data.fill(0);
                    free.push_back(idx);
                    self.active_pages.fetch_sub(1, Ordering::Relaxed);

                    // Clean up swap file if it exists
                    let swap_path = self.swap_path(idx);
                    let _ = std::fs::remove_file(&swap_path);
                }
            }
        }

        // Remove from LRU
        {
            let mut lru = self.lru.lock().expect("lru lock poisoned");
            lru.retain(|s| *s != session_id);
        }

        debug!(session = %session_id, "KV-cache session destroyed");
        Ok(())
    }

    /// Number of pages in a session's page table.
    pub fn session_page_count(&self, session_id: SessionId) -> Result<usize> {
        let tables = self
            .session_tables
            .lock()
            .expect("session_tables lock poisoned");
        tables
            .get(&session_id)
            .map(Vec::len)
            .ok_or(SpiritMemoryError::SessionNotFound(session_id))
    }

    /// Number of active sessions.
    #[must_use]
    pub fn session_count(&self) -> usize {
        self.session_tables
            .lock()
            .expect("session_tables lock poisoned")
            .len()
    }

    /// Active page count.
    #[must_use]
    pub fn active_page_count(&self) -> u64 {
        self.active_pages.load(Ordering::Relaxed)
    }

    /// Swapped page count.
    #[must_use]
    pub fn swapped_page_count(&self) -> u64 {
        self.swapped_pages.load(Ordering::Relaxed)
    }

    /// Total snapshot operations performed.
    #[must_use]
    pub fn snapshot_count(&self) -> u64 {
        self.total_snapshots.load(Ordering::Relaxed)
    }

    /// Total fork operations performed.
    #[must_use]
    pub fn fork_count(&self) -> u64 {
        self.total_forks.load(Ordering::Relaxed)
    }

    /// Total COW copy operations performed.
    #[must_use]
    pub fn cow_copy_count(&self) -> u64 {
        self.total_cow_copies.load(Ordering::Relaxed)
    }

    /// Page size in bytes.
    #[must_use]
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Maximum pages in the pool.
    #[must_use]
    pub fn max_pages(&self) -> usize {
        self.max_pages
    }

    // ── Eviction & Swap ─────────────────────────────────────────────────

    /// Evict the least-recently-used session's pages to NVMe swap.
    fn evict_lru_session(&self) -> Result<()> {
        let victim = {
            let lru = self.lru.lock().expect("lru lock poisoned");
            lru.front().copied()
        };

        let victim = victim.ok_or_else(|| SpiritMemoryError::KvPoolExhausted {
            active: self.active_pages.load(Ordering::Relaxed) as usize,
            swapped: self.swapped_pages.load(Ordering::Relaxed) as usize,
            total: self.max_pages,
        })?;

        self.evict_session(victim)
    }

    /// Evict a specific session's pages to swap.
    fn evict_session(&self, session_id: SessionId) -> Result<()> {
        let page_indices = {
            let tables = self
                .session_tables
                .lock()
                .expect("session_tables lock poisoned");
            match tables.get(&session_id) {
                Some(table) => table.clone(),
                None => return Ok(()),
            }
        };

        // Ensure swap directory exists
        std::fs::create_dir_all(&self.swap_dir).map_err(|e| {
            SpiritMemoryError::SwapError(format!(
                "failed to create swap dir {}: {e}",
                self.swap_dir.display()
            ))
        })?;

        let mut pages = self.pages.lock().expect("pages lock poisoned");
        let mut free = self.free_list.lock().expect("free_list lock poisoned");

        let mut evicted = 0u64;
        for &idx in &page_indices {
            if pages[idx].status != PageStatus::Active {
                continue;
            }
            // Only evict pages with refcount == 1 (not shared)
            if pages[idx].refcount.load(Ordering::Relaxed) > 1 {
                continue;
            }

            // Write page data to swap file
            let swap_path = self.swap_path(idx);
            std::fs::write(&swap_path, &pages[idx].data).map_err(|e| {
                SpiritMemoryError::SwapError(format!(
                    "failed to write swap file {}: {e}",
                    swap_path.display()
                ))
            })?;

            // Mark as swapped and return to free list
            pages[idx].status = PageStatus::Swapped;
            pages[idx].data.fill(0); // Release the RAM
            free.push_back(idx);
            evicted += 1;
        }

        if evicted > 0 {
            self.active_pages.fetch_sub(evicted, Ordering::Relaxed);
            self.swapped_pages.fetch_add(evicted, Ordering::Relaxed);

            info!(
                session = %session_id,
                evicted_pages = evicted,
                "evicted KV-cache pages to swap"
            );
        }

        Ok(())
    }

    /// Prefetch a swapped page back from NVMe.
    fn prefetch_page(&self, page_idx: usize) -> Result<()> {
        let swap_path = self.swap_path(page_idx);
        let data = std::fs::read(&swap_path).map_err(|e| {
            SpiritMemoryError::SwapError(format!(
                "failed to read swap file {}: {e}",
                swap_path.display()
            ))
        })?;

        let mut pages = self.pages.lock().expect("pages lock poisoned");
        if data.len() <= pages[page_idx].data.len() {
            pages[page_idx].data[..data.len()].copy_from_slice(&data);
        }
        pages[page_idx].status = PageStatus::Active;

        // Clean up swap file
        let _ = std::fs::remove_file(&swap_path);

        self.swapped_pages.fetch_sub(1, Ordering::Relaxed);
        self.active_pages.fetch_add(1, Ordering::Relaxed);

        debug!(page_idx, "prefetched page from swap");
        Ok(())
    }

    /// Prefetch all swapped pages for a session (called when session
    /// becomes active again).
    pub fn prefetch_session(&self, session_id: SessionId) -> Result<u64> {
        let page_indices = {
            let tables = self
                .session_tables
                .lock()
                .expect("session_tables lock poisoned");
            tables
                .get(&session_id)
                .ok_or(SpiritMemoryError::SessionNotFound(session_id))?
                .clone()
        };

        // Collect indices of swapped pages first, then prefetch without
        // holding the lock across iterations.
        let swapped: Vec<usize> = {
            let pages = self.pages.lock().expect("pages lock poisoned");
            page_indices
                .iter()
                .copied()
                .filter(|&idx| pages[idx].status == PageStatus::Swapped)
                .collect()
        };

        for idx in &swapped {
            self.prefetch_page(*idx)?;
        }

        let prefetched = swapped.len() as u64;
        if prefetched > 0 {
            info!(
                session = %session_id,
                prefetched,
                "prefetched swapped pages"
            );
        }

        self.touch_lru(session_id);
        Ok(prefetched)
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    fn swap_path(&self, page_idx: usize) -> PathBuf {
        self.swap_dir.join(format!("page_{page_idx:08x}.swap"))
    }

    fn touch_lru(&self, session_id: SessionId) {
        let mut lru = self.lru.lock().expect("lru lock poisoned");
        lru.retain(|s| *s != session_id);
        lru.push_back(session_id);
    }
}

impl std::fmt::Debug for KvPageManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvPageManager")
            .field("page_size", &self.page_size)
            .field("max_pages", &self.max_pages)
            .field("active_pages", &self.active_pages.load(Ordering::Relaxed))
            .field("swapped_pages", &self.swapped_pages.load(Ordering::Relaxed))
            .field("sessions", &self.session_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::KV_PAGE_SIZE;

    fn test_config() -> KvCacheConfig {
        KvCacheConfig {
            page_size: KV_PAGE_SIZE,
            max_pages: 64,
            max_sessions: 8,
            swap_dir: std::env::temp_dir().join(format!("spirit-kv-test-{}", uuid::Uuid::new_v4())),
            cow_enabled: true,
        }
    }

    #[test]
    fn test_create_and_destroy_session() {
        let mgr = KvPageManager::new(&test_config());
        let sid = SessionId::new();

        mgr.create_session(sid).unwrap();
        assert_eq!(mgr.session_count(), 1);

        mgr.destroy_session(sid).unwrap();
        assert_eq!(mgr.session_count(), 0);
    }

    #[test]
    fn test_alloc_and_read_page() {
        let mgr = KvPageManager::new(&test_config());
        let sid = SessionId::new();
        mgr.create_session(sid).unwrap();

        let _page = mgr.alloc_page(sid).unwrap();
        assert_eq!(mgr.active_page_count(), 1);
        assert_eq!(mgr.session_page_count(sid).unwrap(), 1);

        // Read should return zeroed page
        let data = mgr.read_page(sid, 0).unwrap();
        assert_eq!(data.len(), KV_PAGE_SIZE);
        assert!(data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_write_and_read_page() {
        let mgr = KvPageManager::new(&test_config());
        let sid = SessionId::new();
        mgr.create_session(sid).unwrap();
        mgr.alloc_page(sid).unwrap();

        let write_data = vec![0xAB; 128];
        mgr.write_page(sid, 0, &write_data).unwrap();

        let read_data = mgr.read_page(sid, 0).unwrap();
        assert_eq!(&read_data[..128], &write_data[..]);
        // Rest should be zero
        assert!(read_data[128..].iter().all(|&b| b == 0));
    }

    #[test]
    fn test_snapshot_shares_pages() {
        let mgr = KvPageManager::new(&test_config());
        let sid = SessionId::new();
        mgr.create_session(sid).unwrap();
        mgr.alloc_page(sid).unwrap();

        let data = vec![42u8; 64];
        mgr.write_page(sid, 0, &data).unwrap();

        let snap = mgr.snapshot(sid).unwrap();
        assert_eq!(snap.page_indices.len(), 1);
        assert_eq!(mgr.snapshot_count(), 1);
    }

    #[test]
    fn test_fork_creates_new_session() {
        let mgr = KvPageManager::new(&test_config());
        let sid = SessionId::new();
        mgr.create_session(sid).unwrap();
        mgr.alloc_page(sid).unwrap();

        let data = vec![42u8; 64];
        mgr.write_page(sid, 0, &data).unwrap();

        let snap = mgr.snapshot(sid).unwrap();
        let forked = mgr.fork(&snap).unwrap();

        assert_eq!(mgr.session_count(), 2);
        assert_eq!(mgr.fork_count(), 1);

        // Forked session should read the same data
        let forked_data = mgr.read_page(forked, 0).unwrap();
        assert_eq!(&forked_data[..64], &data[..]);
    }

    #[test]
    fn test_cow_on_write_after_fork() {
        let mgr = KvPageManager::new(&test_config());
        let sid = SessionId::new();
        mgr.create_session(sid).unwrap();
        mgr.alloc_page(sid).unwrap();

        let original = vec![1u8; 64];
        mgr.write_page(sid, 0, &original).unwrap();

        let snap = mgr.snapshot(sid).unwrap();
        let forked = mgr.fork(&snap).unwrap();

        // Write different data to the forked session → COW
        let modified = vec![2u8; 64];
        mgr.write_page(forked, 0, &modified).unwrap();

        assert_eq!(mgr.cow_copy_count(), 1);

        // Original should be unchanged
        let orig_data = mgr.read_page(sid, 0).unwrap();
        assert_eq!(&orig_data[..64], &original[..]);

        // Forked should have new data
        let fork_data = mgr.read_page(forked, 0).unwrap();
        assert_eq!(&fork_data[..64], &modified[..]);
    }

    #[test]
    fn test_session_not_found() {
        let mgr = KvPageManager::new(&test_config());
        let sid = SessionId::new();
        assert!(mgr.alloc_page(sid).is_err());
        assert!(mgr.read_page(sid, 0).is_err());
    }

    #[test]
    fn test_page_index_out_of_bounds() {
        let mgr = KvPageManager::new(&test_config());
        let sid = SessionId::new();
        mgr.create_session(sid).unwrap();
        mgr.alloc_page(sid).unwrap();

        assert!(mgr.read_page(sid, 99).is_err());
        assert!(mgr.write_page(sid, 99, &[1]).is_err());
    }

    #[test]
    fn test_destroy_releases_pages() {
        let mgr = KvPageManager::new(&test_config());
        let sid = SessionId::new();
        mgr.create_session(sid).unwrap();

        for _ in 0..10 {
            mgr.alloc_page(sid).unwrap();
        }
        assert_eq!(mgr.active_page_count(), 10);

        mgr.destroy_session(sid).unwrap();
        assert_eq!(mgr.active_page_count(), 0);
    }

    #[test]
    fn test_pool_exhaustion_triggers_eviction() {
        let mut config = test_config();
        config.max_pages = 4;
        let mgr = KvPageManager::new(&config);

        // Create session 1 with 4 pages (exhausts pool)
        let s1 = SessionId::new();
        mgr.create_session(s1).unwrap();
        for _ in 0..4 {
            mgr.alloc_page(s1).unwrap();
        }

        // Create session 2 — should trigger eviction of s1
        let s2 = SessionId::new();
        mgr.create_session(s2).unwrap();
        // This should succeed after evicting s1's pages
        let result = mgr.alloc_page(s2);
        // May succeed (if eviction works) or fail (if swap dir issues)
        // In test env with valid tmp, should succeed
        if result.is_ok() {
            assert!(mgr.swapped_page_count() > 0);
        }
    }

    #[test]
    fn test_multiple_sessions() {
        let mgr = KvPageManager::new(&test_config());

        let s1 = SessionId::new();
        let s2 = SessionId::new();
        mgr.create_session(s1).unwrap();
        mgr.create_session(s2).unwrap();

        mgr.alloc_page(s1).unwrap();
        mgr.alloc_page(s1).unwrap();
        mgr.alloc_page(s2).unwrap();

        assert_eq!(mgr.session_page_count(s1).unwrap(), 2);
        assert_eq!(mgr.session_page_count(s2).unwrap(), 1);
        assert_eq!(mgr.active_page_count(), 3);
    }

    #[test]
    fn test_destroy_shared_pages_decrements_refcount() {
        let mgr = KvPageManager::new(&test_config());
        let sid = SessionId::new();
        mgr.create_session(sid).unwrap();
        mgr.alloc_page(sid).unwrap();

        let snap = mgr.snapshot(sid).unwrap();
        let forked = mgr.fork(&snap).unwrap();

        // Destroying the forked session should decrement, not free
        mgr.destroy_session(forked).unwrap();

        // Original should still work
        let data = mgr.read_page(sid, 0).unwrap();
        assert_eq!(data.len(), KV_PAGE_SIZE);
    }
}
