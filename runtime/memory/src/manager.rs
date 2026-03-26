//! The `MemoryManager` — public API surface for the TensorFirst memory system.
//!
//! This is the primary entry point for all memory operations in SpiritROS.
//! Other components (inference coordinator, scheduler, shell) interact with
//! the memory subsystem exclusively through this API.
//!
//! The manager owns and coordinates:
//! - cgroups v2 zone enforcement
//! - Hugepage-backed model weight mappings
//! - GPU VRAM reservation tracking
//! - KV-cache page pool with COW semantics
//! - Zone allocation tracking with admission control
//! - PSI pressure monitoring
//! - Prometheus metrics
//!
//! // PHASE1-SHIM: In Phase 2, this becomes the kernel's memory subsystem
//! // with hardware-level zone enforcement and zero-copy GPU transfers.

use std::path::Path;
use tracing::{info, warn};

use crate::cgroups::{self, CgroupManager};
use crate::config::MemoryConfig;
use crate::error::{Result, SpiritMemoryError};
use crate::hugepages::HugepageManager;
use crate::kv_cache::KvPageManager;
use crate::metrics::MemoryMetrics;
use crate::pressure::PressureMonitor;
use crate::types::{
    AllZoneStats, AllocationId, GpuDevice, HugepageStats, KvSnapshot, MemoryZone, ModelMapping,
    PressureLevel, SessionId, VramReservation, VramStats, ZoneStats,
};
use crate::vram::VramManager;
use crate::zones::ZoneTracker;

/// The TensorFirst memory manager.
///
/// Thread-safe: can be wrapped in `Arc` and shared across async tasks.
/// All internal state uses `Mutex` or atomic operations.
pub struct MemoryManager {
    config: MemoryConfig,
    cgroups: CgroupManager,
    hugepages: HugepageManager,
    vram: VramManager,
    kv_cache: KvPageManager,
    zones: ZoneTracker,
    pressure: PressureMonitor,
    metrics: MemoryMetrics,
    /// Total system RAM in bytes (detected at init).
    system_ram: u64,
}

impl MemoryManager {
    /// Create and initialize the memory manager.
    ///
    /// Performs the following steps:
    /// 1. Detect system memory
    /// 2. Auto-size zones if configured for auto-detect
    /// 3. Validate configuration
    /// 4. Set up cgroups v2 zone hierarchy
    /// 5. Initialize hugepage manager
    /// 6. Detect GPUs and initialize VRAM manager
    /// 7. Pre-allocate KV-cache page pool
    /// 8. Start pressure monitoring
    /// 9. Register Prometheus metrics
    ///
    /// Gracefully degrades when subsystems are unavailable (no root,
    /// no GPUs, no hugepages, no PSI).
    #[allow(clippy::too_many_lines)]
    pub fn new(mut config: MemoryConfig) -> Result<Self> {
        // 1. Detect system memory
        let system_ram = cgroups::detect_system_memory().unwrap_or_else(|e| {
            warn!("could not detect system memory: {e}; using 32 GiB default");
            32 * crate::types::GIB
        });

        info!(
            system_ram_gib = system_ram / crate::types::GIB,
            "detected system memory"
        );

        // 2. Auto-size zones if any are zero
        config.auto_size(system_ram);

        // 3. Validate
        config.validate()?;
        let zone_sum = config.total_zone_bytes();
        let budget = (system_ram as f64 * config.budget_fraction) as u64;
        if zone_sum > budget {
            return Err(SpiritMemoryError::ZoneSumExceedsMemory {
                zone_sum_bytes: zone_sum,
                available_bytes: budget,
            });
        }

        info!(
            model_gib = config.model_zone_bytes / crate::types::GIB,
            kv_gib = config.kv_zone_bytes / crate::types::GIB,
            tensor_gib = config.tensor_zone_bytes / crate::types::GIB,
            general_gib = config.general_zone_bytes / crate::types::GIB,
            total_gib = zone_sum / crate::types::GIB,
            budget_gib = budget / crate::types::GIB,
            "zone configuration"
        );

        // 4. cgroups v2
        let cgroups = CgroupManager::new(&config.cgroup_root);
        if let Err(e) = cgroups.setup_zones(&config) {
            warn!("cgroups zone setup failed: {e}; continuing without enforcement");
        }

        // 5. Hugepages
        let hugepages = HugepageManager::new(config.hugepage_size, config.use_hugepages);

        // 6. VRAM
        let vram = VramManager::new(config.vram.memory_fraction);

        // 7. KV-cache page pool
        let kv_cache = KvPageManager::new(&config.kv_cache);

        // 8. Zone tracker
        let zones = ZoneTracker::new(&config);

        // 9. Pressure monitoring
        let pressure = PressureMonitor::new(&config.pressure);

        // 10. Metrics
        let metrics = MemoryMetrics::new();

        info!("TensorFirst memory manager initialized");

        Ok(Self {
            config,
            cgroups,
            hugepages,
            vram,
            kv_cache,
            zones,
            pressure,
            metrics,
            system_ram,
        })
    }

    // ── Zone Allocation ─────────────────────────────────────────────────

    /// Track a new allocation in the specified zone.
    ///
    /// Returns an `AllocationId` for later release. The allocation is
    /// checked against the zone's limit before being accepted.
    pub fn track_allocation(&self, zone: MemoryZone, size: u64) -> Result<AllocationId> {
        let id = self.zones.track(zone, size)?;
        self.refresh_zone_metrics(zone);
        Ok(id)
    }

    /// Release a previously tracked allocation.
    pub fn release_allocation(&self, id: AllocationId) -> Result<()> {
        self.zones.release(id)?;
        // Refresh all zones since we don't know which zone the alloc was in
        for zone in MemoryZone::ALL {
            self.refresh_zone_metrics(zone);
        }
        Ok(())
    }

    /// Check whether `size` bytes can be allocated in `zone`.
    #[must_use]
    pub fn can_allocate(&self, zone: MemoryZone, size: u64) -> bool {
        self.zones.can_allocate(zone, size)
    }

    // ── Model Weight Mapping ────────────────────────────────────────────

    /// Memory-map a model weight file (read-only, hugepage-backed).
    ///
    /// The file is mapped with `MAP_POPULATE` to pre-fault all pages,
    /// eliminating page faults during inference. Uses hugepages when
    /// available for reduced TLB pressure.
    ///
    /// Also tracks the mapping as a ZONE_MODEL allocation.
    pub fn mmap_model_weights(&self, path: &Path) -> Result<(ModelMapping, AllocationId)> {
        let mapping = self.hugepages.mmap_model_file(path)?;
        let size = mapping.len() as u64;

        let alloc_id = self.zones.track(MemoryZone::Model, size)?;
        self.refresh_zone_metrics(MemoryZone::Model);

        info!(
            path = %path.display(),
            size_mib = size / crate::types::MIB,
            hugepage = mapping.is_hugepage_backed(),
            "model weights mapped"
        );

        Ok((mapping, alloc_id))
    }

    // ── VRAM Management ─────────────────────────────────────────────────

    /// Reserve VRAM on a specific GPU.
    pub fn reserve_vram(&self, device_index: u32, size: u64) -> Result<VramReservation> {
        let reservation = self.vram.reserve(device_index, size)?;
        self.refresh_vram_metrics(device_index);
        Ok(reservation)
    }

    /// Release a VRAM reservation.
    pub fn release_vram(&self, reservation: &VramReservation) -> Result<()> {
        let device = reservation.device_index();
        self.vram.release(reservation)?;
        self.refresh_vram_metrics(device);
        Ok(())
    }

    /// Get VRAM statistics for a specific GPU.
    pub fn vram_stats(&self, device_index: u32) -> Result<VramStats> {
        self.vram.stats(device_index)
    }

    /// Get VRAM statistics for all GPUs.
    pub fn all_vram_stats(&self) -> Vec<VramStats> {
        self.vram.all_stats()
    }

    /// Number of detected GPU devices.
    #[must_use]
    pub fn gpu_count(&self) -> u32 {
        self.vram.device_count()
    }

    /// Get GPU device info.
    pub fn gpu_device(&self, index: u32) -> Result<&GpuDevice> {
        self.vram.device(index)
    }

    // ── KV-Cache Operations ─────────────────────────────────────────────

    /// Create a new KV-cache session.
    pub fn kv_create_session(&self, session_id: SessionId) -> Result<()> {
        self.kv_cache.create_session(session_id)?;
        self.refresh_kv_metrics();
        Ok(())
    }

    /// Allocate a KV-cache page for a session.
    pub fn kv_alloc_page(&self, session_id: SessionId) -> Result<usize> {
        let idx = self.kv_cache.alloc_page(session_id)?;
        self.refresh_kv_metrics();
        Ok(idx)
    }

    /// Write data to a KV-cache page (COW if shared).
    pub fn kv_write_page(
        &self,
        session_id: SessionId,
        page_offset: usize,
        data: &[u8],
    ) -> Result<()> {
        self.kv_cache.write_page(session_id, page_offset, data)
    }

    /// Read data from a KV-cache page.
    pub fn kv_read_page(&self, session_id: SessionId, page_offset: usize) -> Result<Vec<u8>> {
        self.kv_cache.read_page(session_id, page_offset)
    }

    /// Snapshot a session's KV-cache (COW, O(n) refcount bumps).
    pub fn kv_snapshot(&self, session_id: SessionId) -> Result<KvSnapshot> {
        self.kv_cache.snapshot(session_id)
    }

    /// Fork a new session from a snapshot (instant, COW).
    pub fn kv_fork(&self, snapshot: &KvSnapshot) -> Result<SessionId> {
        let new_id = self.kv_cache.fork(snapshot)?;
        self.refresh_kv_metrics();
        Ok(new_id)
    }

    /// Destroy a KV-cache session, releasing its pages.
    pub fn kv_destroy_session(&self, session_id: SessionId) -> Result<()> {
        self.kv_cache.destroy_session(session_id)?;
        self.refresh_kv_metrics();
        Ok(())
    }

    /// Prefetch swapped pages for a session.
    pub fn kv_prefetch_session(&self, session_id: SessionId) -> Result<u64> {
        let count = self.kv_cache.prefetch_session(session_id)?;
        self.refresh_kv_metrics();
        Ok(count)
    }

    /// Number of pages in a session.
    pub fn kv_session_page_count(&self, session_id: SessionId) -> Result<usize> {
        self.kv_cache.session_page_count(session_id)
    }

    // ── Telemetry ───────────────────────────────────────────────────────

    /// Get statistics for a single zone.
    pub fn zone_stats(&self, zone: MemoryZone) -> ZoneStats {
        let mut stats = self.zones.zone_stats(zone);
        // Enrich with live cgroup data if available
        if let Ok(swap) = self.cgroups.get_swap_usage(zone) {
            stats.swap_bytes = swap;
        }
        stats
    }

    /// Get statistics for all zones plus system-wide info.
    pub fn all_zone_stats(&self) -> AllZoneStats {
        let zones: Vec<ZoneStats> = MemoryZone::ALL
            .iter()
            .map(|z| self.zone_stats(*z))
            .collect();
        let total_used = zones.iter().map(|z| z.used_bytes).sum();

        AllZoneStats {
            zones,
            total_system_bytes: self.system_ram,
            total_used_bytes: total_used,
            hugepages: self.hugepage_stats(),
            vram: self.all_vram_stats(),
        }
    }

    /// Get hugepage statistics.
    pub fn hugepage_stats(&self) -> HugepageStats {
        self.hugepages.stats()
    }

    /// Current memory pressure level.
    #[must_use]
    pub fn pressure_level(&self) -> PressureLevel {
        self.pressure.level()
    }

    /// Poll pressure once and return the updated level.
    pub fn poll_pressure(&self) -> PressureLevel {
        let level = self.pressure.poll();
        self.metrics.update_pressure(level as i64);
        level
    }

    // ── Rebalancing ─────────────────────────────────────────────────────

    /// Rebalance memory between two zones at runtime.
    ///
    /// Moves `amount` bytes of capacity from `from` to `to`. Updates
    /// both the logical tracker and the cgroups limits.
    ///
    /// # Errors
    ///
    /// - `RebalanceDenied` if the source would drop below its minimum.
    /// - `ZoneExhausted` if the source doesn't have enough free capacity.
    pub fn rebalance(&self, from: MemoryZone, to: MemoryZone, amount: u64) -> Result<()> {
        self.zones.rebalance(from, to, amount, &self.cgroups)?;
        self.refresh_zone_metrics(from);
        self.refresh_zone_metrics(to);
        Ok(())
    }

    // ── Metrics ─────────────────────────────────────────────────────────

    /// Get a reference to the Prometheus metrics registry.
    ///
    /// Use this to serve metrics at an HTTP `/metrics` endpoint.
    #[must_use]
    pub fn metrics_registry(&self) -> &prometheus_client::registry::Registry {
        &self.metrics.registry
    }

    /// Encode all metrics as Prometheus text format.
    #[must_use]
    pub fn encode_metrics(&self) -> String {
        self.refresh_all_metrics();
        self.metrics.encode()
    }

    /// Start the background pressure monitoring task.
    ///
    /// Returns a `JoinHandle` for the spawned tokio task, or `None` if
    /// PSI monitoring is not available.
    pub fn start_pressure_monitor(&self) -> Option<tokio::task::JoinHandle<()>> {
        self.pressure.spawn_monitor()
    }

    // ── Configuration ───────────────────────────────────────────────────

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }

    /// Total system RAM detected at initialization.
    #[must_use]
    pub fn system_ram(&self) -> u64 {
        self.system_ram
    }

    /// Whether cgroups v2 enforcement is active.
    #[must_use]
    pub fn cgroups_available(&self) -> bool {
        self.cgroups.is_available()
    }

    /// Whether hugepage support is active.
    #[must_use]
    pub fn hugepages_available(&self) -> bool {
        self.hugepages.is_available()
    }

    /// Whether PSI monitoring is active.
    #[must_use]
    pub fn pressure_monitoring_available(&self) -> bool {
        self.pressure.is_available()
    }

    /// Assign a process to a memory zone's cgroup.
    pub fn assign_process_to_zone(&self, zone: MemoryZone, pid: u32) -> Result<()> {
        self.cgroups.assign_process(zone, pid)
    }

    // ── Internal metric refresh ─────────────────────────────────────────

    fn refresh_zone_metrics(&self, zone: MemoryZone) {
        let stats = self.zones.zone_stats(zone);
        self.metrics.update_zone(
            zone,
            stats.used_bytes as i64,
            stats.limit_bytes as i64,
            stats.allocation_count as i64,
        );
    }

    fn refresh_vram_metrics(&self, device_index: u32) {
        if let Ok(stats) = self.vram.stats(device_index) {
            self.metrics.update_vram(
                &format!("gpu{device_index}"),
                stats.used_bytes as i64,
                stats.reserved_bytes as i64,
            );
        }
    }

    fn refresh_kv_metrics(&self) {
        self.metrics.update_kv(
            self.kv_cache.active_page_count() as i64,
            self.kv_cache.swapped_page_count() as i64,
            self.kv_cache.session_count() as i64,
        );
    }

    fn refresh_all_metrics(&self) {
        for zone in MemoryZone::ALL {
            self.refresh_zone_metrics(zone);
        }
        for i in 0..self.vram.device_count() {
            self.refresh_vram_metrics(i);
        }
        self.refresh_kv_metrics();

        let hp = self.hugepages.stats();
        self.metrics
            .update_hugepages(hp.used_bytes as i64, hp.free_pages as i64);

        self.metrics.update_pressure(self.pressure.level() as i64);
    }
}

impl std::fmt::Debug for MemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryManager")
            .field("system_ram", &self.system_ram)
            .field("cgroups_available", &self.cgroups.is_available())
            .field("hugepages_available", &self.hugepages.is_available())
            .field("gpu_count", &self.vram.device_count())
            .field("pressure_available", &self.pressure.is_available())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn test_manager() -> MemoryManager {
        MemoryManager::new(MemoryConfig::for_test()).unwrap()
    }

    #[test]
    fn test_manager_creation() {
        let mgr = test_manager();
        assert!(mgr.system_ram() > 0);
    }

    #[test]
    fn test_zone_allocation_tracking() {
        let mgr = test_manager();

        let id = mgr.track_allocation(MemoryZone::Tensor, 1024).unwrap();
        assert!(mgr.can_allocate(MemoryZone::Tensor, 1024));

        let stats = mgr.zone_stats(MemoryZone::Tensor);
        assert_eq!(stats.used_bytes, 1024);

        mgr.release_allocation(id).unwrap();
        let stats = mgr.zone_stats(MemoryZone::Tensor);
        assert_eq!(stats.used_bytes, 0);
    }

    #[test]
    fn test_zone_exhaustion() {
        let mgr = test_manager();
        let limit = mgr.zone_stats(MemoryZone::General).limit_bytes;
        assert!(mgr
            .track_allocation(MemoryZone::General, limit + 1)
            .is_err());
    }

    #[test]
    fn test_all_zone_stats() {
        let mgr = test_manager();
        let stats = mgr.all_zone_stats();
        assert_eq!(stats.zones.len(), 4);
        assert!(stats.total_system_bytes > 0);
    }

    #[test]
    fn test_model_mapping() {
        let mgr = test_manager();

        // Create a temp model file
        let dir = std::env::temp_dir().join(format!("spirit-mgr-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.bin");
        std::fs::write(&path, vec![0xABu8; 8192]).unwrap();

        let (mapping, alloc_id) = mgr.mmap_model_weights(&path).unwrap();
        assert_eq!(mapping.len(), 8192);
        assert_eq!(mapping.as_slice()[0], 0xAB);

        let stats = mgr.zone_stats(MemoryZone::Model);
        assert_eq!(stats.used_bytes, 8192);

        mgr.release_allocation(alloc_id).unwrap();
        drop(mapping);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_kv_cache_lifecycle() {
        let mgr = test_manager();
        let sid = SessionId::new();

        mgr.kv_create_session(sid).unwrap();
        mgr.kv_alloc_page(sid).unwrap();
        mgr.kv_write_page(sid, 0, &[42u8; 64]).unwrap();

        let data = mgr.kv_read_page(sid, 0).unwrap();
        assert_eq!(&data[..64], &[42u8; 64]);

        mgr.kv_destroy_session(sid).unwrap();
    }

    #[test]
    fn test_kv_snapshot_and_fork() {
        let mgr = test_manager();
        let sid = SessionId::new();

        mgr.kv_create_session(sid).unwrap();
        mgr.kv_alloc_page(sid).unwrap();
        mgr.kv_write_page(sid, 0, &[1u8; 32]).unwrap();

        let snap = mgr.kv_snapshot(sid).unwrap();
        let forked = mgr.kv_fork(&snap).unwrap();

        // Both should read same data
        let orig = mgr.kv_read_page(sid, 0).unwrap();
        let fork = mgr.kv_read_page(forked, 0).unwrap();
        assert_eq!(&orig[..32], &fork[..32]);

        // Write to fork → COW
        mgr.kv_write_page(forked, 0, &[2u8; 32]).unwrap();
        let orig = mgr.kv_read_page(sid, 0).unwrap();
        let fork = mgr.kv_read_page(forked, 0).unwrap();
        assert_eq!(&orig[..32], &[1u8; 32]);
        assert_eq!(&fork[..32], &[2u8; 32]);

        mgr.kv_destroy_session(sid).unwrap();
        mgr.kv_destroy_session(forked).unwrap();
    }

    #[test]
    fn test_rebalance() {
        let mgr = test_manager();
        let amount = 256 * crate::types::MIB;

        let before_tensor = mgr.zone_stats(MemoryZone::Tensor).limit_bytes;

        mgr.rebalance(MemoryZone::Tensor, MemoryZone::General, amount)
            .unwrap();

        let after_tensor = mgr.zone_stats(MemoryZone::Tensor).limit_bytes;
        assert_eq!(after_tensor, before_tensor - amount);
    }

    #[test]
    fn test_pressure_level() {
        let mgr = test_manager();
        // In test config, pressure is disabled → always None
        assert_eq!(mgr.pressure_level(), PressureLevel::None);
    }

    #[test]
    fn test_encode_metrics() {
        let mgr = test_manager();
        mgr.track_allocation(MemoryZone::Model, 1024).unwrap();

        let encoded = mgr.encode_metrics();
        assert!(encoded.contains("spirit_memory_zone_used_bytes"));
    }

    #[test]
    fn test_config_access() {
        let mgr = test_manager();
        assert_eq!(mgr.config().kv_cache.max_pages, 1024);
    }
}
