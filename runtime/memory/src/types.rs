//! Core type definitions for the TensorFirst memory manager.
//!
//! This module defines all shared types used across the memory subsystem:
//! memory zones, statistics, allocations, GPU devices, KV-cache pages,
//! and pressure levels.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

// ── Constants ───────────────────────────────────────────────────────────────

/// Page size for KV-cache management (4 KiB, matching OS page size).
pub const KV_PAGE_SIZE: usize = 4096;

/// Default hugepage size: 2 MiB.
pub const HUGEPAGE_SIZE_2M: u64 = 2 * 1024 * 1024;

/// Large hugepage size: 1 GiB.
pub const HUGEPAGE_SIZE_1G: u64 = 1024 * 1024 * 1024;

/// Bytes per mebibyte.
pub const MIB: u64 = 1024 * 1024;

/// Bytes per gibibyte.
pub const GIB: u64 = 1024 * 1024 * 1024;

// ── Memory Zones ────────────────────────────────────────────────────────────

/// Identifies one of the four TensorFirst memory zones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryZone {
    /// Read-only model weight storage (hugepage-backed).
    Model,
    /// KV-cache pages with COW semantics.
    KvCache,
    /// Scratch space for activation tensors.
    Tensor,
    /// General-purpose allocations.
    General,
}

impl MemoryZone {
    /// All four zones in priority order (highest first).
    pub const ALL: [Self; 4] = [Self::Model, Self::KvCache, Self::Tensor, Self::General];

    /// Returns the cgroup subtree name for this zone.
    #[must_use]
    pub fn cgroup_name(&self) -> &'static str {
        match self {
            Self::Model => "spirit/zone_model",
            Self::KvCache => "spirit/zone_kv",
            Self::Tensor => "spirit/zone_tensor",
            Self::General => "spirit/zone_general",
        }
    }

    /// Short label for metrics and logging.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::KvCache => "kv",
            Self::Tensor => "tensor",
            Self::General => "general",
        }
    }
}

impl fmt::Display for MemoryZone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ── Zone Statistics ─────────────────────────────────────────────────────────

/// Statistics for a single memory zone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneStats {
    /// Which zone this refers to.
    pub zone: MemoryZone,
    /// Hard limit (memory.max) in bytes.
    pub limit_bytes: u64,
    /// Throttling threshold (memory.high) in bytes.
    pub high_bytes: u64,
    /// Currently used bytes (memory.current).
    pub used_bytes: u64,
    /// Guaranteed minimum bytes (memory.min).
    pub min_bytes: u64,
    /// Current swap usage in bytes.
    pub swap_bytes: u64,
    /// Number of tracked allocations in this zone.
    pub allocation_count: u64,
}

impl ZoneStats {
    /// Utilization as a percentage of the hard limit.
    #[must_use]
    pub fn utilization_percent(&self) -> f64 {
        if self.limit_bytes == 0 {
            return 0.0;
        }
        (self.used_bytes as f64 / self.limit_bytes as f64) * 100.0
    }

    /// Bytes remaining before hitting the hard limit.
    #[must_use]
    pub fn available_bytes(&self) -> u64 {
        self.limit_bytes.saturating_sub(self.used_bytes)
    }
}

/// Aggregate statistics for all four zones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllZoneStats {
    /// Per-zone stats in `MemoryZone::ALL` order.
    pub zones: Vec<ZoneStats>,
    /// Total physical RAM detected on the system.
    pub total_system_bytes: u64,
    /// Sum of all zone used_bytes.
    pub total_used_bytes: u64,
    /// Hugepage statistics.
    pub hugepages: HugepageStats,
    /// Per-GPU VRAM statistics.
    pub vram: Vec<VramStats>,
}

// ── GPU / VRAM ──────────────────────────────────────────────────────────────

/// Identifies a GPU device.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GpuDevice {
    /// Zero-based GPU index.
    pub index: u32,
    /// Human-readable name (e.g. "NVIDIA RTX 4090").
    pub name: String,
    /// Total VRAM in bytes.
    pub total_vram_bytes: u64,
}

/// VRAM utilization for one GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VramStats {
    pub device: GpuDevice,
    /// Bytes actively in use.
    pub used_bytes: u64,
    /// Bytes reserved (may include unused reserved space).
    pub reserved_bytes: u64,
    /// Bytes available for new allocations.
    pub free_bytes: u64,
}

/// Handle to a contiguous VRAM reservation. Dropping releases the reservation.
#[derive(Debug)]
pub struct VramReservation {
    pub(crate) device_index: u32,
    pub(crate) size: u64,
    pub(crate) id: u64,
}

impl VramReservation {
    /// GPU that holds this reservation.
    #[must_use]
    pub fn device_index(&self) -> u32 {
        self.device_index
    }

    /// Size in bytes.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }
}

// ── Hugepages ───────────────────────────────────────────────────────────────

/// Hugepage pool statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HugepageStats {
    /// Total hugepages available on the system.
    pub total_pages: u64,
    /// Free (unallocated) hugepages.
    pub free_pages: u64,
    /// Size of each hugepage in bytes.
    pub page_size_bytes: u64,
    /// Total hugepage pool in bytes (`total_pages * page_size_bytes`).
    pub total_bytes: u64,
    /// Bytes consumed by hugepage-backed mappings.
    pub used_bytes: u64,
}

// ── Model Mapping ───────────────────────────────────────────────────────────

/// Read-only memory-mapped model weights.
///
/// Created via [`MemoryManager::mmap_model_weights`]. The mapping is backed
/// by hugepages when available and `MAP_POPULATE`d to pre-fault all pages.
/// The region is unmapped on drop.
pub struct ModelMapping {
    ptr: *mut u8,
    len: usize,
    path: PathBuf,
    hugepage_backed: bool,
}

// SAFETY: The mapped region is read-only and immutable after creation.
// The pointer is valid for the lifetime of the ModelMapping.
unsafe impl Send for ModelMapping {}
unsafe impl Sync for ModelMapping {}

impl ModelMapping {
    pub(crate) fn new(ptr: *mut u8, len: usize, path: PathBuf, hugepage_backed: bool) -> Self {
        Self {
            ptr,
            len,
            path,
            hugepage_backed,
        }
    }

    /// View the model weights as a byte slice.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for len bytes and the region is read-only.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Size in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the mapping is zero-length.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Filesystem path of the mapped model file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Whether this mapping is backed by hugepages.
    #[must_use]
    pub fn is_hugepage_backed(&self) -> bool {
        self.hugepage_backed
    }
}

impl Drop for ModelMapping {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            // SAFETY: ptr and len were produced by a successful mmap call.
            unsafe {
                libc::munmap(self.ptr.cast(), self.len);
            }
        }
    }
}

impl fmt::Debug for ModelMapping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelMapping")
            .field("path", &self.path)
            .field("len", &self.len)
            .field("hugepage_backed", &self.hugepage_backed)
            .finish()
    }
}

// ── KV-Cache Types ──────────────────────────────────────────────────────────

/// Unique session identifier for KV-cache tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub uuid::Uuid);

impl SessionId {
    /// Generate a new random session ID.
    #[must_use]
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Status of a KV-cache page in the page pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum PageStatus {
    /// Available for allocation.
    Free = 0,
    /// Actively held by one or more sessions.
    Active = 1,
    /// Evicted to NVMe swap.
    Swapped = 2,
}

/// A snapshot of a session's KV-cache state (COW reference).
///
/// Creating a snapshot is O(n) in the number of pages but does NOT copy
/// page data — it increments reference counts so pages are shared until
/// mutated (copy-on-write).
#[derive(Debug, Clone)]
pub struct KvSnapshot {
    /// Session this snapshot was taken from.
    pub session_id: SessionId,
    /// Page indices in the global page pool.
    pub page_indices: Vec<usize>,
    /// When the snapshot was created.
    pub created_at: std::time::Instant,
}

// ── Allocation Tracking ─────────────────────────────────────────────────────

/// Opaque handle to a tracked zone allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AllocationId(pub(crate) u64);

/// Monotonic allocation ID generator.
pub(crate) struct IdGenerator(AtomicU64);

impl IdGenerator {
    pub(crate) const fn new() -> Self {
        Self(AtomicU64::new(1))
    }

    pub(crate) fn next(&self) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed)
    }
}

// ── Pressure ────────────────────────────────────────────────────────────────

/// Memory pressure severity level, derived from Linux PSI metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PressureLevel {
    /// No pressure detected.
    None,
    /// Mild pressure (some stalls < 10%).
    Low,
    /// Moderate pressure (some stalls 10–25%). Shrink tensor scratch.
    Medium,
    /// High pressure (full stalls > 5%). Evict cold KV-cache.
    High,
    /// Critical pressure (full stalls > 20%). Emergency measures.
    Critical,
}

impl fmt::Display for PressureLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::Low => f.write_str("low"),
            Self::Medium => f.write_str("medium"),
            Self::High => f.write_str("high"),
            Self::Critical => f.write_str("critical"),
        }
    }
}

/// Raw PSI (Pressure Stall Information) readings from
/// `/proc/pressure/memory`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PressureInfo {
    /// Percentage of time *some* tasks are stalled (10s window).
    pub some_avg10: f64,
    /// Percentage of time *some* tasks are stalled (60s window).
    pub some_avg60: f64,
    /// Percentage of time *some* tasks are stalled (300s window).
    pub some_avg300: f64,
    /// Percentage of time *all* tasks are stalled (10s window).
    pub full_avg10: f64,
    /// Percentage of time *all* tasks are stalled (60s window).
    pub full_avg60: f64,
    /// Percentage of time *all* tasks are stalled (300s window).
    pub full_avg300: f64,
}

impl PressureInfo {
    /// Derive a discrete pressure level from the raw PSI numbers.
    #[must_use]
    pub fn level(&self) -> PressureLevel {
        if self.full_avg10 > 20.0 {
            PressureLevel::Critical
        } else if self.full_avg10 > 5.0 {
            PressureLevel::High
        } else if self.some_avg10 > 25.0 {
            PressureLevel::Medium
        } else if self.some_avg10 > 10.0 {
            PressureLevel::Low
        } else {
            PressureLevel::None
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Format bytes as a human-readable string (e.g. "4.30 GiB").
#[must_use]
pub fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB_U: u64 = 1024 * 1024;
    const GIB_U: u64 = 1024 * 1024 * 1024;
    const TIB: u64 = 1024 * 1024 * 1024 * 1024;

    if bytes >= TIB {
        format!("{:.2} TiB", bytes as f64 / TIB as f64)
    } else if bytes >= GIB_U {
        format!("{:.2} GiB", bytes as f64 / GIB_U as f64)
    } else if bytes >= MIB_U {
        format!("{:.2} MiB", bytes as f64 / MIB_U as f64)
    } else if bytes >= KIB {
        format!("{:.2} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zone_labels_are_unique() {
        let labels: Vec<&str> = MemoryZone::ALL.iter().map(MemoryZone::label).collect();
        let mut deduped = labels.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(labels.len(), deduped.len());
    }

    #[test]
    fn test_zone_cgroup_names_are_unique() {
        let names: Vec<&str> = MemoryZone::ALL
            .iter()
            .map(MemoryZone::cgroup_name)
            .collect();
        let mut deduped = names.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(names.len(), deduped.len());
    }

    #[test]
    fn test_zone_stats_utilization() {
        let stats = ZoneStats {
            zone: MemoryZone::Model,
            limit_bytes: 1000,
            high_bytes: 900,
            used_bytes: 500,
            min_bytes: 200,
            swap_bytes: 0,
            allocation_count: 1,
        };
        assert!((stats.utilization_percent() - 50.0).abs() < f64::EPSILON);
        assert_eq!(stats.available_bytes(), 500);
    }

    #[test]
    fn test_zone_stats_utilization_zero_limit() {
        let stats = ZoneStats {
            zone: MemoryZone::General,
            limit_bytes: 0,
            high_bytes: 0,
            used_bytes: 0,
            min_bytes: 0,
            swap_bytes: 0,
            allocation_count: 0,
        };
        assert!((stats.utilization_percent()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_memory_zone_serialization_roundtrip() {
        for zone in MemoryZone::ALL {
            let json = serde_json::to_string(&zone).expect("serialize");
            let back: MemoryZone = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(zone, back);
        }
    }

    #[test]
    fn test_session_id_uniqueness() {
        let a = SessionId::new();
        let b = SessionId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn test_pressure_info_levels() {
        let none = PressureInfo::default();
        assert_eq!(none.level(), PressureLevel::None);

        let low = PressureInfo {
            some_avg10: 15.0,
            ..Default::default()
        };
        assert_eq!(low.level(), PressureLevel::Low);

        let medium = PressureInfo {
            some_avg10: 30.0,
            ..Default::default()
        };
        assert_eq!(medium.level(), PressureLevel::Medium);

        let high = PressureInfo {
            full_avg10: 10.0,
            ..Default::default()
        };
        assert_eq!(high.level(), PressureLevel::High);

        let critical = PressureInfo {
            full_avg10: 25.0,
            ..Default::default()
        };
        assert_eq!(critical.level(), PressureLevel::Critical);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(2048), "2.00 KiB");
        assert_eq!(format_bytes(5 * 1024 * 1024), "5.00 MiB");
        assert_eq!(format_bytes(4 * GIB), "4.00 GiB");
    }

    #[test]
    fn test_page_status_values() {
        assert_eq!(PageStatus::Free as u8, 0);
        assert_eq!(PageStatus::Active as u8, 1);
        assert_eq!(PageStatus::Swapped as u8, 2);
    }

    #[test]
    fn test_id_generator_monotonic() {
        let gen = IdGenerator::new();
        let a = gen.next();
        let b = gen.next();
        let c = gen.next();
        assert_eq!(a, 1);
        assert_eq!(b, 2);
        assert_eq!(c, 3);
    }
}
