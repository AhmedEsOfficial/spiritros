//! Prometheus metrics for the TensorFirst memory manager.
//!
//! Exposes the following metrics:
//! - `spirit_memory_zone_used_bytes{zone}` — current usage per zone
//! - `spirit_memory_zone_limit_bytes{zone}` — hard limit per zone
//! - `spirit_vram_used_bytes{device}` — VRAM usage per GPU
//! - `spirit_vram_reserved_bytes{device}` — VRAM reserved per GPU
//! - `spirit_kv_pages_active` — active KV-cache pages
//! - `spirit_kv_pages_swapped` — swapped KV-cache pages
//! - `spirit_kv_sessions` — active KV-cache sessions
//! - `spirit_kv_snapshots_total` — cumulative snapshot count
//! - `spirit_kv_forks_total` — cumulative fork count
//! - `spirit_kv_cow_copies_total` — cumulative COW copy count
//! - `spirit_hugepages_used` — hugepages in use
//! - `spirit_hugepages_free` — hugepages available
//! - `spirit_pressure_level` — current pressure level (0–4)
//! - `spirit_memory_allocations_total{zone}` — allocation count per zone

use prometheus_client::encoding::{EncodeLabelSet, EncodeLabelValue};
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::registry::Registry;

use crate::types::MemoryZone;

/// Label set for zone-scoped metrics.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ZoneLabel {
    pub zone: ZoneName,
}

/// Zone name as a label value.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelValue)]
pub enum ZoneName {
    Model,
    Kv,
    Tensor,
    General,
}

impl From<MemoryZone> for ZoneName {
    fn from(z: MemoryZone) -> Self {
        match z {
            MemoryZone::Model => Self::Model,
            MemoryZone::KvCache => Self::Kv,
            MemoryZone::Tensor => Self::Tensor,
            MemoryZone::General => Self::General,
        }
    }
}

/// Label set for device-scoped metrics.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct DeviceLabel {
    pub device: String,
}

/// All Prometheus metrics for the memory subsystem.
#[allow(dead_code)]
pub struct MemoryMetrics {
    /// The metrics registry (used for encoding).
    pub(crate) registry: Registry,

    // ── Zone metrics ────────────────────────────────────────────────
    pub(crate) zone_used: Family<ZoneLabel, Gauge>,
    pub(crate) zone_limit: Family<ZoneLabel, Gauge>,
    pub(crate) zone_allocs: Family<ZoneLabel, Gauge>,

    // ── VRAM metrics ────────────────────────────────────────────────
    pub(crate) vram_used: Family<DeviceLabel, Gauge>,
    pub(crate) vram_reserved: Family<DeviceLabel, Gauge>,

    // ── KV-cache metrics ────────────────────────────────────────────
    pub(crate) kv_pages_active: Gauge,
    pub(crate) kv_pages_swapped: Gauge,
    pub(crate) kv_sessions: Gauge,
    pub(crate) kv_snapshots_total: Counter,
    pub(crate) kv_forks_total: Counter,
    pub(crate) kv_cow_copies_total: Counter,

    // ── Hugepage metrics ────────────────────────────────────────────
    pub(crate) hugepages_used: Gauge,
    pub(crate) hugepages_free: Gauge,

    // ── Pressure metrics ────────────────────────────────────────────
    pub(crate) pressure_level: Gauge,
}

impl MemoryMetrics {
    /// Create and register all memory metrics.
    #[allow(clippy::too_many_lines)]
    pub fn new() -> Self {
        let mut registry = Registry::default();

        let zone_used = Family::default();
        let zone_limit = Family::default();
        let zone_allocs = Family::default();
        let vram_used = Family::default();
        let vram_reserved = Family::default();
        let kv_pages_active = Gauge::default();
        let kv_pages_swapped = Gauge::default();
        let kv_sessions = Gauge::default();
        let kv_snapshots_total = Counter::default();
        let kv_forks_total = Counter::default();
        let kv_cow_copies_total = Counter::default();
        let hugepages_used = Gauge::default();
        let hugepages_free = Gauge::default();
        let pressure_level = Gauge::default();

        registry.register(
            "spirit_memory_zone_used_bytes",
            "Current memory usage per zone in bytes",
            zone_used.clone(),
        );
        registry.register(
            "spirit_memory_zone_limit_bytes",
            "Memory hard limit per zone in bytes",
            zone_limit.clone(),
        );
        registry.register(
            "spirit_memory_zone_allocations",
            "Number of tracked allocations per zone",
            zone_allocs.clone(),
        );
        registry.register(
            "spirit_vram_used_bytes",
            "GPU VRAM currently in use per device",
            vram_used.clone(),
        );
        registry.register(
            "spirit_vram_reserved_bytes",
            "GPU VRAM reserved per device",
            vram_reserved.clone(),
        );
        registry.register(
            "spirit_kv_pages_active",
            "Number of active KV-cache pages",
            kv_pages_active.clone(),
        );
        registry.register(
            "spirit_kv_pages_swapped",
            "Number of KV-cache pages swapped to NVMe",
            kv_pages_swapped.clone(),
        );
        registry.register(
            "spirit_kv_sessions",
            "Number of active KV-cache sessions",
            kv_sessions.clone(),
        );
        registry.register(
            "spirit_kv_snapshots_total",
            "Total KV-cache snapshots created",
            kv_snapshots_total.clone(),
        );
        registry.register(
            "spirit_kv_forks_total",
            "Total KV-cache forks created",
            kv_forks_total.clone(),
        );
        registry.register(
            "spirit_kv_cow_copies_total",
            "Total KV-cache copy-on-write copies",
            kv_cow_copies_total.clone(),
        );
        registry.register(
            "spirit_hugepages_used",
            "Number of hugepages currently in use",
            hugepages_used.clone(),
        );
        registry.register(
            "spirit_hugepages_free",
            "Number of free hugepages available",
            hugepages_free.clone(),
        );
        registry.register(
            "spirit_memory_pressure_level",
            "Current memory pressure level (0=none, 4=critical)",
            pressure_level.clone(),
        );

        Self {
            registry,
            zone_used,
            zone_limit,
            zone_allocs,
            vram_used,
            vram_reserved,
            kv_pages_active,
            kv_pages_swapped,
            kv_sessions,
            kv_snapshots_total,
            kv_forks_total,
            kv_cow_copies_total,
            hugepages_used,
            hugepages_free,
            pressure_level,
        }
    }

    /// Update zone metrics from zone stats.
    pub fn update_zone(&self, zone: MemoryZone, used: i64, limit: i64, allocs: i64) {
        let label = ZoneLabel {
            zone: ZoneName::from(zone),
        };
        self.zone_used.get_or_create(&label).set(used);
        self.zone_limit.get_or_create(&label).set(limit);
        self.zone_allocs.get_or_create(&label).set(allocs);
    }

    /// Update VRAM metrics for a device.
    pub fn update_vram(&self, device_name: &str, used: i64, reserved: i64) {
        let label = DeviceLabel {
            device: device_name.to_string(),
        };
        self.vram_used.get_or_create(&label).set(used);
        self.vram_reserved.get_or_create(&label).set(reserved);
    }

    /// Update KV-cache metrics.
    pub fn update_kv(&self, active: i64, swapped: i64, sessions: i64) {
        self.kv_pages_active.set(active);
        self.kv_pages_swapped.set(swapped);
        self.kv_sessions.set(sessions);
    }

    /// Update hugepage metrics.
    pub fn update_hugepages(&self, used: i64, free: i64) {
        self.hugepages_used.set(used);
        self.hugepages_free.set(free);
    }

    /// Update pressure level metric.
    pub fn update_pressure(&self, level: i64) {
        self.pressure_level.set(level);
    }

    /// Encode all metrics as Prometheus text format.
    pub fn encode(&self) -> String {
        let mut buf = String::new();
        prometheus_client::encoding::text::encode(&mut buf, &self.registry)
            .expect("encoding metrics should not fail");
        buf
    }
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MemoryMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryMetrics")
            .field("registered", &true)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = MemoryMetrics::new();
        // Registry is created and can encode without panic
        let _encoded = metrics.encode();
    }

    #[test]
    fn test_zone_metrics_update() {
        let metrics = MemoryMetrics::new();

        metrics.update_zone(MemoryZone::Model, 1000, 2000, 1);
        metrics.update_zone(MemoryZone::KvCache, 500, 1000, 3);

        let encoded = metrics.encode();
        assert!(encoded.contains("spirit_memory_zone_used_bytes"));
        assert!(encoded.contains("spirit_memory_zone_limit_bytes"));
    }

    #[test]
    fn test_kv_metrics_update() {
        let metrics = MemoryMetrics::new();
        metrics.update_kv(100, 20, 4);

        let encoded = metrics.encode();
        assert!(encoded.contains("spirit_kv_pages_active"));
        assert!(encoded.contains("spirit_kv_pages_swapped"));
    }

    #[test]
    fn test_pressure_metric() {
        let metrics = MemoryMetrics::new();
        metrics.update_pressure(3); // High

        let encoded = metrics.encode();
        assert!(encoded.contains("spirit_memory_pressure_level"));
    }

    #[test]
    fn test_vram_metrics() {
        let metrics = MemoryMetrics::new();
        metrics.update_vram("gpu0", 4000, 8000);

        let encoded = metrics.encode();
        assert!(encoded.contains("spirit_vram_used_bytes"));
        assert!(encoded.contains("spirit_vram_reserved_bytes"));
    }
}
