//! cgroups v2 interface for memory zone enforcement.
//!
//! Creates and manages the cgroup hierarchy under
//! `/sys/fs/cgroup/spirit/zone_{model,kv,tensor,general}`.
//!
//! Each zone maps to a cgroup subtree with:
//! - `memory.max`  — hard limit (OOM kill threshold)
//! - `memory.high` — throttling threshold (90% of max)
//! - `memory.min`  — guaranteed minimum (kernel never reclaims)
//! - `memory.swap.max` — swap limit (0 for inference zones)
//!
//! // PHASE1-SHIM: Replace with kernel-native zone enforcement in Phase 2

use std::path::{Path, PathBuf};

use tracing::{debug, info, warn};

use crate::config::MemoryConfig;
use crate::error::{Result, SpiritMemoryError};
use crate::types::MemoryZone;

/// Manages cgroups v2 memory zones for the TensorFirst allocator.
#[derive(Debug)]
pub struct CgroupManager {
    root: PathBuf,
    available: bool,
}

impl CgroupManager {
    /// Create a new cgroup manager rooted at the given path.
    ///
    /// Probes whether cgroups v2 is available. If not, the manager
    /// operates in degraded mode (logging warnings, but not failing).
    pub fn new(root: &Path) -> Self {
        let available = Self::probe_cgroupv2(root);
        if available {
            debug!("cgroups v2 available at {}", root.display());
        } else {
            warn!(
                "cgroups v2 not available at {}; memory zone enforcement \
                 will be advisory only",
                root.display()
            );
        }
        Self {
            root: root.to_path_buf(),
            available,
        }
    }

    /// Whether cgroups v2 is available on this system.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Set up the full zone hierarchy and apply limits from config.
    pub fn setup_zones(&self, config: &MemoryConfig) -> Result<()> {
        if !self.available {
            warn!("skipping cgroup zone setup (cgroups v2 not available)");
            return Ok(());
        }

        // Ensure parent "spirit" cgroup exists
        self.ensure_cgroup("spirit")?;

        // Enable memory controller in the parent
        self.enable_controller("spirit", "memory");

        // Create and configure each zone
        let zones = [
            (MemoryZone::Model, config.model_zone_bytes),
            (MemoryZone::KvCache, config.kv_zone_bytes),
            (MemoryZone::Tensor, config.tensor_zone_bytes),
            (MemoryZone::General, config.general_zone_bytes),
        ];

        for (zone, limit) in &zones {
            self.setup_zone(*zone, *limit, config)?;
        }

        info!("cgroups v2 memory zones configured successfully");
        Ok(())
    }

    /// Configure a single zone cgroup with appropriate limits.
    fn setup_zone(&self, zone: MemoryZone, limit_bytes: u64, config: &MemoryConfig) -> Result<()> {
        let cgroup_name = zone.cgroup_name();
        self.ensure_cgroup(cgroup_name)?;

        // memory.max — hard limit
        self.write_cgroup_file(cgroup_name, "memory.max", limit_bytes)?;

        // memory.high — throttling at 90% of max
        let high = (limit_bytes as f64 * 0.90) as u64;
        self.write_cgroup_file(cgroup_name, "memory.high", high)?;

        // memory.min — guaranteed minimum for inference-critical zones
        let min = match zone {
            // Model weights must NEVER be reclaimed
            MemoryZone::Model => limit_bytes,
            // KV-cache gets 50% guaranteed
            MemoryZone::KvCache => limit_bytes / 2,
            // Tensor and general have no guaranteed minimum
            MemoryZone::Tensor | MemoryZone::General => 0,
        };
        if min > 0 {
            self.write_cgroup_file(cgroup_name, "memory.min", min)?;
        }

        // memory.swap.max — disable swap for model and tensor zones
        let swap_max = match zone {
            MemoryZone::Model | MemoryZone::Tensor => 0,
            // KV-cache can overflow to NVMe swap
            MemoryZone::KvCache => config.kv_zone_bytes / 2,
            // General can swap freely
            MemoryZone::General => u64::MAX,
        };
        self.write_cgroup_file(cgroup_name, "memory.swap.max", swap_max)?;

        info!(
            zone = zone.label(),
            limit_bytes, high, min, swap_max, "configured cgroup zone"
        );

        Ok(())
    }

    /// Read the current memory usage for a zone.
    pub fn get_memory_usage(&self, zone: MemoryZone) -> Result<u64> {
        if !self.available {
            return Ok(0);
        }
        self.read_cgroup_u64(zone.cgroup_name(), "memory.current")
    }

    /// Read the memory.max limit for a zone.
    pub fn get_memory_limit(&self, zone: MemoryZone) -> Result<u64> {
        if !self.available {
            return Ok(u64::MAX);
        }
        self.read_cgroup_u64(zone.cgroup_name(), "memory.max")
    }

    /// Read the memory.high threshold for a zone.
    pub fn get_memory_high(&self, zone: MemoryZone) -> Result<u64> {
        if !self.available {
            return Ok(u64::MAX);
        }
        self.read_cgroup_u64(zone.cgroup_name(), "memory.high")
    }

    /// Read the memory.min guaranteed floor for a zone.
    pub fn get_memory_min(&self, zone: MemoryZone) -> Result<u64> {
        if !self.available {
            return Ok(0);
        }
        self.read_cgroup_u64(zone.cgroup_name(), "memory.min")
    }

    /// Read the current swap usage for a zone.
    pub fn get_swap_usage(&self, zone: MemoryZone) -> Result<u64> {
        if !self.available {
            return Ok(0);
        }
        self.read_cgroup_u64(zone.cgroup_name(), "memory.swap.current")
    }

    /// Update the memory.max for a zone (used during rebalancing).
    pub fn set_memory_limit(&self, zone: MemoryZone, limit_bytes: u64) -> Result<()> {
        if !self.available {
            warn!(
                zone = zone.label(),
                limit_bytes, "cgroups unavailable; limit change is advisory"
            );
            return Ok(());
        }
        self.write_cgroup_file(zone.cgroup_name(), "memory.max", limit_bytes)?;
        // Also update memory.high to 90% of the new limit
        let high = (limit_bytes as f64 * 0.90) as u64;
        self.write_cgroup_file(zone.cgroup_name(), "memory.high", high)?;
        info!(
            zone = zone.label(),
            limit_bytes, high, "updated cgroup zone limits"
        );
        Ok(())
    }

    /// Move a process into a zone's cgroup.
    pub fn assign_process(&self, zone: MemoryZone, pid: u32) -> Result<()> {
        if !self.available {
            return Ok(());
        }
        let path = self.root.join(zone.cgroup_name()).join("cgroup.procs");
        std::fs::write(&path, pid.to_string()).map_err(|e| {
            SpiritMemoryError::CgroupsError(format!(
                "failed to assign PID {pid} to zone {}: {e}",
                zone.label()
            ))
        })?;
        debug!(zone = zone.label(), pid, "assigned process to zone");
        Ok(())
    }

    // ── Internal helpers ────────────────────────────────────────────────

    fn probe_cgroupv2(root: &Path) -> bool {
        root.join("cgroup.controllers").exists()
    }

    fn ensure_cgroup(&self, name: &str) -> Result<()> {
        let path = self.root.join(name);
        if !path.exists() {
            std::fs::create_dir_all(&path).map_err(|e| {
                SpiritMemoryError::CgroupsError(format!(
                    "failed to create cgroup {}: {e}",
                    path.display()
                ))
            })?;
            debug!("created cgroup directory {}", path.display());
        }
        Ok(())
    }

    fn enable_controller(&self, cgroup: &str, controller: &str) {
        let path = self.root.join(cgroup).join("cgroup.subtree_control");
        let content = format!("+{controller}");
        // This may fail if the controller is already enabled or not
        // delegated — that's fine.
        if let Err(e) = std::fs::write(&path, &content) {
            debug!(
                "could not enable controller '{}' in {}: {} (may already \
                 be enabled)",
                controller,
                path.display(),
                e
            );
        }
    }

    fn write_cgroup_file(&self, cgroup: &str, file: &str, value: u64) -> Result<()> {
        let path = self.root.join(cgroup).join(file);
        let content = if value == u64::MAX {
            "max".to_string()
        } else {
            value.to_string()
        };
        std::fs::write(&path, &content).map_err(|e| {
            SpiritMemoryError::CgroupsError(format!(
                "failed to write '{}' to {}: {e}",
                content,
                path.display()
            ))
        })?;
        Ok(())
    }

    fn read_cgroup_u64(&self, cgroup: &str, file: &str) -> Result<u64> {
        let path = self.root.join(cgroup).join(file);
        let content = std::fs::read_to_string(&path)?;
        let trimmed = content.trim();
        if trimmed == "max" {
            return Ok(u64::MAX);
        }
        trimmed.parse::<u64>().map_err(|e| {
            SpiritMemoryError::CgroupsError(format!(
                "failed to parse '{}' from {}: {e}",
                trimmed,
                path.display()
            ))
        })
    }
}

/// Detect total system RAM from `/proc/meminfo`.
pub fn detect_system_memory() -> Result<u64> {
    let content = std::fs::read_to_string("/proc/meminfo")?;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb_str = rest.trim().trim_end_matches("kB").trim();
            let kb: u64 = kb_str.parse().map_err(|e| {
                SpiritMemoryError::CgroupsError(format!("failed to parse MemTotal: {e}"))
            })?;
            return Ok(kb * 1024);
        }
    }
    Err(SpiritMemoryError::CgroupsError(
        "MemTotal not found in /proc/meminfo".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Minimal temp dir for testing (avoids external tempfile crate).
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let id = uuid::Uuid::new_v4();
            let path = std::env::temp_dir().join(format!("spirit-cgroup-test-{id}"));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn make_fake_cgroup() -> (TempDir, CgroupManager) {
        let dir = TempDir::new();
        fs::write(dir.path().join("cgroup.controllers"), "memory cpu").unwrap();
        let mgr = CgroupManager::new(dir.path());
        (dir, mgr)
    }

    #[test]
    fn test_probe_detects_cgroupv2() {
        let (dir, mgr) = make_fake_cgroup();
        assert!(mgr.is_available());
        drop(dir);
    }

    #[test]
    fn test_probe_missing_returns_false() {
        let dir = TempDir::new();
        // Don't create cgroup.controllers
        let mgr = CgroupManager::new(dir.path());
        assert!(!mgr.is_available());
    }

    #[test]
    fn test_setup_zones_creates_directories() {
        let (dir, mgr) = make_fake_cgroup();
        let config = MemoryConfig::for_test();
        mgr.setup_zones(&config).unwrap();

        for zone in MemoryZone::ALL {
            let zone_path = dir.path().join(zone.cgroup_name());
            assert!(zone_path.exists(), "zone dir missing: {zone}");
            assert!(zone_path.join("memory.max").exists());
            assert!(zone_path.join("memory.high").exists());
            assert!(zone_path.join("memory.swap.max").exists());
        }
    }

    #[test]
    fn test_read_write_roundtrip() {
        let (dir, mgr) = make_fake_cgroup();
        let config = MemoryConfig::for_test();
        mgr.setup_zones(&config).unwrap();

        let limit = mgr
            .read_cgroup_u64("spirit/zone_model", "memory.max")
            .unwrap();
        assert_eq!(limit, config.model_zone_bytes);
    }

    #[test]
    fn test_model_zone_has_memory_min() {
        let (dir, mgr) = make_fake_cgroup();
        let config = MemoryConfig::for_test();
        mgr.setup_zones(&config).unwrap();

        // Model zone memory.min = limit (full protection)
        let min = mgr
            .read_cgroup_u64("spirit/zone_model", "memory.min")
            .unwrap();
        assert_eq!(min, config.model_zone_bytes);
    }

    #[test]
    fn test_model_zone_swap_disabled() {
        let (dir, mgr) = make_fake_cgroup();
        let config = MemoryConfig::for_test();
        mgr.setup_zones(&config).unwrap();

        let swap = mgr
            .read_cgroup_u64("spirit/zone_model", "memory.swap.max")
            .unwrap();
        assert_eq!(swap, 0);
    }

    #[test]
    fn test_general_zone_swap_is_max() {
        let (dir, mgr) = make_fake_cgroup();
        let config = MemoryConfig::for_test();
        mgr.setup_zones(&config).unwrap();

        let swap = mgr
            .read_cgroup_u64("spirit/zone_general", "memory.swap.max")
            .unwrap();
        assert_eq!(swap, u64::MAX);
    }

    #[test]
    fn test_set_memory_limit_updates() {
        let (dir, mgr) = make_fake_cgroup();
        let config = MemoryConfig::for_test();
        mgr.setup_zones(&config).unwrap();

        let new_limit = 8 * crate::types::GIB;
        mgr.set_memory_limit(MemoryZone::Model, new_limit).unwrap();

        let read_back = mgr
            .read_cgroup_u64("spirit/zone_model", "memory.max")
            .unwrap();
        assert_eq!(read_back, new_limit);
    }

    #[test]
    fn test_unavailable_cgroups_dont_fail() {
        let dir = TempDir::new();
        let mgr = CgroupManager::new(dir.path());
        assert!(!mgr.is_available());

        let config = MemoryConfig::for_test();
        // Should not fail, just warn
        assert!(mgr.setup_zones(&config).is_ok());
        assert_eq!(mgr.get_memory_usage(MemoryZone::Model).unwrap(), 0);
    }

    #[test]
    fn test_detect_system_memory() {
        // This reads actual /proc/meminfo, so it should work on Linux
        if std::path::Path::new("/proc/meminfo").exists() {
            let mem = detect_system_memory().unwrap();
            assert!(mem > 0, "system memory should be > 0");
        }
    }
}
