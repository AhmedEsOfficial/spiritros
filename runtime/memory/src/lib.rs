//! # Spirit Memory — `TensorFirst` Memory Manager
//!
//! This crate implements the Phase 1 userspace shim for the `TensorFirst` memory
//! architecture described in the `SpiritROS` Architecture Plan. It uses Linux
//! cgroups v2 and hugepages to enforce four distinct memory zones:
//!
//! - **`ZONE_MODEL`**: Read-only region for model weights (hugepage-backed)
//! - **`ZONE_KV`**: Copy-on-write KV-cache pages with refcounting
//! - **`ZONE_TENSOR`**: Scratch space for intermediate activation tensors
//! - **`ZONE_GENERAL`**: General-purpose allocations for non-inference workloads
//!
//! In Phase 2+, this logic moves into the `SpiritKernel`'s `mem` subsystem.
//!
//! // PHASE1-SHIM: Replace with kernel-native implementation in Phase 2

#![warn(clippy::all, clippy::pedantic)]

// ── Modules ──────────────────────────────────────────────────────────────────

/// Core type definitions for memory zones, pages, and allocations.
pub mod types {
    use serde::{Deserialize, Serialize};

    /// Identifies one of the four `TensorFirst` memory zones.
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

    /// Statistics for a single memory zone.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ZoneStats {
        /// Zone identifier.
        pub zone: MemoryZone,
        /// Total bytes allocated to this zone.
        pub total_bytes: u64,
        /// Bytes currently in use.
        pub used_bytes: u64,
        /// Number of active allocations.
        pub allocation_count: u64,
    }
}

/// Error types for memory operations.
pub mod error {
    use super::types::MemoryZone;

    /// Errors that can occur during Spirit memory operations.
    #[derive(Debug, thiserror::Error)]
    pub enum SpiritMemoryError {
        /// Attempted allocation exceeds zone capacity.
        #[error(
            "Zone {zone:?} exhausted: requested {requested} bytes, \
             available {available} bytes"
        )]
        ZoneExhausted {
            zone: MemoryZone,
            requested: u64,
            available: u64,
        },

        /// cgroups v2 operation failed.
        #[error("cgroups operation failed: {0}")]
        CgroupsError(String),

        /// Hugepage allocation failed.
        #[error("Hugepage allocation failed: requested {requested} pages, got {allocated}")]
        HugepageAllocationFailed { requested: u64, allocated: u64 },

        /// Attempted to mutate a read-only model zone.
        #[error("Zone ZONE_MODEL is read-only after model load")]
        ModelZoneReadOnly,

        /// Configuration error.
        #[error("Memory configuration error: {0}")]
        ConfigError(String),

        /// Underlying OS error.
        #[error("OS error: {0}")]
        OsError(#[from] nix::Error),

        /// I/O error interacting with cgroup filesystem.
        #[error("I/O error: {0}")]
        IoError(#[from] std::io::Error),
    }
}

pub mod zones {
    //! Memory zone management — allocation, enforcement, and statistics.
    //!
    //! Manages the four `TensorFirst` memory zones via cgroups v2 limits.
    //!
    //! Each zone maps to a cgroup under `/sys/fs/cgroup/spirit/`.

    use crate::types::MemoryZone;

    /// Returns the cgroup path suffix for the given zone.
    #[must_use]
    pub fn cgroup_path_for_zone(zone: MemoryZone) -> &'static str {
        match zone {
            MemoryZone::Model => "spirit/zone_model",
            MemoryZone::KvCache => "spirit/zone_kv",
            MemoryZone::Tensor => "spirit/zone_tensor",
            MemoryZone::General => "spirit/zone_general",
        }
    }
}

pub mod cgroups {
    //! Low-level cgroups v2 interface.
    //!
    //! Thin wrapper around cgroups v2 filesystem operations.
    //!
    //! Reads and writes to `/sys/fs/cgroup/` to enforce memory limits,
    //! monitor usage, and manage zone hierarchies.

    use crate::error::SpiritMemoryError;

    /// Sets the `memory.max` limit for a cgroup.
    ///
    /// # Errors
    ///
    /// Returns `SpiritMemoryError::IoError` if the cgroup file cannot be written.
    pub fn set_memory_limit(cgroup_path: &str, limit_bytes: u64) -> Result<(), SpiritMemoryError> {
        let path = format!("/sys/fs/cgroup/{cgroup_path}/memory.max");
        std::fs::write(&path, limit_bytes.to_string())?;
        Ok(())
    }

    /// Reads the current `memory.current` usage for a cgroup.
    ///
    /// # Errors
    ///
    /// Returns `SpiritMemoryError::IoError` if the cgroup file cannot be read.
    pub fn get_memory_usage(cgroup_path: &str) -> Result<u64, SpiritMemoryError> {
        let path = format!("/sys/fs/cgroup/{cgroup_path}/memory.current");
        let content = std::fs::read_to_string(&path)?;
        content
            .trim()
            .parse::<u64>()
            .map_err(|e| SpiritMemoryError::CgroupsError(e.to_string()))
    }
}

pub mod hugepages {
    //! Hugepage allocation and management.
    //!
    //! Manages hugepage (2 MiB / 1 GiB) reservations for `ZONE_MODEL`.
    //!
    //! Model weights benefit from hugepages because they reduce TLB pressure
    //! during the large sequential reads that dominate the prefill phase.
}

/// Configuration for the memory subsystem.
pub mod config {
    use serde::{Deserialize, Serialize};

    /// Configuration for the `TensorFirst` memory manager.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MemoryConfig {
        /// Maximum bytes for `ZONE_MODEL`.
        pub model_zone_bytes: u64,
        /// Maximum bytes for `ZONE_KV`.
        pub kv_zone_bytes: u64,
        /// Maximum bytes for `ZONE_TENSOR`.
        pub tensor_zone_bytes: u64,
        /// Maximum bytes for `ZONE_GENERAL`.
        pub general_zone_bytes: u64,
        /// Whether to use hugepages for the model zone.
        pub use_hugepages: bool,
        /// Hugepage size in bytes (default: 2 MiB).
        pub hugepage_size: u64,
    }

    impl Default for MemoryConfig {
        fn default() -> Self {
            Self {
                model_zone_bytes: 16 * 1024 * 1024 * 1024,  // 16 GiB
                kv_zone_bytes: 8 * 1024 * 1024 * 1024,      // 8 GiB
                tensor_zone_bytes: 4 * 1024 * 1024 * 1024,  // 4 GiB
                general_zone_bytes: 4 * 1024 * 1024 * 1024, // 4 GiB
                use_hugepages: true,
                hugepage_size: 2 * 1024 * 1024, // 2 MiB
            }
        }
    }
}

// ── Re-exports ───────────────────────────────────────────────────────────────

pub use config::MemoryConfig;
pub use error::SpiritMemoryError;
pub use types::{MemoryZone, ZoneStats};

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_zone_cgroup_paths_are_distinct() {
        let zones = [
            MemoryZone::Model,
            MemoryZone::KvCache,
            MemoryZone::Tensor,
            MemoryZone::General,
        ];
        let paths: Vec<&str> = zones
            .iter()
            .map(|z| zones::cgroup_path_for_zone(*z))
            .collect();
        // All paths must be unique.
        let mut deduped = paths.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(
            paths.len(),
            deduped.len(),
            "Zone cgroup paths must be unique"
        );
    }

    #[test]
    fn test_memory_config_default_has_sane_values() {
        let cfg = MemoryConfig::default();
        assert!(cfg.model_zone_bytes > 0);
        assert!(cfg.kv_zone_bytes > 0);
        assert!(cfg.tensor_zone_bytes > 0);
        assert!(cfg.general_zone_bytes > 0);
        assert!(cfg.use_hugepages);
        assert_eq!(cfg.hugepage_size, 2 * 1024 * 1024);
    }

    #[test]
    fn test_memory_zone_serialization_roundtrip() {
        let zone = MemoryZone::KvCache;
        let json = serde_json::to_string(&zone).expect("serialize");
        let back: MemoryZone = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(zone, back);
    }

    #[test]
    fn test_zone_stats_construction() {
        let stats = ZoneStats {
            zone: MemoryZone::Tensor,
            total_bytes: 4 * 1024 * 1024 * 1024,
            used_bytes: 1024 * 1024 * 512,
            allocation_count: 42,
        };
        assert_eq!(stats.zone, MemoryZone::Tensor);
        assert!(stats.used_bytes < stats.total_bytes);
    }

    #[test]
    fn test_spirit_memory_error_display() {
        let err = SpiritMemoryError::ZoneExhausted {
            zone: MemoryZone::KvCache,
            requested: 1024,
            available: 512,
        };
        let msg = err.to_string();
        assert!(msg.contains("KvCache"));
        assert!(msg.contains("1024"));
        assert!(msg.contains("512"));
    }
}
