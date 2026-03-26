//! Configuration for the TensorFirst memory manager.
//!
//! Supports a three-level hierarchy:
//! 1. `spirit.yaml` defaults (via figment YAML provider)
//! 2. Environment variable overrides (`SPIRIT_ZONE_MODEL`, etc.)
//! 3. Runtime API calls (`rebalance`)

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::types::{GIB, HUGEPAGE_SIZE_2M, KV_PAGE_SIZE};

/// Top-level memory configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum bytes for ZONE_MODEL. 0 = auto-detect (40% of budget).
    #[serde(default = "defaults::model_zone_bytes")]
    pub model_zone_bytes: u64,

    /// Maximum bytes for ZONE_KV. 0 = auto-detect (30% of budget).
    #[serde(default = "defaults::kv_zone_bytes")]
    pub kv_zone_bytes: u64,

    /// Maximum bytes for ZONE_TENSOR. 0 = auto-detect (20% of budget).
    #[serde(default = "defaults::tensor_zone_bytes")]
    pub tensor_zone_bytes: u64,

    /// Maximum bytes for ZONE_GENERAL. 0 = auto-detect (10% of budget).
    #[serde(default = "defaults::general_zone_bytes")]
    pub general_zone_bytes: u64,

    /// Fraction of total system RAM to use (0.0–1.0). Default: 0.90.
    #[serde(default = "defaults::budget_fraction")]
    pub budget_fraction: f64,

    /// Whether to attempt hugepage-backed model loading.
    #[serde(default = "defaults::use_hugepages")]
    pub use_hugepages: bool,

    /// Hugepage size in bytes. Default: 2 MiB.
    #[serde(default = "defaults::hugepage_size")]
    pub hugepage_size: u64,

    /// KV-cache page pool configuration.
    #[serde(default)]
    pub kv_cache: KvCacheConfig,

    /// VRAM management configuration.
    #[serde(default)]
    pub vram: VramConfig,

    /// Pressure monitoring configuration.
    #[serde(default)]
    pub pressure: PressureConfig,

    /// Metrics configuration.
    #[serde(default)]
    pub metrics: MetricsConfig,

    /// Root path for cgroups v2 filesystem.
    #[serde(default = "defaults::cgroup_root")]
    pub cgroup_root: PathBuf,
}

/// KV-cache specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheConfig {
    /// Size of each KV-cache page in bytes. Default: 4096.
    #[serde(default = "defaults::kv_page_size")]
    pub page_size: usize,

    /// Maximum pages in the KV-cache pool. Default: 262144 (1 GiB at 4K).
    #[serde(default = "defaults::kv_max_pages")]
    pub max_pages: usize,

    /// Maximum cached sessions. Default: 64.
    #[serde(default = "defaults::kv_max_sessions")]
    pub max_sessions: usize,

    /// Directory for NVMe swap. Default: /var/lib/spiritros/kv-swap.
    #[serde(default = "defaults::kv_swap_dir")]
    pub swap_dir: PathBuf,

    /// Whether copy-on-write is enabled for snapshots/forks.
    #[serde(default = "defaults::kv_cow_enabled")]
    pub cow_enabled: bool,
}

/// VRAM configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VramConfig {
    /// Fraction of each GPU's VRAM to reserve (0.0–1.0). Default: 0.90.
    #[serde(default = "defaults::vram_fraction")]
    pub memory_fraction: f64,

    /// Whether to enable CUDA MPS for multi-process sharing.
    #[serde(default = "defaults::vram_mps_enabled")]
    pub mps_enabled: bool,
}

/// Memory pressure monitoring configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PressureConfig {
    /// Whether to monitor PSI and respond to pressure.
    #[serde(default = "defaults::pressure_enabled")]
    pub enabled: bool,

    /// Poll interval in milliseconds. Default: 1000.
    #[serde(default = "defaults::pressure_poll_ms")]
    pub poll_interval_ms: u64,
}

/// Prometheus metrics configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Whether to expose Prometheus metrics.
    #[serde(default = "defaults::metrics_enabled")]
    pub enabled: bool,
}

// ── Defaults ────────────────────────────────────────────────────────────────

mod defaults {
    use super::{PathBuf, HUGEPAGE_SIZE_2M, KV_PAGE_SIZE};

    pub fn model_zone_bytes() -> u64 {
        0
    }
    pub fn kv_zone_bytes() -> u64 {
        0
    }
    pub fn tensor_zone_bytes() -> u64 {
        0
    }
    pub fn general_zone_bytes() -> u64 {
        0
    }
    pub fn budget_fraction() -> f64 {
        0.90
    }
    pub fn use_hugepages() -> bool {
        true
    }
    pub fn hugepage_size() -> u64 {
        HUGEPAGE_SIZE_2M
    }
    pub fn cgroup_root() -> PathBuf {
        PathBuf::from("/sys/fs/cgroup")
    }

    pub fn kv_page_size() -> usize {
        KV_PAGE_SIZE
    }
    pub fn kv_max_pages() -> usize {
        262_144
    } // 1 GiB at 4K pages
    pub fn kv_max_sessions() -> usize {
        64
    }
    pub fn kv_swap_dir() -> PathBuf {
        PathBuf::from("/var/lib/spiritros/kv-swap")
    }
    pub fn kv_cow_enabled() -> bool {
        true
    }

    pub fn vram_fraction() -> f64 {
        0.90
    }
    pub fn vram_mps_enabled() -> bool {
        true
    }

    pub fn pressure_enabled() -> bool {
        true
    }
    pub fn pressure_poll_ms() -> u64 {
        1000
    }

    pub fn metrics_enabled() -> bool {
        true
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            model_zone_bytes: defaults::model_zone_bytes(),
            kv_zone_bytes: defaults::kv_zone_bytes(),
            tensor_zone_bytes: defaults::tensor_zone_bytes(),
            general_zone_bytes: defaults::general_zone_bytes(),
            budget_fraction: defaults::budget_fraction(),
            use_hugepages: defaults::use_hugepages(),
            hugepage_size: defaults::hugepage_size(),
            kv_cache: KvCacheConfig::default(),
            vram: VramConfig::default(),
            pressure: PressureConfig::default(),
            metrics: MetricsConfig::default(),
            cgroup_root: defaults::cgroup_root(),
        }
    }
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            page_size: defaults::kv_page_size(),
            max_pages: defaults::kv_max_pages(),
            max_sessions: defaults::kv_max_sessions(),
            swap_dir: defaults::kv_swap_dir(),
            cow_enabled: defaults::kv_cow_enabled(),
        }
    }
}

impl Default for VramConfig {
    fn default() -> Self {
        Self {
            memory_fraction: defaults::vram_fraction(),
            mps_enabled: defaults::vram_mps_enabled(),
        }
    }
}

impl Default for PressureConfig {
    fn default() -> Self {
        Self {
            enabled: defaults::pressure_enabled(),
            poll_interval_ms: defaults::pressure_poll_ms(),
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: defaults::metrics_enabled(),
        }
    }
}

impl MemoryConfig {
    /// Load configuration from figment providers: YAML file + env vars.
    ///
    /// Environment variables override YAML values. The prefix is `SPIRIT_`:
    /// - `SPIRIT_ZONE_MODEL` → `model_zone_bytes`
    /// - `SPIRIT_ZONE_KV` → `kv_zone_bytes`
    /// - `SPIRIT_ZONE_TENSOR` → `tensor_zone_bytes`
    /// - `SPIRIT_ZONE_GENERAL` → `general_zone_bytes`
    /// - `SPIRIT_HUGEPAGES` → `use_hugepages`
    /// - `SPIRIT_BUDGET_FRACTION` → `budget_fraction`
    ///
    /// # Errors
    ///
    /// Returns `ConfigError` if the YAML file is malformed or values are
    /// invalid.
    pub fn from_file(yaml_path: &std::path::Path) -> crate::error::Result<Self> {
        use figment::providers::{Env, Format, Serialized, Yaml};
        use figment::Figment;

        let config: Self = Figment::new()
            .merge(Serialized::defaults(Self::default()))
            .merge(Yaml::file(yaml_path).nested())
            .merge(
                Env::prefixed("SPIRIT_")
                    .map(|key| key.as_str().replace("ZONE_", "").to_lowercase().into()),
            )
            .extract()
            .map_err(|e| crate::error::SpiritMemoryError::ConfigError(e.to_string()))?;

        config.validate()?;
        Ok(config)
    }

    /// Auto-size zones based on detected system memory.
    ///
    /// If any zone size is 0, it is calculated as a percentage of the total
    /// memory budget (total RAM × `budget_fraction`).
    pub fn auto_size(&mut self, total_system_ram: u64) {
        let budget = (total_system_ram as f64 * self.budget_fraction) as u64;

        if self.model_zone_bytes == 0 {
            self.model_zone_bytes = (budget as f64 * 0.40) as u64;
        }
        if self.kv_zone_bytes == 0 {
            self.kv_zone_bytes = (budget as f64 * 0.30) as u64;
        }
        if self.tensor_zone_bytes == 0 {
            self.tensor_zone_bytes = (budget as f64 * 0.20) as u64;
        }
        if self.general_zone_bytes == 0 {
            self.general_zone_bytes = (budget as f64 * 0.10) as u64;
        }
    }

    /// Total bytes committed across all four zones.
    #[must_use]
    pub fn total_zone_bytes(&self) -> u64 {
        self.model_zone_bytes
            + self.kv_zone_bytes
            + self.tensor_zone_bytes
            + self.general_zone_bytes
    }

    /// Validate that the configuration is internally consistent.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.budget_fraction <= 0.0 || self.budget_fraction > 1.0 {
            return Err(crate::error::SpiritMemoryError::ConfigError(format!(
                "budget_fraction must be in (0.0, 1.0], got {}",
                self.budget_fraction
            )));
        }
        if self.hugepage_size != HUGEPAGE_SIZE_2M && self.hugepage_size != 1 * GIB {
            return Err(crate::error::SpiritMemoryError::ConfigError(format!(
                "hugepage_size must be {} (2M) or {} (1G), got {}",
                HUGEPAGE_SIZE_2M,
                1 * GIB,
                self.hugepage_size
            )));
        }
        if self.kv_cache.page_size == 0 {
            return Err(crate::error::SpiritMemoryError::ConfigError(
                "kv_cache.page_size must be > 0".into(),
            ));
        }
        if self.vram.memory_fraction <= 0.0 || self.vram.memory_fraction > 1.0 {
            return Err(crate::error::SpiritMemoryError::ConfigError(format!(
                "vram.memory_fraction must be in (0.0, 1.0], got {}",
                self.vram.memory_fraction
            )));
        }
        Ok(())
    }

    /// Build an explicit-size config for testing (all zones set, no auto).
    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        Self {
            model_zone_bytes: 4 * GIB,
            kv_zone_bytes: 2 * GIB,
            tensor_zone_bytes: 1 * GIB,
            general_zone_bytes: 1 * GIB,
            budget_fraction: 0.90,
            use_hugepages: false, // not available in test env
            hugepage_size: HUGEPAGE_SIZE_2M,
            kv_cache: KvCacheConfig {
                page_size: KV_PAGE_SIZE,
                max_pages: 1024,
                max_sessions: 8,
                swap_dir: std::env::temp_dir().join("spirit-kv-swap-test"),
                cow_enabled: true,
            },
            vram: VramConfig::default(),
            pressure: PressureConfig {
                enabled: false,
                ..Default::default()
            },
            metrics: MetricsConfig { enabled: false },
            cgroup_root: PathBuf::from("/tmp/spirit-cgroup-test"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let cfg = MemoryConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_auto_size_fills_zero_zones() {
        let mut cfg = MemoryConfig::default();
        assert_eq!(cfg.model_zone_bytes, 0);

        let ram = 64 * GIB;
        cfg.auto_size(ram);

        let budget = (ram as f64 * 0.90) as u64;
        assert_eq!(cfg.model_zone_bytes, (budget as f64 * 0.40) as u64);
        assert_eq!(cfg.kv_zone_bytes, (budget as f64 * 0.30) as u64);
        assert_eq!(cfg.tensor_zone_bytes, (budget as f64 * 0.20) as u64);
        assert_eq!(cfg.general_zone_bytes, (budget as f64 * 0.10) as u64);
    }

    #[test]
    fn test_auto_size_preserves_explicit_values() {
        let mut cfg = MemoryConfig::default();
        cfg.model_zone_bytes = 8 * GIB;
        cfg.auto_size(64 * GIB);
        assert_eq!(cfg.model_zone_bytes, 8 * GIB);
    }

    #[test]
    fn test_invalid_budget_fraction() {
        let mut cfg = MemoryConfig::default();
        cfg.budget_fraction = 0.0;
        assert!(cfg.validate().is_err());

        cfg.budget_fraction = 1.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_invalid_hugepage_size() {
        let mut cfg = MemoryConfig::default();
        cfg.hugepage_size = 4096; // not 2M or 1G
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_total_zone_bytes() {
        let cfg = MemoryConfig::for_test();
        assert_eq!(cfg.total_zone_bytes(), 8 * GIB);
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let cfg = MemoryConfig::for_test();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: MemoryConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.model_zone_bytes, cfg.model_zone_bytes);
        assert_eq!(back.kv_cache.max_pages, cfg.kv_cache.max_pages);
    }
}
