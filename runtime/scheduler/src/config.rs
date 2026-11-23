//! Configuration for the Spirit Scheduler.
//!
//! Supports loading from YAML files, environment variables, and
//! programmatic construction via `figment`.

use serde::{Deserialize, Serialize};

use crate::error::{Result, SchedulerError};

/// Configuration for the Spirit Scheduler daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)] // config flags are naturally booleans
pub struct SchedulerConfig {
    /// Percentage of total cores to dedicate to inference (0.0-1.0).
    /// Default: 0.75 (75%).
    #[serde(default = "default_inference_ratio")]
    pub inference_core_ratio: f64,

    /// `SCHED_FIFO` priority for P0 (`InferenceCritical`) threads.
    #[serde(default = "default_p0_priority")]
    pub p0_fifo_priority: u32,

    /// `SCHED_FIFO` priority for P1 (`InferenceBatch`) threads.
    #[serde(default = "default_p1_priority")]
    pub p1_fifo_priority: u32,

    /// Nice value for P2 (`SystemServices`) threads.
    #[serde(default = "default_p2_nice")]
    pub p2_nice: i32,

    /// Nice value for P3 (`Compatibility`) threads.
    #[serde(default = "default_p3_nice")]
    pub p3_nice: i32,

    /// Nice value for P4 (`IdleMaintenance`) threads.
    #[serde(default = "default_p4_nice")]
    pub p4_nice: i32,

    /// Whether to enable GPU exclusive compute mode during inference.
    #[serde(default = "default_true")]
    pub gpu_exclusive_mode: bool,

    /// Minimum cores to always reserve for system services (P2),
    /// even during prefill burst.
    #[serde(default = "default_reserved_system_cores")]
    pub reserved_system_cores: usize,

    /// Watchdog timeout in milliseconds. If a P0 thread holds CPU
    /// for longer than this without yielding, it is demoted.
    #[serde(default = "default_watchdog_timeout_ms")]
    pub watchdog_timeout_ms: u64,

    /// Whether to enable watchdog monitoring.
    #[serde(default = "default_true")]
    pub watchdog_enabled: bool,

    /// Path to the cgroups v2 mount point.
    #[serde(default = "default_cgroup_root")]
    pub cgroup_root: String,

    /// Name of the cgroup subtree for Spirit scheduler.
    #[serde(default = "default_cgroup_name")]
    pub cgroup_name: String,

    /// Path for the scheduler IPC socket.
    #[serde(default = "default_socket_path")]
    pub socket_path: String,

    /// Whether to attempt `SCHED_FIFO` (requires `CAP_SYS_NICE`).
    /// Falls back to nice values if unavailable.
    #[serde(default = "default_true")]
    pub enable_realtime: bool,

    /// Whether to enable NUMA-aware core selection.
    #[serde(default = "default_true")]
    pub numa_aware: bool,

    /// Explicit inference core list (overrides `inference_core_ratio`).
    #[serde(default)]
    pub inference_cores_override: Option<Vec<usize>>,

    /// Explicit system core list (overrides `inference_core_ratio`).
    #[serde(default)]
    pub system_cores_override: Option<Vec<usize>>,
}

fn default_inference_ratio() -> f64 {
    0.75
}
fn default_p0_priority() -> u32 {
    99
}
fn default_p1_priority() -> u32 {
    50
}
fn default_p2_nice() -> i32 {
    -10
}
fn default_p3_nice() -> i32 {
    0
}
fn default_p4_nice() -> i32 {
    19
}
fn default_true() -> bool {
    true
}
fn default_reserved_system_cores() -> usize {
    1
}
fn default_watchdog_timeout_ms() -> u64 {
    5000
}
fn default_cgroup_root() -> String {
    "/sys/fs/cgroup".to_string()
}
fn default_cgroup_name() -> String {
    "spirit".to_string()
}
fn default_socket_path() -> String {
    "/run/spirit/scheduler.sock".to_string()
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            inference_core_ratio: default_inference_ratio(),
            p0_fifo_priority: default_p0_priority(),
            p1_fifo_priority: default_p1_priority(),
            p2_nice: default_p2_nice(),
            p3_nice: default_p3_nice(),
            p4_nice: default_p4_nice(),
            gpu_exclusive_mode: true,
            reserved_system_cores: default_reserved_system_cores(),
            watchdog_timeout_ms: default_watchdog_timeout_ms(),
            watchdog_enabled: true,
            cgroup_root: default_cgroup_root(),
            cgroup_name: default_cgroup_name(),
            socket_path: default_socket_path(),
            enable_realtime: true,
            numa_aware: true,
            inference_cores_override: None,
            system_cores_override: None,
        }
    }
}

impl SchedulerConfig {
    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns `SchedulerError::ConfigError` if any configuration value is
    /// out of range or violates invariants.
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.inference_core_ratio) {
            return Err(SchedulerError::ConfigError(format!(
                "inference_core_ratio must be 0.0-1.0, got {}",
                self.inference_core_ratio
            )));
        }
        if !(1..=99).contains(&self.p0_fifo_priority) {
            return Err(SchedulerError::ConfigError(format!(
                "p0_fifo_priority must be 1-99, got {}",
                self.p0_fifo_priority
            )));
        }
        if !(1..=99).contains(&self.p1_fifo_priority) {
            return Err(SchedulerError::ConfigError(format!(
                "p1_fifo_priority must be 1-99, got {}",
                self.p1_fifo_priority
            )));
        }
        if self.p0_fifo_priority <= self.p1_fifo_priority {
            return Err(SchedulerError::ConfigError(
                "p0_fifo_priority must be higher than p1_fifo_priority".to_string(),
            ));
        }
        if self.watchdog_timeout_ms < 100 {
            return Err(SchedulerError::ConfigError(
                "watchdog_timeout_ms must be >= 100".to_string(),
            ));
        }
        if self.reserved_system_cores == 0 {
            return Err(SchedulerError::ConfigError(
                "reserved_system_cores must be >= 1 to prevent system starvation".to_string(),
            ));
        }
        Ok(())
    }

    /// Load configuration from a YAML file, with environment variable overrides.
    ///
    /// # Errors
    ///
    /// Returns `SchedulerError::ConfigError` if the file cannot be read,
    /// parsed, or if validation fails.
    pub fn load(path: Option<&str>) -> Result<Self> {
        use figment::providers::{Env, Format, Serialized, Yaml};
        use figment::Figment;

        let mut figment = Figment::from(Serialized::defaults(Self::default()));

        if let Some(p) = path {
            figment = figment.merge(Yaml::file(p).nested());
        }

        figment = figment.merge(Env::prefixed("SPIRIT_SCHEDULER_").split("_"));

        let config: Self = figment
            .extract()
            .map_err(|e| SchedulerError::ConfigError(e.to_string()))?;

        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validates() {
        let cfg = SchedulerConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_invalid_inference_ratio() {
        let mut cfg = SchedulerConfig::default();
        cfg.inference_core_ratio = 1.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_invalid_fifo_priority() {
        let mut cfg = SchedulerConfig::default();
        cfg.p0_fifo_priority = 100;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_p0_must_exceed_p1() {
        let mut cfg = SchedulerConfig::default();
        cfg.p0_fifo_priority = 50;
        cfg.p1_fifo_priority = 50;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_reserved_cores_nonzero() {
        let mut cfg = SchedulerConfig::default();
        cfg.reserved_system_cores = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let cfg = SchedulerConfig::default();
        let yaml = serde_yaml::to_string(&cfg).expect("serialize");
        let back: SchedulerConfig = serde_yaml::from_str(&yaml).expect("deserialize");
        assert_eq!(back.p0_fifo_priority, cfg.p0_fifo_priority);
    }
}
