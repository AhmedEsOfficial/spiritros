//! Configuration for the inference subsystem.
//!
//! Loads from `spirit.yaml` inference section with environment variable
//! overrides (prefixed `SPIRIT_INFERENCE_`). Uses `figment` for multi-source
//! configuration merging.

use std::path::{Path, PathBuf};

use figment::providers::{Env, Format, Serialized, Yaml};
use figment::Figment;
use serde::{Deserialize, Serialize};

use crate::error::InferenceError;

/// Top-level configuration for the inference coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Backend type: "ollama" (Phase 1 default).
    #[serde(default = "default_backend")]
    pub backend: String,

    /// Ollama-specific configuration.
    #[serde(default)]
    pub ollama: OllamaConfig,

    /// Coordinator settings.
    #[serde(default)]
    pub coordinator: CoordinatorConfig,

    /// KV-cache management.
    #[serde(default)]
    pub kv_cache: KvCacheConfig,

    /// GPU settings.
    #[serde(default)]
    pub gpu: GpuConfig,
}

/// Configuration for the Ollama backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    /// Base URL for the Ollama API.
    #[serde(default = "default_ollama_url")]
    pub url: String,

    /// Maximum number of concurrent inference sessions.
    #[serde(default = "default_max_sessions")]
    pub max_sessions: usize,

    /// Default model to load on startup.
    #[serde(default)]
    pub default_model: Option<String>,

    /// Request timeout in seconds.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,

    /// Keep-alive duration for models (seconds, 0 = never unload).
    #[serde(default = "default_keep_alive")]
    pub keep_alive_secs: u64,
}

/// Coordinator process settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorConfig {
    /// Listen address for the IPC socket.
    #[serde(default = "default_socket_path")]
    pub socket_path: String,

    /// gRPC listen address.
    #[serde(default = "default_grpc_addr")]
    pub grpc_addr: String,

    /// Maximum queued requests before rejecting.
    #[serde(default = "default_max_queue_depth")]
    pub max_queue_depth: usize,

    /// Pre-allocated token buffer size per session (in tokens).
    #[serde(default = "default_token_buffer_size")]
    pub token_buffer_size: usize,

    /// Use SCHED_FIFO for decode threads.
    #[serde(default)]
    pub realtime_scheduling: bool,

    /// Decode thread FIFO priority (1-99).
    #[serde(default = "default_realtime_priority")]
    pub realtime_priority: u32,

    /// Shared memory path for zero-copy token streaming.
    #[serde(default = "default_shm_path")]
    pub shm_path: String,

    /// Shared memory ring buffer capacity (number of `TokenEntry` slots).
    /// Must be a power of 2.
    #[serde(default = "default_shm_capacity")]
    pub shm_capacity: usize,
}

/// KV-cache management configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheConfig {
    /// Eviction policy: "lru", "lfu", or "size-aware".
    #[serde(default = "default_eviction_policy")]
    pub eviction_policy: String,

    /// Maximum KV-cache size per session in MB (0 = unlimited within zone).
    #[serde(default)]
    pub max_per_session_mb: u64,

    /// Enable snapshot/restore for session continuity.
    #[serde(default = "default_true")]
    pub snapshots_enabled: bool,

    /// Directory for persisted KV-cache snapshots.
    #[serde(default = "default_snapshot_dir")]
    pub snapshot_dir: PathBuf,
}

/// GPU configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Layer offloading strategy: "auto", "full", or integer.
    #[serde(default = "default_offload")]
    pub offload: String,

    /// Enable CUDA MPS for multi-session GPU sharing.
    #[serde(default = "default_true")]
    pub mps_enabled: bool,

    /// GPU memory fraction to reserve (0.0-1.0).
    #[serde(default = "default_memory_fraction")]
    pub memory_fraction: f64,
}

// ── Defaults ────────────────────────────────────────────────────────────────

fn default_backend() -> String {
    "ollama".into()
}
fn default_ollama_url() -> String {
    "http://127.0.0.1:11434".into()
}
fn default_max_sessions() -> usize {
    8
}
fn default_timeout_secs() -> u64 {
    300
}
fn default_keep_alive() -> u64 {
    86400
}
fn default_socket_path() -> String {
    "/run/spirit/infer.sock".into()
}
fn default_grpc_addr() -> String {
    "0.0.0.0:50051".into()
}
fn default_max_queue_depth() -> usize {
    256
}
fn default_token_buffer_size() -> usize {
    4096
}
fn default_realtime_priority() -> u32 {
    80
}
fn default_shm_path() -> String {
    "/dev/shm/spirit_infer".into()
}
fn default_shm_capacity() -> usize {
    8192
}
fn default_eviction_policy() -> String {
    "lru".into()
}
fn default_true() -> bool {
    true
}
fn default_snapshot_dir() -> PathBuf {
    PathBuf::from("/var/lib/spiritros/data/kv-snapshots")
}
fn default_offload() -> String {
    "auto".into()
}
fn default_memory_fraction() -> f64 {
    0.9
}

// ── Impl ────────────────────────────────────────────────────────────────────

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            backend: default_backend(),
            ollama: OllamaConfig::default(),
            coordinator: CoordinatorConfig::default(),
            kv_cache: KvCacheConfig::default(),
            gpu: GpuConfig::default(),
        }
    }
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            url: default_ollama_url(),
            max_sessions: default_max_sessions(),
            default_model: None,
            timeout_secs: default_timeout_secs(),
            keep_alive_secs: default_keep_alive(),
        }
    }
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            socket_path: default_socket_path(),
            grpc_addr: default_grpc_addr(),
            max_queue_depth: default_max_queue_depth(),
            token_buffer_size: default_token_buffer_size(),
            realtime_scheduling: false,
            realtime_priority: default_realtime_priority(),
            shm_path: default_shm_path(),
            shm_capacity: default_shm_capacity(),
        }
    }
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            eviction_policy: default_eviction_policy(),
            max_per_session_mb: 0,
            snapshots_enabled: true,
            snapshot_dir: default_snapshot_dir(),
        }
    }
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            offload: default_offload(),
            mps_enabled: true,
            memory_fraction: default_memory_fraction(),
        }
    }
}

impl InferenceConfig {
    /// Load configuration from a YAML file with environment variable overrides.
    ///
    /// Environment variables are prefixed with `SPIRIT_INFERENCE_` and use
    /// double underscores for nesting (e.g., `SPIRIT_INFERENCE_OLLAMA__URL`).
    ///
    /// # Errors
    ///
    /// Returns `InferenceError::ConfigError` if the file cannot be read or
    /// parsed, or if validation fails.
    pub fn from_file(path: &Path) -> crate::error::Result<Self> {
        let config: Self = Figment::new()
            .merge(Serialized::defaults(Self::default()))
            .merge(Yaml::file(path).nested())
            .merge(Env::prefixed("SPIRIT_INFERENCE_").split("__"))
            .extract()
            .map_err(|e| InferenceError::ConfigError(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.ollama.max_sessions == 0 {
            return Err(InferenceError::ConfigError(
                "max_sessions must be > 0".into(),
            ));
        }
        if self.coordinator.token_buffer_size == 0 {
            return Err(InferenceError::ConfigError(
                "token_buffer_size must be > 0".into(),
            ));
        }
        if self.coordinator.realtime_priority > 99 {
            return Err(InferenceError::ConfigError(
                "realtime_priority must be 1-99".into(),
            ));
        }
        if !self.coordinator.shm_capacity.is_power_of_two() {
            return Err(InferenceError::ConfigError(
                "shm_capacity must be a power of 2".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.gpu.memory_fraction) {
            return Err(InferenceError::ConfigError(
                "gpu.memory_fraction must be 0.0-1.0".into(),
            ));
        }
        Ok(())
    }

    /// Create a minimal configuration for unit tests.
    #[must_use]
    pub fn for_test() -> Self {
        Self {
            backend: "ollama".into(),
            ollama: OllamaConfig {
                url: "http://127.0.0.1:11434".into(),
                max_sessions: 4,
                default_model: None,
                timeout_secs: 30,
                keep_alive_secs: 60,
            },
            coordinator: CoordinatorConfig {
                socket_path: "/tmp/spirit-test-infer.sock".into(),
                grpc_addr: "127.0.0.1:0".into(),
                max_queue_depth: 16,
                token_buffer_size: 256,
                realtime_scheduling: false,
                realtime_priority: 1,
                shm_path: "/dev/shm/spirit_infer_test".into(),
                shm_capacity: 1024,
            },
            kv_cache: KvCacheConfig {
                eviction_policy: "lru".into(),
                max_per_session_mb: 0,
                snapshots_enabled: true,
                snapshot_dir: PathBuf::from("/tmp/spirit-test-snapshots"),
            },
            gpu: GpuConfig {
                offload: "auto".into(),
                mps_enabled: false,
                memory_fraction: 0.9,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let cfg = InferenceConfig::default();
        cfg.validate().expect("default config should be valid");
    }

    #[test]
    fn test_test_config_is_valid() {
        let cfg = InferenceConfig::for_test();
        cfg.validate().expect("test config should be valid");
    }

    #[test]
    fn test_validation_rejects_zero_sessions() {
        let mut cfg = InferenceConfig::for_test();
        cfg.ollama.max_sessions = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_bad_shm_capacity() {
        let mut cfg = InferenceConfig::for_test();
        cfg.coordinator.shm_capacity = 1000; // not power of 2
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_bad_memory_fraction() {
        let mut cfg = InferenceConfig::for_test();
        cfg.gpu.memory_fraction = 1.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_default_ollama_url() {
        let cfg = InferenceConfig::default();
        assert!(cfg.ollama.url.contains("127.0.0.1"));
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let cfg = InferenceConfig::for_test();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: InferenceConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg.ollama.max_sessions, back.ollama.max_sessions);
    }
}
