//! Configuration for the SpiritFS filesystem.
//!
//! Loads from YAML config file, environment variables, and CLI arguments
//! using `figment` for layered configuration.

use std::path::PathBuf;

use figment::providers::{Env, Format, Serialized, Yaml};
use figment::Figment;
use serde::{Deserialize, Serialize};

/// Configuration for the SpiritFS filesystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpiritFsConfig {
    /// Mount point path for FUSE.
    pub mount_point: PathBuf,

    /// Backing directory (the actual storage on ext4/btrfs).
    pub backing_dir: PathBuf,

    /// CAS blob storage directory.
    pub cas_dir: PathBuf,

    /// SQLite metadata database path.
    pub metadata_db: PathBuf,

    /// Qdrant gRPC endpoint URL.
    pub qdrant_url: String,

    /// Qdrant collection name.
    pub qdrant_collection: String,

    /// Ollama API URL for embedding generation.
    pub ollama_url: String,

    /// Model to use for embedding generation.
    pub embedding_model: String,

    /// Embedding vector dimensions (must match model output).
    pub embedding_dimensions: u32,

    /// Maximum file size eligible for automatic embedding (bytes).
    pub max_embedding_file_size: u64,

    /// Maximum tokens per text chunk for embedding.
    pub chunk_max_tokens: usize,

    /// Token overlap between consecutive chunks.
    pub chunk_overlap_tokens: usize,

    /// Maximum number of concurrent embedding requests.
    pub max_concurrent_embeddings: usize,

    /// Index queue capacity (backpressure threshold).
    pub index_queue_capacity: usize,

    /// Enable file-system watcher for automatic indexing.
    pub enable_watcher: bool,

    /// Allow direct FUSE mount (requires root or fuse group).
    pub allow_other: bool,

    /// Enable model weight caching.
    pub enable_model_cache: bool,

    /// Maximum model cache size in bytes.
    pub model_cache_max_bytes: u64,

    /// Prometheus metrics bind address.
    pub metrics_bind: String,
}

impl Default for SpiritFsConfig {
    fn default() -> Self {
        Self {
            mount_point: PathBuf::from("/mnt/spiritfs"),
            backing_dir: PathBuf::from("/var/lib/spiritfs/data"),
            cas_dir: PathBuf::from("/var/lib/spiritfs/cas"),
            metadata_db: PathBuf::from("/var/lib/spiritfs/metadata.db"),
            qdrant_url: "http://127.0.0.1:6334".to_string(),
            qdrant_collection: "spiritfs".to_string(),
            ollama_url: "http://127.0.0.1:11434".to_string(),
            embedding_model: "nomic-embed-text".to_string(),
            embedding_dimensions: 768,
            max_embedding_file_size: 10 * 1024 * 1024, // 10 MiB
            chunk_max_tokens: 512,
            chunk_overlap_tokens: 64,
            max_concurrent_embeddings: 4,
            index_queue_capacity: 10_000,
            enable_watcher: true,
            allow_other: false,
            enable_model_cache: true,
            model_cache_max_bytes: 10 * 1024 * 1024 * 1024, // 10 GiB
            metrics_bind: "127.0.0.1:9104".to_string(),
        }
    }
}

impl SpiritFsConfig {
    /// Load configuration from YAML file + environment variables.
    ///
    /// Environment variables are prefixed with `SPIRITFS_` and use `__` for
    /// nesting (e.g., `SPIRITFS_QDRANT_URL`).
    #[allow(clippy::result_large_err)]
    pub fn load(config_path: Option<&str>) -> Result<Self, figment::Error> {
        let mut figment = Figment::from(Serialized::defaults(Self::default()));

        if let Some(path) = config_path {
            figment = figment.merge(Yaml::file(path));
        }

        figment.merge(Env::prefixed("SPIRITFS_")).extract()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = SpiritFsConfig::default();
        assert_eq!(cfg.mount_point, PathBuf::from("/mnt/spiritfs"));
        assert_eq!(cfg.embedding_model, "nomic-embed-text");
        assert_eq!(cfg.embedding_dimensions, 768);
        assert_eq!(cfg.max_embedding_file_size, 10 * 1024 * 1024);
        assert_eq!(cfg.chunk_max_tokens, 512);
        assert_eq!(cfg.chunk_overlap_tokens, 64);
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let cfg = SpiritFsConfig::default();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: SpiritFsConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg.embedding_model, back.embedding_model);
        assert_eq!(cfg.qdrant_url, back.qdrant_url);
    }

    #[test]
    fn test_config_load_defaults() {
        let cfg = SpiritFsConfig::load(None).expect("load defaults");
        assert_eq!(cfg.qdrant_collection, "spiritfs");
    }
}
