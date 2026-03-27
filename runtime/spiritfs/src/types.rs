//! Core type definitions for the SpiritFS semantic filesystem.

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique inode identifier within SpiritFS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InodeId(pub u64);

impl std::fmt::Display for InodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Content hash used as the CAS key (SHA-256, hex-encoded).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentHash(pub String);

impl std::fmt::Display for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Semantic tag automatically assigned by the embedding pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticTag {
    /// Human-readable tag label.
    pub label: String,
    /// Confidence score in [0.0, 1.0].
    pub confidence: f32,
}

/// File type tracked by SpiritFS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileKind {
    RegularFile,
    Directory,
    Symlink,
}

/// Metadata for a file in SpiritFS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMeta {
    /// Inode number.
    pub inode: InodeId,
    /// Parent inode number.
    pub parent_inode: InodeId,
    /// File name (basename only).
    pub name: String,
    /// Full path relative to mount point.
    pub path: PathBuf,
    /// File kind.
    pub kind: FileKind,
    /// Size in bytes.
    pub size: u64,
    /// MIME type (detected on write).
    pub mime_type: Option<String>,
    /// Content hash for CAS lookup.
    pub content_hash: Option<ContentHash>,
    /// Qdrant point ID for the primary embedding.
    pub embedding_id: Option<Uuid>,
    /// Auto-generated semantic tags (JSON-serialized).
    pub tags: Vec<SemanticTag>,
    /// Created timestamp.
    pub created_at: DateTime<Utc>,
    /// Last modified timestamp.
    pub modified_at: DateTime<Utc>,
}

/// Result from a semantic search query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// File path relative to mount point.
    pub path: PathBuf,
    /// Similarity score (higher is more similar).
    pub score: f32,
    /// Matched chunk text (if available).
    pub chunk_text: Option<String>,
    /// Chunk index within the file.
    pub chunk_index: Option<u32>,
}

/// A chunk of text extracted from a file for embedding.
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// Source file path.
    pub source_path: PathBuf,
    /// Chunk index within the file.
    pub chunk_index: u32,
    /// Byte offset in the source file.
    pub byte_offset: usize,
    /// The chunk text.
    pub text: String,
}

/// A job queued for the async indexing pipeline.
#[derive(Debug, Clone)]
pub enum IndexJob {
    /// Index or re-index a file.
    Update(PathBuf),
    /// Remove a file from the index.
    Remove(PathBuf),
    /// Rebuild the entire index.
    RebuildAll,
}

/// Statistics about the index state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    /// Total files tracked in metadata.
    pub total_files: u64,
    /// Files with embeddings in the vector store.
    pub indexed_files: u64,
    /// Files pending indexing.
    pub pending_files: u64,
    /// Total vectors in Qdrant.
    pub total_vectors: u64,
    /// Queue depth of pending index jobs.
    pub queue_depth: usize,
}

/// Model cache entry for frequency-weighted LRU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCacheEntry {
    /// Content hash of the model file.
    pub blob_hash: ContentHash,
    /// Model name (e.g., "llama3.1:8b").
    pub model_name: String,
    /// Number of times accessed.
    pub access_count: u64,
    /// Last access timestamp.
    pub last_accessed: DateTime<Utc>,
    /// Size in bytes.
    pub size: u64,
}

/// Extensions that are eligible for semantic indexing.
pub const INDEXABLE_EXTENSIONS: &[&str] = &[
    "txt", "md", "rst", "rs", "py", "js", "ts", "tsx", "jsx", "c", "cpp", "cc", "h", "hpp", "go",
    "java", "kt", "swift", "rb", "php", "cs", "toml", "yaml", "yml", "json", "csv", "html", "htm",
    "xml", "sh", "bash", "zsh", "fish", "log", "conf", "cfg", "ini", "env", "sql", "r", "jl",
    "lua", "vim", "el", "clj", "ex", "erl", "hs", "ml", "tex", "bib", "org", "adoc",
];

/// Check if a file should be indexed based on extension and size.
pub fn should_index(path: &std::path::Path, size: u64, max_size: u64) -> bool {
    if size > max_size || size == 0 {
        return false;
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    INDEXABLE_EXTENSIONS.contains(&ext.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inode_id_equality_and_display() {
        let a = InodeId(42);
        let b = InodeId(42);
        assert_eq!(a, b);
        assert_eq!(a.to_string(), "42");
    }

    #[test]
    fn test_content_hash_display() {
        let hash = ContentHash("abcdef1234".to_string());
        assert_eq!(hash.to_string(), "abcdef1234");
    }

    #[test]
    fn test_semantic_tag_serialization() {
        let tag = SemanticTag {
            label: "rust-code".to_string(),
            confidence: 0.95,
        };
        let json = serde_json::to_string(&tag).expect("serialize");
        let back: SemanticTag = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(tag.label, back.label);
        assert!((tag.confidence - back.confidence).abs() < f32::EPSILON);
    }

    #[test]
    fn test_should_index_valid_extensions() {
        use std::path::Path;
        let max = 10 * 1024 * 1024;
        assert!(should_index(Path::new("foo.rs"), 1000, max));
        assert!(should_index(Path::new("readme.md"), 500, max));
        assert!(should_index(Path::new("config.yaml"), 200, max));
        assert!(should_index(Path::new("main.py"), 3000, max));
    }

    #[test]
    fn test_should_index_rejects_binary() {
        use std::path::Path;
        let max = 10 * 1024 * 1024;
        assert!(!should_index(Path::new("image.png"), 1000, max));
        assert!(!should_index(Path::new("model.bin"), 1000, max));
        assert!(!should_index(Path::new("archive.tar.gz"), 1000, max));
    }

    #[test]
    fn test_should_index_rejects_large_files() {
        use std::path::Path;
        let max = 10 * 1024 * 1024;
        assert!(!should_index(Path::new("huge.rs"), max + 1, max));
    }

    #[test]
    fn test_should_index_rejects_empty_files() {
        use std::path::Path;
        assert!(!should_index(Path::new("empty.rs"), 0, 10_000_000));
    }
}
