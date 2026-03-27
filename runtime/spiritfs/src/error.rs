//! Error types for SpiritFS operations.

/// Result alias for SpiritFS operations.
pub type Result<T> = std::result::Result<T, SpiritFsError>;

/// Errors that can occur during SpiritFS operations.
#[derive(Debug, thiserror::Error)]
pub enum SpiritFsError {
    /// FUSE operation failed.
    #[error("FUSE error: {0}")]
    Fuse(String),

    /// Content-addressable storage error.
    #[error("CAS error: {0}")]
    Cas(String),

    /// Inode not found.
    #[error("inode {0} not found")]
    InodeNotFound(u64),

    /// File not found at path.
    #[error("file not found: {0}")]
    FileNotFound(String),

    /// Semantic index operation failed.
    #[error("semantic index error: {0}")]
    SemanticIndex(String),

    /// Embedding generation failed.
    #[error("embedding generation failed: {0}")]
    Embedding(String),

    /// Qdrant vector store error.
    #[error("qdrant error: {0}")]
    Qdrant(String),

    /// SQLite metadata store error.
    #[error("metadata store error: {0}")]
    Metadata(String),

    /// Configuration error.
    #[error("configuration error: {0}")]
    Config(String),

    /// File too large for embedding.
    #[error("file too large for embedding: {size} bytes (max {max} bytes)")]
    FileTooLarge { size: u64, max: u64 },

    /// Unsupported file type for indexing.
    #[error("unsupported file type: {0}")]
    UnsupportedFileType(String),

    /// Underlying I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// HTTP client error (Ollama API).
    #[error("HTTP error: {0}")]
    Http(String),

    /// Serialization error.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Channel send/receive error.
    #[error("channel error: {0}")]
    Channel(String),

    /// Mount point error.
    #[error("mount error: {0}")]
    Mount(String),
}

impl From<rusqlite::Error> for SpiritFsError {
    fn from(e: rusqlite::Error) -> Self {
        Self::Metadata(e.to_string())
    }
}

impl From<reqwest::Error> for SpiritFsError {
    fn from(e: reqwest::Error) -> Self {
        Self::Http(e.to_string())
    }
}

impl From<serde_json::Error> for SpiritFsError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serialization(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = SpiritFsError::InodeNotFound(42);
        assert!(err.to_string().contains("42"));
    }

    #[test]
    fn test_file_too_large_display() {
        let err = SpiritFsError::FileTooLarge {
            size: 20_000_000,
            max: 10_000_000,
        };
        let msg = err.to_string();
        assert!(msg.contains("20000000"));
        assert!(msg.contains("10000000"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
        let err: SpiritFsError = io_err.into();
        assert!(matches!(err, SpiritFsError::Io(_)));
    }
}
