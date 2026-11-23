//! Error types for the inference subsystem.
//!
//! Follows the SpiritROS error classification:
//! - Operational errors (expected, recoverable)
//! - Programming errors (bugs, should panic)
//! - Security violations (logged, session terminated)

use crate::types::SessionId;

/// Errors that can occur during inference operations.
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    // ── Model errors ────────────────────────────────────────────────────
    /// The requested model is not available in the registry.
    #[error("Model '{0}' not found in registry")]
    ModelNotFound(String),

    /// The model is registered but not currently loaded.
    #[error("Model '{0}' is not loaded")]
    ModelNotLoaded(String),

    /// The model is already loaded.
    #[error("Model '{0}' is already loaded")]
    ModelAlreadyLoaded(String),

    /// Insufficient VRAM to load the requested model.
    #[error("Insufficient VRAM: need {needed} MiB, available {available} MiB")]
    InsufficientVram { needed: u64, available: u64 },

    /// Model load timed out.
    #[error("Model load timed out after {timeout_secs}s")]
    ModelLoadTimeout { timeout_secs: u64 },

    // ── Session errors ──────────────────────────────────────────────────
    /// The referenced session does not exist.
    #[error("Session {0} not found")]
    SessionNotFound(SessionId),

    /// Maximum concurrent sessions exceeded.
    #[error("Session limit reached: {active} active, max {limit}")]
    SessionLimitReached { active: usize, limit: usize },

    /// The session is in the wrong state for the requested operation.
    #[error("Session {session_id} is in state {state:?}, expected one of {expected:?}")]
    InvalidSessionState {
        session_id: SessionId,
        state: crate::types::SessionState,
        expected: Vec<crate::types::SessionState>,
    },

    /// Session has been terminated.
    #[error("Session {0} has been terminated")]
    SessionTerminated(SessionId),

    // ── KV-cache errors ─────────────────────────────────────────────────
    /// KV-cache snapshot not found.
    #[error("KV-cache snapshot {0} not found")]
    SnapshotNotFound(String),

    /// KV-cache zone exhausted.
    #[error("KV-cache zone exhausted: {active_sessions} sessions active")]
    KvCacheExhausted { active_sessions: usize },

    // ── Batch errors ────────────────────────────────────────────────────
    /// Request queue is full.
    #[error("Request queue full: {queue_depth} pending, max {max_depth}")]
    QueueFull {
        queue_depth: usize,
        max_depth: usize,
    },

    // ── Backend errors ──────────────────────────────────────────────────
    /// Ollama backend returned an error.
    #[error("Ollama backend error: {0}")]
    OllamaError(String),

    /// HTTP request to Ollama failed.
    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    /// Backend is not available.
    #[error("Backend '{0}' is not available")]
    BackendUnavailable(String),

    /// Generation was cancelled.
    #[error("Generation cancelled for session {0}")]
    Cancelled(SessionId),

    // ── IPC errors ──────────────────────────────────────────────────────
    /// Unix domain socket error.
    #[error("IPC socket error: {0}")]
    SocketError(String),

    /// Shared memory error.
    #[error("Shared memory error: {0}")]
    SharedMemoryError(String),

    // ── Serialization errors ────────────────────────────────────────────
    /// JSON serialization/deserialization failed.
    #[error("Serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),

    // ── Configuration errors ────────────────────────────────────────────
    /// Configuration error.
    #[error("Inference configuration error: {0}")]
    ConfigError(String),

    // ── Security errors ─────────────────────────────────────────────────
    /// Capability denied for the requested operation.
    #[error("Capability denied: {action} requires {capability}")]
    CapabilityDenied { action: String, capability: String },

    // ── I/O errors ──────────────────────────────────────────────────────
    /// Generic I/O error.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Convenience alias for `Result<T, InferenceError>`.
pub type Result<T> = std::result::Result<T, InferenceError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_model_not_found() {
        let err = InferenceError::ModelNotFound("llama3:70b".into());
        assert!(err.to_string().contains("llama3:70b"));
    }

    #[test]
    fn test_error_display_session_limit() {
        let err = InferenceError::SessionLimitReached {
            active: 8,
            limit: 8,
        };
        let s = err.to_string();
        assert!(s.contains('8'));
    }

    #[test]
    fn test_error_display_insufficient_vram() {
        let err = InferenceError::InsufficientVram {
            needed: 8192,
            available: 4096,
        };
        let s = err.to_string();
        assert!(s.contains("8192"));
        assert!(s.contains("4096"));
    }

    #[test]
    fn test_error_display_capability_denied() {
        let err = InferenceError::CapabilityDenied {
            action: "model_load".into(),
            capability: "CAP_MODEL_LOAD".into(),
        };
        assert!(err.to_string().contains("CAP_MODEL_LOAD"));
    }
}
