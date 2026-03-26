//! # Spirit Inference — Inference Coordinator
//!
//! This crate is the central orchestrator for all LLM inference in `SpiritROS`.
//! It manages model loading, session lifecycle, token streaming, and integration
//! with the Ollama backend (Phase 1).
//!
//! ## Key responsibilities
//!
//! - **Session management**: Create, track, and tear down inference sessions.
//! - **Model registry**: Discover available models and their resource requirements.
//! - **Ollama client**: Communicate with the Ollama HTTP API for model pulls,
//!   generation, and embedding requests.
//! - **Coordinator**: Route inference requests to the appropriate backend,
//!   enforce concurrency limits, and collect metrics.
//!
//! ## Hot-path contract
//!
//! The token streaming path (from Ollama response to consumer) must **never**
//! allocate heap memory. All buffers are pre-allocated at session creation time.
//! See the `session` module for details.
//!
//! // PHASE1-SHIM: Ollama backend; Phase 2+ uses direct llama.cpp via FFI.

#![warn(clippy::all, clippy::pedantic)]

// ── Modules ──────────────────────────────────────────────────────────────────

/// Core type definitions for inference sessions and tokens.
pub mod types {
    use serde::{Deserialize, Serialize};
    use uuid::Uuid;

    /// Unique identifier for an inference session.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct SessionId(pub Uuid);

    impl SessionId {
        /// Creates a new random session ID.
        #[must_use]
        pub fn new() -> Self {
            Self(Uuid::new_v4())
        }
    }

    impl Default for SessionId {
        fn default() -> Self {
            Self::new()
        }
    }

    /// The current state of an inference session.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum SessionState {
        /// Session created, model not yet loaded.
        Pending,
        /// Model loaded and ready for inference.
        Ready,
        /// Actively generating tokens.
        Generating,
        /// Generation complete, session still alive.
        Idle,
        /// Session terminated.
        Terminated,
    }

    /// A single generated token from the inference backend.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Token {
        /// The token text.
        pub text: String,
        /// Log-probability of this token, if available.
        pub logprob: Option<f32>,
        /// Whether this is the final token in the sequence.
        pub done: bool,
    }
}

/// Error types for inference operations.
pub mod error {
    use crate::types::SessionId;

    /// Errors that can occur during inference operations.
    #[derive(Debug, thiserror::Error)]
    pub enum InferenceError {
        /// The requested model is not available.
        #[error("Model '{0}' not found in registry")]
        ModelNotFound(String),

        /// Insufficient VRAM for the requested model.
        #[error("Insufficient VRAM: need {needed} MiB, available {available} MiB")]
        InsufficientVram { needed: u64, available: u64 },

        /// The referenced session does not exist.
        #[error("Session {0:?} not found")]
        SessionNotFound(SessionId),

        /// Maximum concurrent sessions exceeded.
        #[error("Session limit reached: {active} active, max {limit}")]
        SessionLimitReached { active: usize, limit: usize },

        /// Ollama backend returned an error.
        #[error("Ollama backend error: {0}")]
        OllamaError(String),

        /// HTTP request to Ollama failed.
        #[error("HTTP error: {0}")]
        HttpError(#[from] reqwest::Error),

        /// JSON serialization/deserialization failed.
        #[error("Serialization error: {0}")]
        SerdeError(#[from] serde_json::Error),

        /// Configuration error.
        #[error("Inference configuration error: {0}")]
        ConfigError(String),
    }
}

pub mod coordinator {
    //! The inference coordinator — routes requests and manages sessions.
    //!
    //! Central dispatch for inference requests.
    //!
    //! The coordinator maintains a session table, enforces concurrency limits,
    //! and delegates actual inference to the configured backend (Ollama in Phase 1).
}

pub mod ollama_client {
    //! Ollama HTTP API client.
    //!
    //! Async client for the Ollama REST API.
    //!
    //! Handles `/api/generate`, `/api/chat`, `/api/embeddings`, `/api/pull`,
    //! and `/api/tags` endpoints. Streams tokens via chunked HTTP responses.
}

pub mod session {
    //! Inference session lifecycle management.
    //!
    //! Each inference session owns pre-allocated buffers for token streaming.
    //!
    //! Buffers are allocated once at session creation and reused across
    //! multiple generation requests within the same session. This ensures
    //! the hot path (token receive loop) never allocates.
}

pub mod model_registry {
    //! Model registry — discovery and metadata.
    //!
    //! Tracks available models, their sizes, quantization formats, and
    //! resource requirements. In Phase 1 this queries Ollama's `/api/tags`.
}

/// Configuration for the inference subsystem.
pub mod config {
    use serde::{Deserialize, Serialize};

    /// Configuration for the inference coordinator.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InferenceConfig {
        /// Base URL for the Ollama API.
        pub ollama_url: String,
        /// Maximum number of concurrent inference sessions.
        pub max_sessions: usize,
        /// Pre-allocated token buffer size per session (in tokens).
        pub token_buffer_size: usize,
        /// Request timeout in seconds.
        pub timeout_secs: u64,
    }

    impl Default for InferenceConfig {
        fn default() -> Self {
            Self {
                ollama_url: "http://127.0.0.1:11434".to_string(),
                max_sessions: 8,
                token_buffer_size: 4096,
                timeout_secs: 300,
            }
        }
    }
}

// ── Re-exports ───────────────────────────────────────────────────────────────

pub use config::InferenceConfig;
pub use error::InferenceError;
pub use types::{SessionId, SessionState, Token};

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_id_uniqueness() {
        let a = SessionId::new();
        let b = SessionId::new();
        assert_ne!(a, b, "Two new session IDs must be distinct");
    }

    #[test]
    fn test_session_id_serialization_roundtrip() {
        let id = SessionId::new();
        let json = serde_json::to_string(&id).expect("serialize");
        let back: SessionId = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(id, back);
    }

    #[test]
    fn test_inference_config_default_points_to_localhost() {
        let cfg = InferenceConfig::default();
        assert!(cfg.ollama_url.contains("127.0.0.1"));
        assert!(cfg.max_sessions > 0);
        assert!(cfg.token_buffer_size > 0);
    }

    #[test]
    fn test_session_state_transitions_are_distinct() {
        let states = [
            SessionState::Pending,
            SessionState::Ready,
            SessionState::Generating,
            SessionState::Idle,
            SessionState::Terminated,
        ];
        for (i, a) in states.iter().enumerate() {
            for (j, b) in states.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_inference_error_display_model_not_found() {
        let err = InferenceError::ModelNotFound("llama3:70b".to_string());
        assert!(err.to_string().contains("llama3:70b"));
    }

    #[test]
    fn test_token_done_flag() {
        let token = Token {
            text: "Hello".to_string(),
            logprob: Some(-0.5),
            done: false,
        };
        assert!(!token.done);
    }
}
