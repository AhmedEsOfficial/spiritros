//! Core types shared across the Spirit SDK.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for an inference session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub Uuid);

impl SessionId {
    /// Create a new random session ID.
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

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A request to generate a completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// The model to use for inference.
    pub model: String,
    /// The prompt text.
    pub prompt: String,
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
    /// Sampling temperature (0.0 = greedy, higher = more random).
    pub temperature: f32,
}

/// A single token from a streaming inference response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenResponse {
    /// The session this token belongs to.
    pub session_id: SessionId,
    /// The generated token text.
    pub token: String,
    /// Whether this is the final token in the sequence.
    pub done: bool,
}

/// Status of the `SpiritROS` system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    /// `SpiritROS` version.
    pub version: String,
    /// Number of active inference sessions.
    pub active_sessions: u32,
    /// Available VRAM in megabytes.
    pub available_vram_mb: u64,
    /// Loaded models.
    pub loaded_models: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_id_uniqueness() {
        let a = SessionId::new();
        let b = SessionId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn test_session_id_display() {
        let id = SessionId::new();
        let display = format!("{id}");
        assert!(!display.is_empty());
        // UUID v4 format: 8-4-4-4-12 hex chars
        assert_eq!(display.len(), 36);
    }

    #[test]
    fn test_inference_request_serialization() {
        let req = InferenceRequest {
            model: "llama3:8b".to_string(),
            prompt: "Hello".to_string(),
            max_tokens: 100,
            temperature: 0.7,
        };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: InferenceRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.model, "llama3:8b");
        assert_eq!(deserialized.max_tokens, 100);
    }
}
