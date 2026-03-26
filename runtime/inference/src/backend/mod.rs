//! Backend interface for inference engines.
//!
//! Defines the [`InferenceBackend`] trait that all inference backends must
//! implement. The coordinator dispatches requests to whichever backend is
//! configured (Ollama in Phase 1, with vLLM/candle as future options).

pub mod ollama;

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::error::InferenceError;
use crate::types::{
    InferencePhase, KvSnapshot, ModelId, ResourceReport, SamplingConfig, SessionId, Token,
};

/// Configuration for loading a model.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model name (e.g., "llama3.1:8b").
    pub name: String,
    /// GPU layer offload strategy.
    pub gpu_offload: String,
    /// Keep-alive duration in seconds.
    pub keep_alive_secs: u64,
}

/// Handle to a loaded model in the backend.
#[derive(Debug, Clone)]
pub struct ModelHandle {
    /// Internal model ID.
    pub model_id: ModelId,
    /// Model name as known by the backend.
    pub backend_name: String,
}

/// Result of the prefill (prompt processing) phase.
#[derive(Debug)]
pub struct PrefillResult {
    /// Number of prompt tokens processed.
    pub prompt_tokens: u32,
    /// Time spent on prefill in milliseconds.
    pub prefill_ms: u64,
}

/// Session state as tracked by the backend.
#[derive(Debug)]
pub struct BackendSessionState {
    /// Session ID.
    pub session_id: SessionId,
    /// Number of tokens in the KV-cache.
    pub kv_token_count: usize,
    /// Current inference phase.
    pub phase: InferencePhase,
}

/// Abstract interface for inference backends.
///
/// All backends implement this trait. The coordinator uses it to dispatch
/// requests without knowing the underlying engine.
///
/// # Hot-path contract
///
/// The `stream()` method is on the hot path. Implementations must ensure
/// that token streaming does not allocate per-token. Use pre-allocated
/// buffers and lock-free channels.
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Human-readable backend name (e.g., "ollama", "vllm", "candle").
    fn name(&self) -> &str;

    /// Check if the backend is healthy and ready to serve requests.
    async fn health_check(&self) -> bool;

    /// Load a model into the backend.
    async fn load_model(&self, config: &ModelConfig) -> Result<ModelHandle, InferenceError>;

    /// Unload a model from the backend.
    async fn unload_model(&self, handle: &ModelHandle) -> Result<(), InferenceError>;

    /// List models currently available in the backend.
    async fn list_models(&self) -> Result<Vec<String>, InferenceError>;

    /// Generate a complete response (non-streaming).
    async fn generate(
        &self,
        handle: &ModelHandle,
        prompt: &str,
        system: Option<&str>,
        sampling: &SamplingConfig,
    ) -> Result<(String, GenerationStats), InferenceError>;

    /// Generate tokens in a streaming fashion.
    ///
    /// Returns a stream of tokens. The stream must be cancel-safe.
    async fn stream(
        &self,
        handle: &ModelHandle,
        prompt: &str,
        system: Option<&str>,
        sampling: &SamplingConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Token, InferenceError>> + Send>>, InferenceError>;

    /// Snapshot KV-cache for a session (if supported).
    async fn kv_snapshot(&self, session_id: &SessionId) -> Result<KvSnapshot, InferenceError>;

    /// Restore KV-cache from a snapshot (COW semantics).
    async fn kv_restore(
        &self,
        session_id: &SessionId,
        snapshot: &KvSnapshot,
    ) -> Result<(), InferenceError>;

    /// Report current resource usage.
    fn resource_usage(&self) -> ResourceReport;
}

/// Statistics from a generation run.
#[derive(Debug, Clone)]
pub struct GenerationStats {
    /// Time to first token in milliseconds.
    pub first_token_ms: u64,
    /// Total generation time in milliseconds.
    pub total_ms: u64,
    /// Number of prompt tokens processed.
    pub prompt_tokens: u32,
    /// Number of tokens generated.
    pub generated_tokens: u32,
    /// Tokens per second during decode.
    pub tokens_per_second: f32,
}
