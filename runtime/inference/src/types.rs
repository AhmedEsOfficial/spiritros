//! Core type definitions for the inference subsystem.
//!
//! This module defines the data structures used across the inference
//! coordinator, backends, session manager, and IPC layer.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Session Types ───────────────────────────────────────────────────────────

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

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The current state of an inference session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionState {
    /// Session created, model not yet loaded.
    Pending,
    /// Model loaded and ready for inference.
    Ready,
    /// Actively generating tokens (prefill phase).
    Prefilling,
    /// Actively generating tokens (decode phase).
    Decoding,
    /// Generation complete, session still alive.
    Idle,
    /// Session terminated.
    Terminated,
}

impl SessionState {
    /// Returns `true` if the session is actively generating.
    #[must_use]
    pub fn is_generating(&self) -> bool {
        matches!(self, Self::Prefilling | Self::Decoding)
    }

    /// Returns `true` if the session can accept new prompts.
    #[must_use]
    pub fn can_prompt(&self) -> bool {
        matches!(self, Self::Ready | Self::Idle)
    }
}

/// Priority level for inference requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    /// Background tasks (e.g., SpiritFS indexing). Batched aggressively.
    Background = 0,
    /// Normal priority (default).
    Normal = 1,
    /// High priority (e.g., SDK API calls).
    High = 2,
    /// Interactive priority (e.g., Spirit Shell). Lowest latency.
    Interactive = 3,
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

// ── Token Types ─────────────────────────────────────────────────────────────

/// A single generated token from the inference backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    /// The token text.
    pub text: String,
    /// The token ID from the model's vocabulary.
    pub token_id: u32,
    /// Log-probability of this token, if available.
    pub logprob: Option<f32>,
    /// Whether this is the final token in the sequence.
    pub done: bool,
}

/// Flags for token entries in the shared memory ring buffer.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenFlag {
    /// Normal token.
    Normal = 0,
    /// End of sequence.
    Eos = 1,
    /// Error occurred during generation.
    Error = 2,
}

/// A single entry in the shared memory token ring buffer.
///
/// Exactly 16 bytes for cache-line friendliness.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TokenEntry {
    /// Token ID from the model vocabulary.
    pub token_id: u32,
    /// Log-probability of this token.
    pub logprob: f32,
    /// Token flags (EOS, error, etc.).
    pub flags: u32,
    /// Reserved for alignment.
    pub _pad: u32,
}

const _: () = assert!(std::mem::size_of::<TokenEntry>() == 16);

// ── Model Types ─────────────────────────────────────────────────────────────

/// Unique identifier for a loaded model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(pub Uuid);

impl ModelId {
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ModelId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Information about a model in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Internal model ID.
    pub id: ModelId,
    /// Human-readable model name (e.g., "llama3.1:8b").
    pub name: String,
    /// Model family (e.g., "llama", "mistral").
    pub family: String,
    /// Parameter count label (e.g., "8B", "70B").
    pub parameter_size: String,
    /// Quantization format (e.g., "Q4_K_M", "F16").
    pub quantization: String,
    /// Size on disk in bytes.
    pub size_bytes: u64,
    /// SHA-256 digest of the model blob.
    pub digest: String,
    /// Format (e.g., "gguf").
    pub format: String,
    /// When the model was registered.
    pub registered_at: DateTime<Utc>,
}

/// State of a model in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelLoadState {
    /// Model is registered but not loaded.
    Unloaded,
    /// Model is currently being loaded.
    Loading,
    /// Model is loaded and ready for inference.
    Loaded,
    /// Model is being unloaded.
    Unloading,
    /// Load failed.
    Failed,
}

/// Information about a loaded model including resource usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedModelInfo {
    /// Model metadata.
    pub info: ModelInfo,
    /// Current load state.
    pub state: ModelLoadState,
    /// GPU device indices where the model is loaded.
    pub gpu_devices: Vec<u32>,
    /// VRAM usage in bytes.
    pub vram_usage_bytes: u64,
    /// RAM usage in bytes (for CPU layers).
    pub ram_usage_bytes: u64,
    /// Number of active sessions using this model.
    pub active_sessions: usize,
}

// ── Batch Types ─────────────────────────────────────────────────────────────

/// A unique identifier for a batch of inference requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchId(u64);

impl BatchId {
    #[must_use]
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    #[must_use]
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

/// A request to be included in a batch.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    /// The session this request belongs to.
    pub session_id: SessionId,
    /// Priority of this request.
    pub priority: Priority,
    /// Prompt tokens (for prefill requests).
    pub prompt_tokens: Option<Vec<u32>>,
    /// When this request was enqueued.
    pub enqueued_at: Instant,
}

/// A completed batch with results for each request.
#[derive(Debug)]
pub struct BatchResult {
    /// Batch identifier.
    pub batch_id: BatchId,
    /// Results for each session in the batch.
    pub results: Vec<(SessionId, Result<Vec<Token>, String>)>,
    /// Total time for the batch in microseconds.
    pub duration_us: u64,
}

// ── Inference Phase Hints ───────────────────────────────────────────────────

/// Scheduling phase hint for the inference pipeline.
///
/// Emitted by the backend so the scheduler can adjust CPU/GPU priorities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferencePhase {
    /// About to run the prefill (prompt processing) phase.
    /// This is compute-bound and benefits from burst CPU/GPU allocation.
    Prefill,
    /// About to run the autoregressive decode loop.
    /// This is memory-bandwidth-bound and benefits from SCHED_FIFO.
    Decode,
    /// Inference is complete; resources can be reclaimed.
    Idle,
}

// ── Sampling Configuration ──────────────────────────────────────────────────

/// Sampling parameters for token generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Temperature for sampling (0.0 = greedy, higher = more random).
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold.
    pub top_p: f32,
    /// Top-k sampling (0 = disabled).
    pub top_k: u32,
    /// Repetition penalty (1.0 = none).
    pub repeat_penalty: f32,
    /// Maximum tokens to generate (0 = model default).
    pub max_tokens: u32,
    /// Stop sequences.
    pub stop: Vec<String>,
    /// Random seed (0 = random).
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            max_tokens: 2048,
            stop: Vec::new(),
            seed: 0,
        }
    }
}

// ── Request / Response Types ────────────────────────────────────────────────

/// An inference request submitted to the coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Model to use (by name, e.g. "llama3.1:8b").
    pub model: String,
    /// The prompt text.
    pub prompt: String,
    /// Optional system prompt.
    pub system: Option<String>,
    /// Sampling configuration.
    pub sampling: SamplingConfig,
    /// Request priority.
    pub priority: Priority,
    /// Whether to stream tokens as they are generated.
    pub stream: bool,
}

/// A complete inference response (non-streaming).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// The session that handled this request.
    pub session_id: SessionId,
    /// The generated text.
    pub text: String,
    /// Individual tokens (if requested).
    pub tokens: Vec<Token>,
    /// Time to first token in milliseconds.
    pub first_token_ms: u64,
    /// Total generation time in milliseconds.
    pub total_ms: u64,
    /// Total tokens generated.
    pub tokens_generated: u32,
    /// Tokens per second during decode.
    pub tokens_per_second: f32,
    /// Whether generation was stopped by a stop sequence.
    pub stop_reason: StopReason,
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// End-of-sequence token was generated.
    EndOfSequence,
    /// Max tokens limit reached.
    MaxTokens,
    /// A stop sequence was matched.
    StopSequence,
    /// Generation was cancelled.
    Cancelled,
}

// ── Resource Report ─────────────────────────────────────────────────────────

/// Resource usage report from a backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceReport {
    /// Number of loaded models.
    pub loaded_models: usize,
    /// Number of active sessions.
    pub active_sessions: usize,
    /// Total VRAM used in bytes.
    pub vram_used_bytes: u64,
    /// Total VRAM available in bytes.
    pub vram_total_bytes: u64,
    /// Total RAM used for inference in bytes.
    pub ram_used_bytes: u64,
    /// Current tokens per second (across all sessions).
    pub aggregate_tokens_per_sec: f32,
}

// ── KV-Cache Snapshot ───────────────────────────────────────────────────────

/// A snapshot of a session's KV-cache state (COW semantics).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvSnapshot {
    /// Unique snapshot identifier.
    pub id: Uuid,
    /// Session this snapshot was taken from.
    pub session_id: SessionId,
    /// Number of tokens in the KV-cache at snapshot time.
    pub token_count: usize,
    /// When the snapshot was taken.
    pub created_at: DateTime<Utc>,
}

// ── ID Generation ───────────────────────────────────────────────────────────

/// Thread-safe monotonic ID generator for batches.
pub struct IdGenerator {
    next: AtomicU64,
}

impl IdGenerator {
    /// Creates a new generator starting at 1.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next: AtomicU64::new(1),
        }
    }

    /// Returns the next unique ID.
    pub fn next_id(&self) -> u64 {
        self.next.fetch_add(1, Ordering::Relaxed)
    }
}

impl Default for IdGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

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
    fn test_session_id_serialization_roundtrip() {
        let id = SessionId::new();
        let json = serde_json::to_string(&id).expect("serialize");
        let back: SessionId = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(id, back);
    }

    #[test]
    fn test_session_state_generating_check() {
        assert!(SessionState::Prefilling.is_generating());
        assert!(SessionState::Decoding.is_generating());
        assert!(!SessionState::Ready.is_generating());
        assert!(!SessionState::Idle.is_generating());
    }

    #[test]
    fn test_session_state_can_prompt() {
        assert!(SessionState::Ready.can_prompt());
        assert!(SessionState::Idle.can_prompt());
        assert!(!SessionState::Pending.can_prompt());
        assert!(!SessionState::Decoding.can_prompt());
        assert!(!SessionState::Terminated.can_prompt());
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Interactive > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Background);
    }

    #[test]
    fn test_token_entry_size_is_16_bytes() {
        assert_eq!(std::mem::size_of::<TokenEntry>(), 16);
    }

    #[test]
    fn test_model_id_display() {
        let id = ModelId::new();
        let s = format!("{id}");
        assert!(!s.is_empty());
    }

    #[test]
    fn test_sampling_config_defaults() {
        let cfg = SamplingConfig::default();
        assert!(cfg.temperature > 0.0);
        assert!(cfg.top_p > 0.0);
        assert!(cfg.max_tokens > 0);
    }

    #[test]
    fn test_id_generator_monotonic() {
        let gen = IdGenerator::new();
        let a = gen.next_id();
        let b = gen.next_id();
        let c = gen.next_id();
        assert_eq!(a, 1);
        assert_eq!(b, 2);
        assert_eq!(c, 3);
    }

    #[test]
    fn test_batch_id_roundtrip() {
        let id = BatchId::new(42);
        assert_eq!(id.as_u64(), 42);
    }

    #[test]
    fn test_stop_reason_variants() {
        let reasons = [
            StopReason::EndOfSequence,
            StopReason::MaxTokens,
            StopReason::StopSequence,
            StopReason::Cancelled,
        ];
        for (i, a) in reasons.iter().enumerate() {
            for (j, b) in reasons.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }
}
