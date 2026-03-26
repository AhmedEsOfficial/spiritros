//! # Spirit Inference — Inference Coordinator
//!
//! This crate is the central orchestrator for all LLM inference in `SpiritROS`.
//! It manages model loading, session lifecycle, token streaming, and integration
//! with the Ollama backend (Phase 1).
//!
//! ## Architecture
//!
//! ```text
//! Client Request (HTTP / gRPC / Unix Socket / Shared Memory)
//!         │
//!         ▼
//! ┌───────────────────────────────────────┐
//! │     InferenceCoordinator              │  ← Rust daemon
//! │  ┌─────────────────────────────────┐  │
//! │  │  SessionManager                 │  │  Tracks active sessions
//! │  │  ├── session pool               │  │
//! │  │  ├── KV-cache tracker           │  │  Manages COW snapshots
//! │  │  └── pre-allocated buffers      │  │  Zero-alloc hot path
//! │  └─────────────────────────────────┘  │
//! │  ┌─────────────────────────────────┐  │
//! │  │  BatchScheduler                 │  │  Continuous batching
//! │  │  ├── priority queue             │  │  Interactive → Background
//! │  │  └── model grouping             │  │  Same-model requests batched
//! │  └─────────────────────────────────┘  │
//! │  ┌─────────────────────────────────┐  │
//! │  │  ModelRegistry                  │  │  Content-addressable store
//! │  │  ├── name index                 │  │  name → ModelId
//! │  │  └── digest index               │  │  SHA-256 dedup
//! │  └─────────────────────────────────┘  │
//! │  ┌─────────────────────────────────┐  │
//! │  │  InferenceBackend               │  │  Pluggable engine trait
//! │  │  └── OllamaBackend (Phase 1)    │  │  HTTP API delegation
//! │  └─────────────────────────────────┘  │
//! │  ┌─────────────────────────────────┐  │
//! │  │  IPC Layer                      │  │
//! │  │  ├── Unix Domain Socket         │  │  Control messages
//! │  │  └── Shared Memory Ring Buffer  │  │  Zero-copy tokens
//! │  └─────────────────────────────────┘  │
//! │  ┌─────────────────────────────────┐  │
//! │  │  InferenceMetrics               │  │  Prometheus metrics
//! │  └─────────────────────────────────┘  │
//! └───────────────────────────────────────┘
//!         │
//!         ▼
//!     Ollama / llama-server (patched for Spirit memory & scheduling)
//!         │
//!         ▼
//!     GPU / CPU (via CUDA / ROCm / Vulkan)
//! ```
//!
//! ## Hot-path contract
//!
//! The token streaming path (from Ollama response to consumer) must **never**
//! allocate heap memory. All buffers are pre-allocated at session creation time.
//! See the [`session`] module for details.
//!
//! The shared memory ring buffer ([`ipc::shm`]) provides zero-copy token
//! transfer using lock-free SPSC queues. Each `TokenEntry` is exactly 16 bytes,
//! cache-line friendly.
//!
//! // PHASE1-SHIM: Ollama HTTP backend; Phase 2+ uses direct llama.cpp via FFI.

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::manual_inspect)]
#![allow(clippy::new_ret_no_self)]
#![allow(clippy::pub_underscore_fields)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::ptr_as_ptr)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_ptr_alignment)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::ptr_cast_constness)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::if_not_else)]
#![allow(clippy::single_match_else)]
#![allow(clippy::unnecessary_literal_bound)]
#![allow(clippy::manual_string_new)]

// ── Modules ──────────────────────────────────────────────────────────────────

pub mod backend;
pub mod batch;
pub mod config;
pub mod coordinator;
pub mod error;
pub mod ipc;
pub mod metrics;
pub mod model_registry;
pub mod ollama_client;
pub mod session;
pub mod types;

// ── Re-exports ───────────────────────────────────────────────────────────────

pub use config::InferenceConfig;
pub use coordinator::InferenceCoordinator;
pub use error::InferenceError;
pub use types::{
    InferenceRequest, InferenceResponse, ModelId, ModelInfo, Priority, SamplingConfig, SessionId,
    SessionState, StopReason, Token,
};
