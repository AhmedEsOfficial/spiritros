//! # SpiritFS — Semantic Filesystem
//!
//! SpiritFS is a FUSE-based filesystem that augments the traditional POSIX file
//! interface with semantic operations: embedding-based search, automatic tagging,
//! content-addressable storage, and model-aware caching.
//!
//! ## Design principles
//!
//! - **POSIX first**: All standard file operations (`open`, `read`, `write`,
//!   `readdir`, etc.) work exactly as expected. SpiritFS is a drop-in replacement
//!   for any directory.
//! - **Async semantics**: Embedding generation and semantic indexing happen
//!   asynchronously in background tasks. A `write()` returns immediately; the
//!   embedding is computed later. This preserves invariant #6.
//! - **CAS backend**: File contents are stored in a content-addressable store
//!   (SHA-256 keyed), enabling deduplication and efficient snapshotting.
//! - **Graceful degradation**: When Qdrant is unavailable, falls back to SQLite
//!   FTS5 for text search. Semantic features degrade, POSIX never breaks.
//!
//! // PHASE1-SHIM: Uses FUSE via the `fuser` crate. Phase 2+ integrates
//! // directly with the SpiritKernel VFS layer.

#![warn(clippy::all, clippy::pedantic)]
// Allow common patterns used throughout the crate
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::doc_markdown)]

// ── Modules ──────────────────────────────────────────────────────────────────

pub mod cache;
pub mod cas;
pub mod config;
pub mod embedding;
pub mod error;
pub mod index;
pub mod metadata;
pub mod metrics;
pub mod qdrant;
pub mod semantic;
pub mod types;

#[cfg(feature = "fuse")]
pub mod fuse;

// ── Re-exports ───────────────────────────────────────────────────────────────

pub use cas::ContentAddressableStore;
pub use config::SpiritFsConfig;
pub use error::SpiritFsError;
pub use types::{ContentHash, FileMeta, IndexJob, InodeId, SearchResult, SemanticTag};
