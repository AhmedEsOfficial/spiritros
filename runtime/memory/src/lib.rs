//! # Spirit Memory — TensorFirst Memory Manager
//!
//! This crate implements the Phase 1 userspace shim for the TensorFirst memory
//! architecture described in the SpiritROS Architecture Plan §4.
//!
//! ## Memory Zones
//!
//! SpiritROS partitions system memory into four zones, enforced via
//! cgroups v2 in Phase 1:
//!
//! | Zone | Purpose | Swap | Protected |
//! |------|---------|------|-----------|
//! | **ZONE_MODEL** | Model weights (read-only, hugepage-backed) | No | `memory.min = limit` |
//! | **ZONE_KV** | KV-cache pages (COW, LRU eviction) | NVMe | `memory.min = 50%` |
//! | **ZONE_TENSOR** | Activation scratch space (arena-pooled) | No | None |
//! | **ZONE_GENERAL** | Everything else | Yes | None |
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │                     MemoryManager                           │
//! │  (public API: allocate, mmap, reserve, rebalance, stats)    │
//! ├──────────────────────────────────────────────────────────────┤
//! │ ZoneTracker │ CgroupManager │ HugepageManager │ VramManager │
//! │ KvPageManager │ PressureMonitor │ MemoryMetrics             │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,no_run
//! use spirit_memory::{MemoryConfig, MemoryManager, MemoryZone};
//!
//! let config = MemoryConfig::default();
//! let mgr = MemoryManager::new(config).expect("init memory manager");
//!
//! // Track a zone allocation
//! let id = mgr.track_allocation(MemoryZone::Tensor, 1024).unwrap();
//!
//! // Check zone stats
//! let stats = mgr.zone_stats(MemoryZone::Tensor);
//! println!("Tensor zone: {:.1}% used", stats.utilization_percent());
//!
//! // Release
//! mgr.release_allocation(id).unwrap();
//! ```
//!
//! // PHASE1-SHIM: Replace with kernel-native implementation in Phase 2

#![warn(clippy::all, clippy::pedantic)]
// Allow certain pedantic lints that are too noisy for this crate
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_fields_in_debug)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::identity_op)]

// ── Modules ────────────────────────────────────────────────────────────────

pub mod cgroups;
pub mod config;
pub mod error;
pub mod hugepages;
pub mod kv_cache;
pub mod manager;
pub mod metrics;
pub mod pressure;
pub mod types;
pub mod vram;
pub mod zones;

// ── Re-exports ─────────────────────────────────────────────────────────────

pub use config::MemoryConfig;
pub use error::SpiritMemoryError;
pub use manager::MemoryManager;
pub use types::{
    AllZoneStats, AllocationId, GpuDevice, HugepageStats, KvSnapshot, MemoryZone, ModelMapping,
    PageStatus, PressureInfo, PressureLevel, SessionId, VramReservation, VramStats, ZoneStats,
};
