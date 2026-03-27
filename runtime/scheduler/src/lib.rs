//! # Spirit Scheduler — Inference-Aware Process Scheduling
//!
//! This crate implements the Phase 1 userspace scheduler shim for `SpiritROS`.
//! It uses Linux cpusets (cgroups v2), `SCHED_FIFO`, nice levels, and GPU
//! compute mode management to give inference threads deterministic scheduling
//! priority over general workloads.
//!
//! ## Five-Tier Priority System
//!
//! | Tier | Linux Mechanism |
//! |------|----------------|
//! | P0: `INFERENCE_CRITICAL` | `SCHED_FIFO` 99, dedicated CPU cores |
//! | P1: `INFERENCE_BATCH`    | `SCHED_FIFO` 50, large time quanta |
//! | P2: `SYSTEM_SERVICES`    | `SCHED_OTHER` nice -10 |
//! | P3: COMPATIBILITY      | `SCHED_OTHER` nice 0–5 |
//! | P4: `IDLE_MAINTENANCE`   | `SCHED_IDLE` or nice 19 |
//!
//! ## Phase-Aware Scheduling
//!
//! The scheduler dynamically adjusts CPU core assignments based on the
//! current inference phase:
//!
//! - **Idle**: All cores available for system work
//! - **Prefill**: Burst mode — claim all cores for inference (compute-bound)
//! - **Decode**: Latency mode — dedicated cores with real-time priority
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     SpiritScheduler                             │
//! │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
//! │  │ Topology  │  │  Phase   │  │ Process  │  │   Watchdog    │  │
//! │  │ Discovery │  │ Manager  │  │  Table   │  │               │  │
//! │  └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │
//! │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
//! │  │  Cpuset  │  │ Priority │  │   GPU    │  │   Metrics     │  │
//! │  │ Manager  │  │ Manager  │  │ Scheduler│  │  (Prometheus)  │  │
//! │  └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │
//! │  ┌──────────────────────────────────────────────────────────┐  │
//! │  │              IPC Server (Unix Socket)                     │  │
//! │  └──────────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! // PHASE1-SHIM: Replace with kernel-native scheduler class in Phase 2.

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

// ── Modules ──────────────────────────────────────────────────────────────────

pub mod config;
pub mod cpuset;
pub mod error;
pub mod gpu;
pub mod ipc;
pub mod metrics;
pub mod phase;
pub mod priority;
pub mod process;
pub mod scheduler;
pub mod topology;
pub mod types;
pub mod watchdog;

// ── Re-exports ───────────────────────────────────────────────────────────────

pub use config::SchedulerConfig;
pub use error::SchedulerError;
pub use ipc::{IpcClient, IpcServer};
pub use metrics::SchedulerMetrics;
pub use scheduler::SpiritScheduler;
pub use types::{
    CpuId, CpuTopology, GpuDevice, InferencePhase, NumaNode, PriorityTier, ProcessEntry,
    SchedulerMessage, SchedulerResponse, SchedulerStats,
};
