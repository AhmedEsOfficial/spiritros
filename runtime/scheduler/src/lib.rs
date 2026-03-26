//! # Spirit Scheduler — Inference-Aware Process Scheduling
//!
//! This crate implements the Phase 1 userspace scheduler shim. It uses Linux
//! cpusets, nice levels, and `SCHED_FIFO` to give inference threads deterministic
//! scheduling priority over general workloads.
//!
//! ## Strategy
//!
//! - **Inference threads** are pinned to dedicated CPU cores via cpusets and
//!   run at `SCHED_FIFO` priority to minimize scheduling jitter.
//! - **Decode threads** (the latency-critical token generation loop) get the
//!   highest FIFO priority and are never preempted by non-inference work.
//! - **General processes** are confined to a separate cpuset and use the
//!   default `SCHED_OTHER` policy.
//!
//! // PHASE1-SHIM: Replace with kernel-native scheduler class in Phase 2.

#![warn(clippy::all, clippy::pedantic)]

// ── Modules ──────────────────────────────────────────────────────────────────

/// Core type definitions for scheduling.
pub mod types {
    use serde::{Deserialize, Serialize};

    /// Scheduling class for a process or thread.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum SchedClass {
        /// Inference decode thread — highest priority, `SCHED_FIFO`.
        InferenceDecode,
        /// Inference prefill thread — high priority, `SCHED_FIFO`.
        InferencePrefill,
        /// Background inference work (embedding, indexing).
        InferenceBackground,
        /// General-purpose process — default scheduler.
        General,
    }

    /// Describes the CPU affinity assignment for a scheduling class.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CpuAssignment {
        /// Scheduling class this assignment applies to.
        pub sched_class: SchedClass,
        /// List of CPU core indices assigned to this class.
        pub cores: Vec<usize>,
        /// `SCHED_FIFO` priority (1-99), or `None` for `SCHED_OTHER`.
        pub fifo_priority: Option<u32>,
    }
}

/// Error types for scheduler operations.
pub mod error {
    /// Errors that can occur during scheduling operations.
    #[derive(Debug, thiserror::Error)]
    pub enum SchedulerError {
        /// Failed to set CPU affinity.
        #[error("Failed to set cpuset for pid {pid}: {reason}")]
        CpusetError { pid: u32, reason: String },

        /// Failed to set scheduling policy.
        #[error("Failed to set scheduling policy: {0}")]
        PolicyError(String),

        /// Invalid priority value.
        #[error("Invalid FIFO priority {0}: must be 1-99")]
        InvalidPriority(u32),

        /// No cores available for the requested class.
        #[error("No CPU cores assigned to scheduling class {0:?}")]
        NoCoresAvailable(super::types::SchedClass),

        /// Configuration error.
        #[error("Scheduler configuration error: {0}")]
        ConfigError(String),

        /// Underlying OS error.
        #[error("OS error: {0}")]
        OsError(#[from] nix::Error),
    }
}

pub mod cpuset {
    //! cpuset management for CPU partitioning.
    //!
    //! Manages cpuset cgroups to partition CPU cores between inference
    //! and general workloads. Inference cores are isolated from the
    //! general scheduler to reduce jitter.
}

pub mod priority {
    //! Thread priority management (nice, `SCHED_FIFO`).
    //!
    //! Sets scheduling policies and priorities for inference threads.
    //!
    //! Decode threads run at `SCHED_FIFO` with the highest priority.
    //! This ensures they are never preempted by `SCHED_OTHER` processes,
    //! which is critical for consistent token generation latency.

    use crate::error::SchedulerError;

    /// Validates that a FIFO priority value is in the allowed range (1-99).
    ///
    /// # Errors
    ///
    /// Returns `SchedulerError::InvalidPriority` if the value is out of range.
    pub fn validate_fifo_priority(priority: u32) -> Result<u32, SchedulerError> {
        if (1..=99).contains(&priority) {
            Ok(priority)
        } else {
            Err(SchedulerError::InvalidPriority(priority))
        }
    }
}

pub mod inference_aware {
    //! Inference-aware scheduling decisions.
    //!
    //! High-level scheduling logic that maps inference session states
    //! to scheduling classes and CPU assignments.
    //!
    //! When a session enters the decode phase, its threads are promoted
    //! to `SchedClass::InferenceDecode`. When the session goes idle,
    //! threads are demoted back to `InferenceBackground`.
}

/// Configuration for the scheduler subsystem.
pub mod config {
    use serde::{Deserialize, Serialize};

    /// Configuration for the Spirit scheduler shim.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SchedulerConfig {
        /// CPU cores reserved for inference decode threads.
        pub decode_cores: Vec<usize>,
        /// CPU cores reserved for inference prefill threads.
        pub prefill_cores: Vec<usize>,
        /// CPU cores for general-purpose workloads.
        pub general_cores: Vec<usize>,
        /// `SCHED_FIFO` priority for decode threads (1-99).
        pub decode_fifo_priority: u32,
        /// `SCHED_FIFO` priority for prefill threads (1-99).
        pub prefill_fifo_priority: u32,
    }

    impl Default for SchedulerConfig {
        fn default() -> Self {
            Self {
                decode_cores: vec![0, 1],
                prefill_cores: vec![2, 3],
                general_cores: vec![4, 5, 6, 7],
                decode_fifo_priority: 99,
                prefill_fifo_priority: 80,
            }
        }
    }
}

// ── Re-exports ───────────────────────────────────────────────────────────────

pub use config::SchedulerConfig;
pub use error::SchedulerError;
pub use types::{CpuAssignment, SchedClass};

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_config_default_has_non_overlapping_cores() {
        let cfg = SchedulerConfig::default();
        let mut all_cores: Vec<usize> = Vec::new();
        all_cores.extend_from_slice(&cfg.decode_cores);
        all_cores.extend_from_slice(&cfg.prefill_cores);
        all_cores.extend_from_slice(&cfg.general_cores);
        let len_before = all_cores.len();
        all_cores.sort();
        all_cores.dedup();
        assert_eq!(
            len_before,
            all_cores.len(),
            "Default core assignments must not overlap"
        );
    }

    #[test]
    fn test_validate_fifo_priority_valid() {
        assert!(priority::validate_fifo_priority(1).is_ok());
        assert!(priority::validate_fifo_priority(50).is_ok());
        assert!(priority::validate_fifo_priority(99).is_ok());
    }

    #[test]
    fn test_validate_fifo_priority_invalid() {
        assert!(priority::validate_fifo_priority(0).is_err());
        assert!(priority::validate_fifo_priority(100).is_err());
    }

    #[test]
    fn test_sched_class_equality() {
        assert_eq!(SchedClass::InferenceDecode, SchedClass::InferenceDecode);
        assert_ne!(SchedClass::InferenceDecode, SchedClass::General);
    }

    #[test]
    fn test_cpu_assignment_construction() {
        let assignment = CpuAssignment {
            sched_class: SchedClass::InferenceDecode,
            cores: vec![0, 1],
            fifo_priority: Some(99),
        };
        assert_eq!(assignment.cores.len(), 2);
        assert_eq!(assignment.fifo_priority, Some(99));
    }

    #[test]
    fn test_scheduler_error_display() {
        let err = SchedulerError::InvalidPriority(0);
        assert!(err.to_string().contains('0'));
        assert!(err.to_string().contains("1-99"));
    }
}
