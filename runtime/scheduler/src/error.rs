//! Error types for the Spirit Scheduler.

use crate::types::PriorityTier;

/// Errors that can occur during scheduling operations.
#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    /// Failed to set CPU affinity for a process.
    #[error("Failed to set cpuset for pid {pid}: {reason}")]
    CpusetError { pid: u32, reason: String },

    /// Failed to set scheduling policy.
    #[error("Failed to set scheduling policy for pid {pid}: {reason}")]
    PolicyError { pid: u32, reason: String },

    /// Invalid FIFO priority value.
    #[error("Invalid FIFO priority {0}: must be 1-99")]
    InvalidPriority(u32),

    /// No cores available for the requested tier.
    #[error("No CPU cores assigned to tier {0}")]
    NoCoresAvailable(PriorityTier),

    /// Configuration error.
    #[error("Scheduler configuration error: {0}")]
    ConfigError(String),

    /// Process not found in the process table.
    #[error("Process {0} not registered with scheduler")]
    ProcessNotFound(u32),

    /// Topology detection failed.
    #[error("CPU topology detection failed: {0}")]
    TopologyError(String),

    /// GPU management error.
    #[error("GPU scheduling error: {0}")]
    GpuError(String),

    /// Phase transition error.
    #[error("Phase transition failed: {0}")]
    PhaseTransitionError(String),

    /// IPC error.
    #[error("Scheduler IPC error: {0}")]
    IpcError(String),

    /// Cgroup management error.
    #[error("Cgroup error: {0}")]
    CgroupError(String),

    /// Watchdog detected a starvation condition.
    #[error("Watchdog: {0}")]
    WatchdogAlert(String),

    /// Insufficient capabilities (e.g., missing `CAP_SYS_NICE`).
    #[error("Insufficient capability: {0}")]
    InsufficientCapability(String),

    /// Underlying OS error.
    #[error("OS error: {0}")]
    OsError(#[from] nix::Error),

    /// I/O error.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type alias for scheduler operations.
pub type Result<T> = std::result::Result<T, SchedulerError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = SchedulerError::InvalidPriority(0);
        assert!(err.to_string().contains("1-99"));

        let err = SchedulerError::ProcessNotFound(42);
        assert!(err.to_string().contains("42"));

        let err = SchedulerError::NoCoresAvailable(PriorityTier::InferenceCritical);
        assert!(err.to_string().contains("P0:INFERENCE_CRITICAL"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let sched_err: SchedulerError = io_err.into();
        assert!(matches!(sched_err, SchedulerError::IoError(_)));
    }
}
