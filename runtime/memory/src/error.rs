//! Error types for Spirit memory operations.

use crate::types::MemoryZone;

/// Errors that can occur during Spirit memory operations.
#[derive(Debug, thiserror::Error)]
pub enum SpiritMemoryError {
    /// Attempted allocation exceeds zone capacity.
    #[error("zone {zone} exhausted: requested {requested} bytes, available {available} bytes")]
    ZoneExhausted {
        zone: MemoryZone,
        requested: u64,
        available: u64,
    },

    /// cgroups v2 is not available on this system.
    #[error("cgroups v2 not available: {0}")]
    CgroupsUnavailable(String),

    /// cgroups v2 operation failed.
    #[error("cgroups operation failed: {0}")]
    CgroupsError(String),

    /// Hugepage allocation failed.
    #[error(
        "hugepage allocation failed: requested {requested} pages, \
         got {allocated}"
    )]
    HugepageAllocationFailed { requested: u64, allocated: u64 },

    /// Hugepages not available on this system.
    #[error("hugepages not available: {0}")]
    HugepagesUnavailable(String),

    /// Attempted to mutate a model zone after load (invariant violation).
    #[error("ZONE_MODEL is read-only after model load")]
    ModelZoneReadOnly,

    /// mmap operation failed.
    #[error("mmap failed for {path}: {source}")]
    MmapFailed {
        path: String,
        source: std::io::Error,
    },

    /// Insufficient GPU VRAM for the requested reservation.
    #[error(
        "insufficient VRAM on GPU {device}: need {needed_mb} MB, \
         available {available_mb} MB"
    )]
    InsufficientVram {
        device: u32,
        needed_mb: u64,
        available_mb: u64,
    },

    /// No GPU devices detected.
    #[error("no GPU devices detected")]
    NoGpuDevices,

    /// GPU device index out of range.
    #[error("GPU device {index} not found (have {count} devices)")]
    GpuDeviceNotFound { index: u32, count: u32 },

    /// KV-cache page pool exhausted.
    #[error(
        "KV-cache page pool exhausted: {active} active pages, \
         {swapped} swapped, {total} total"
    )]
    KvPoolExhausted {
        active: usize,
        swapped: usize,
        total: usize,
    },

    /// Session not found in KV-cache manager.
    #[error("KV-cache session {0} not found")]
    SessionNotFound(crate::types::SessionId),

    /// KV-cache page index out of bounds.
    #[error("KV-cache page index {index} out of bounds (session has {count} pages)")]
    PageIndexOutOfBounds { index: usize, count: usize },

    /// KV-cache swap I/O error.
    #[error("KV-cache swap I/O error: {0}")]
    SwapError(String),

    /// Configuration error.
    #[error("memory configuration error: {0}")]
    ConfigError(String),

    /// Zone sum exceeds available physical memory.
    #[error(
        "zone sum ({zone_sum_bytes} bytes) exceeds available memory \
         ({available_bytes} bytes)"
    )]
    ZoneSumExceedsMemory {
        zone_sum_bytes: u64,
        available_bytes: u64,
    },

    /// Rebalance would leave source zone below its minimum guarantee.
    #[error(
        "rebalance denied: moving {amount} bytes from {from} would \
         leave it below minimum ({min_bytes} bytes)"
    )]
    RebalanceDenied {
        from: MemoryZone,
        amount: u64,
        min_bytes: u64,
    },

    /// Underlying OS error.
    #[error("OS error: {0}")]
    OsError(#[from] nix::Error),

    /// I/O error interacting with cgroup or proc filesystem.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, SpiritMemoryError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_zone_exhausted() {
        let err = SpiritMemoryError::ZoneExhausted {
            zone: MemoryZone::KvCache,
            requested: 1024,
            available: 512,
        };
        let msg = err.to_string();
        assert!(msg.contains("kv"));
        assert!(msg.contains("1024"));
        assert!(msg.contains("512"));
    }

    #[test]
    fn test_error_display_insufficient_vram() {
        let err = SpiritMemoryError::InsufficientVram {
            device: 0,
            needed_mb: 4096,
            available_mb: 2048,
        };
        let msg = err.to_string();
        assert!(msg.contains("GPU 0"));
        assert!(msg.contains("4096"));
    }

    #[test]
    fn test_error_display_model_readonly() {
        let err = SpiritMemoryError::ModelZoneReadOnly;
        assert!(err.to_string().contains("read-only"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
        let err: SpiritMemoryError = io_err.into();
        assert!(err.to_string().contains("gone"));
    }
}
