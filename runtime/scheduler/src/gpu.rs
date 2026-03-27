//! GPU scheduling and exclusive compute mode management.
//!
//! During active inference, the GPU is set to exclusive-compute mode
//! via `nvidia-smi` to prevent context switches from competing CUDA
//! consumers. The GPU is released to default mode when idle.
//!
//! // PHASE1-SHIM: In Phase 2, GPU scheduling is handled by the kernel.

use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

use tracing::{debug, info, warn};

use crate::error::{Result, SchedulerError};

/// NVIDIA GPU compute modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuComputeMode {
    /// Default mode: multiple contexts allowed.
    Default,
    /// Exclusive process mode: only one CUDA context at a time.
    ExclusiveProcess,
}

impl GpuComputeMode {
    fn nvidia_smi_arg(self) -> &'static str {
        match self {
            Self::Default => "DEFAULT",
            Self::ExclusiveProcess => "EXCLUSIVE_PROCESS",
        }
    }
}

/// Manages GPU scheduling and compute mode for inference workloads.
pub struct GpuScheduler {
    /// Whether GPU management is enabled.
    enabled: bool,
    /// Number of detected NVIDIA GPUs.
    gpu_count: usize,
    /// Whether GPU is currently in exclusive mode.
    exclusive: AtomicBool,
    /// Whether nvidia-smi is available.
    #[allow(dead_code)]
    nvidia_smi_available: bool,
}

impl GpuScheduler {
    /// Create a new GPU scheduler.
    pub fn new(enabled: bool) -> Self {
        let nvidia_smi_available = check_nvidia_smi();
        let gpu_count = if nvidia_smi_available {
            detect_gpu_count()
        } else {
            0
        };

        if enabled && !nvidia_smi_available {
            warn!("GPU exclusive mode requested but nvidia-smi not available");
        }

        if gpu_count > 0 {
            info!(gpu_count, "NVIDIA GPUs detected for scheduling");
        }

        Self {
            enabled: enabled && nvidia_smi_available && gpu_count > 0,
            gpu_count,
            exclusive: AtomicBool::new(false),
            nvidia_smi_available,
        }
    }

    /// Enter exclusive compute mode on all GPUs.
    ///
    /// Called when inference becomes active. Prevents other processes
    /// from creating GPU contexts.
    ///
    /// # Errors
    ///
    /// Returns `SchedulerError::GpuError` if `nvidia-smi` fails to set
    /// exclusive compute mode.
    pub fn enter_exclusive_mode(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        if self.exclusive.load(Ordering::Relaxed) {
            return Ok(()); // Already exclusive
        }

        for gpu_id in 0..self.gpu_count {
            set_compute_mode(gpu_id, GpuComputeMode::ExclusiveProcess)?;
        }

        self.exclusive.store(true, Ordering::Relaxed);
        info!("GPU entered exclusive compute mode");
        Ok(())
    }

    /// Release exclusive compute mode on all GPUs.
    ///
    /// Called when inference goes idle. Allows other processes to
    /// access the GPU.
    ///
    /// # Errors
    ///
    /// Returns `SchedulerError::GpuError` if `nvidia-smi` fails to reset
    /// the compute mode.
    pub fn release_exclusive_mode(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        if !self.exclusive.load(Ordering::Relaxed) {
            return Ok(()); // Already default
        }

        for gpu_id in 0..self.gpu_count {
            set_compute_mode(gpu_id, GpuComputeMode::Default)?;
        }

        self.exclusive.store(false, Ordering::Relaxed);
        info!("GPU released to default compute mode");
        Ok(())
    }

    /// Returns whether GPU is currently in exclusive mode.
    #[must_use]
    pub fn is_exclusive(&self) -> bool {
        self.exclusive.load(Ordering::Relaxed)
    }

    /// Returns the number of managed GPUs.
    #[must_use]
    pub fn gpu_count(&self) -> usize {
        self.gpu_count
    }

    /// Returns whether GPU management is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Ensure GPUs are released on shutdown.
    pub fn cleanup(&self) {
        if self.exclusive.load(Ordering::Relaxed) {
            if let Err(e) = self.release_exclusive_mode() {
                warn!(error = %e, "Failed to release GPU exclusive mode during cleanup");
            }
        }
    }
}

/// Set the compute mode for a specific GPU.
fn set_compute_mode(gpu_id: usize, mode: GpuComputeMode) -> Result<()> {
    let output = Command::new("nvidia-smi")
        .args(["-i", &gpu_id.to_string(), "-c", mode.nvidia_smi_arg()])
        .output()
        .map_err(|e| SchedulerError::GpuError(format!("nvidia-smi execution failed: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(SchedulerError::GpuError(format!(
            "nvidia-smi -c {} failed for GPU {gpu_id}: {stderr}",
            mode.nvidia_smi_arg()
        )));
    }

    debug!(
        gpu = gpu_id,
        mode = ?mode,
        "GPU compute mode set"
    );
    Ok(())
}

/// Check if nvidia-smi is available on the system.
fn check_nvidia_smi() -> bool {
    Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Detect the number of NVIDIA GPUs via nvidia-smi.
fn detect_gpu_count() -> usize {
    Command::new("nvidia-smi")
        .args(["--query-gpu=count", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8_lossy(&o.stdout)
                    .trim()
                    .lines()
                    .next()
                    .and_then(|s| s.trim().parse::<usize>().ok())
            } else {
                None
            }
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_scheduler_disabled() {
        let sched = GpuScheduler::new(false);
        assert!(!sched.is_enabled());
        assert!(!sched.is_exclusive());
        // Should be no-ops
        assert!(sched.enter_exclusive_mode().is_ok());
        assert!(sched.release_exclusive_mode().is_ok());
    }

    #[test]
    fn test_compute_mode_args() {
        assert_eq!(GpuComputeMode::Default.nvidia_smi_arg(), "DEFAULT");
        assert_eq!(
            GpuComputeMode::ExclusiveProcess.nvidia_smi_arg(),
            "EXCLUSIVE_PROCESS"
        );
    }
}
