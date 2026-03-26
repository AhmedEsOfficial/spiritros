//! # `SpiritKernel`
//!
//! The `SpiritKernel` is a hybrid microkernel designed from the ground up for LLM inference
//! workloads. It provides inference-aware scheduling, `TensorFirst` memory management, GPU
//! resource arbitration, and capability-based security.
//!
//! ## Phase Status
//!
//! This crate is a **Phase 2+ component**. In Phase 1, all kernel-level features are
//! approximated by userspace shims running on a stock Linux host:
//!
//! - Memory zones -> cgroups v2 + hugepages
//! - Scheduler -> cpuset + nice + `SCHED_FIFO`
//! - GPU management -> CUDA MPS + nvidia-container-toolkit
//! - Inference syscalls -> Unix domain socket IPC
//! - `SpiritFS` -> FUSE filesystem
//!
//! Do not build or depend on this crate in Phase 1 unless running tests.

// PHASE1-SHIM: This entire crate is a placeholder. The kernel will be implemented in Phase 2
// when we move from a virtualized Docker deployment to bare-metal / hypervisor execution.
// Until then, all "kernel" functionality lives in runtime/ as userspace shims over Linux.

#![cfg_attr(not(test), no_std)]
#![warn(clippy::all, clippy::pedantic)]

/// Kernel version string, matching the workspace version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Placeholder for the kernel's boot info structure.
///
/// In Phase 2, this will contain UEFI memory map, ACPI tables, and GPU BAR addresses
/// passed from the bootloader.
#[derive(Debug, Clone, Copy)]
pub struct BootInfo {
    /// Physical memory available in bytes.
    pub total_memory: u64,
    /// Number of CPU cores detected.
    pub cpu_count: u32,
    /// Whether a compatible GPU was detected during early init.
    pub gpu_present: bool,
}

/// Placeholder for kernel initialization.
///
/// # Safety
///
/// In Phase 2, this function will be called from the bootloader with a valid `BootInfo`
/// populated from hardware discovery. It must be called exactly once.
pub fn kernel_init(_boot_info: &BootInfo) {
    // PHASE1-SHIM: No-op in Phase 1. Kernel init will set up memory zones,
    // initialize the scheduler, probe GPU devices, and start the capability manager.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_version_matches_package() {
        assert_eq!(VERSION, "0.1.0");
    }

    #[test]
    fn test_kernel_init_does_not_panic() {
        let boot_info = BootInfo {
            total_memory: 16 * 1024 * 1024 * 1024, // 16 GiB
            cpu_count: 8,
            gpu_present: true,
        };
        kernel_init(&boot_info);
    }

    #[test]
    fn test_boot_info_debug_impl() {
        let info = BootInfo {
            total_memory: 0,
            cpu_count: 1,
            gpu_present: false,
        };
        let debug_str = format!("{info:?}");
        assert!(debug_str.contains("BootInfo"));
    }
}
