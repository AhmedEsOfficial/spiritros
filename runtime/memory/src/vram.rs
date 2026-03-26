//! GPU VRAM tracking and reservation.
//!
//! In Phase 1, actual GPU memory is managed by Ollama / llama.cpp through
//! the CUDA/ROCm runtime. This module provides:
//!
//! - GPU device detection (via `nvidia-smi` parsing)
//! - Logical VRAM reservation tracking (prevents over-subscription)
//! - Per-device utilization statistics
//! - CUDA MPS awareness
//!
//! The VRAM manager does NOT call CUDA APIs directly — it tracks what
//! *should* be allocated and enforces limits at the coordinator level.
//!
//! // PHASE1-SHIM: In Phase 2, the kernel's GPU driver subsystem manages
//! // VRAM directly with contiguous block allocation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use tracing::{debug, info, warn};

use crate::error::{Result, SpiritMemoryError};
use crate::types::{GpuDevice, IdGenerator, VramReservation, VramStats, MIB};

/// Tracks GPU VRAM reservations to prevent over-subscription.
pub struct VramManager {
    /// Detected GPU devices.
    devices: Vec<GpuDevice>,
    /// Per-device allocated bytes (atomic for lock-free reads).
    allocated: Vec<AtomicU64>,
    /// Per-device reservation map: id → size.
    reservations: Vec<Mutex<HashMap<u64, u64>>>,
    /// Fraction of total VRAM available for Spirit use.
    memory_fraction: f64,
    /// Monotonic ID generator for reservations.
    id_gen: IdGenerator,
}

impl VramManager {
    /// Create a new VRAM manager, detecting available GPUs.
    pub fn new(memory_fraction: f64) -> Self {
        let devices = Self::detect_gpus();

        if devices.is_empty() {
            warn!("no GPU devices detected; VRAM management disabled");
        } else {
            for dev in &devices {
                info!(
                    gpu = dev.index,
                    name = %dev.name,
                    vram_mb = dev.total_vram_bytes / MIB,
                    "detected GPU"
                );
            }
        }

        let count = devices.len();
        let allocated = (0..count).map(|_| AtomicU64::new(0)).collect();
        let reservations = (0..count).map(|_| Mutex::new(HashMap::new())).collect();

        Self {
            devices,
            allocated,
            reservations,
            memory_fraction,
            id_gen: IdGenerator::new(),
        }
    }

    /// Number of detected GPUs.
    #[must_use]
    pub fn device_count(&self) -> u32 {
        self.devices.len() as u32
    }

    /// Get device info by index.
    pub fn device(&self, index: u32) -> Result<&GpuDevice> {
        self.devices
            .get(index as usize)
            .ok_or(SpiritMemoryError::GpuDeviceNotFound {
                index,
                count: self.device_count(),
            })
    }

    /// Usable VRAM for a device (total × memory_fraction).
    pub fn usable_vram(&self, index: u32) -> Result<u64> {
        let dev = self.device(index)?;
        Ok((dev.total_vram_bytes as f64 * self.memory_fraction) as u64)
    }

    /// Reserve a contiguous block of VRAM on a specific GPU.
    ///
    /// This is a logical reservation — it tracks the allocation and
    /// ensures we don't over-subscribe. The actual CUDA allocation
    /// happens when Ollama loads the model.
    pub fn reserve(&self, device_index: u32, size: u64) -> Result<VramReservation> {
        let usable = self.usable_vram(device_index)?;
        let idx = device_index as usize;

        // Check available space
        let current = self.allocated[idx].load(Ordering::Relaxed);
        if current + size > usable {
            return Err(SpiritMemoryError::InsufficientVram {
                device: device_index,
                needed_mb: size / MIB,
                available_mb: (usable.saturating_sub(current)) / MIB,
            });
        }

        // Reserve
        self.allocated[idx].fetch_add(size, Ordering::SeqCst);
        let id = self.id_gen.next();

        let mut map = self.reservations[idx]
            .lock()
            .expect("vram reservation lock poisoned");
        map.insert(id, size);

        debug!(
            device = device_index,
            size_mb = size / MIB,
            id,
            "VRAM reserved"
        );

        Ok(VramReservation {
            device_index,
            size,
            id,
        })
    }

    /// Release a VRAM reservation.
    pub fn release(&self, reservation: &VramReservation) -> Result<()> {
        let idx = reservation.device_index as usize;
        if idx >= self.devices.len() {
            return Err(SpiritMemoryError::GpuDeviceNotFound {
                index: reservation.device_index,
                count: self.device_count(),
            });
        }

        let mut map = self.reservations[idx]
            .lock()
            .expect("vram reservation lock poisoned");

        if map.remove(&reservation.id).is_some() {
            self.allocated[idx].fetch_sub(reservation.size, Ordering::SeqCst);
            debug!(
                device = reservation.device_index,
                size_mb = reservation.size / MIB,
                id = reservation.id,
                "VRAM released"
            );
        } else {
            warn!(
                id = reservation.id,
                "VRAM release: reservation not found (double release?)"
            );
        }

        Ok(())
    }

    /// Get VRAM statistics for a specific GPU.
    pub fn stats(&self, device_index: u32) -> Result<VramStats> {
        let dev = self.device(device_index)?.clone();
        let idx = device_index as usize;
        let used = self.allocated[idx].load(Ordering::Relaxed);
        let usable = self.usable_vram(device_index)?;

        Ok(VramStats {
            device: dev,
            used_bytes: used,
            reserved_bytes: used,
            free_bytes: usable.saturating_sub(used),
        })
    }

    /// Get VRAM statistics for all GPUs.
    pub fn all_stats(&self) -> Vec<VramStats> {
        (0..self.device_count())
            .filter_map(|i| self.stats(i).ok())
            .collect()
    }

    // ── GPU Detection ───────────────────────────────────────────────────

    fn detect_gpus() -> Vec<GpuDevice> {
        // Try nvidia-smi first (most common)
        if let Ok(gpus) = Self::detect_nvidia_gpus() {
            if !gpus.is_empty() {
                return gpus;
            }
        }

        // Try sysfs fallback
        if let Ok(gpus) = Self::detect_gpus_sysfs() {
            if !gpus.is_empty() {
                return gpus;
            }
        }

        Vec::new()
    }

    fn detect_nvidia_gpus() -> std::result::Result<Vec<GpuDevice>, String> {
        let output = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .map_err(|e| format!("nvidia-smi not found: {e}"))?;

        if !output.status.success() {
            return Err("nvidia-smi failed".into());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut devices = Vec::new();

        for line in stdout.lines() {
            let parts: Vec<&str> = line.split(',').map(str::trim).collect();
            if parts.len() >= 3 {
                let index: u32 = parts[0].parse().unwrap_or(0);
                let name = parts[1].to_string();
                // nvidia-smi reports memory in MiB
                let vram_mib: u64 = parts[2].parse().unwrap_or(0);
                devices.push(GpuDevice {
                    index,
                    name,
                    total_vram_bytes: vram_mib * MIB,
                });
            }
        }

        Ok(devices)
    }

    fn detect_gpus_sysfs() -> std::result::Result<Vec<GpuDevice>, String> {
        // Fallback: check /proc/driver/nvidia/gpus/
        let nvidia_path = std::path::Path::new("/proc/driver/nvidia/gpus");
        if !nvidia_path.exists() {
            return Ok(Vec::new());
        }

        let mut devices = Vec::new();
        let entries = std::fs::read_dir(nvidia_path).map_err(|e| e.to_string())?;

        for (idx, entry) in entries.enumerate() {
            let entry = entry.map_err(|e| e.to_string())?;
            let info_path = entry.path().join("information");
            if let Ok(info) = std::fs::read_to_string(&info_path) {
                let name = info.lines().find(|l| l.starts_with("Model:")).map_or_else(
                    || format!("GPU {idx}"),
                    |l| l.trim_start_matches("Model:").trim().to_string(),
                );

                devices.push(GpuDevice {
                    index: idx as u32,
                    name,
                    // sysfs doesn't easily expose VRAM; default to 0
                    total_vram_bytes: 0,
                });
            }
        }

        Ok(devices)
    }
}

impl std::fmt::Debug for VramManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VramManager")
            .field("device_count", &self.devices.len())
            .field("memory_fraction", &self.memory_fraction)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_manager() -> VramManager {
        // Create a manager with fake devices for testing
        let devices = vec![
            GpuDevice {
                index: 0,
                name: "Test GPU 0".into(),
                total_vram_bytes: 24 * 1024 * MIB, // 24 GiB
            },
            GpuDevice {
                index: 1,
                name: "Test GPU 1".into(),
                total_vram_bytes: 16 * 1024 * MIB, // 16 GiB
            },
        ];

        let count = devices.len();
        VramManager {
            devices,
            allocated: (0..count).map(|_| AtomicU64::new(0)).collect(),
            reservations: (0..count).map(|_| Mutex::new(HashMap::new())).collect(),
            memory_fraction: 0.90,
            id_gen: IdGenerator::new(),
        }
    }

    #[test]
    fn test_device_count() {
        let mgr = make_test_manager();
        assert_eq!(mgr.device_count(), 2);
    }

    #[test]
    fn test_device_lookup() {
        let mgr = make_test_manager();
        let dev = mgr.device(0).unwrap();
        assert_eq!(dev.name, "Test GPU 0");
        assert!(mgr.device(99).is_err());
    }

    #[test]
    fn test_usable_vram() {
        let mgr = make_test_manager();
        let usable = mgr.usable_vram(0).unwrap();
        let expected = (24.0 * 1024.0 * MIB as f64 * 0.90) as u64;
        assert_eq!(usable, expected);
    }

    #[test]
    fn test_reserve_and_release() {
        let mgr = make_test_manager();

        let res = mgr.reserve(0, 4 * 1024 * MIB).unwrap();
        assert_eq!(res.device_index(), 0);
        assert_eq!(res.size(), 4 * 1024 * MIB);

        let stats = mgr.stats(0).unwrap();
        assert_eq!(stats.used_bytes, 4 * 1024 * MIB);

        mgr.release(&res).unwrap();
        let stats = mgr.stats(0).unwrap();
        assert_eq!(stats.used_bytes, 0);
    }

    #[test]
    fn test_reserve_exceeds_capacity() {
        let mgr = make_test_manager();
        let usable = mgr.usable_vram(0).unwrap();
        let result = mgr.reserve(0, usable + 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_reservations() {
        let mgr = make_test_manager();
        let r1 = mgr.reserve(0, 2 * 1024 * MIB).unwrap();
        let r2 = mgr.reserve(0, 2 * 1024 * MIB).unwrap();

        let stats = mgr.stats(0).unwrap();
        assert_eq!(stats.used_bytes, 4 * 1024 * MIB);

        mgr.release(&r1).unwrap();
        let stats = mgr.stats(0).unwrap();
        assert_eq!(stats.used_bytes, 2 * 1024 * MIB);

        mgr.release(&r2).unwrap();
        let stats = mgr.stats(0).unwrap();
        assert_eq!(stats.used_bytes, 0);
    }

    #[test]
    fn test_cross_device_isolation() {
        let mgr = make_test_manager();
        let _r0 = mgr.reserve(0, 4 * 1024 * MIB).unwrap();
        let _r1 = mgr.reserve(1, 2 * 1024 * MIB).unwrap();

        assert_eq!(mgr.stats(0).unwrap().used_bytes, 4 * 1024 * MIB);
        assert_eq!(mgr.stats(1).unwrap().used_bytes, 2 * 1024 * MIB);
    }

    #[test]
    fn test_all_stats() {
        let mgr = make_test_manager();
        let all = mgr.all_stats();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_invalid_device() {
        let mgr = make_test_manager();
        assert!(mgr.reserve(99, 1024).is_err());
        assert!(mgr.stats(99).is_err());
    }

    #[test]
    fn test_double_release_is_safe() {
        let mgr = make_test_manager();
        let res = mgr.reserve(0, 1024 * MIB).unwrap();
        mgr.release(&res).unwrap();
        // Second release should not panic — just warn
        mgr.release(&res).unwrap();
        assert_eq!(mgr.stats(0).unwrap().used_bytes, 0);
    }
}
