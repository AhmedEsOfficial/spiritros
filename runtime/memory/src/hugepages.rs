//! Hugepage allocation and management for ZONE_MODEL.
//!
//! Model weights benefit enormously from hugepages:
//! - A 7B Q4_K_M model (~4.3 GB) with 4K pages = ~1.1M page table entries
//! - With 2 MB hugepages = ~2,150 entries (500× fewer TLB misses)
//! - With 1 GB hugepages = ~5 entries
//!
//! This module provides:
//! - Hugepage availability detection
//! - `mmap` with `MAP_HUGETLB | MAP_POPULATE` for model files
//! - Automatic fallback to standard pages when hugepages unavailable
//!
//! // PHASE1-SHIM: In Phase 2, the kernel allocator hands out hugepages
//! // directly without going through the host's hugetlbfs.

use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

use tracing::{debug, info, warn};

use crate::error::{Result, SpiritMemoryError};
use crate::types::{HugepageStats, ModelMapping, HUGEPAGE_SIZE_2M};

/// Manages hugepage resources for model weight mappings.
#[derive(Debug)]
pub struct HugepageManager {
    /// Configured hugepage size (2M or 1G).
    page_size: u64,
    /// Whether hugepages are available on this system.
    available: bool,
    /// Whether to attempt hugepage-backed mappings.
    enabled: bool,
}

impl HugepageManager {
    /// Create a new hugepage manager.
    ///
    /// Probes the system for hugepage availability. If hugepages are not
    /// pre-reserved on the host, the manager falls back to standard pages.
    pub fn new(page_size: u64, enabled: bool) -> Self {
        let available = if enabled {
            Self::probe_hugepages(page_size)
        } else {
            false
        };

        if enabled && !available {
            warn!(
                page_size,
                "hugepages not available; model mappings will use \
                 standard pages (higher TLB pressure)"
            );
        } else if available {
            info!(page_size, "hugepage support detected");
        }

        Self {
            page_size,
            available,
            enabled,
        }
    }

    /// Whether hugepages are available and enabled.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.available && self.enabled
    }

    /// Memory-map a model weight file with hugepages (or standard fallback).
    ///
    /// The mapping is:
    /// - **Read-only** (`PROT_READ`) — model weights are immutable after load
    /// - **Pre-faulted** (`MAP_POPULATE`) — no page faults during inference
    /// - **Hugepage-backed** (`MAP_HUGETLB`) — when available
    /// - **Sequential advice** (`MADV_SEQUENTIAL`) — prefetch hint
    ///
    /// # Errors
    ///
    /// Returns `MmapFailed` if the file cannot be opened or mapped.
    pub fn mmap_model_file(&self, path: &Path) -> Result<ModelMapping> {
        let file = std::fs::File::open(path).map_err(|e| SpiritMemoryError::MmapFailed {
            path: path.display().to_string(),
            source: e,
        })?;

        let metadata = file.metadata().map_err(|e| SpiritMemoryError::MmapFailed {
            path: path.display().to_string(),
            source: e,
        })?;

        let len = metadata.len() as usize;
        if len == 0 {
            return Ok(ModelMapping::new(
                std::ptr::null_mut(),
                0,
                path.to_path_buf(),
                false,
            ));
        }

        let fd = file.as_raw_fd();

        // Try hugepage-backed mapping first
        if self.is_available() {
            match self.mmap_hugepage(fd, len, path) {
                Ok(mapping) => return Ok(mapping),
                Err(e) => {
                    warn!(
                        path = %path.display(),
                        error = %e,
                        "hugepage mmap failed, falling back to standard pages"
                    );
                }
            }
        }

        // Standard page fallback
        self.mmap_standard(fd, len, path)
    }

    /// Query hugepage statistics from the kernel.
    pub fn stats(&self) -> HugepageStats {
        if !self.available {
            return HugepageStats::default();
        }
        self.read_hugepage_stats().unwrap_or_default()
    }

    // ── Internal ────────────────────────────────────────────────────────

    fn mmap_hugepage(&self, fd: i32, len: usize, path: &Path) -> Result<ModelMapping> {
        // SAFETY: We pass a valid fd, len > 0, and appropriate flags.
        // The resulting pointer is valid for `len` bytes.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED | libc::MAP_POPULATE | libc::MAP_HUGETLB,
                fd,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(SpiritMemoryError::MmapFailed {
                path: path.display().to_string(),
                source: std::io::Error::last_os_error(),
            });
        }

        // Advise kernel about sequential access pattern
        // SAFETY: ptr is valid from a successful mmap.
        unsafe {
            libc::madvise(ptr, len, libc::MADV_SEQUENTIAL);
        }

        info!(
            path = %path.display(),
            len,
            page_size = self.page_size,
            "mapped model weights with hugepages"
        );

        Ok(ModelMapping::new(ptr.cast(), len, path.to_path_buf(), true))
    }

    fn mmap_standard(&self, fd: i32, len: usize, path: &Path) -> Result<ModelMapping> {
        let _ = &self; // used for method receiver consistency
                       // SAFETY: Valid fd, len > 0, standard flags.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED | libc::MAP_POPULATE,
                fd,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(SpiritMemoryError::MmapFailed {
                path: path.display().to_string(),
                source: std::io::Error::last_os_error(),
            });
        }

        // SAFETY: ptr is valid from a successful mmap.
        unsafe {
            libc::madvise(ptr, len, libc::MADV_SEQUENTIAL);
        }

        debug!(
            path = %path.display(),
            len,
            "mapped model weights with standard pages"
        );

        Ok(ModelMapping::new(
            ptr.cast(),
            len,
            path.to_path_buf(),
            false,
        ))
    }

    fn probe_hugepages(page_size: u64) -> bool {
        let size_label = if page_size == HUGEPAGE_SIZE_2M {
            "hugepages-2048kB"
        } else {
            "hugepages-1048576kB"
        };

        let nr_path = format!("/sys/kernel/mm/hugepages/{size_label}/nr_hugepages");
        let free_path = format!("/sys/kernel/mm/hugepages/{size_label}/free_hugepages");

        let nr: u64 = std::fs::read_to_string(&nr_path)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        let free: u64 = std::fs::read_to_string(&free_path)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        nr > 0 && free > 0
    }

    fn read_hugepage_stats(&self) -> Result<HugepageStats> {
        let size_label = if self.page_size == HUGEPAGE_SIZE_2M {
            "hugepages-2048kB"
        } else {
            "hugepages-1048576kB"
        };

        let base = PathBuf::from("/sys/kernel/mm/hugepages").join(size_label);

        let total = Self::read_sysfs_u64(&base.join("nr_hugepages"))?;
        let free = Self::read_sysfs_u64(&base.join("free_hugepages"))?;

        Ok(HugepageStats {
            total_pages: total,
            free_pages: free,
            page_size_bytes: self.page_size,
            total_bytes: total * self.page_size,
            used_bytes: (total - free) * self.page_size,
        })
    }

    fn read_sysfs_u64(path: &Path) -> Result<u64> {
        let content = std::fs::read_to_string(path)?;
        content
            .trim()
            .parse::<u64>()
            .map_err(|e| SpiritMemoryError::CgroupsError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_hugepage_manager_disabled() {
        let mgr = HugepageManager::new(HUGEPAGE_SIZE_2M, false);
        assert!(!mgr.is_available());
    }

    #[test]
    fn test_hugepage_stats_default_when_unavailable() {
        let mgr = HugepageManager::new(HUGEPAGE_SIZE_2M, false);
        let stats = mgr.stats();
        assert_eq!(stats.total_pages, 0);
        assert_eq!(stats.free_pages, 0);
    }

    #[test]
    fn test_mmap_standard_fallback() {
        // Create a temp file to map
        let dir = std::env::temp_dir().join(format!("spirit-hp-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("test_model.bin");

        // Write some test data
        let data = vec![0xABu8; 4096];
        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(&data).unwrap();
        drop(file);

        // Map with hugepages disabled → should use standard fallback
        let mgr = HugepageManager::new(HUGEPAGE_SIZE_2M, false);
        let mapping = mgr.mmap_model_file(&file_path).unwrap();

        assert_eq!(mapping.len(), 4096);
        assert!(!mapping.is_hugepage_backed());
        assert_eq!(mapping.as_slice()[0], 0xAB);
        assert_eq!(mapping.as_slice()[4095], 0xAB);

        // Cleanup
        drop(mapping);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_mmap_empty_file() {
        let dir = std::env::temp_dir().join(format!("spirit-hp-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("empty.bin");
        std::fs::File::create(&file_path).unwrap();

        let mgr = HugepageManager::new(HUGEPAGE_SIZE_2M, false);
        let mapping = mgr.mmap_model_file(&file_path).unwrap();
        assert!(mapping.is_empty());
        assert_eq!(mapping.len(), 0);

        drop(mapping);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_mmap_nonexistent_file() {
        let mgr = HugepageManager::new(HUGEPAGE_SIZE_2M, false);
        let result = mgr.mmap_model_file(Path::new("/nonexistent/model.bin"));
        assert!(result.is_err());
    }

    #[test]
    fn test_model_mapping_is_readonly() {
        let dir = std::env::temp_dir().join(format!("spirit-hp-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("readonly.bin");
        let data = vec![42u8; 8192];
        std::fs::write(&file_path, &data).unwrap();

        let mgr = HugepageManager::new(HUGEPAGE_SIZE_2M, false);
        let mapping = mgr.mmap_model_file(&file_path).unwrap();

        // Can read
        assert_eq!(mapping.as_slice().len(), 8192);
        assert_eq!(mapping.path(), file_path);

        drop(mapping);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
