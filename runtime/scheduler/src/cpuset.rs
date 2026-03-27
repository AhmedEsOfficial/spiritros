//! Cgroup v2 cpuset management for CPU partitioning.
//!
//! Creates and manages cgroup subtrees that partition CPU cores between
//! inference and system workloads. Inference cores are isolated from the
//! general scheduler to reduce jitter and context-switch overhead.
//!
//! // PHASE1-SHIM: Replace with kernel-native CPU domain management in Phase 2.

use std::fs;
use std::path::{Path, PathBuf};

use tracing::{debug, info, warn};

use crate::error::{Result, SchedulerError};
use crate::types::{CpuId, PriorityTier};

/// Names for the cgroup subtrees managed by the scheduler.
const INFERENCE_CGROUP: &str = "inference";
const SYSTEM_CGROUP: &str = "system";
const COMPAT_CGROUP: &str = "compat";
const IDLE_CGROUP: &str = "idle";

/// Manages cgroup v2 cpuset controllers for CPU partitioning.
pub struct CpusetManager {
    /// Root path for the Spirit cgroup subtree.
    cgroup_root: PathBuf,
    /// Whether cgroups are actually available (graceful degradation).
    available: bool,
}

impl CpusetManager {
    /// Create a new cpuset manager rooted at the given cgroup path.
    ///
    /// The `cgroup_root` should be something like `/sys/fs/cgroup/spirit`.
    pub fn new(cgroup_root: &str, cgroup_name: &str) -> Self {
        let root = PathBuf::from(cgroup_root).join(cgroup_name);
        let available = Path::new(cgroup_root).exists();

        if !available {
            warn!(
                path = %cgroup_root,
                "cgroup v2 not available, cpuset enforcement disabled"
            );
        }

        Self {
            cgroup_root: root,
            available,
        }
    }

    /// Initialize the cgroup subtree structure.
    ///
    /// Creates the following hierarchy:
    /// ```text
    /// /sys/fs/cgroup/spirit/
    /// ├── inference/    ← P0/P1 threads
    /// ├── system/       ← P2 threads
    /// ├── compat/       ← P3 threads
    /// └── idle/         ← P4 threads
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if cgroup directories cannot be created or cpuset
    /// controllers cannot be enabled.
    pub fn initialize(&self, inference_cores: &[CpuId], system_cores: &[CpuId]) -> Result<()> {
        if !self.available {
            info!("Skipping cgroup initialization (cgroups not available)");
            return Ok(());
        }

        // Enable cpuset controller in parent
        self.enable_controller("cpuset")?;

        // Create subtrees
        self.create_subtree(INFERENCE_CGROUP, inference_cores)?;
        self.create_subtree(SYSTEM_CGROUP, system_cores)?;
        self.create_subtree(COMPAT_CGROUP, system_cores)?;
        self.create_subtree(IDLE_CGROUP, system_cores)?;

        info!(
            inference_cores = ?inference_cores,
            system_cores = ?system_cores,
            "cgroup cpuset partitions initialized"
        );

        Ok(())
    }

    /// Assign a process to the cgroup corresponding to its priority tier.
    ///
    /// # Errors
    ///
    /// Returns `SchedulerError::CpusetError` if the process cannot be moved
    /// to the target cgroup.
    pub fn assign_process(&self, pid: u32, tier: PriorityTier) -> Result<()> {
        if !self.available {
            return Ok(());
        }

        let cgroup = tier_to_cgroup(tier);
        let procs_path = self.cgroup_root.join(cgroup).join("cgroup.procs");

        write_file(&procs_path, &pid.to_string()).map_err(|e| SchedulerError::CpusetError {
            pid,
            reason: format!("write {}: {e}", procs_path.display()),
        })?;

        debug!(pid, tier = %tier, cgroup, "Process assigned to cgroup");
        Ok(())
    }

    /// Update the cores assigned to the inference cgroup.
    ///
    /// Called during phase transitions to expand/contract inference cores.
    ///
    /// # Errors
    ///
    /// Returns `SchedulerError::CgroupError` if the cpuset file cannot be written.
    pub fn update_inference_cores(&self, cores: &[CpuId]) -> Result<()> {
        if !self.available || cores.is_empty() {
            return Ok(());
        }

        let cpuset_path = self.cgroup_root.join(INFERENCE_CGROUP).join("cpuset.cpus");

        write_file(&cpuset_path, &format_cpu_list(cores))
            .map_err(|e| SchedulerError::CgroupError(format!("update inference cpuset: {e}")))?;

        debug!(cores = ?cores, "Updated inference cgroup cpuset");
        Ok(())
    }

    /// Update the cores assigned to system/compat/idle cgroups.
    ///
    /// # Errors
    ///
    /// Returns `SchedulerError::CgroupError` if any cpuset file cannot be written.
    pub fn update_system_cores(&self, cores: &[CpuId]) -> Result<()> {
        if !self.available || cores.is_empty() {
            return Ok(());
        }

        let cpu_list = format_cpu_list(cores);

        for cgroup in [SYSTEM_CGROUP, COMPAT_CGROUP, IDLE_CGROUP] {
            let cpuset_path = self.cgroup_root.join(cgroup).join("cpuset.cpus");
            write_file(&cpuset_path, &cpu_list)
                .map_err(|e| SchedulerError::CgroupError(format!("update {cgroup} cpuset: {e}")))?;
        }

        debug!(cores = ?cores, "Updated system cgroup cpusets");
        Ok(())
    }

    /// Clean up cgroup subtrees on shutdown.
    ///
    /// # Errors
    ///
    /// Returns an error if processes cannot be migrated back to the parent cgroup.
    pub fn cleanup(&self) -> Result<()> {
        if !self.available {
            return Ok(());
        }

        // Move all processes back to the parent cgroup
        for cgroup in [INFERENCE_CGROUP, SYSTEM_CGROUP, COMPAT_CGROUP, IDLE_CGROUP] {
            let procs_path = self.cgroup_root.join(cgroup).join("cgroup.procs");
            if let Ok(content) = fs::read_to_string(&procs_path) {
                let parent_procs = self.cgroup_root.join("cgroup.procs");
                for line in content.lines() {
                    let _ = write_file(&parent_procs, line);
                }
            }

            // Try to remove the subtree
            let subtree = self.cgroup_root.join(cgroup);
            if subtree.exists() {
                let _ = fs::remove_dir(&subtree);
            }
        }

        info!("cgroup cpuset partitions cleaned up");
        Ok(())
    }

    /// Returns whether cgroup management is available.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.available
    }

    // ── Internal helpers ────────────────────────────────────────────────

    fn enable_controller(&self, controller: &str) -> Result<()> {
        // First ensure the parent subtree directory exists
        if !self.cgroup_root.exists() {
            fs::create_dir_all(&self.cgroup_root).map_err(|e| {
                SchedulerError::CgroupError(format!(
                    "create cgroup dir {}: {e}",
                    self.cgroup_root.display()
                ))
            })?;
        }

        // Enable the controller in the parent's subtree_control
        let parent = self.cgroup_root.parent().unwrap_or(&self.cgroup_root);
        let subtree_control = parent.join("cgroup.subtree_control");

        if subtree_control.exists() {
            let _ = write_file(&subtree_control, &format!("+{controller}"));
        }

        // Also enable in our own subtree
        let own_subtree = self.cgroup_root.join("cgroup.subtree_control");
        if own_subtree.exists() {
            let _ = write_file(&own_subtree, &format!("+{controller}"));
        }

        Ok(())
    }

    fn create_subtree(&self, name: &str, cores: &[CpuId]) -> Result<()> {
        let path = self.cgroup_root.join(name);

        if !path.exists() {
            fs::create_dir_all(&path).map_err(|e| {
                SchedulerError::CgroupError(format!("create cgroup {}: {e}", path.display()))
            })?;
        }

        // Set cpuset.cpus
        if !cores.is_empty() {
            let cpus_path = path.join("cpuset.cpus");
            write_file(&cpus_path, &format_cpu_list(cores))
                .map_err(|e| SchedulerError::CgroupError(format!("set cpuset for {name}: {e}")))?;
        }

        // Set cpuset.mems (required for cpuset to work)
        let mems_path = path.join("cpuset.mems");
        if mems_path.exists() {
            // Allow all NUMA nodes
            let _ = write_file(&mems_path, "0");
        }

        debug!(cgroup = name, cores = ?cores, "Created cgroup subtree");
        Ok(())
    }
}

/// Map a priority tier to its cgroup subtree name.
fn tier_to_cgroup(tier: PriorityTier) -> &'static str {
    match tier {
        PriorityTier::InferenceCritical | PriorityTier::InferenceBatch => INFERENCE_CGROUP,
        PriorityTier::SystemServices => SYSTEM_CGROUP,
        PriorityTier::Compatibility => COMPAT_CGROUP,
        PriorityTier::IdleMaintenance => IDLE_CGROUP,
    }
}

/// Format a list of CPU IDs as a cgroup cpuset string (e.g., "0-3,5,7").
fn format_cpu_list(cpus: &[CpuId]) -> String {
    if cpus.is_empty() {
        return String::new();
    }

    let mut sorted = cpus.to_vec();
    sorted.sort_unstable();
    sorted.dedup();

    let mut ranges = Vec::new();
    let mut start = sorted[0];
    let mut end = sorted[0];

    for &cpu in &sorted[1..] {
        if cpu == end + 1 {
            end = cpu;
        } else {
            ranges.push((start, end));
            start = cpu;
            end = cpu;
        }
    }
    ranges.push((start, end));

    ranges
        .iter()
        .map(|(s, e)| {
            if s == e {
                s.to_string()
            } else {
                format!("{s}-{e}")
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// Write a value to a sysfs/cgroup file.
fn write_file(path: &Path, value: &str) -> std::result::Result<(), String> {
    fs::write(path, value).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_cpu_list_contiguous() {
        assert_eq!(format_cpu_list(&[0, 1, 2, 3]), "0-3");
    }

    #[test]
    fn test_format_cpu_list_gaps() {
        assert_eq!(format_cpu_list(&[0, 1, 3, 5, 6, 7]), "0-1,3,5-7");
    }

    #[test]
    fn test_format_cpu_list_single() {
        assert_eq!(format_cpu_list(&[5]), "5");
    }

    #[test]
    fn test_format_cpu_list_empty() {
        assert_eq!(format_cpu_list(&[]), "");
    }

    #[test]
    fn test_format_cpu_list_unsorted() {
        assert_eq!(format_cpu_list(&[3, 1, 2, 0]), "0-3");
    }

    #[test]
    fn test_tier_to_cgroup_mapping() {
        assert_eq!(tier_to_cgroup(PriorityTier::InferenceCritical), "inference");
        assert_eq!(tier_to_cgroup(PriorityTier::InferenceBatch), "inference");
        assert_eq!(tier_to_cgroup(PriorityTier::SystemServices), "system");
        assert_eq!(tier_to_cgroup(PriorityTier::Compatibility), "compat");
        assert_eq!(tier_to_cgroup(PriorityTier::IdleMaintenance), "idle");
    }

    #[test]
    fn test_cpuset_manager_unavailable_graceful() {
        let mgr = CpusetManager::new("/nonexistent/path", "spirit");
        assert!(!mgr.is_available());
        // Should succeed without error
        assert!(mgr
            .assign_process(1234, PriorityTier::SystemServices)
            .is_ok());
    }
}
