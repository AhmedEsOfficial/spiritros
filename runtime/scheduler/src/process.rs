//! Process table for tracking registered processes and their tiers.
//!
//! All processes interacting with the Spirit runtime register with
//! the scheduler so it can manage their CPU affinity and priority.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use tracing::{debug, info};

use crate::config::SchedulerConfig;
use crate::cpuset::CpusetManager;
use crate::error::{Result, SchedulerError};
use crate::priority;
use crate::types::{PriorityTier, ProcessEntry};

/// Manages the set of processes tracked by the scheduler.
pub struct ProcessTable {
    /// PID → process entry mapping.
    entries: HashMap<u32, ProcessEntry>,
    /// Total priority adjustments performed.
    adjustment_count: AtomicU64,
    /// Whether real-time scheduling is available.
    realtime_available: bool,
}

impl ProcessTable {
    /// Create a new empty process table.
    #[must_use]
    pub fn new(realtime_available: bool) -> Self {
        Self {
            entries: HashMap::new(),
            adjustment_count: AtomicU64::new(0),
            realtime_available,
        }
    }

    /// Register a process with the scheduler.
    ///
    /// # Errors
    ///
    /// Returns an error if cgroup assignment or priority setting fails.
    pub fn register(
        &mut self,
        pid: u32,
        tier: PriorityTier,
        label: &str,
        session_id: Option<String>,
        config: &SchedulerConfig,
        cpuset_mgr: &CpusetManager,
    ) -> Result<()> {
        let entry = ProcessEntry {
            pid,
            tier,
            session_id,
            registered_at: Instant::now(),
            label: label.to_string(),
        };

        // Set CPU affinity via cgroups
        cpuset_mgr.assign_process(pid, tier)?;

        // Set scheduling priority
        self.apply_priority(pid, tier, config)?;

        info!(pid, tier = %tier, label, "Process registered");
        self.entries.insert(pid, entry);
        Ok(())
    }

    /// Change the priority tier of a registered process.
    ///
    /// # Errors
    ///
    /// Returns `SchedulerError::ProcessNotFound` if the PID is not registered,
    /// or a policy error if the priority cannot be applied.
    pub fn set_tier(
        &mut self,
        pid: u32,
        new_tier: PriorityTier,
        config: &SchedulerConfig,
        cpuset_mgr: &CpusetManager,
    ) -> Result<()> {
        let entry = self
            .entries
            .get(&pid)
            .ok_or(SchedulerError::ProcessNotFound(pid))?;

        let old_tier = entry.tier;
        let label = entry.label.clone();
        if old_tier == new_tier {
            return Ok(());
        }

        cpuset_mgr.assign_process(pid, new_tier)?;
        self.apply_priority(pid, new_tier, config)?;

        // Now mutably borrow to update the tier
        if let Some(entry) = self.entries.get_mut(&pid) {
            entry.tier = new_tier;
        }
        self.adjustment_count.fetch_add(1, Ordering::Relaxed);

        debug!(
            pid,
            from = %old_tier,
            to = %new_tier,
            label = %label,
            "Process tier changed"
        );

        Ok(())
    }

    /// Unregister a process from the scheduler.
    ///
    /// # Errors
    ///
    /// This function currently always succeeds but returns `Result` for
    /// forward compatibility.
    pub fn unregister(&mut self, pid: u32) -> Result<()> {
        if let Some(entry) = self.entries.remove(&pid) {
            info!(pid, label = %entry.label, "Process unregistered");
        } else {
            debug!(
                pid,
                "Unregister called for unknown process (may have exited)"
            );
        }
        Ok(())
    }

    /// Returns the number of registered processes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if there are no registered processes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the number of processes in each tier.
    #[must_use]
    pub fn counts_per_tier(&self) -> [usize; 5] {
        let mut counts = [0usize; 5];
        for entry in self.entries.values() {
            counts[entry.tier as usize] += 1;
        }
        counts
    }

    /// Returns the total number of priority adjustments.
    #[must_use]
    pub fn adjustment_count(&self) -> u64 {
        self.adjustment_count.load(Ordering::Relaxed)
    }

    /// Returns an iterator over all registered entries.
    pub fn iter(&self) -> impl Iterator<Item = (&u32, &ProcessEntry)> {
        self.entries.iter()
    }

    /// Returns all PIDs with the specified tier.
    pub fn pids_with_tier(&self, tier: PriorityTier) -> Vec<u32> {
        self.entries
            .iter()
            .filter(|(_, e)| e.tier == tier)
            .map(|(pid, _)| *pid)
            .collect()
    }

    /// Remove processes that no longer exist (cleanup stale entries).
    pub fn reap_dead_processes(&mut self) {
        let dead: Vec<u32> = self
            .entries
            .keys()
            .copied()
            .filter(|pid| !process_exists(*pid))
            .collect();

        for pid in &dead {
            if let Some(entry) = self.entries.remove(pid) {
                debug!(
                    pid,
                    label = %entry.label,
                    "Reaped dead process from scheduler"
                );
            }
        }

        if !dead.is_empty() {
            info!(count = dead.len(), "Reaped dead processes");
        }
    }

    /// Apply the scheduling priority for a process based on its tier.
    fn apply_priority(&self, pid: u32, tier: PriorityTier, config: &SchedulerConfig) -> Result<()> {
        priority::set_process_priority(
            pid,
            tier,
            config.p0_fifo_priority,
            config.p1_fifo_priority,
            config.p2_nice,
            config.p3_nice,
            config.p4_nice,
            self.realtime_available && config.enable_realtime,
        )
    }
}

/// Check if a process exists via /proc.
fn process_exists(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{pid}")).exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_table_counts() {
        let table = ProcessTable::new(false);
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn test_counts_per_tier_empty() {
        let table = ProcessTable::new(false);
        assert_eq!(table.counts_per_tier(), [0, 0, 0, 0, 0]);
    }
}
