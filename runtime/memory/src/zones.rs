//! Zone allocation tracker.
//!
//! Tracks logical allocations within each memory zone, independent of
//! the cgroups enforcement layer. This enables:
//! - Per-zone allocation counting and size tracking
//! - Admission control (reject allocations that would exceed the zone)
//! - Rebalancing between zones at runtime
//!
//! // PHASE1-SHIM: In Phase 2, zone tracking is kernel-native.

use std::collections::HashMap;
use std::sync::Mutex;

use tracing::{debug, info, warn};

use crate::cgroups::CgroupManager;
use crate::config::MemoryConfig;
use crate::error::{Result, SpiritMemoryError};
use crate::types::{AllocationId, IdGenerator, MemoryZone, ZoneStats};

/// Tracks a single allocation.
#[derive(Debug)]
struct Allocation {
    zone: MemoryZone,
    size: u64,
}

/// Per-zone bookkeeping.
#[derive(Debug)]
struct ZoneState {
    limit_bytes: u64,
    used_bytes: u64,
    high_bytes: u64,
    min_bytes: u64,
    allocation_count: u64,
}

/// Tracks allocations across all four memory zones.
pub struct ZoneTracker {
    /// Per-zone state.
    state: Mutex<HashMap<MemoryZone, ZoneState>>,
    /// All tracked allocations: id → allocation.
    allocations: Mutex<HashMap<AllocationId, Allocation>>,
    /// ID generator.
    id_gen: IdGenerator,
}

impl ZoneTracker {
    /// Create a zone tracker from configuration.
    pub fn new(config: &MemoryConfig) -> Self {
        let mut state = HashMap::new();

        state.insert(
            MemoryZone::Model,
            ZoneState {
                limit_bytes: config.model_zone_bytes,
                used_bytes: 0,
                high_bytes: (config.model_zone_bytes as f64 * 0.90) as u64,
                min_bytes: config.model_zone_bytes, // fully protected
                allocation_count: 0,
            },
        );
        state.insert(
            MemoryZone::KvCache,
            ZoneState {
                limit_bytes: config.kv_zone_bytes,
                used_bytes: 0,
                high_bytes: (config.kv_zone_bytes as f64 * 0.90) as u64,
                min_bytes: config.kv_zone_bytes / 2,
                allocation_count: 0,
            },
        );
        state.insert(
            MemoryZone::Tensor,
            ZoneState {
                limit_bytes: config.tensor_zone_bytes,
                used_bytes: 0,
                high_bytes: (config.tensor_zone_bytes as f64 * 0.90) as u64,
                min_bytes: 0,
                allocation_count: 0,
            },
        );
        state.insert(
            MemoryZone::General,
            ZoneState {
                limit_bytes: config.general_zone_bytes,
                used_bytes: 0,
                high_bytes: (config.general_zone_bytes as f64 * 0.90) as u64,
                min_bytes: 0,
                allocation_count: 0,
            },
        );

        Self {
            state: Mutex::new(state),
            allocations: Mutex::new(HashMap::new()),
            id_gen: IdGenerator::new(),
        }
    }

    /// Track a new allocation in the specified zone.
    ///
    /// Returns an `AllocationId` that must be passed to `release` when
    /// the allocation is freed.
    ///
    /// # Errors
    ///
    /// Returns `ZoneExhausted` if the allocation would exceed the zone's
    /// limit.
    pub fn track(&self, zone: MemoryZone, size: u64) -> Result<AllocationId> {
        let mut state = self.state.lock().expect("zone state lock poisoned");
        let zs = state.get_mut(&zone).expect("zone missing from state map");

        if zs.used_bytes + size > zs.limit_bytes {
            return Err(SpiritMemoryError::ZoneExhausted {
                zone,
                requested: size,
                available: zs.limit_bytes.saturating_sub(zs.used_bytes),
            });
        }

        zs.used_bytes += size;
        zs.allocation_count += 1;

        if zs.used_bytes > zs.high_bytes {
            warn!(
                zone = zone.label(),
                used = zs.used_bytes,
                high = zs.high_bytes,
                "zone usage exceeds high threshold"
            );
        }

        let id = AllocationId(self.id_gen.next());
        let mut allocs = self.allocations.lock().expect("allocations lock poisoned");
        allocs.insert(id, Allocation { zone, size });

        debug!(
            zone = zone.label(),
            size,
            id = id.0,
            used = zs.used_bytes,
            "tracked allocation"
        );

        Ok(id)
    }

    /// Release a previously tracked allocation.
    pub fn release(&self, id: AllocationId) -> Result<()> {
        let mut allocs = self.allocations.lock().expect("allocations lock poisoned");
        let alloc = allocs.remove(&id).ok_or_else(|| {
            SpiritMemoryError::CgroupsError(format!("allocation {} not found", id.0))
        })?;

        let mut state = self.state.lock().expect("zone state lock poisoned");
        let zs = state
            .get_mut(&alloc.zone)
            .expect("zone missing from state map");

        zs.used_bytes = zs.used_bytes.saturating_sub(alloc.size);
        zs.allocation_count = zs.allocation_count.saturating_sub(1);

        debug!(
            zone = alloc.zone.label(),
            size = alloc.size,
            id = id.0,
            used = zs.used_bytes,
            "released allocation"
        );

        Ok(())
    }

    /// Check whether `size` bytes can be allocated in `zone`.
    #[must_use]
    pub fn can_allocate(&self, zone: MemoryZone, size: u64) -> bool {
        let state = self.state.lock().expect("zone state lock poisoned");
        state
            .get(&zone)
            .is_some_and(|zs| zs.used_bytes + size <= zs.limit_bytes)
    }

    /// Get statistics for a single zone.
    pub fn zone_stats(&self, zone: MemoryZone) -> ZoneStats {
        let state = self.state.lock().expect("zone state lock poisoned");
        let zs = state.get(&zone).expect("zone missing from state map");
        ZoneStats {
            zone,
            limit_bytes: zs.limit_bytes,
            high_bytes: zs.high_bytes,
            used_bytes: zs.used_bytes,
            min_bytes: zs.min_bytes,
            swap_bytes: 0, // filled in by cgroup reads
            allocation_count: zs.allocation_count,
        }
    }

    /// Get statistics for all zones.
    pub fn all_zone_stats(&self) -> Vec<ZoneStats> {
        MemoryZone::ALL
            .iter()
            .map(|z| self.zone_stats(*z))
            .collect()
    }

    /// Rebalance memory between two zones.
    ///
    /// Moves `amount` bytes of limit from `from` to `to`. The source
    /// zone's limit is decreased and the destination's is increased.
    /// Also updates the cgroups limits if available.
    pub fn rebalance(
        &self,
        from: MemoryZone,
        to: MemoryZone,
        amount: u64,
        cgroups: &CgroupManager,
    ) -> Result<()> {
        if from == to {
            return Ok(());
        }

        let mut state = self.state.lock().expect("zone state lock poisoned");

        let from_state = state.get(&from).expect("zone missing");
        let available = from_state.limit_bytes.saturating_sub(from_state.used_bytes);
        let min_required = from_state.min_bytes;

        // Don't allow rebalance below the minimum guarantee
        if from_state.limit_bytes.saturating_sub(amount) < min_required {
            return Err(SpiritMemoryError::RebalanceDenied {
                from,
                amount,
                min_bytes: min_required,
            });
        }

        // Don't allow rebalance below current usage
        if amount > available {
            return Err(SpiritMemoryError::ZoneExhausted {
                zone: from,
                requested: amount,
                available,
            });
        }

        // Perform the rebalance
        {
            let from_s = state.get_mut(&from).expect("zone missing");
            from_s.limit_bytes -= amount;
            from_s.high_bytes = (from_s.limit_bytes as f64 * 0.90) as u64;
        }
        {
            let to_s = state.get_mut(&to).expect("zone missing");
            to_s.limit_bytes += amount;
            to_s.high_bytes = (to_s.limit_bytes as f64 * 0.90) as u64;
        }

        let from_new = state.get(&from).expect("zone missing").limit_bytes;
        let to_new = state.get(&to).expect("zone missing").limit_bytes;

        drop(state);

        // Update cgroups
        cgroups.set_memory_limit(from, from_new)?;
        cgroups.set_memory_limit(to, to_new)?;

        info!(
            from = from.label(),
            to = to.label(),
            amount,
            from_new_limit = from_new,
            to_new_limit = to_new,
            "rebalanced memory zones"
        );

        Ok(())
    }

    /// Update zone limits (e.g. after auto-sizing).
    pub fn update_limits(&self, config: &MemoryConfig) {
        let mut state = self.state.lock().expect("zone state lock poisoned");

        if let Some(zs) = state.get_mut(&MemoryZone::Model) {
            zs.limit_bytes = config.model_zone_bytes;
            zs.high_bytes = (config.model_zone_bytes as f64 * 0.90) as u64;
            zs.min_bytes = config.model_zone_bytes;
        }
        if let Some(zs) = state.get_mut(&MemoryZone::KvCache) {
            zs.limit_bytes = config.kv_zone_bytes;
            zs.high_bytes = (config.kv_zone_bytes as f64 * 0.90) as u64;
            zs.min_bytes = config.kv_zone_bytes / 2;
        }
        if let Some(zs) = state.get_mut(&MemoryZone::Tensor) {
            zs.limit_bytes = config.tensor_zone_bytes;
            zs.high_bytes = (config.tensor_zone_bytes as f64 * 0.90) as u64;
        }
        if let Some(zs) = state.get_mut(&MemoryZone::General) {
            zs.limit_bytes = config.general_zone_bytes;
            zs.high_bytes = (config.general_zone_bytes as f64 * 0.90) as u64;
        }
    }
}

impl std::fmt::Debug for ZoneTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZoneTracker")
            .field("zones", &self.all_zone_stats())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn test_tracker() -> ZoneTracker {
        ZoneTracker::new(&MemoryConfig::for_test())
    }

    fn fake_cgroups() -> CgroupManager {
        // Use a temp dir so cgroups operations are no-ops
        CgroupManager::new(std::path::Path::new("/tmp/nonexistent-cgroup"))
    }

    #[test]
    fn test_track_allocation() {
        let tracker = test_tracker();
        let id = tracker.track(MemoryZone::Model, 1024).unwrap();
        let stats = tracker.zone_stats(MemoryZone::Model);
        assert_eq!(stats.used_bytes, 1024);
        assert_eq!(stats.allocation_count, 1);

        tracker.release(id).unwrap();
        let stats = tracker.zone_stats(MemoryZone::Model);
        assert_eq!(stats.used_bytes, 0);
        assert_eq!(stats.allocation_count, 0);
    }

    #[test]
    fn test_zone_exhaustion() {
        let tracker = test_tracker();
        let limit = tracker.zone_stats(MemoryZone::Tensor).limit_bytes;
        assert!(tracker.track(MemoryZone::Tensor, limit + 1).is_err());
    }

    #[test]
    fn test_can_allocate() {
        let tracker = test_tracker();
        assert!(tracker.can_allocate(MemoryZone::Model, 1024));
        assert!(!tracker.can_allocate(MemoryZone::Model, u64::MAX));
    }

    #[test]
    fn test_all_zone_stats() {
        let tracker = test_tracker();
        let stats = tracker.all_zone_stats();
        assert_eq!(stats.len(), 4);
    }

    #[test]
    fn test_rebalance() {
        let tracker = test_tracker();
        let cgroups = fake_cgroups();

        let before_tensor = tracker.zone_stats(MemoryZone::Tensor).limit_bytes;
        let before_general = tracker.zone_stats(MemoryZone::General).limit_bytes;

        let amount = 512 * crate::types::MIB;
        tracker
            .rebalance(MemoryZone::Tensor, MemoryZone::General, amount, &cgroups)
            .unwrap();

        let after_tensor = tracker.zone_stats(MemoryZone::Tensor).limit_bytes;
        let after_general = tracker.zone_stats(MemoryZone::General).limit_bytes;

        assert_eq!(after_tensor, before_tensor - amount);
        assert_eq!(after_general, before_general + amount);
    }

    #[test]
    fn test_rebalance_below_min_denied() {
        let tracker = test_tracker();
        let cgroups = fake_cgroups();

        // Model zone has min == limit, so any rebalance FROM it is denied
        let result = tracker.rebalance(MemoryZone::Model, MemoryZone::General, 1024, &cgroups);
        assert!(result.is_err());
    }

    #[test]
    fn test_rebalance_same_zone_noop() {
        let tracker = test_tracker();
        let cgroups = fake_cgroups();
        let before = tracker.zone_stats(MemoryZone::Tensor).limit_bytes;
        tracker
            .rebalance(MemoryZone::Tensor, MemoryZone::Tensor, 1024, &cgroups)
            .unwrap();
        let after = tracker.zone_stats(MemoryZone::Tensor).limit_bytes;
        assert_eq!(before, after);
    }

    #[test]
    fn test_multiple_allocations() {
        let tracker = test_tracker();
        let ids: Vec<_> = (0..10)
            .map(|_| tracker.track(MemoryZone::Tensor, 1024).unwrap())
            .collect();

        let stats = tracker.zone_stats(MemoryZone::Tensor);
        assert_eq!(stats.used_bytes, 10 * 1024);
        assert_eq!(stats.allocation_count, 10);

        for id in ids {
            tracker.release(id).unwrap();
        }
        let stats = tracker.zone_stats(MemoryZone::Tensor);
        assert_eq!(stats.used_bytes, 0);
    }

    #[test]
    fn test_release_unknown_id() {
        let tracker = test_tracker();
        assert!(tracker.release(AllocationId(999)).is_err());
    }
}
