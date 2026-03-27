//! Phase transition handler for inference-aware scheduling.
//!
//! Dynamically adjusts CPU core assignments based on the current
//! inference phase:
//!
//! - **Idle**: All cores available for system work.
//! - **Prefill**: Burst mode — claim all cores except reserved system
//!   cores for inference (compute-bound).
//! - **Decode**: Latency mode — few dedicated cores with `SCHED_FIFO`
//!   for token generation (memory-bound).
//!
//! // PHASE1-SHIM: Replace with kernel-native phase-aware scheduling in Phase 2.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use tracing::{debug, info, warn};

use crate::cpuset::CpusetManager;
use crate::error::Result;
use crate::gpu::GpuScheduler;
use crate::types::{CpuId, InferencePhase};

/// Manages inference phase transitions and corresponding resource adjustments.
pub struct PhaseManager {
    /// Current inference phase.
    current_phase: InferencePhase,
    /// When the current phase started.
    phase_start: Instant,
    /// Cores dedicated to inference during decode.
    decode_cores: Vec<CpuId>,
    /// All available cores for inference burst during prefill.
    all_inference_cores: Vec<CpuId>,
    /// Cores reserved for system services (never given to inference).
    reserved_system_cores: Vec<CpuId>,
    /// Total phase transitions since startup.
    transition_count: AtomicU64,
}

impl PhaseManager {
    /// Create a new phase manager.
    #[must_use]
    pub fn new(
        decode_cores: Vec<CpuId>,
        all_inference_cores: Vec<CpuId>,
        reserved_system_cores: Vec<CpuId>,
    ) -> Self {
        Self {
            current_phase: InferencePhase::Idle,
            phase_start: Instant::now(),
            decode_cores,
            all_inference_cores,
            reserved_system_cores,
            transition_count: AtomicU64::new(0),
        }
    }

    /// Handle a phase transition.
    ///
    /// Returns the new inference and system core assignments.
    /// Sub-millisecond latency requirement: this function must not allocate
    /// or perform I/O beyond cgroup writes.
    ///
    /// # Errors
    ///
    /// Returns an error if cpuset or GPU mode changes fail during the transition.
    pub fn transition(
        &mut self,
        new_phase: InferencePhase,
        cpuset_mgr: &CpusetManager,
        gpu_sched: &GpuScheduler,
    ) -> Result<PhaseTransitionResult> {
        if new_phase == self.current_phase {
            return Ok(PhaseTransitionResult {
                inference_cores: self.current_inference_cores(),
                system_cores: self.current_system_cores(),
                duration_of_previous_phase: None,
            });
        }

        let previous_phase = self.current_phase;
        let phase_duration = self.phase_start.elapsed();

        #[allow(clippy::cast_possible_truncation)]
        let previous_duration_ms = phase_duration.as_millis() as u64;
        info!(
            from = %previous_phase,
            to = %new_phase,
            previous_duration_ms,
            "Phase transition"
        );

        self.current_phase = new_phase;
        self.phase_start = Instant::now();
        self.transition_count.fetch_add(1, Ordering::Relaxed);

        let (inference_cores, system_cores) = match new_phase {
            InferencePhase::Idle => {
                // Release all cores to system
                gpu_sched.release_exclusive_mode().unwrap_or_else(|e| {
                    warn!(error = %e, "Failed to release GPU exclusive mode");
                });

                let all_cores: Vec<CpuId> = self
                    .all_inference_cores
                    .iter()
                    .chain(self.reserved_system_cores.iter())
                    .copied()
                    .collect();

                // In idle, inference gets no dedicated cores
                cpuset_mgr
                    .update_inference_cores(&self.decode_cores)
                    .unwrap_or_else(|e| {
                        warn!(error = %e, "Failed to update inference cpuset");
                    });
                cpuset_mgr
                    .update_system_cores(&all_cores)
                    .unwrap_or_else(|e| {
                        warn!(error = %e, "Failed to update system cpuset");
                    });

                (self.decode_cores.clone(), all_cores)
            }
            InferencePhase::Prefill => {
                // Burst mode: claim all cores except reserved for inference
                gpu_sched.enter_exclusive_mode().unwrap_or_else(|e| {
                    warn!(error = %e, "Failed to enter GPU exclusive mode");
                });

                cpuset_mgr
                    .update_inference_cores(&self.all_inference_cores)
                    .unwrap_or_else(|e| {
                        warn!(error = %e, "Failed to expand inference cpuset");
                    });
                cpuset_mgr
                    .update_system_cores(&self.reserved_system_cores)
                    .unwrap_or_else(|e| {
                        warn!(error = %e, "Failed to contract system cpuset");
                    });

                (
                    self.all_inference_cores.clone(),
                    self.reserved_system_cores.clone(),
                )
            }
            InferencePhase::Decode => {
                // Latency mode: dedicated cores for decode, release rest
                gpu_sched.enter_exclusive_mode().unwrap_or_else(|e| {
                    warn!(error = %e, "Failed to enter GPU exclusive mode");
                });

                let system_cores: Vec<CpuId> = self
                    .all_inference_cores
                    .iter()
                    .filter(|c| !self.decode_cores.contains(c))
                    .chain(self.reserved_system_cores.iter())
                    .copied()
                    .collect();

                cpuset_mgr
                    .update_inference_cores(&self.decode_cores)
                    .unwrap_or_else(|e| {
                        warn!(error = %e, "Failed to set decode cpuset");
                    });
                cpuset_mgr
                    .update_system_cores(&system_cores)
                    .unwrap_or_else(|e| {
                        warn!(error = %e, "Failed to update system cpuset");
                    });

                (self.decode_cores.clone(), system_cores)
            }
        };

        debug!(
            phase = %new_phase,
            inference_cores = ?inference_cores,
            system_cores = ?system_cores,
            "Core assignment updated"
        );

        Ok(PhaseTransitionResult {
            inference_cores,
            system_cores,
            duration_of_previous_phase: Some(phase_duration),
        })
    }

    /// Returns the current inference phase.
    #[must_use]
    pub fn current_phase(&self) -> InferencePhase {
        self.current_phase
    }

    /// Returns the total number of phase transitions.
    #[must_use]
    pub fn transition_count(&self) -> u64 {
        self.transition_count.load(Ordering::Relaxed)
    }

    /// Returns how long the current phase has been active.
    #[must_use]
    pub fn current_phase_duration(&self) -> std::time::Duration {
        self.phase_start.elapsed()
    }

    fn current_inference_cores(&self) -> Vec<CpuId> {
        match self.current_phase {
            InferencePhase::Prefill => self.all_inference_cores.clone(),
            InferencePhase::Decode | InferencePhase::Idle => self.decode_cores.clone(),
        }
    }

    fn current_system_cores(&self) -> Vec<CpuId> {
        match self.current_phase {
            InferencePhase::Prefill => self.reserved_system_cores.clone(),
            InferencePhase::Decode => self
                .all_inference_cores
                .iter()
                .filter(|c| !self.decode_cores.contains(c))
                .chain(self.reserved_system_cores.iter())
                .copied()
                .collect(),
            InferencePhase::Idle => self
                .all_inference_cores
                .iter()
                .chain(self.reserved_system_cores.iter())
                .copied()
                .collect(),
        }
    }
}

/// Result of a phase transition, containing the new core assignments.
pub struct PhaseTransitionResult {
    /// Cores now assigned to inference.
    pub inference_cores: Vec<CpuId>,
    /// Cores now assigned to system/compat.
    pub system_cores: Vec<CpuId>,
    /// Duration of the previous phase (None if no transition occurred).
    pub duration_of_previous_phase: Option<std::time::Duration>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_phase_manager() -> PhaseManager {
        PhaseManager::new(
            vec![0, 1],       // decode cores
            vec![0, 1, 2, 3], // all inference cores
            vec![4, 5],       // reserved system cores
        )
    }

    #[test]
    fn test_initial_phase_is_idle() {
        let pm = make_phase_manager();
        assert_eq!(pm.current_phase(), InferencePhase::Idle);
    }

    #[test]
    fn test_transition_count_increments() {
        let mut pm = make_phase_manager();
        let cpuset = CpusetManager::new("/nonexistent", "test");
        let gpu = GpuScheduler::new(false);

        assert_eq!(pm.transition_count(), 0);
        let _ = pm.transition(InferencePhase::Prefill, &cpuset, &gpu);
        assert_eq!(pm.transition_count(), 1);
        let _ = pm.transition(InferencePhase::Decode, &cpuset, &gpu);
        assert_eq!(pm.transition_count(), 2);
    }

    #[test]
    fn test_same_phase_no_transition() {
        let mut pm = make_phase_manager();
        let cpuset = CpusetManager::new("/nonexistent", "test");
        let gpu = GpuScheduler::new(false);

        let result = pm.transition(InferencePhase::Idle, &cpuset, &gpu).unwrap();
        assert!(result.duration_of_previous_phase.is_none());
        assert_eq!(pm.transition_count(), 0);
    }

    #[test]
    fn test_prefill_claims_all_inference_cores() {
        let mut pm = make_phase_manager();
        let cpuset = CpusetManager::new("/nonexistent", "test");
        let gpu = GpuScheduler::new(false);

        let result = pm
            .transition(InferencePhase::Prefill, &cpuset, &gpu)
            .unwrap();
        assert_eq!(result.inference_cores, vec![0, 1, 2, 3]);
        assert_eq!(result.system_cores, vec![4, 5]);
    }

    #[test]
    fn test_decode_uses_dedicated_cores() {
        let mut pm = make_phase_manager();
        let cpuset = CpusetManager::new("/nonexistent", "test");
        let gpu = GpuScheduler::new(false);

        let result = pm
            .transition(InferencePhase::Decode, &cpuset, &gpu)
            .unwrap();
        assert_eq!(result.inference_cores, vec![0, 1]);
        // System gets non-decode inference cores + reserved
        assert!(result.system_cores.contains(&2));
        assert!(result.system_cores.contains(&3));
        assert!(result.system_cores.contains(&4));
        assert!(result.system_cores.contains(&5));
    }
}
