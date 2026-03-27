//! Prometheus metrics for the Spirit Scheduler.
//!
//! Exposes the following metrics:
//! - `spirit_scheduler_phase` — current inference phase gauge
//! - `spirit_scheduler_phase_duration_seconds` — histogram of phase durations
//! - `spirit_scheduler_inference_cores` — cores assigned to inference
//! - `spirit_scheduler_system_cores` — cores assigned to system
//! - `spirit_scheduler_context_switches_total` — priority adjustment count
//! - `spirit_scheduler_queue_depth` — processes per tier
//! - `spirit_scheduler_phase_transitions_total` — total phase transitions
//! - `spirit_scheduler_gpu_exclusive` — GPU exclusive mode gauge
//! - `spirit_scheduler_watchdog_triggers_total` — watchdog activations

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{exponential_buckets, Histogram};
use prometheus_client::registry::Registry;

/// Label set for phase-scoped metrics.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct PhaseLabel {
    pub phase: String,
}

/// Label set for tier-scoped metrics.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct TierLabel {
    pub tier: String,
}

/// All Prometheus metrics for the scheduler.
pub struct SchedulerMetrics {
    /// The metrics registry.
    pub(crate) registry: Registry,

    // ── Phase metrics ──────────────────────────────────────────────
    /// Current inference phase (0=idle, 1=prefill, 2=decode).
    pub(crate) phase: Gauge,

    /// Duration of each phase in seconds (histogram).
    pub(crate) phase_duration_seconds: Family<PhaseLabel, Histogram>,

    /// Total phase transitions.
    pub(crate) phase_transitions_total: Counter,

    // ── Core assignment metrics ────────────────────────────────────
    /// Number of cores assigned to inference.
    pub(crate) inference_cores: Gauge,

    /// Number of cores assigned to system.
    pub(crate) system_cores: Gauge,

    // ── Process metrics ────────────────────────────────────────────
    /// Number of processes per tier.
    pub(crate) queue_depth: Family<TierLabel, Gauge>,

    /// Total priority adjustments.
    #[allow(dead_code)]
    pub(crate) priority_adjustments_total: Counter,

    // ── GPU metrics ────────────────────────────────────────────────
    /// Whether GPU is in exclusive mode (1 = yes, 0 = no).
    pub(crate) gpu_exclusive: Gauge,

    // ── Watchdog metrics ───────────────────────────────────────────
    /// Total watchdog trigger events.
    pub(crate) watchdog_triggers_total: Counter,
}

impl SchedulerMetrics {
    /// Create and register all scheduler metrics.
    #[must_use]
    pub fn new() -> Self {
        let mut registry = Registry::default();
        let sub = registry.sub_registry_with_prefix("spirit_scheduler");

        let phase = Gauge::default();
        sub.register(
            "phase",
            "Current inference phase (0=idle, 1=prefill, 2=decode)",
            phase.clone(),
        );

        let phase_duration_seconds: Family<PhaseLabel, Histogram> =
            Family::new_with_constructor(|| Histogram::new(exponential_buckets(0.001, 2.0, 16)));
        sub.register(
            "phase_duration_seconds",
            "Duration of inference phases in seconds",
            phase_duration_seconds.clone(),
        );

        let phase_transitions_total = Counter::default();
        sub.register(
            "phase_transitions_total",
            "Total inference phase transitions",
            phase_transitions_total.clone(),
        );

        let inference_cores = Gauge::default();
        sub.register(
            "inference_cores",
            "Number of CPU cores assigned to inference",
            inference_cores.clone(),
        );

        let system_cores = Gauge::default();
        sub.register(
            "system_cores",
            "Number of CPU cores assigned to system/compat",
            system_cores.clone(),
        );

        let queue_depth = Family::default();
        sub.register(
            "queue_depth",
            "Number of registered processes per priority tier",
            queue_depth.clone(),
        );

        let priority_adjustments_total = Counter::default();
        sub.register(
            "priority_adjustments_total",
            "Total scheduling priority adjustments",
            priority_adjustments_total.clone(),
        );

        let gpu_exclusive = Gauge::default();
        sub.register(
            "gpu_exclusive",
            "Whether GPU is in exclusive compute mode (1=yes, 0=no)",
            gpu_exclusive.clone(),
        );

        let watchdog_triggers_total = Counter::default();
        sub.register(
            "watchdog_triggers_total",
            "Total watchdog starvation prevention triggers",
            watchdog_triggers_total.clone(),
        );

        Self {
            registry,
            phase,
            phase_duration_seconds,
            phase_transitions_total,
            inference_cores,
            system_cores,
            queue_depth,
            priority_adjustments_total,
            gpu_exclusive,
            watchdog_triggers_total,
        }
    }

    /// Update the current phase gauge.
    pub fn set_phase(&self, phase: crate::types::InferencePhase) {
        let val = match phase {
            crate::types::InferencePhase::Idle => 0,
            crate::types::InferencePhase::Prefill => 1,
            crate::types::InferencePhase::Decode => 2,
        };
        self.phase.set(val);
    }

    /// Record a completed phase duration.
    pub fn record_phase_duration(&self, phase: crate::types::InferencePhase, duration_secs: f64) {
        let label = PhaseLabel {
            phase: phase.to_string(),
        };
        self.phase_duration_seconds
            .get_or_create(&label)
            .observe(duration_secs);
    }

    /// Update per-tier process counts.
    pub fn update_queue_depths(&self, counts: &[usize; 5]) {
        for tier in crate::types::PriorityTier::all() {
            let label = TierLabel {
                tier: tier.label().to_string(),
            };
            #[allow(clippy::cast_possible_wrap)]
            let count = counts[*tier as usize] as i64;
            self.queue_depth.get_or_create(&label).set(count);
        }
    }

    /// Update core assignment gauges.
    pub fn update_core_counts(&self, inference: usize, system: usize) {
        #[allow(clippy::cast_possible_wrap)]
        let inference_i64 = inference as i64;
        #[allow(clippy::cast_possible_wrap)]
        let system_i64 = system as i64;
        self.inference_cores.set(inference_i64);
        self.system_cores.set(system_i64);
    }

    /// Update GPU exclusive mode gauge.
    pub fn set_gpu_exclusive(&self, exclusive: bool) {
        self.gpu_exclusive.set(i64::from(exclusive));
    }

    /// Encode all metrics as Prometheus text format.
    ///
    /// # Panics
    ///
    /// Panics if the Prometheus text encoder fails, which should not happen
    /// under normal operation.
    #[must_use]
    pub fn encode(&self) -> String {
        let mut buf = String::new();
        prometheus_client::encoding::text::encode(&mut buf, &self.registry)
            .expect("encoding metrics should not fail");
        buf
    }
}

impl Default for SchedulerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SchedulerMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchedulerMetrics")
            .field("registered", &true)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::InferencePhase;

    #[test]
    fn test_metrics_creation() {
        let metrics = SchedulerMetrics::new();
        let encoded = metrics.encode();
        assert!(encoded.contains("spirit_scheduler"));
    }

    #[test]
    fn test_phase_gauge() {
        let metrics = SchedulerMetrics::new();

        metrics.set_phase(InferencePhase::Idle);
        assert_eq!(metrics.phase.get(), 0);

        metrics.set_phase(InferencePhase::Prefill);
        assert_eq!(metrics.phase.get(), 1);

        metrics.set_phase(InferencePhase::Decode);
        assert_eq!(metrics.phase.get(), 2);
    }

    #[test]
    fn test_core_count_gauges() {
        let metrics = SchedulerMetrics::new();
        metrics.update_core_counts(6, 2);
        assert_eq!(metrics.inference_cores.get(), 6);
        assert_eq!(metrics.system_cores.get(), 2);
    }

    #[test]
    fn test_queue_depth_update() {
        let metrics = SchedulerMetrics::new();
        metrics.update_queue_depths(&[2, 1, 3, 0, 5]);

        let encoded = metrics.encode();
        assert!(encoded.contains("spirit_scheduler_queue_depth"));
    }

    #[test]
    fn test_gpu_exclusive_gauge() {
        let metrics = SchedulerMetrics::new();
        metrics.set_gpu_exclusive(true);
        assert_eq!(metrics.gpu_exclusive.get(), 1);
        metrics.set_gpu_exclusive(false);
        assert_eq!(metrics.gpu_exclusive.get(), 0);
    }
}
