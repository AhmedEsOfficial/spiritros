//! Main `SpiritScheduler` — composes all scheduling subsystems.
//!
//! This is the primary entry point for the scheduler. It owns the
//! process table, phase manager, cpuset manager, GPU scheduler,
//! watchdog, and metrics. The scheduler runs as an event-driven
//! daemon that responds to IPC commands and phase hints.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::config::SchedulerConfig;
use crate::cpuset::CpusetManager;
use crate::error::{Result, SchedulerError};
use crate::gpu::GpuScheduler;
use crate::ipc::{IpcCommand, IpcServer};
use crate::metrics::SchedulerMetrics;
use crate::phase::PhaseManager;
use crate::priority;
use crate::process::ProcessTable;
use crate::topology;
use crate::types::{
    CpuId, CpuTopology, InferencePhase, PriorityTier, SchedulerMessage, SchedulerResponse,
    SchedulerStats,
};
use crate::watchdog;

/// The main Spirit Scheduler.
///
/// Composes all scheduling subsystems and runs the event loop
/// that processes IPC commands.
pub struct SpiritScheduler {
    config: SchedulerConfig,
    topology: CpuTopology,
    inference_cores: Vec<CpuId>,
    system_cores: Vec<CpuId>,
    cpuset_mgr: CpusetManager,
    gpu_sched: GpuScheduler,
    phase_mgr: PhaseManager,
    process_table: ProcessTable,
    metrics: Arc<SchedulerMetrics>,
    #[allow(dead_code)]
    realtime_available: bool,
}

impl SpiritScheduler {
    /// Create a new scheduler from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration validation, topology discovery,
    /// or cgroup initialization fails.
    pub fn new(config: SchedulerConfig) -> Result<Self> {
        config.validate()?;

        // Discover system topology
        let topo = topology::discover()?;

        // Determine core partitioning
        let (inference_cores, system_cores) = if let (Some(ref inf), Some(ref sys)) = (
            &config.inference_cores_override,
            &config.system_cores_override,
        ) {
            (inf.clone(), sys.clone())
        } else {
            topology::partition_cores(
                &topo,
                config.inference_core_ratio,
                config.reserved_system_cores,
                config.numa_aware,
            )
        };

        if inference_cores.is_empty() {
            return Err(SchedulerError::ConfigError(
                "No cores available for inference".to_string(),
            ));
        }

        // Check real-time capability
        let realtime_available = config.enable_realtime && priority::check_realtime_capability();

        // Initialize cpuset cgroups
        let cpuset_mgr = CpusetManager::new(&config.cgroup_root, &config.cgroup_name);
        cpuset_mgr.initialize(&inference_cores, &system_cores)?;

        // Initialize GPU scheduling
        let gpu_sched = GpuScheduler::new(config.gpu_exclusive_mode);

        // Select decode cores (first N cores from inference set)
        let decode_core_count = (inference_cores.len() / 3).max(1);
        let decode_cores = inference_cores[..decode_core_count].to_vec();

        // Determine reserved system cores
        let reserved_system_cores = if system_cores.len() >= config.reserved_system_cores {
            system_cores[..config.reserved_system_cores].to_vec()
        } else {
            system_cores.clone()
        };

        // Initialize phase manager
        let phase_mgr =
            PhaseManager::new(decode_cores, inference_cores.clone(), reserved_system_cores);

        // Initialize process table
        let process_table = ProcessTable::new(realtime_available);

        // Initialize metrics
        let metrics = Arc::new(SchedulerMetrics::new());
        metrics.update_core_counts(inference_cores.len(), system_cores.len());
        metrics.set_phase(InferencePhase::Idle);

        info!(
            total_cores = topo.total_cores,
            inference_cores = ?inference_cores,
            system_cores = ?system_cores,
            realtime = realtime_available,
            gpu_exclusive = config.gpu_exclusive_mode,
            "Spirit Scheduler initialized"
        );

        Ok(Self {
            config,
            topology: topo,
            inference_cores,
            system_cores,
            cpuset_mgr,
            gpu_sched,
            phase_mgr,
            process_table,
            metrics,
            realtime_available,
        })
    }

    /// Run the scheduler event loop.
    ///
    /// This is the main entry point for the scheduler daemon. It starts
    /// the IPC server, watchdog, and processes incoming commands.
    ///
    /// # Errors
    ///
    /// Returns an error if the IPC server fails to start or the event
    /// loop encounters a fatal error.
    pub async fn run(&mut self) -> Result<()> {
        let (cmd_tx, mut cmd_rx) = mpsc::channel::<IpcCommand>(256);

        // Start IPC server
        let ipc_server = IpcServer::new(&self.config.socket_path);
        ipc_server.start(cmd_tx).await?;

        // Start watchdog
        let metrics_for_watchdog = self.metrics.clone();
        let watchdog_handle = if self.config.watchdog_enabled {
            let timeout = Duration::from_millis(self.config.watchdog_timeout_ms);
            let on_timeout: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
                warn!("Watchdog triggered — P0 thread starvation detected");
                metrics_for_watchdog.watchdog_triggers_total.inc();
            });
            Some(watchdog::spawn_watchdog(timeout, on_timeout))
        } else {
            None
        };

        // Start periodic maintenance task
        let metrics_for_maintenance = self.metrics.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            loop {
                interval.tick().await;
                // Metrics refresh happens in the main loop
                let _ = &metrics_for_maintenance;
            }
        });

        info!("Spirit Scheduler event loop started");

        // Main event loop
        loop {
            tokio::select! {
                Some(cmd) = cmd_rx.recv() => {
                    let response = self.handle_command(cmd.message);

                    // Send heartbeat on inference activity
                    if let Some(ref handle) = watchdog_handle {
                        handle.heartbeat().await;
                    }

                    // Update metrics
                    self.refresh_metrics();

                    let _ = cmd.reply_tx.send(response);
                }
                else => {
                    info!("Command channel closed, shutting down");
                    break;
                }
            }
        }

        // Cleanup
        self.shutdown();
        if let Some(handle) = watchdog_handle {
            handle.shutdown().await;
        }

        Ok(())
    }

    /// Handle a single scheduler command.
    fn handle_command(&mut self, msg: SchedulerMessage) -> SchedulerResponse {
        match msg {
            SchedulerMessage::PhaseHint {
                phase,
                session_id: _,
            } => match self.hint_phase(phase) {
                Ok(()) => SchedulerResponse::Ok,
                Err(e) => SchedulerResponse::Error(e.to_string()),
            },
            SchedulerMessage::RegisterProcess {
                pid,
                tier,
                label,
                session_id,
            } => match self.register_process(pid, tier, &label, session_id) {
                Ok(()) => SchedulerResponse::Ok,
                Err(e) => SchedulerResponse::Error(e.to_string()),
            },
            SchedulerMessage::SetTier { pid, tier } => match self.set_tier(pid, tier) {
                Ok(()) => SchedulerResponse::Ok,
                Err(e) => SchedulerResponse::Error(e.to_string()),
            },
            SchedulerMessage::UnregisterProcess { pid } => {
                match self.process_table.unregister(pid) {
                    Ok(()) => SchedulerResponse::Ok,
                    Err(e) => SchedulerResponse::Error(e.to_string()),
                }
            }
            SchedulerMessage::GetStats => SchedulerResponse::Stats(self.stats()),
        }
    }

    /// Process a phase hint from the inference coordinator.
    ///
    /// # Errors
    ///
    /// Returns an error if the phase transition fails.
    pub fn hint_phase(&mut self, phase: InferencePhase) -> Result<()> {
        let result = self
            .phase_mgr
            .transition(phase, &self.cpuset_mgr, &self.gpu_sched)?;

        if let Some(duration) = result.duration_of_previous_phase {
            let prev_phase = match phase {
                InferencePhase::Idle => self.phase_mgr.current_phase(),
                _ => InferencePhase::Idle, // approximate
            };
            self.metrics
                .record_phase_duration(prev_phase, duration.as_secs_f64());
            self.metrics.phase_transitions_total.inc();
        }

        self.metrics.set_phase(phase);
        self.metrics
            .update_core_counts(result.inference_cores.len(), result.system_cores.len());
        self.metrics
            .set_gpu_exclusive(self.gpu_sched.is_exclusive());

        Ok(())
    }

    /// Register a process with the scheduler.
    ///
    /// # Errors
    ///
    /// Returns an error if cgroup assignment or priority setting fails.
    pub fn register_process(
        &mut self,
        pid: u32,
        tier: PriorityTier,
        label: &str,
        session_id: Option<String>,
    ) -> Result<()> {
        self.process_table
            .register(pid, tier, label, session_id, &self.config, &self.cpuset_mgr)
    }

    /// Change a process's priority tier.
    ///
    /// # Errors
    ///
    /// Returns an error if the process is not registered or the tier change fails.
    pub fn set_tier(&mut self, pid: u32, tier: PriorityTier) -> Result<()> {
        self.process_table
            .set_tier(pid, tier, &self.config, &self.cpuset_mgr)
    }

    /// Get a snapshot of scheduler statistics.
    #[must_use]
    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            phase: self.phase_mgr.current_phase(),
            inference_core_count: self.inference_cores.len(),
            system_core_count: self.system_cores.len(),
            processes_per_tier: self.process_table.counts_per_tier(),
            phase_transitions: self.phase_mgr.transition_count(),
            priority_adjustments: self.process_table.adjustment_count(),
            gpu_exclusive: self.gpu_sched.is_exclusive(),
        }
    }

    /// Returns a reference to the CPU topology.
    #[must_use]
    pub fn cpu_topology(&self) -> &CpuTopology {
        &self.topology
    }

    /// Returns the inference core assignments.
    #[must_use]
    pub fn inference_cores(&self) -> &[CpuId] {
        &self.inference_cores
    }

    /// Returns the system core assignments.
    #[must_use]
    pub fn system_cores(&self) -> &[CpuId] {
        &self.system_cores
    }

    /// Returns a reference to the metrics.
    #[must_use]
    pub fn metrics(&self) -> &Arc<SchedulerMetrics> {
        &self.metrics
    }

    /// Refresh all metrics from current state.
    fn refresh_metrics(&self) {
        self.metrics
            .update_queue_depths(&self.process_table.counts_per_tier());
        self.metrics
            .set_gpu_exclusive(self.gpu_sched.is_exclusive());
    }

    /// Graceful shutdown.
    fn shutdown(&mut self) {
        info!("Spirit Scheduler shutting down");

        // Release GPU
        self.gpu_sched.cleanup();

        // Clean up cgroups
        if let Err(e) = self.cpuset_mgr.cleanup() {
            warn!(error = %e, "Error during cgroup cleanup");
        }

        // Reap remaining processes
        self.process_table.reap_dead_processes();

        info!("Spirit Scheduler shutdown complete");
    }
}

impl Drop for SpiritScheduler {
    fn drop(&mut self) {
        self.gpu_sched.cleanup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SchedulerConfig {
        SchedulerConfig {
            // Use non-existent cgroup root so tests don't need privileges
            cgroup_root: "/nonexistent".to_string(),
            cgroup_name: "test".to_string(),
            enable_realtime: false,
            gpu_exclusive_mode: false,
            socket_path: format!("/tmp/spirit-sched-test-{}.sock", std::process::id()),
            ..SchedulerConfig::default()
        }
    }

    #[test]
    fn test_scheduler_creation() {
        let config = test_config();
        let scheduler = SpiritScheduler::new(config);
        // May fail on CI without /sys, but should not panic
        if let Ok(sched) = scheduler {
            assert!(!sched.inference_cores().is_empty());
            assert_eq!(sched.stats().phase, InferencePhase::Idle);
        }
    }

    #[test]
    fn test_scheduler_stats_initial() {
        let config = test_config();
        if let Ok(sched) = SpiritScheduler::new(config) {
            let stats = sched.stats();
            assert_eq!(stats.phase, InferencePhase::Idle);
            assert_eq!(stats.phase_transitions, 0);
            assert!(!stats.gpu_exclusive);
        }
    }
}
