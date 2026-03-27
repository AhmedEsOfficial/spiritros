//! Core type definitions for the Spirit Scheduler.
//!
//! Defines the five-tier priority system, inference phases, CPU topology
//! structures, process tracking entries, and scheduler statistics.

use std::fmt;
use std::time::Instant;

use serde::{Deserialize, Serialize};

// ── Priority Tiers ─────────────────────────────────────────────────────────

/// Five-tier priority system mapped to Linux scheduling primitives.
///
/// | Tier | Linux Mechanism |
/// |------|----------------|
/// | P0: `InferenceCritical` | `SCHED_FIFO` priority 99, dedicated CPU cores |
/// | P1: `InferenceBatch`    | `SCHED_FIFO` priority 50, large time quanta |
/// | P2: `SystemServices`    | `SCHED_OTHER` nice -10 |
/// | P3: `Compatibility`     | `SCHED_OTHER` nice 0–5 |
/// | P4: `IdleMaintenance`   | `SCHED_IDLE` or nice 19 |
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum PriorityTier {
    /// P0: GPU kernel launch threads, decode loops. `SCHED_FIFO` 99.
    InferenceCritical = 0,
    /// P1: Prefill computation, batching. `SCHED_FIFO` 50.
    InferenceBatch = 1,
    /// P2: Shell, `SpiritFS`, networking, monitoring. `SCHED_OTHER` nice -10.
    SystemServices = 2,
    /// P3: Legacy Linux/Windows/Android apps. `SCHED_OTHER` nice 0-5.
    Compatibility = 3,
    /// P4: GC, log rotation, model prefetching. `SCHED_IDLE` or nice 19.
    IdleMaintenance = 4,
}

impl PriorityTier {
    /// Returns all tiers in priority order (highest first).
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::InferenceCritical,
            Self::InferenceBatch,
            Self::SystemServices,
            Self::Compatibility,
            Self::IdleMaintenance,
        ]
    }

    /// Returns the short label for metrics and logging.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::InferenceCritical => "P0",
            Self::InferenceBatch => "P1",
            Self::SystemServices => "P2",
            Self::Compatibility => "P3",
            Self::IdleMaintenance => "P4",
        }
    }

    /// Returns `true` if this tier uses `SCHED_FIFO`.
    #[must_use]
    pub fn is_realtime(&self) -> bool {
        matches!(self, Self::InferenceCritical | Self::InferenceBatch)
    }
}

impl fmt::Display for PriorityTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InferenceCritical => write!(f, "P0:INFERENCE_CRITICAL"),
            Self::InferenceBatch => write!(f, "P1:INFERENCE_BATCH"),
            Self::SystemServices => write!(f, "P2:SYSTEM_SERVICES"),
            Self::Compatibility => write!(f, "P3:COMPATIBILITY"),
            Self::IdleMaintenance => write!(f, "P4:IDLE_MAINTENANCE"),
        }
    }
}

// ── Inference Phase ────────────────────────────────────────────────────────

/// Current phase of the inference pipeline.
///
/// Emitted by the inference coordinator so the scheduler can adjust
/// CPU/GPU resource allocation dynamically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferencePhase {
    /// No inference running. All cores available for general work.
    Idle,
    /// Prompt processing. Compute-bound — claim all cores for inference.
    Prefill,
    /// Token generation. Memory-bound — few dedicated cores with `SCHED_FIFO`.
    Decode,
}

impl fmt::Display for InferencePhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "idle"),
            Self::Prefill => write!(f, "prefill"),
            Self::Decode => write!(f, "decode"),
        }
    }
}

// ── CPU Topology ───────────────────────────────────────────────────────────

/// Unique CPU identifier (Linux CPU index).
pub type CpuId = usize;

/// Information about a single NUMA node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaNode {
    /// NUMA node index.
    pub id: usize,
    /// CPU cores belonging to this node.
    pub cpus: Vec<CpuId>,
}

/// GPU device information relevant to scheduling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    /// GPU index (as seen by nvidia-smi / CUDA).
    pub index: usize,
    /// PCI bus address (e.g., "0000:01:00.0").
    pub pci_address: String,
    /// NUMA node this GPU is attached to (None if unknown).
    pub numa_node: Option<usize>,
}

/// Discovered CPU/GPU topology of the system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuTopology {
    /// Total number of online CPU cores.
    pub total_cores: usize,
    /// NUMA nodes (may be a single node on non-NUMA systems).
    pub numa_nodes: Vec<NumaNode>,
    /// Detected GPU devices.
    pub gpus: Vec<GpuDevice>,
}

impl CpuTopology {
    /// Returns all CPU IDs across all NUMA nodes.
    #[must_use]
    pub fn all_cpus(&self) -> Vec<CpuId> {
        self.numa_nodes
            .iter()
            .flat_map(|n| n.cpus.iter().copied())
            .collect()
    }

    /// Returns CPUs on the same NUMA node as the primary GPU.
    /// Falls back to all CPUs if no GPU or NUMA info is available.
    #[must_use]
    pub fn gpu_local_cpus(&self) -> Vec<CpuId> {
        if let Some(gpu) = self.gpus.first() {
            if let Some(numa_id) = gpu.numa_node {
                if let Some(node) = self.numa_nodes.iter().find(|n| n.id == numa_id) {
                    return node.cpus.clone();
                }
            }
        }
        self.all_cpus()
    }
}

// ── Process Tracking ───────────────────────────────────────────────────────

/// A process or thread registered with the scheduler.
#[derive(Debug, Clone)]
pub struct ProcessEntry {
    /// Linux process/thread ID.
    pub pid: u32,
    /// Assigned priority tier.
    pub tier: PriorityTier,
    /// Optional session ID (for inference processes).
    pub session_id: Option<String>,
    /// When this process was registered.
    pub registered_at: Instant,
    /// Human-readable label for logging.
    pub label: String,
}

// ── Scheduler Statistics ───────────────────────────────────────────────────

/// Snapshot of scheduler state for monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerStats {
    /// Current inference phase.
    pub phase: InferencePhase,
    /// Number of cores assigned to inference.
    pub inference_core_count: usize,
    /// Number of cores assigned to system/compat.
    pub system_core_count: usize,
    /// Number of registered processes per tier.
    pub processes_per_tier: [usize; 5],
    /// Total phase transitions since startup.
    pub phase_transitions: u64,
    /// Total context switch adjustments.
    pub priority_adjustments: u64,
    /// Whether GPU is in exclusive mode.
    pub gpu_exclusive: bool,
}

// ── Phase Hint IPC ─────────────────────────────────────────────────────────

/// Message sent over the scheduler IPC socket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulerMessage {
    /// Hint about a phase transition.
    PhaseHint {
        phase: InferencePhase,
        session_id: Option<String>,
    },
    /// Register a process with a priority tier.
    RegisterProcess {
        pid: u32,
        tier: PriorityTier,
        label: String,
        session_id: Option<String>,
    },
    /// Change a process's priority tier.
    SetTier { pid: u32, tier: PriorityTier },
    /// Unregister a process.
    UnregisterProcess { pid: u32 },
    /// Request current scheduler stats.
    GetStats,
}

/// Response from the scheduler daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulerResponse {
    /// Operation completed successfully.
    Ok,
    /// Current scheduler statistics.
    Stats(SchedulerStats),
    /// An error occurred.
    Error(String),
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_tier_ordering() {
        assert!(PriorityTier::InferenceCritical < PriorityTier::InferenceBatch);
        assert!(PriorityTier::InferenceBatch < PriorityTier::SystemServices);
        assert!(PriorityTier::SystemServices < PriorityTier::Compatibility);
        assert!(PriorityTier::Compatibility < PriorityTier::IdleMaintenance);
    }

    #[test]
    fn test_priority_tier_labels() {
        assert_eq!(PriorityTier::InferenceCritical.label(), "P0");
        assert_eq!(PriorityTier::IdleMaintenance.label(), "P4");
    }

    #[test]
    fn test_priority_tier_realtime() {
        assert!(PriorityTier::InferenceCritical.is_realtime());
        assert!(PriorityTier::InferenceBatch.is_realtime());
        assert!(!PriorityTier::SystemServices.is_realtime());
        assert!(!PriorityTier::Compatibility.is_realtime());
        assert!(!PriorityTier::IdleMaintenance.is_realtime());
    }

    #[test]
    fn test_inference_phase_display() {
        assert_eq!(InferencePhase::Idle.to_string(), "idle");
        assert_eq!(InferencePhase::Prefill.to_string(), "prefill");
        assert_eq!(InferencePhase::Decode.to_string(), "decode");
    }

    #[test]
    fn test_cpu_topology_all_cpus() {
        let topo = CpuTopology {
            total_cores: 4,
            numa_nodes: vec![
                NumaNode {
                    id: 0,
                    cpus: vec![0, 1],
                },
                NumaNode {
                    id: 1,
                    cpus: vec![2, 3],
                },
            ],
            gpus: vec![],
        };
        assert_eq!(topo.all_cpus(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_cpu_topology_gpu_local_cpus_with_numa() {
        let topo = CpuTopology {
            total_cores: 4,
            numa_nodes: vec![
                NumaNode {
                    id: 0,
                    cpus: vec![0, 1],
                },
                NumaNode {
                    id: 1,
                    cpus: vec![2, 3],
                },
            ],
            gpus: vec![GpuDevice {
                index: 0,
                pci_address: "0000:01:00.0".to_string(),
                numa_node: Some(1),
            }],
        };
        assert_eq!(topo.gpu_local_cpus(), vec![2, 3]);
    }

    #[test]
    fn test_cpu_topology_gpu_local_fallback() {
        let topo = CpuTopology {
            total_cores: 4,
            numa_nodes: vec![NumaNode {
                id: 0,
                cpus: vec![0, 1, 2, 3],
            }],
            gpus: vec![],
        };
        assert_eq!(topo.gpu_local_cpus(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_priority_tier_all_returns_five() {
        assert_eq!(PriorityTier::all().len(), 5);
    }

    #[test]
    fn test_scheduler_message_serialization() {
        let msg = SchedulerMessage::PhaseHint {
            phase: InferencePhase::Decode,
            session_id: Some("test-123".to_string()),
        };
        let json = serde_json::to_string(&msg).expect("serialize");
        let back: SchedulerMessage = serde_json::from_str(&json).expect("deserialize");
        match back {
            SchedulerMessage::PhaseHint { phase, session_id } => {
                assert_eq!(phase, InferencePhase::Decode);
                assert_eq!(session_id.as_deref(), Some("test-123"));
            }
            _ => panic!("wrong variant"),
        }
    }
}
