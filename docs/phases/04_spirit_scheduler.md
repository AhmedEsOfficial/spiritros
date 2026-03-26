# Phase 04: Spirit Scheduler — Inference-Aware Process Management

**Roadmap Alignment:** Phase 1 — Virtual Proof of Concept, Milestone M1.3
**Estimated Duration:** Month 3
**Dependencies:** Phase 01 (Foundation), Phase 02 (Memory Manager), Phase 03 (Inference Engine)
**Architecture Plan Sections:** §6

---

## Context & Overview

Linux's Completely Fair Scheduler (CFS) treats all processes equally — which is actively harmful for inference workloads. During prefill, the LLM needs burst CPU access; CFS will happily preempt it for a cron job. During decode, the token generation loop needs real-time latency; CFS doesn't prioritize it. GPU kernel launch threads need to run *immediately* when the GPU signals completion — any scheduler delay creates a bubble in the GPU pipeline.

In Phase 1, the Spirit Scheduler is a **userspace scheduling shim** that uses Linux's existing scheduling APIs (cgroups cpuset, `SCHED_FIFO`, `sched_setaffinity`, nice values, CPU pinning) to approximate the five-tier priority system described in the architecture plan. The native kernel scheduler (Phase 12) will implement this natively.

---

## Goals

1. **Five-tier priority queue** — P0 (INFERENCE_CRITICAL) through P4 (IDLE_MAINTENANCE) mapped to Linux scheduling primitives
2. **CPU pinning** — Dedicate specific cores to inference, with remaining cores for system/compat work
3. **Phase-aware scheduling** — React to prefill/decode hints from llama.cpp to adjust resource allocation
4. **GPU scheduling** — Exclusive GPU mode during inference; throttle competing GPU consumers
5. **NUMA awareness** — Pin inference threads to the NUMA node closest to the GPU
6. **Scheduler telemetry** — Priority queue depths, context switch rates, phase durations via Prometheus

---

## Design Decisions

### DD-04.1: Linux Scheduling Primitives for Phase 1
**Decision:** Map Spirit priority tiers to Linux facilities:
| Spirit Tier | Linux Mechanism |
|---|---|
| P0: INFERENCE_CRITICAL | `SCHED_FIFO` priority 99, dedicated CPU cores |
| P1: INFERENCE_BATCH | `SCHED_FIFO` priority 50, large time quanta |
| P2: SYSTEM_SERVICES | `SCHED_OTHER` nice -10 |
| P3: COMPATIBILITY | `SCHED_OTHER` nice 0–5 |
| P4: IDLE_MAINTENANCE | `SCHED_IDLE` or nice 19 |
**Rationale:** These are the strongest scheduling guarantees available in userspace Linux without a custom kernel. `SCHED_FIFO` provides real-time priority for inference-critical threads.

### DD-04.2: CPU Pinning Strategy
**Decision:** At startup, partition available CPUs: N cores for inference (P0/P1), remaining for everything else (P2-P4). Default: 75% inference, 25% system.
**Rationale:** Prevents interference between inference and system workloads. NUMA-aware pinning ensures inference cores are on the same socket as the GPU for minimal memory latency.

### DD-04.3: Phase Transition Handling
**Decision:** When llama.cpp emits `PHASE_PREFILL`, the scheduler temporarily claims all CPU cores for inference (suspending P3/P4 work). On `PHASE_DECODE`, it releases non-essential cores.
**Rationale:** Prefill is compute-bound and benefits from maximum parallelism. Decode is memory-bound and needs fewer cores but with real-time latency. This dynamic adjustment maximizes throughput without permanent starvation.

### DD-04.4: GPU Exclusive Mode
**Decision:** During active inference, set NVIDIA GPU to exclusive-compute mode via `nvidia-smi -c EXCLUSIVE_PROCESS`. Release on idle.
**Rationale:** Prevents GPU context switches from desktop compositing or competing CUDA apps. In Phase 1 (container), there's typically no desktop, but other containers or processes might access the GPU.

### DD-04.5: Scheduler as Event-Driven Daemon
**Decision:** The scheduler runs as a long-lived daemon that receives events (phase hints, process creation, timer ticks) and adjusts Linux scheduling parameters in response.
**Rationale:** Centralized control point ensures consistent scheduling policy. Event-driven design avoids polling overhead.

---

## Components & Tasks

### 1. spirit-scheduler Crate (`runtime/scheduler/`)
```rust
pub struct SpiritScheduler {
    config: SchedulerConfig,
    cpu_topology: CpuTopology,
    inference_cores: Vec<CpuId>,
    system_cores: Vec<CpuId>,
    phase: AtomicInferencePhase,
    processes: ProcessTable,
}

impl SpiritScheduler {
    pub fn new(config: SchedulerConfig) -> Result<Self>;

    // Phase transitions (called by Inference Coordinator via IPC)
    pub fn hint_phase(&self, phase: InferencePhase) -> Result<()>;

    // Process management
    pub fn register_process(&self, pid: Pid, tier: PriorityTier) -> Result<()>;
    pub fn set_tier(&self, pid: Pid, tier: PriorityTier) -> Result<()>;

    // CPU topology
    pub fn cpu_topology(&self) -> &CpuTopology;
    pub fn inference_cores(&self) -> &[CpuId];

    // Stats
    pub fn stats(&self) -> SchedulerStats;
}

pub enum InferencePhase {
    Idle,
    Prefill,  // Burst: claim all cores
    Decode,   // Latency: few dedicated cores
}

pub enum PriorityTier {
    InferenceCritical,  // P0
    InferenceBatch,     // P1
    SystemServices,     // P2
    Compatibility,      // P3
    IdleMaintenance,    // P4
}
```

### 2. CPU Topology Discovery
- Parse `/sys/devices/system/cpu/` for core count, NUMA topology, cache hierarchy
- Detect which NUMA nodes GPUs are attached to via `/sys/bus/pci/devices/*/numa_node`
- Select inference cores on the same NUMA node as the primary GPU
- Handle systems with no NUMA (single-socket) gracefully

### 3. CPU Pinning & Affinity
- Use `sched_setaffinity()` to pin inference threads to inference cores
- Use cgroups `cpuset` controller to confine compat/system processes to system cores
- Dynamic re-pinning on phase transitions:
  - `PREFILL` → all cores to inference cpuset
  - `DECODE` → release non-essential cores back to system cpuset
  - `IDLE` → release all cores to system cpuset

### 4. Real-Time Priority Assignment
- Use `sched_setscheduler(SCHED_FIFO)` for P0/P1 threads
- Requires `CAP_SYS_NICE` capability in the container
- Fallback: `nice` values if real-time scheduling is unavailable
- Safety: watchdog timer to prevent P0 threads from starving the system entirely

### 5. GPU Scheduling
- Set GPU compute mode via `nvidia-smi -c EXCLUSIVE_PROCESS` during inference
- Release to `DEFAULT` mode when inference is idle
- For multi-GPU: manage each GPU's compute mode independently
- CUDA MPS integration (coordinated with Phase 02 memory manager)

### 6. Phase Hint IPC
- Spirit Scheduler listens on `/run/spirit/scheduler.sock` for phase hints
- llama.cpp patch (from Phase 03) sends hints via this socket before each phase
- Protocol: simple binary messages (phase enum + session_id)
- Sub-millisecond latency requirement for hint processing

### 7. Prometheus Metrics
- `spirit_scheduler_phase{phase="idle|prefill|decode"}` — gauge: current phase
- `spirit_scheduler_phase_duration_seconds{phase="..."}` — histogram
- `spirit_scheduler_inference_cores` — gauge: number of cores assigned to inference
- `spirit_scheduler_context_switches_total{tier="..."}` — counter
- `spirit_scheduler_queue_depth{tier="P0|P1|P2|P3|P4"}` — gauge

---

## Acceptance Criteria

- [ ] CPU topology is correctly detected (NUMA nodes, core count, GPU affinity)
- [ ] Inference threads run on dedicated cores (verifiable via `taskset -p <pid>`)
- [ ] P0 threads run with `SCHED_FIFO` priority (verifiable via `chrt -p <pid>`)
- [ ] Phase transition from IDLE → PREFILL reassigns all cores to inference within <1ms
- [ ] Phase transition from PREFILL → DECODE releases non-essential cores within <1ms
- [ ] GPU enters exclusive compute mode during inference (verifiable via `nvidia-smi`)
- [ ] Compatibility layer processes (P3) cannot preempt inference threads
- [ ] System does not become unresponsive (watchdog prevents full starvation)
- [ ] Scheduler metrics visible in Prometheus and Grafana
- [ ] Inter-token latency improves measurably vs. stock Linux CFS (target: 20%+ improvement)
- [ ] Prefill throughput improves measurably when all cores are claimed (target: 15%+ improvement)

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| `SCHED_FIFO` causes system lockup if inference thread hangs | Container becomes unresponsive | Watchdog thread at highest priority monitors for liveness; auto-demote after timeout |
| `CAP_SYS_NICE` not available in container | Can't set real-time priority | Detect at startup; fall back to nice values with warning |
| NUMA pinning wrong (GPU on different socket) | Increased memory latency | Auto-detect GPU NUMA node; validate with benchmark at startup |
| Phase hints arrive too late (IPC latency) | Scheduler reacts after the phase has started | Use shared memory flag instead of socket for phase hints (faster) |
| CPU starvation of system services during prefill | Monitoring/networking disrupted | Always reserve at least 1 core for P2 system services, even during prefill |

---

## Key References

- Architecture Plan §6 (The Spirit Scheduler)
- Architecture Plan §6.1 (Why CFS Fails for Inference)
- Architecture Plan §6.2 (Inference-Aware Scheduling)
- Architecture Plan §6.3 (Phase-Aware Scheduling)
- Architecture Plan §6.4 (GPU Scheduling)
- Architecture Plan §14.2 (Phase 1 performance: CPU pinning, CUDA MPS)
