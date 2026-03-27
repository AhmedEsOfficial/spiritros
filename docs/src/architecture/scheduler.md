# Spirit Scheduler — Inference-Aware Process Scheduling

The Spirit Scheduler is a userspace scheduling shim that uses Linux's existing scheduling APIs to give inference threads deterministic priority over general workloads.

## Why CFS Fails for Inference

Linux's Completely Fair Scheduler (CFS) treats all processes equally — which is actively harmful for inference workloads:

- **Prefill phase**: The LLM needs burst CPU access; CFS will preempt it for a cron job
- **Decode phase**: The token generation loop needs real-time latency; CFS doesn't prioritize it
- **GPU kernel launches**: GPU completion signal threads must run *immediately* — any scheduler delay creates a GPU pipeline bubble

## Five-Tier Priority System

| Tier | Name | Linux Mechanism | Use Case |
|------|------|----------------|----------|
| P0 | `INFERENCE_CRITICAL` | `SCHED_FIFO` priority 99, dedicated cores | GPU kernel launch threads, decode loops |
| P1 | `INFERENCE_BATCH` | `SCHED_FIFO` priority 50 | Prefill computation, batching |
| P2 | `SYSTEM_SERVICES` | `SCHED_OTHER` nice -10 | Shell, SpiritFS, monitoring |
| P3 | `COMPATIBILITY` | `SCHED_OTHER` nice 0-5 | Legacy Linux/Windows/Android apps |
| P4 | `IDLE_MAINTENANCE` | `SCHED_IDLE` or nice 19 | GC, log rotation, model prefetching |

## Phase-Aware Scheduling

The scheduler dynamically adjusts CPU core assignments based on the current inference phase:

### Idle Phase
All cores are available for system work. GPU is in default compute mode.

### Prefill Phase (Burst Mode)
Compute-bound prompt processing. The scheduler:
- Claims **all** inference cores (all except reserved system cores)
- Suspends P3/P4 workloads
- Sets GPU to exclusive compute mode
- Maximizes parallelism for prompt encoding

### Decode Phase (Latency Mode)
Memory-bandwidth-bound token generation. The scheduler:
- Restricts inference to dedicated decode cores with `SCHED_FIFO`
- Releases non-essential cores back to system workloads
- Maintains GPU exclusive mode
- Ensures sub-20ms inter-token latency

## CPU Topology Discovery

At startup, the scheduler:
1. Parses `/sys/devices/system/cpu/online` for total core count
2. Reads NUMA topology from `/sys/devices/system/node/`
3. Detects GPU NUMA affinity via `/sys/bus/pci/devices/*/numa_node`
4. Selects inference cores on the same NUMA node as the primary GPU
5. Partitions cores: 75% inference (default), 25% system

## Core Partitioning via cgroups v2

```text
/sys/fs/cgroup/spirit/
├── inference/    ← P0/P1 threads (dedicated cpuset)
├── system/       ← P2 threads
├── compat/       ← P3 threads
└── idle/         ← P4 threads
```

Each cgroup subtree has its own `cpuset.cpus` limiting which cores processes can use. During phase transitions, the scheduler updates these cpuset files to expand or contract the inference domain.

## GPU Exclusive Mode

During active inference, NVIDIA GPUs are set to `EXCLUSIVE_PROCESS` compute mode via `nvidia-smi`. This prevents GPU context switches from competing CUDA consumers. The GPU is released to `DEFAULT` mode when inference goes idle.

## Watchdog

A watchdog monitors `SCHED_FIFO` threads to prevent system starvation. If a P0 thread exceeds the configured timeout (default: 5s) without yielding, the watchdog triggers a demotion to prevent system lockup.

## IPC Protocol

The scheduler listens on `/run/spirit/scheduler.sock` for length-prefixed JSON messages. The inference coordinator sends phase hints before each inference phase so the scheduler can adjust resources proactively.

## Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `spirit_scheduler_phase` | Gauge | Current phase (0=idle, 1=prefill, 2=decode) |
| `spirit_scheduler_phase_duration_seconds` | Histogram | Duration per phase |
| `spirit_scheduler_inference_cores` | Gauge | Cores assigned to inference |
| `spirit_scheduler_queue_depth{tier}` | Gauge | Processes per priority tier |
| `spirit_scheduler_gpu_exclusive` | Gauge | GPU exclusive mode status |
| `spirit_scheduler_watchdog_triggers_total` | Counter | Watchdog activations |

## Phase 1 vs Phase 2

| Aspect | Phase 1 (Current) | Phase 2 (Kernel) |
|--------|-------------------|------------------|
| Scheduling | cgroups cpuset + `SCHED_FIFO` | Native kernel scheduler class |
| Core isolation | cgroup cpuset files | Hardware CPU domain partitioning |
| Phase hints | Unix socket IPC | Kernel syscall |
| GPU management | `nvidia-smi` CLI | Kernel GPU driver interface |
| Latency | ~0.5ms transitions | ~10μs transitions |
