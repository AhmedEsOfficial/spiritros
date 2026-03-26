# Architecture Overview

SpiritROS is built on the principle that LLM inference should be a first-class operating system concern, not an application running on top of a general-purpose OS.

## Design Philosophy

Traditional operating systems were designed for a world of many small processes competing for CPU time. LLM inference inverts this model: a single inference workload dominates memory (tens of gigabytes for model weights), requires sustained GPU compute, and is extremely sensitive to latency jitter.

SpiritROS addresses this mismatch at every layer of the OS stack.

## System Architecture

```
+----------------------------------------------------------+
|                      Spirit Shell                         |
|              (LLM-native CLI / Agent interface)           |
+----------------------------------------------------------+
|     SpiritFS      |   Inference    |    Spirit Guard      |
|  (Semantic FUSE)  |  Coordinator   | (Capability Security)|
+----------------------------------------------------------+
|       Memory Zone Manager       |     Scheduler Shim      |
|      (cgroups v2 / hugepages)   |   (cpuset / SCHED_FIFO) |
+----------------------------------------------------------+
|                    Linux Host Kernel                       |
|              (Docker / nvidia-container-toolkit)           |
+----------------------------------------------------------+
|                     GPU Hardware                          |
|                  (CUDA / ROCm / Vulkan)                   |
+----------------------------------------------------------+
```

## Core Subsystems

### TensorFirst Memory Zones

Memory is partitioned into four zones, each with distinct eviction policies:

| Zone | Purpose | Eviction Policy |
|---|---|---|
| `ZONE_MODEL` | Model weights (read-only after load) | LRU by last-access, never OOM-killed |
| `ZONE_KV` | KV-cache pages (COW-capable) | Page-granular eviction, oldest session first |
| `ZONE_COMPUTE` | Scratch buffers, activations | Freed after each forward pass |
| `ZONE_GENERAL` | Everything else (standard processes) | Standard Linux OOM policy |

In Phase 1, these zones are enforced via cgroups v2 memory controllers with hugepage backing for `ZONE_MODEL`.

### Inference-Aware Scheduler

The scheduler ensures that inference decode threads always have CPU priority. In Phase 1, this is implemented via:

- `SCHED_FIFO` policy for inference threads (priority 90)
- Dedicated CPU cores via `cpuset` cgroup controller
- Background tasks deprioritized with `nice +19`

### Inference Coordinator

The central daemon managing all inference operations:

- Model lifecycle (load, unload, hot-swap)
- Session management with KV-cache sharing
- Copy-on-write KV-cache forking for branching conversations
- GPU resource allocation via CUDA MPS
- Token streaming over zero-copy IPC (Unix domain sockets with shared memory)

### SpiritFS (Semantic Filesystem)

A FUSE filesystem that augments standard POSIX operations with semantic capabilities:

- Files are automatically embedded on write (async, non-blocking)
- `search` operations query by meaning, not filename
- Backed by Qdrant for vector storage
- Full POSIX compatibility for standard file operations

### Spirit Guard (Security)

Capability-based security with no ambient authority:

- `CAP_INFERENCE` -- permission to run inference sessions
- `CAP_GPU_COMPUTE` -- permission to access GPU resources
- `CAP_MODEL_LOAD` -- permission to load models into ZONE_MODEL
- `CAP_AGENTIC` -- permission to execute LLM-generated actions
- `CAP_NETWORK` -- permission for network access

Every IPC message between SpiritROS components carries capability tokens.

## Phase Roadmap

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Virtualized deployment (Docker + userspace shims) | **Active** |
| Phase 2 | Custom kernel module for memory zones and scheduling | Planned |
| Phase 3 | Full microkernel with hardware memory zone support | Planned |
| Phase 4 | Multi-node distributed inference | Planned |

For the complete architecture specification, see `SpiritROS_Architecture_Plan.md` in the repository root.
