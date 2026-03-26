# SpiritROS — The First LLM-Native Operating System

**Version:** 0.1.0-alpha · **Codename:** *Ghost in the Machine*
**Classification:** Architecture & Implementation Plan
**Date:** March 2026

---

## Executive Summary

Every operating system in existence was designed around a single assumption: **the CPU is the brain.** SpiritROS rejects that premise. It is the first operating system architected from the ground up where **the LLM is the kernel's first-class citizen** — not an application running on top of an OS designed for spreadsheets and web browsers, but the central intelligence that the entire system is built to serve.

Traditional operating systems waste enormous resources on abstractions the LLM never needs (display servers, desktop metaphors, legacy syscall overhead) while starving it of what it *does* need: contiguous GPU memory, zero-copy tensor pipelines, predictive model caching, and an execution model that treats inference as a **native system primitive** rather than a userspace afterthought.

SpiritROS is not a Linux distro with Ollama bolted on. It is a new kind of operating system — one where the scheduler understands token generation, where memory management speaks in KV-cache pages, where the filesystem is a semantic graph, and where every subsystem is purpose-built to make LLM inference blindingly fast while remaining fully compatible with the software ecosystems developers already depend on.

---

## Table of Contents

1. [Design Philosophy & Principles](#1-design-philosophy--principles)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [SpiritKernel — The Hybrid Microkernel](#3-spiritkernel--the-hybrid-microkernel)
4. [Memory Architecture — TensorFirst Memory](#4-memory-architecture--tensorfirst-memory)
5. [Inference Engine Integration (Ollama / llama.cpp)](#5-inference-engine-integration)
6. [The Spirit Scheduler — Inference-Aware Process Management](#6-the-spirit-scheduler)
7. [SpiritFS — The Semantic Filesystem](#7-spiritfs--the-semantic-filesystem)
8. [Networking & Distributed Inference](#8-networking--distributed-inference)
9. [Compatibility Layers — Universal Software Integration](#9-compatibility-layers--universal-software-integration)
10. [Security Architecture — Zero-Trust Inference](#10-security-architecture--zero-trust-inference)
11. [The Spirit Shell — LLM-Native Interface](#11-the-spirit-shell--llm-native-interface)
12. [Developer Experience & SDK](#12-developer-experience--sdk)
13. [Virtualized Deployment (Phase 1)](#13-virtualized-deployment-phase-1)
14. [Performance Benchmarks & Targets](#14-performance-benchmarks--targets)
15. [Open Source Stack & Dependencies](#15-open-source-stack--dependencies)
16. [Roadmap & Milestones](#16-roadmap--milestones)
17. [Appendices](#17-appendices)

---

## 1. Design Philosophy & Principles

### 1.1 The Core Thesis

> *"An operating system should be shaped by its most important workload. For the next era of computing, that workload is inference."*

Every design decision in SpiritROS flows from five foundational principles:

**Principle 1 — Inference is a Syscall, Not an App.**
Token generation is elevated to a kernel-level primitive. Any process on the system can invoke `spirit_infer()` with the same ease and speed as `read()` or `write()`. The LLM is not a service you call over HTTP — it is woven into the fabric of the OS itself.

**Principle 2 — Memory Belongs to the Model First.**
Traditional OS memory management treats all processes equally. SpiritROS inverts this: the LLM's KV-cache, model weights, and tensor scratch space are **pinned, pre-allocated, and protected**. Everything else — containers, compatibility layers, userspace apps — lives in what's left. The model never swaps. The model never waits.

**Principle 3 — Zero Abstraction Tax.**
Every layer of abstraction between the model and the hardware is a layer of latency. SpiritROS eliminates unnecessary indirection: no display server the model doesn't need, no audio subsystem burning cycles, no legacy syscall translation. The kernel exposes hardware to the inference engine through the thinnest possible interface.

**Principle 4 — Compatible, Not Captive.**
SpiritROS runs Linux binaries, Windows applications (via Wine/Proton), and Android apps (via Waydroid) without modification. It speaks OCI containers natively. It mounts NFS, SMB, and S3. Developers bring their existing tools; SpiritROS doesn't ask them to start over.

**Principle 5 — Security Through Isolation, Not Obscurity.**
Every inference request is sandboxed. Model weights are integrity-verified. Prompt injection is treated as a kernel-level threat vector. The security model is built for a world where the most powerful process on the system can generate arbitrary text.

### 1.2 What SpiritROS Is NOT

- It is **not a Linux distribution**. It uses Linux-compatible interfaces but has its own kernel architecture.
- It is **not an inference server**. It is a complete operating system that happens to be optimized for inference.
- It is **not model-specific**. Any GGUF, safetensors, or ONNX model runs natively. Ollama is the default frontend, not the only path.
- It is **not GPU-only**. It scales from CPU-only inference on a Raspberry Pi to multi-node GPU clusters.

---

## 2. System Architecture Overview

### 2.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SPIRIT SHELL (LLM-Native CLI/API)             │
│                   Natural language + traditional commands           │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────┐ │
│  │  Ollama /     │  │ Compatibility │  │   Spirit Services         │ │
│  │  llama.cpp    │  │ Layers        │  │   (Semantic FS, Net,      │ │
│  │  (Inference)  │  │ (Linux/Win/   │  │    Orchestrator, Cron,    │ │
│  │              │  │  Android)     │  │    Monitoring)            │ │
│  └──────┬───────┘  └──────┬───────┘  └───────────┬───────────────┘ │
│         │                 │                       │                 │
├─────────┴─────────────────┴───────────────────────┴─────────────────┤
│                        SPIRIT RUNTIME                               │
│         IPC · Capability System · Event Bus · Plugin API            │
├─────────────────────────────────────────────────────────────────────┤
│                        SPIRITKERNEL                                 │
│    ┌─────────────┐ ┌──────────────┐ ┌────────────┐ ┌────────────┐  │
│    │ TensorFirst  │ │ Inference    │ │ Device     │ │ Security   │  │
│    │ Memory Mgr   │ │ Scheduler    │ │ Drivers    │ │ Module     │  │
│    └─────────────┘ └──────────────┘ └────────────┘ └────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                      HARDWARE / HYPERVISOR                          │
│         CPU · GPU (CUDA/ROCm/Vulkan) · RAM · NVMe · NIC            │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Subsystems

| Subsystem | Purpose | Key Innovation |
|---|---|---|
| **SpiritKernel** | Hybrid microkernel managing hardware, memory, scheduling | Inference-aware scheduling, tensor memory primitives |
| **TensorFirst Memory** | Memory allocator purpose-built for LLM workloads | KV-cache page management, model weight pinning, zero-copy VRAM mapping |
| **Spirit Scheduler** | Process/thread scheduler | Understands token generation phases, preempts non-inference work during generation |
| **SpiritFS** | Semantic + POSIX-compatible filesystem | Content-addressable storage, embedding-indexed retrieval, model-aware caching |
| **Inference Subsystem** | Ollama / llama.cpp integrated at kernel boundary | Direct GPU memory access, shared KV-cache pools, batched inference multiplexing |
| **Compatibility Engine** | Linux, Windows, Android binary support | Syscall translation, container runtime, Wine/Waydroid integration |
| **Spirit Shell** | Primary user interface | Natural language commands processed by the system's own LLM |
| **Security Module** | Mandatory access control for inference | Prompt sandboxing, output filtering, model integrity verification |

---

## 3. SpiritKernel — The Hybrid Microkernel

### 3.1 Why a New Kernel Architecture

The Linux kernel is a monolithic masterpiece — for general-purpose computing. But its design has specific costs for LLM workloads:

- **The OOM killer doesn't understand KV-cache.** It will happily kill the inference process to save a logging daemon.
- **The CFS scheduler treats inference like any other CPU-bound process.** It doesn't know that a prefill phase needs burst compute while decode is memory-bound.
- **GPU memory is managed through userspace drivers (CUDA runtime)** with no kernel-level coordination between competing GPU consumers.
- **The page cache optimizes for file I/O patterns** that are irrelevant to model weight loading (sequential reads of multi-GB files that should be mmapped and pinned, not cached in pages that can be evicted).

SpiritKernel solves these problems not by patching Linux but by building a kernel where inference is a first-class concept.

### 3.2 Architecture: Hybrid Microkernel

SpiritKernel uses a **hybrid microkernel** design inspired by seL4 (for formal verification properties), Zircon/Fuchsia (for capability-based security), and Linux (for driver compatibility through a translation shim).

```
┌─────────────────────────────────────────────────────────────┐
│                     USERSPACE SERVERS                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ FS Server │ │ Net      │ │ Device   │ │ Inference     │  │
│  │ (SpiritFS)│ │ Server   │ │ Manager  │ │ Coordinator   │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───────┬───────┘  │
│       │            │            │               │           │
├───────┴────────────┴────────────┴───────────────┴───────────┤
│                    MICROKERNEL CORE                          │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  • Capability-based IPC (zero-copy, async)            │  │
│  │  • TensorFirst physical memory manager                │  │
│  │  • Inference-aware thread scheduler                   │  │
│  │  • Minimal interrupt handling & timer                 │  │
│  │  • Hardware abstraction (CPU, GPU doorbell, DMA)      │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  Kernel footprint target: < 50,000 lines of Rust + asm      │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Implementation Language: Rust + Minimal Assembly

The kernel is written in **Rust** using `#![no_std]` with thin assembly stubs for architecture-specific boot, context switching, and interrupt vectors. Rust provides:

- Memory safety without garbage collection overhead.
- Zero-cost abstractions that compile to the same machine code as C.
- `unsafe` blocks that are auditable and minimized.
- A mature ecosystem for OS development (`x86_64`, `aarch64` crates, bootloader frameworks).

Key dependencies for kernel development:

| Crate / Tool | Purpose |
|---|---|
| `x86_64` / `aarch64-cpu` | Architecture-specific register manipulation |
| `bootloader` (Rust) | UEFI/BIOS bootloader |
| `spinning_top` | Spinlock primitives |
| `linked_list_allocator` | Early-boot heap allocation |
| `volatile` | MMIO register access |
| `acpi` / `aml` | Hardware enumeration |

### 3.4 Syscall Interface

SpiritKernel exposes a compact syscall table organized into domains. The critical innovation is the **Inference domain** — syscalls that don't exist in any other OS:

**Core Domain (POSIX-compatible)**
```
spirit_read, spirit_write, spirit_open, spirit_close
spirit_mmap, spirit_munmap, spirit_mprotect
spirit_fork, spirit_exec, spirit_exit, spirit_wait
spirit_socket, spirit_bind, spirit_connect, spirit_send, spirit_recv
```

**Inference Domain (SpiritROS-native)**
```
spirit_infer_create(model_id, config) → session_handle
    Create an inference session with a loaded model.

spirit_infer_prompt(session, tokens[], n_tokens, flags) → request_id
    Submit a prompt for inference. Returns immediately (async).

spirit_infer_recv(request_id, buf, max_tokens, flags) → n_generated
    Receive generated tokens. Supports blocking & streaming modes.

spirit_infer_kv_snapshot(session) → snapshot_handle
    Snapshot the current KV-cache state for branching/rollback.

spirit_infer_kv_restore(session, snapshot_handle)
    Restore a previous KV-cache state (conversation branching).

spirit_infer_batch(requests[], n_requests) → batch_id
    Submit multiple inference requests for batched processing.

spirit_model_load(path, memory_policy) → model_id
    Load model weights with specified memory pinning policy.

spirit_model_unload(model_id)
    Release model weights and associated memory.

spirit_tensor_alloc(size, device, flags) → tensor_handle
    Allocate memory in the tensor memory pool (CPU or GPU).

spirit_tensor_free(tensor_handle)
    Free tensor memory.
```

**Capability Domain**
```
spirit_cap_grant(target, capability, scope)
spirit_cap_revoke(target, capability)
spirit_cap_check(capability) → bool
```

### 3.5 Linux Compatibility Shim

To run unmodified Linux binaries, SpiritKernel includes a **syscall translation layer** that maps Linux syscalls to SpiritKernel equivalents. This is implemented as a userspace server (similar to WSL1's lxcore.sys approach but with a microkernel IPC fast-path).

```
Linux Binary → Linux syscall (int 0x80 / syscall) →
  SpiritKernel traps → routes to Linux Compat Server →
    translates to Spirit syscalls → executes → returns
```

For Phase 1 (virtual deployment), this layer runs atop a real Linux kernel using a custom `LD_PRELOAD` library and seccomp-based syscall interception, allowing SpiritROS userspace to run as a containerized environment while the custom kernel is under development.

---

## 4. Memory Architecture — TensorFirst Memory

### 4.1 The Problem with Conventional Memory Management

When Ollama loads a 70B parameter model:

1. **6.5 GB of model weights** need to be loaded into GPU VRAM (at Q4_K_M quantization).
2. **KV-cache grows dynamically** during generation — up to several GB for long contexts.
3. **Tensor scratch space** is needed for intermediate computations during each layer's forward pass.

A conventional OS treats all of this as generic memory allocations. The page cache may evict model weight pages. The OOM killer has no idea that the 4 GB KV-cache is more valuable than the 200 MB logging buffer. GPU memory is managed entirely in userspace with no kernel coordination.

### 4.2 TensorFirst Memory Zones

SpiritROS partitions physical memory into **zones** with different management policies:

```
┌─────────────────────────────────────────────────────────┐
│                    PHYSICAL MEMORY                       │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  ZONE_MODEL  (pinned, read-only after load)     │    │
│  │  Model weights. Never paged. Shared across      │    │
│  │  inference sessions. Memory-mapped from NVMe.   │    │
│  │  Size: configured at boot or model load time.   │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  ZONE_KV  (dynamic, prioritized, swappable)     │    │
│  │  KV-cache pages. Managed with PagedAttention.   │    │
│  │  Eviction policy: LRU by session recency.       │    │
│  │  Can overflow to NVMe with async prefetch.      │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  ZONE_TENSOR  (scratch space, pooled)           │    │
│  │  Intermediate computation buffers. Recycled      │    │
│  │  between inference steps. Arena-allocated.       │    │
│  │  Never fragmented — compacted after each batch.  │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  ZONE_GENERAL  (standard allocation)            │    │
│  │  Everything else: userspace apps, compat layers, │    │
│  │  system services. Standard page allocation       │    │
│  │  with swap. This zone absorbs memory pressure   │    │
│  │  so inference zones don't have to.              │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### 4.3 GPU VRAM Management

SpiritKernel manages GPU VRAM directly through a **kernel-level VRAM allocator** rather than delegating entirely to the CUDA/ROCm runtime:

- **VRAM Reservation:** At model load time, VRAM is pre-allocated in a single contiguous block for model weights. This eliminates fragmentation from competing CUDA applications.
- **Unified Memory Awareness:** On systems with unified memory (Apple Silicon, AMD APUs), the zone boundaries are advisory — the kernel manages a single pool with priority tiers instead of physical partitions.
- **Multi-GPU Sharding:** For models too large for a single GPU, the kernel coordinates tensor parallelism across GPUs, managing the NVLink/PCIe transfers at the memory management layer rather than in userspace.

### 4.4 KV-Cache as a Kernel-Managed Resource

Inspired by **vLLM's PagedAttention**, SpiritROS manages KV-cache at the kernel level:

```
KV-Cache Page Table (per session):
┌────────┬────────────┬──────────┬───────────────┐
│ Page # │ Phys Addr  │ Layer    │ Status        │
├────────┼────────────┼──────────┼───────────────┤
│ 0      │ 0xGPU_A000 │ Layer 0  │ Active        │
│ 1      │ 0xGPU_A100 │ Layer 0  │ Active        │
│ 2      │ 0xGPU_A200 │ Layer 1  │ Active        │
│ ...    │ ...        │ ...      │ ...           │
│ 847    │ 0xNVM_3000 │ Layer 31 │ Swapped(NVMe) │
│ 848    │ NULL       │ Layer 31 │ Free          │
└────────┴────────────┴──────────┴───────────────┘
```

Benefits:
- **No contiguous allocation required.** KV-cache pages can be scattered and managed like virtual memory pages.
- **Cross-session sharing.** Common prompt prefixes (system prompts) can share KV-cache pages via copy-on-write.
- **Instant forking.** Branching a conversation is a page table copy, not a memory copy.
- **Graceful degradation.** Under memory pressure, cold KV-cache pages swap to NVMe with async prefetch on access.

### 4.5 Model Weight Caching & Prefetching

Model weight files (.gguf) are large (1–140 GB). SpiritROS optimizes loading:

- **mmap with `MAP_POPULATE`** for initial load — the kernel pre-faults all pages, avoiding thousands of page faults during first inference.
- **Weight deduplication.** If two sessions load the same model, weights are shared (read-only mapping). No duplication.
- **Predictive prefetch.** If the scheduler detects a model switch is likely (e.g., a routing pipeline), it begins loading the next model's weights asynchronously.
- **NVMe Direct.** On supported hardware, model weights stream directly from NVMe to GPU VRAM via GPUDirect Storage, bypassing CPU entirely.

---

## 5. Inference Engine Integration

### 5.1 Ollama as the Default Inference Frontend

Ollama provides the user-facing model management experience (pulling models, managing Modelfiles, serving an OpenAI-compatible API). In SpiritROS, Ollama runs as a **system service** with enhanced privileges:

```
┌──────────────────────────────────┐
│           Ollama Service          │
│  ┌────────────────────────────┐  │
│  │  Model Registry & Pull     │  │
│  │  Modelfile Management      │  │
│  │  OpenAI-compatible API     │  │
│  │  Session Management        │  │
│  └──────────┬─────────────────┘  │
│             │                    │
│  ┌──────────┴─────────────────┐  │
│  │  llama.cpp / llama-server  │  │
│  │  (patched for SpiritROS)   │  │
│  │                            │  │
│  │  - Uses spirit_tensor_*    │  │
│  │    syscalls for memory     │  │
│  │  - Uses spirit_infer_*     │  │
│  │    for KV-cache management │  │
│  │  - Direct VRAM access via  │  │
│  │    kernel GPU driver       │  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘
```

### 5.2 Modifications to llama.cpp

SpiritROS maintains a fork of `llama.cpp` with the following kernel-integrated modifications:

1. **Memory allocator replacement.** `ggml_backend_alloc` calls route to `spirit_tensor_alloc()` instead of `malloc`/`cudaMalloc`. This ensures all tensor memory comes from the managed zones.

2. **KV-cache delegation.** Instead of managing its own KV-cache arrays, the patched llama.cpp uses `spirit_infer_kv_*` syscalls. The kernel manages KV-cache pages across sessions.

3. **Scheduler hints.** Before each phase (prefill vs. decode), llama.cpp issues `spirit_sched_hint(PHASE_PREFILL)` or `spirit_sched_hint(PHASE_DECODE)` so the kernel can adjust CPU/GPU scheduling.

4. **Batching integration.** Continuous batching is coordinated with the kernel — the inference coordinator server multiplexes requests across a shared batch, maximizing GPU utilization.

### 5.3 Multi-Backend Support

While Ollama/llama.cpp is the default, SpiritROS's inference syscalls are backend-agnostic. Alternative backends can be integrated:

| Backend | Use Case | Integration Status |
|---|---|---|
| `llama.cpp` (GGUF) | Default, quantized models, CPU+GPU | Primary, kernel-integrated |
| `vLLM` | High-throughput serving, PagedAttention | Supported via Spirit memory API |
| `TensorRT-LLM` | NVIDIA-optimized inference | Plugin, uses Spirit VRAM manager |
| `ExLlamaV2` | GPTQ/EXL2 quantized models | Plugin |
| `candle` (Rust) | Native Rust inference, experimental | Future primary backend |
| `ONNX Runtime` | Cross-platform, model interchange | Supported |
| `MLX` | Apple Silicon native | Supported on ARM/Apple targets |

### 5.4 Model Format & Registry

SpiritROS maintains a local **model registry** backed by a content-addressable store:

```
/spirit/models/
├── registry.db          # SQLite: model metadata, hashes, configs
├── blobs/               # Content-addressed model weight files
│   ├── sha256-a1b2c3... # llama-3.1-8b-q4_k_m.gguf
│   ├── sha256-d4e5f6... # mistral-7b-v0.3-q5_k_m.gguf
│   └── sha256-g7h8i9... # deepseek-r1-32b-q4_k_m.gguf
└── manifests/           # Model configurations (Modelfile equivalents)
    ├── llama3.1-8b.json
    ├── mistral-7b.json
    └── deepseek-r1-32b.json
```

Models are pulled via Ollama's registry protocol, HuggingFace Hub, or direct file import. The registry tracks which models are currently loaded in which memory zones for instant switching.

---

## 6. The Spirit Scheduler

### 6.1 Why CFS Fails for Inference

Linux's Completely Fair Scheduler (CFS) distributes CPU time "fairly" across all runnable tasks. This is actively harmful for inference because:

- **Prefill is CPU-burst-intensive.** The model processes the entire prompt in one shot. CFS will preempt this for a cron job.
- **Decode is latency-sensitive.** Each token must be generated with minimal delay for streaming UX. CFS doesn't prioritize latency for specific threads.
- **GPU kernel launches are bursty.** The CPU thread launching CUDA kernels needs to run *immediately* when the GPU signals completion — any scheduling delay becomes a bubble in the GPU pipeline.

### 6.2 Inference-Aware Scheduling

The Spirit Scheduler uses a **multi-queue priority scheduler** with inference awareness:

```
Priority Queues:
┌──────────────────────────────────────┐
│  P0: INFERENCE_CRITICAL              │  ← GPU kernel launch threads,
│      Preempts everything.            │    token decode loops
│      Real-time FIFO within queue.    │
├──────────────────────────────────────┤
│  P1: INFERENCE_BATCH                 │  ← Prefill computation,
│      High priority, time-sliced.     │    batch processing
│      Larger time quanta (10ms).      │
├──────────────────────────────────────┤
│  P2: SYSTEM_SERVICES                 │  ← Spirit Shell, SpiritFS,
│      Normal priority.                │    networking, monitoring
│      Standard time quanta (4ms).     │
├──────────────────────────────────────┤
│  P3: COMPATIBILITY                   │  ← Linux/Windows/Android
│      Background priority.            │    apps via compat layers
│      Yields to P0-P2.               │
├──────────────────────────────────────┤
│  P4: IDLE_MAINTENANCE                │  ← Garbage collection,
│      Only runs when nothing else     │    log rotation, model
│      needs the CPU.                  │    prefetching
└──────────────────────────────────────┘
```

### 6.3 Phase-Aware Scheduling

The scheduler understands the two-phase nature of autoregressive inference:

**Prefill Phase:**
- All CPU cores assigned to the inference process.
- Compatibility layer processes are suspended (with nanosecond-granularity resume).
- Memory bandwidth is prioritized for weight loading.
- Duration: typically 50–500ms depending on prompt length.

**Decode Phase:**
- One or two CPU cores dedicated to the decode loop (launching GPU kernels, sampling).
- Remaining cores released for other work.
- The decode thread runs at P0 with real-time priority — it must never wait in a scheduler queue.
- Inter-token latency target: < 20ms for 7B models, < 50ms for 70B models.

### 6.4 GPU Scheduling

SpiritKernel manages GPU compute resources directly:

- **Exclusive mode by default.** During inference, the GPU is not shared with other compute tasks (no desktop compositing, no concurrent CUDA apps unless explicitly allowed).
- **Inference stream priority.** GPU compute streams for inference run at maximum priority. Any other GPU work (e.g., a compatibility layer running a CUDA application) is throttled.
- **Cooperative multi-model.** When multiple models need the same GPU, the scheduler uses time-division multiplexing with model-aware context switching (saving/restoring KV-cache state rather than full GPU context).

---

## 7. SpiritFS — The Semantic Filesystem

### 7.1 Design Goals

SpiritFS is a **dual-layer filesystem** that presents a standard POSIX interface (for compatibility) while maintaining a parallel **semantic index** that enables LLM-native file operations.

### 7.2 Architecture

```
┌────────────────────────────────────────────────┐
│              SpiritFS Userspace Server          │
│                                                │
│  ┌──────────────────┐  ┌────────────────────┐  │
│  │  POSIX VFS Layer  │  │  Semantic Layer     │  │
│  │                  │  │                    │  │
│  │  open/read/write │  │  search(query)     │  │
│  │  mkdir/readdir   │  │  similar(file)     │  │
│  │  stat/chmod      │  │  summarize(dir)    │  │
│  │  mount           │  │  tag(file, auto)   │  │
│  └────────┬─────────┘  └────────┬───────────┘  │
│           │                     │              │
│  ┌────────┴─────────────────────┴───────────┐  │
│  │          Storage Backend                  │  │
│  │                                           │  │
│  │  Content-Addressable Store (CAS)          │  │
│  │  ┌─────────┐ ┌──────────┐ ┌───────────┐  │  │
│  │  │ SQLite  │ │ Qdrant/  │ │ Raw Block │  │  │
│  │  │ Metadata│ │ FAISS    │ │ Storage   │  │  │
│  │  │         │ │ Vectors  │ │ (ext4/    │  │  │
│  │  │         │ │          │ │  btrfs)   │  │  │
│  │  └─────────┘ └──────────┘ └───────────┘  │  │
│  └───────────────────────────────────────────┘  │
└────────────────────────────────────────────────┘
```

### 7.3 Semantic Operations

SpiritFS extends the standard filesystem with these operations:

```bash
# Search files by meaning, not just name
$ spirit find "the quarterly revenue report from Q3"
→ /home/user/reports/q3_2025_financials.xlsx (0.94 similarity)
→ /home/user/docs/revenue_summary_q3.md (0.87 similarity)

# Auto-tag files on creation
$ echo "Meeting notes: agreed to delay launch by 2 weeks" > notes.md
→ Auto-tagged: [meeting, launch, delay, planning]

# Summarize directory contents
$ spirit summarize /var/log/
→ "System logs from the past 48 hours showing 3 warning events
   related to disk I/O latency and 1 OOM event in the compat layer."

# Find similar files
$ spirit similar ./architecture.md
→ /old_projects/system_design_v2.md (0.91)
→ /references/distributed_systems_notes.md (0.82)
```

### 7.4 Implementation Details

- **Embedding generation** uses a small, fast model (e.g., `nomic-embed-text` or `all-MiniLM-L6-v2`) running on CPU, scheduled at P4 priority. Files are embedded asynchronously on write.
- **Vector index** is maintained using **FAISS** (for CPU) or **Qdrant** (for GPU-accelerated similarity search).
- **Metadata store** uses **SQLite** with FTS5 for full-text search as a fallback.
- **Storage backend** can be any Linux filesystem (ext4, btrfs, ZFS) — SpiritFS operates as a FUSE layer in Phase 1, with plans for a native kernel filesystem driver in Phase 3.

### 7.5 Model-Aware Caching

SpiritFS includes a **model weight cache** that understands access patterns:

- Recently used model files are kept in memory (ZONE_MODEL).
- Model files are stored with content-addressing — identical quantizations share storage.
- The cache uses a **frequency-weighted LRU** eviction policy: a model used 50 times per day stays cached over one used once per week, even if the latter was accessed more recently.

---

## 8. Networking & Distributed Inference

### 8.1 Network Stack

SpiritROS uses a userspace network stack based on **smoltcp** (Rust) for maximum control, with a fallback to the Linux compat layer's networking for complex protocols:

```
┌─────────────────────────────────────────┐
│          Application Layer               │
│  HTTP/2, gRPC, WebSocket, MQTT           │
├─────────────────────────────────────────┤
│          Spirit Net Server               │
│  TLS 1.3 (rustls), DNS, mDNS            │
├─────────────────────────────────────────┤
│          smoltcp (TCP/IP stack)           │
│  TCP, UDP, ICMP, ARP, DHCP              │
├─────────────────────────────────────────┤
│          NIC Driver (kernel module)      │
│  virtio-net, e1000e, AF_XDP             │
└─────────────────────────────────────────┘
```

### 8.2 Inference API Endpoints

SpiritROS exposes inference over the network via multiple protocols:

| Protocol | Endpoint | Use Case |
|---|---|---|
| OpenAI-compatible REST | `:11434/v1/chat/completions` | Drop-in replacement for any OpenAI client |
| gRPC | `:50051/spirit.Inference` | Low-latency, streaming, typed |
| WebSocket | `:11434/ws` | Real-time streaming for web UIs |
| Unix Domain Socket | `/run/spirit/infer.sock` | Local IPC, zero network overhead |
| Shared Memory IPC | `/dev/shm/spirit_infer` | Fastest path, zero-copy tokens |

### 8.3 Distributed Inference (Multi-Node)

For models too large for a single machine, SpiritROS supports **tensor parallelism across nodes**:

```
Node 0 (Coordinator)              Node 1 (Worker)
┌─────────────────┐              ┌─────────────────┐
│ Layers 0-15     │  ◄─RDMA──►  │ Layers 16-31    │
│ GPU 0           │              │ GPU 0           │
│ KV-cache (L0-15)│              │ KV-cache (L16+) │
└─────────────────┘              └─────────────────┘
         │                                │
    InfiniBand / RoCE v2 / TCP
    (auto-detected, RDMA preferred)
```

Coordination protocol:

- **etcd** for cluster membership and leader election.
- **NCCL** or **Gloo** for GPU-to-GPU tensor transfers.
- **Custom Spirit Wire Protocol** for inference coordination (prompt routing, KV-cache sync, token aggregation).

### 8.4 Federation — Spirit Mesh

Multiple SpiritROS instances can form a **Spirit Mesh** for workload distribution:

- A mesh-aware load balancer routes inference requests to the node with the requested model already loaded (avoiding cold-start latency).
- Models can be "pinned" to specific nodes based on hardware capabilities (e.g., "deepseek-r1-671b runs only on Node 3 with 8xA100").
- The mesh is self-organizing using mDNS for local discovery and WireGuard for secure inter-node communication.

---

## 9. Compatibility Layers — Universal Software Integration

### 9.1 Design Philosophy

SpiritROS does not require developers to rewrite their software. It brings the entire ecosystem of existing applications to the LLM-native platform through a layered compatibility architecture:

```
┌──────────────────────────────────────────────────────┐
│                    SpiritROS Native                    │
│             (Rust binaries, Spirit syscalls)           │
├──────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌────────────┐  ┌────────────────┐  │
│  │  Linux     │  │  Windows   │  │  Android       │  │
│  │  Compat    │  │  Compat    │  │  Compat        │  │
│  │  Layer     │  │  Layer     │  │  Layer         │  │
│  │            │  │            │  │                │  │
│  │  glibc     │  │  Wine 9.x  │  │  Waydroid      │  │
│  │  shim +    │  │  + Proton  │  │  + Android     │  │
│  │  syscall   │  │  + DXVK    │  │    Runtime     │  │
│  │  translate │  │            │  │                │  │
│  └────────────┘  └────────────┘  └────────────────┘  │
├──────────────────────────────────────────────────────┤
│              OCI Container Runtime                    │
│        (Docker / Podman / containerd compatible)      │
└──────────────────────────────────────────────────────┘
```

### 9.2 Linux Compatibility (Primary)

Since the overwhelming majority of LLM tooling runs on Linux, this is the highest-fidelity compatibility layer.

**Implementation:** A userspace **Linux Compat Server** that translates Linux syscalls to Spirit syscalls. For Phase 1, this is implemented as a modified **gVisor** sandbox (in Go/Rust) that intercepts syscalls via ptrace or seccomp and routes them to SpiritKernel equivalents.

**What works out of the box:**
- All CLI tools (bash, python, node, cargo, gcc, git)
- Docker/Podman containers (via containerd integration)
- CUDA/ROCm applications (GPU driver shim)
- systemd services (via a lightweight init shim)
- Network services (nginx, postgres, redis — anything using BSD sockets)

**What requires adaptation:**
- Direct hardware access (bypassing the kernel) — handled by device capability grants
- Kernel modules (eBPF, custom drivers) — not supported in Phase 1; use native Spirit drivers

### 9.3 Windows Compatibility

Based on **Wine 9.x** with **Proton** enhancements from Valve:

- Windows executables (.exe) run via Wine's PE loader.
- DirectX 9/10/11/12 → Vulkan translation via DXVK/VKD3D-Proton.
- .NET applications via Mono/Wine-Mono.
- Win32 API coverage: ~95% of desktop applications.

**Use case:** Running Windows-only ML tools, proprietary data processing software, or enterprise applications alongside the LLM stack.

### 9.4 Android Compatibility

Based on **Waydroid** (container-based Android runtime):

- Android apps run in an LXC container with a full Android stack.
- GPU passthrough for OpenGL ES / Vulkan.
- Android-native ML frameworks (TFLite, NNAPI) work natively.

**Use case:** Running Android-only apps, testing mobile AI applications, or using Android sensor APIs in robotics contexts (the "ROS" in SpiritROS).

### 9.5 Container Runtime

SpiritROS includes a native **OCI-compatible container runtime** based on **youki** (Rust):

```bash
# Pull and run any Docker image
$ spirit container run -it ubuntu:24.04 /bin/bash

# Run with GPU access
$ spirit container run --gpu all nvidia/cuda:12.8-runtime python inference.py

# Compose multi-service stacks
$ spirit compose up -f docker-compose.yml
```

The container runtime integrates with Spirit's memory zones: containers are allocated from ZONE_GENERAL and cannot access ZONE_MODEL or ZONE_KV unless explicitly granted inference capabilities.

### 9.6 Cross-Platform Package Management

SpiritROS includes **spirit-pkg**, a universal package manager that wraps multiple backends:

```bash
$ spirit-pkg install python3          # → apt/apk backend
$ spirit-pkg install nodejs           # → apt/apk backend
$ spirit-pkg install ollama           # → native Spirit package
$ spirit-pkg install notepad++        # → Wine + Windows installer
$ spirit-pkg install termux           # → Waydroid + F-Droid
```

Native Spirit packages use a **Flatpak-like sandboxed format** with declarative capabilities:

```toml
# spirit-package.toml
[package]
name = "my-inference-app"
version = "1.0.0"
runtime = "spirit-runtime-1.0"

[capabilities]
inference = true       # Can call spirit_infer_*
network = "localhost"   # Network access limited to localhost
filesystem = ["/data"]  # Can read/write /data
gpu = true             # Can access GPU via Spirit API
```

---

## 10. Security Architecture — Zero-Trust Inference

### 10.1 Threat Model

SpiritROS faces unique security challenges because its most powerful subsystem (the LLM) can:

- Generate arbitrary text, including code that will be executed.
- Be manipulated via prompt injection to perform unintended actions.
- Leak sensitive data from its context window.
- Be used as an oracle to exfiltrate information.

The security model treats **every inference output as untrusted** until validated.

### 10.2 Capability-Based Security

All access control in SpiritROS is based on **capabilities** (unforgeable tokens that grant specific permissions). There are no ambient authorities — no process can do anything it hasn't been explicitly granted the capability to do.

```
Capability Hierarchy:
┌─────────────────────────────────────────┐
│  CAP_INFER_ADMIN                        │  Full inference control
│  ├── CAP_INFER_LOAD_MODEL              │  Load/unload models
│  ├── CAP_INFER_CREATE_SESSION          │  Create inference sessions
│  │   ├── CAP_INFER_PROMPT             │  Submit prompts
│  │   └── CAP_INFER_KV_MANAGE          │  Snapshot/restore KV-cache
│  └── CAP_INFER_BATCH                   │  Submit batch requests
├─────────────────────────────────────────┤
│  CAP_MEMORY_ADMIN                       │  Full memory control
│  ├── CAP_TENSOR_ALLOC                  │  Allocate tensor memory
│  ├── CAP_VRAM_ACCESS                   │  Direct VRAM access
│  └── CAP_KV_ZONE_ACCESS               │  Access KV-cache zone
├─────────────────────────────────────────┤
│  CAP_GPU_ADMIN                          │  Full GPU control
│  ├── CAP_GPU_COMPUTE                   │  Launch GPU kernels
│  └── CAP_GPU_EXCLUSIVE                 │  Exclusive GPU access
├─────────────────────────────────────────┤
│  CAP_FS_ADMIN                           │  Full filesystem control
│  ├── CAP_FS_READ(path)                 │  Read specific path
│  ├── CAP_FS_WRITE(path)               │  Write specific path
│  └── CAP_FS_SEMANTIC                   │  Use semantic operations
├─────────────────────────────────────────┤
│  CAP_NET_ADMIN                          │  Full network control
│  ├── CAP_NET_LISTEN(port)              │  Bind to port
│  ├── CAP_NET_CONNECT(host)             │  Connect to host
│  └── CAP_NET_MESH                      │  Join Spirit Mesh
└─────────────────────────────────────────┘
```

### 10.3 Inference Sandboxing

Every inference session runs inside a **sandbox** with configurable policies:

```toml
[sandbox.default]
max_output_tokens = 4096
allowed_tools = ["search", "calculator"]
blocked_patterns = ["ssh_key", "password", "API_KEY"]
output_filter = "spirit-guard"          # Output sanitization
context_isolation = true                # No cross-session context leakage
audit_log = "/var/log/spirit/audit.log" # All prompts and outputs logged
```

### 10.4 Model Integrity

- Model weights are verified against **SHA-256 checksums** on load.
- Models can be signed with **Ed25519 signatures** — unsigned models run in a restricted sandbox.
- The model registry supports **Sigstore/cosign** for supply chain verification.
- Model weights in ZONE_MODEL memory are marked **read-only** — no runtime modification is possible.

### 10.5 Spirit Guard — Output Filtering

Spirit Guard is a lightweight, rule-based output filter that runs on every inference output before it reaches the requesting process:

- **Pattern matching** for sensitive data (API keys, credentials, PII).
- **Semantic analysis** using a small classifier model to detect harmful/unexpected outputs.
- **Rate limiting** per session and per user.
- **Audit trail** — every prompt/response pair is logged (configurable retention).

### 10.6 Network Security

- All inter-node communication in Spirit Mesh uses **WireGuard** tunnels.
- The inference API supports **mTLS** for authenticated access.
- The network stack includes a built-in **firewall** with sensible defaults (deny all inbound except inference API ports).

---

## 11. The Spirit Shell — LLM-Native Interface

### 11.1 Overview

The Spirit Shell is SpiritROS's primary interface — a command-line environment where natural language and traditional commands coexist:

```
spirit@ghost ~ $ What models do I have loaded?

Currently loaded models:
  • llama-3.1-8b-instruct (Q4_K_M) — 4.3 GB VRAM, 2 active sessions
  • nomic-embed-text — 274 MB RAM, idle (semantic indexing)

spirit@ghost ~ $ spin up deepseek-r1 and give it 32k context
Loading deepseek-r1-distill-qwen-32b (Q4_K_M)...
  Weights: 18.5 GB → GPU 0 (ZONE_MODEL)
  KV-cache: pre-allocated 2.1 GB for 32768 context (ZONE_KV)
  Ready in 3.2s (cached weights, NVMe direct)

spirit@ghost ~ $ ls -la /var/log/ | explain what happened today
[runs ls -la, pipes output to LLM for analysis]

System logs today show:
  - 14 INFO messages from the inference scheduler (normal operation)
  - 2 WARN messages at 03:14 about disk I/O latency on /dev/nvme0n1
  - 1 ERROR at 07:22: OOM in compat layer (Linux container exceeded ZONE_GENERAL limit)
  Recommendation: increase ZONE_GENERAL allocation or audit the container's memory usage.

spirit@ghost ~ $ sudo spirit mem rebalance --give-general 2G --from tensor-scratch
Rebalanced: ZONE_TENSOR_SCRATCH reduced by 2 GB → ZONE_GENERAL increased by 2 GB
```

### 11.2 Command Resolution

The Spirit Shell resolves input through a priority chain:

1. **Built-in Spirit commands** (`spirit model`, `spirit mem`, `spirit sched`, etc.)
2. **Standard Unix commands** (ls, cd, grep, etc. — via Linux compat layer)
3. **Natural language fallback** → routed to the system's loaded LLM for interpretation

The shell can also operate in **pure LLM mode** (every input is treated as a natural language instruction) or **pure CLI mode** (no LLM interpretation, traditional shell only).

### 11.3 Agentic Capabilities

The Spirit Shell supports **agentic workflows** where the LLM can autonomously execute multi-step tasks:

```
spirit@ghost ~ $ "Monitor the inference API for the next hour.
                  If latency exceeds 100ms/token, reduce batch size.
                  If it exceeds 200ms/token, alert me and switch to the
                  smaller model."

Agent started: inference-monitor-001
  Monitoring: spirit_infer latency
  Actions: batch_resize, model_switch, alert
  Duration: 1 hour
  Approval required for: model_switch

[Agent runs autonomously, logging actions to /var/log/spirit/agents/]
```

Agentic actions are governed by the capability system — the agent can only perform actions it has been granted capabilities for, and high-impact actions require explicit user approval.

---

## 12. Developer Experience & SDK

### 12.1 Spirit SDK

The Spirit SDK provides native bindings for building applications that leverage SpiritROS's inference primitives:

**Rust (primary):**
```rust
use spirit_sdk::{InferenceSession, Model, Config};

fn main() -> spirit::Result<()> {
    // Load model (uses kernel memory management)
    let model = Model::load("llama-3.1-8b-instruct")?;

    // Create session with 8192 context
    let session = InferenceSession::new(&model, Config {
        context_length: 8192,
        temperature: 0.7,
        ..Default::default()
    })?;

    // Stream inference (uses spirit_infer_* syscalls)
    let stream = session.prompt("Explain quantum computing")?;
    for token in stream {
        print!("{}", token?);
    }

    // Snapshot KV-cache for conversation branching
    let snapshot = session.snapshot()?;

    // Branch the conversation
    let branch = session.restore(&snapshot)?;
    let alt_response = branch.prompt("Now explain it to a 5-year-old")?;

    Ok(())
}
```

**Python (via PyO3 bindings):**
```python
import spirit

model = spirit.load_model("llama-3.1-8b-instruct")
session = spirit.InferenceSession(model, context_length=8192)

# Streaming inference
for token in session.prompt("Explain quantum computing"):
    print(token, end="", flush=True)

# KV-cache snapshot
snapshot = session.snapshot()
branch = session.restore(snapshot)
```

**TypeScript/JavaScript (via Neon bindings):**
```typescript
import { Spirit } from '@spirit-os/sdk';

const model = await Spirit.loadModel('llama-3.1-8b-instruct');
const session = model.createSession({ contextLength: 8192 });

for await (const token of session.prompt('Explain quantum computing')) {
  process.stdout.write(token);
}
```

### 12.2 Spirit Plugin System

Third-party extensions are loaded as **Spirit Plugins** — sandboxed WebAssembly modules that can extend the OS:

```toml
# spirit-plugin.toml
[plugin]
name = "rag-retriever"
version = "1.0.0"
entry = "plugin.wasm"

[capabilities.required]
filesystem = ["/data/documents"]
inference = true
network = ["localhost:6333"]  # Qdrant

[hooks]
on_infer_pre = "augment_prompt"    # Called before inference
on_file_write = "index_document"   # Called on file creation
```

Plugins run in **Wasmtime** with capability-limited access. They cannot escape their sandbox, and all interactions with the system go through the capability-checked Spirit Runtime API.

### 12.3 Spirit Compose — Declarative System Configuration

SpiritROS can be configured declaratively via a YAML manifest:

```yaml
# spirit-compose.yaml
version: "1.0"

models:
  primary:
    name: llama-3.1-70b-instruct
    quantization: Q4_K_M
    context: 32768
    memory:
      zone_model: 38G
      zone_kv: 8G
    gpu: [0, 1]  # Tensor parallel across 2 GPUs

  embedding:
    name: nomic-embed-text
    device: cpu
    memory:
      zone_model: 300M

  guard:
    name: llama-guard-3-1b
    device: gpu:0
    memory:
      zone_model: 1G

services:
  api:
    type: inference-api
    model: primary
    port: 11434
    tls: true
    rate_limit: 100/min

  semantic-index:
    type: spiritfs-indexer
    model: embedding
    watch: ["/home", "/data"]

  output-filter:
    type: spirit-guard
    model: guard
    policy: /etc/spirit/guard-policy.yaml

scheduler:
  inference_priority: maximum
  compat_layer_limit: 4_cores  # Reserve the rest for inference
  gpu_exclusive: true

memory:
  zone_model: 40G
  zone_kv: 12G
  zone_tensor: 4G
  zone_general: 8G  # Everything else gets 8 GB

security:
  model_signing: required
  audit_log: /var/log/spirit/audit.log
  sandbox:
    max_output_tokens: 8192
    blocked_patterns: ["API_KEY", "SECRET", "password"]
```

---

## 13. Virtualized Deployment (Phase 1)

Phase 1 deploys SpiritROS as a **containerized virtual environment** running on a Linux host, enabling development and testing before the custom kernel is ready.

### 13.1 Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     HOST MACHINE                          │
│                     (Linux + Docker)                      │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │              SpiritROS Container                    │  │
│  │              (OCI / Docker image)                   │  │
│  │                                                    │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │  Spirit Userspace                            │  │  │
│  │  │  (Shell, SpiritFS, Inference Coordinator,    │  │  │
│  │  │   Compat Layers, SDK, Scheduler Shim)        │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  │                                                    │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │  Ollama + patched llama.cpp                  │  │  │
│  │  │  (using Spirit memory manager shim)          │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  │                                                    │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │  Spirit Memory Manager (userspace shim)      │  │  │
│  │  │  (cgroups v2 for zone enforcement,           │  │  │
│  │  │   hugepages for model weight pinning,        │  │  │
│  │  │   CUDA MPS for GPU sharing control)          │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  │                                                    │  │
│  └─────────────────────────┬──────────────────────────┘  │
│                            │                              │
│  ┌─────────────────────────┴──────────────────────────┐  │
│  │  NVIDIA Container Toolkit / ROCm Docker            │  │
│  │  (GPU passthrough)                                 │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  Host Linux Kernel (handles actual hardware)             │
└──────────────────────────────────────────────────────────┘
```

### 13.2 Docker Implementation

```dockerfile
# Dockerfile.spiritros-phase1
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake git curl wget \
    python3 python3-pip \
    libfuse3-dev sqlite3 libsqlite3-dev \
    wireguard-tools etcd-client \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Build patched llama.cpp with Spirit memory shim
COPY patches/llama-cpp-spirit.patch /tmp/
RUN git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp && \
    cd /opt/llama.cpp && \
    git apply /tmp/llama-cpp-spirit.patch && \
    cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j$(nproc)

# Build Spirit userspace components
COPY spirit-runtime/ /opt/spirit/
RUN cd /opt/spirit && cargo build --release

# Install SpiritFS (FUSE-based)
RUN cd /opt/spirit/spiritfs && cargo build --release && \
    cp target/release/spiritfs /usr/local/bin/

# Install Spirit Shell
RUN cd /opt/spirit/shell && cargo build --release && \
    cp target/release/spirit-shell /usr/local/bin/

# Install Spirit SDK (Python bindings)
RUN cd /opt/spirit/sdk/python && pip install -e . --break-system-packages

# Install embedding model for SpiritFS
RUN ollama pull nomic-embed-text

# Configure memory zones via cgroups
COPY config/spirit-cgroups.sh /etc/init.d/
COPY config/spirit.yaml /etc/spirit/

# Spirit Shell as default entrypoint
ENTRYPOINT ["/usr/local/bin/spirit-shell"]
```

### 13.3 Docker Compose for Full Stack

```yaml
# docker-compose.yml
version: "3.8"

services:
  spiritros:
    build:
      context: .
      dockerfile: Dockerfile.spiritros-phase1
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - SPIRIT_ZONE_MODEL=32G
      - SPIRIT_ZONE_KV=8G
      - SPIRIT_ZONE_TENSOR=4G
      - SPIRIT_ZONE_GENERAL=8G
      - OLLAMA_NUM_PARALLEL=4
      - OLLAMA_MAX_LOADED_MODELS=3
    volumes:
      - spirit-models:/root/.ollama/models
      - spirit-data:/data
      - spirit-logs:/var/log/spirit
      - /dev/shm:/dev/shm  # Shared memory for IPC
    ports:
      - "11434:11434"   # Ollama API
      - "50051:50051"   # gRPC inference
      - "8080:8080"     # Spirit Shell web UI
      - "51820:51820/udp" # WireGuard (mesh)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
        limits:
          memory: 64G
    cap_add:
      - SYS_ADMIN      # For FUSE (SpiritFS)
      - NET_ADMIN       # For WireGuard
    devices:
      - /dev/fuse       # SpiritFS
    privileged: false
    security_opt:
      - seccomp:unconfined  # Needed for Spirit syscall shim
    stdin_open: true
    tty: true

  qdrant:
    image: qdrant/qdrant:v1.12.0
    volumes:
      - qdrant-data:/qdrant/storage
    ports:
      - "6333:6333"

  etcd:
    image: quay.io/coreos/etcd:v3.5.17
    command: >
      etcd
      --name spirit-node-0
      --listen-client-urls http://0.0.0.0:2379
      --advertise-client-urls http://etcd:2379

  monitoring:
    image: grafana/grafana:11.4.0
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana

  prometheus:
    image: prom/prometheus:v2.55.0
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

volumes:
  spirit-models:
  spirit-data:
  spirit-logs:
  qdrant-data:
  grafana-data:
```

### 13.4 Quick Start

```bash
# Clone SpiritROS
git clone https://github.com/spiritros/spiritros.git
cd spiritros

# Build and launch
docker compose up -d

# Attach to Spirit Shell
docker compose exec spiritros spirit-shell

# Pull a model and start inference
spirit@ghost ~ $ ollama pull llama3.1:8b
spirit@ghost ~ $ spirit infer "Hello, I'm running on SpiritROS!"
```

---

## 14. Performance Benchmarks & Targets

### 14.1 Key Metrics

| Metric | Traditional (Linux + Ollama) | SpiritROS Phase 1 (Container) | SpiritROS Phase 3 (Native Kernel) |
|---|---|---|---|
| **Model cold load (7B, Q4)** | 4.2s | 2.1s (weight pinning) | 0.8s (NVMe Direct) |
| **Model hot swap** | 3.8s (full reload) | 0.3s (shared weights) | 0.1s (pre-mapped) |
| **First token latency (7B)** | 180ms | 95ms | 45ms |
| **Token/s decode (7B, single)** | 65 tok/s | 85 tok/s | 110 tok/s |
| **Token/s decode (7B, batch=8)** | 320 tok/s | 480 tok/s | 620 tok/s |
| **KV-cache fork** | N/A (full copy) | 0.02ms (CoW) | 0.01ms (CoW) |
| **Context switch to inference** | 12ms (scheduler) | 2ms (priority boost) | 0.3ms (dedicated core) |
| **Memory overhead (OS)** | 800MB+ (Ubuntu) | 200MB (Spirit shim) | 45MB (microkernel) |
| **GPU memory waste** | 15-20% (fragmentation) | 5-8% (zone alloc) | < 2% (contiguous) |

### 14.2 How Gains Are Achieved

**Phase 1 (Container, 30-60% improvement):**
- cgroups v2 enforce memory zones, preventing OS from stealing model memory.
- Hugepages (2MB/1GB) for model weights eliminate TLB misses.
- CPU pinning and NUMA-aware allocation reduce cross-socket traffic.
- CUDA MPS prevents GPU context-switch overhead from competing processes.
- Ollama's `OLLAMA_NUM_PARALLEL` tuned with Spirit's batch coordinator.

**Phase 3 (Native Kernel, 2-3x improvement):**
- Zero syscall overhead for inference (kernel-native path).
- GPU memory managed by kernel, no CUDA runtime allocation overhead.
- NVMe → GPU direct transfer for model loading (GPUDirect Storage).
- Inference decode thread runs on a dedicated core with no preemption.
- KV-cache is a kernel data structure with page-granular management.

### 14.3 Benchmark Suite

SpiritROS ships with `spirit-bench`, a comprehensive benchmarking tool:

```bash
$ spirit-bench run --suite full

Running SpiritROS Benchmark Suite v0.1.0
─────────────────────────────────────────
[1/8] Model Load (cold):     llama-3.1-8b-q4_k_m    2.14s ✓
[2/8] Model Load (warm):     llama-3.1-8b-q4_k_m    0.31s ✓
[3/8] First Token Latency:   prompt=256 tokens       94ms  ✓
[4/8] Decode Throughput:      single stream           87 tok/s ✓
[5/8] Decode Throughput:      batch=4                 312 tok/s ✓
[6/8] Decode Throughput:      batch=8                 486 tok/s ✓
[7/8] KV-Cache Fork:          32k context             0.018ms ✓
[8/8] Memory Efficiency:      7B model loaded         94.2% utilization ✓

Overall Score: 847 / 1000
Baseline (Linux + Ollama): 520 / 1000
Improvement: +62.9%
```

---

## 15. Open Source Stack & Dependencies

### 15.1 Core Infrastructure

| Component | Project | License | Role in SpiritROS |
|---|---|---|---|
| Kernel language | **Rust** (stable) | MIT/Apache-2.0 | Primary implementation language |
| Inference engine | **llama.cpp** | MIT | Core inference backend |
| Model management | **Ollama** | MIT | Model registry, API server |
| Container runtime | **youki** | Apache-2.0 | OCI-compatible container execution |
| Container orchestration | **containerd** | Apache-2.0 | Container lifecycle management |
| WASM runtime | **Wasmtime** | Apache-2.0 | Plugin sandboxing |
| Vector database | **Qdrant** | Apache-2.0 | SpiritFS semantic index |
| Vector library | **FAISS** | MIT | CPU-based similarity search |
| Userspace network | **smoltcp** | 0BSD | Kernel network stack |
| TLS | **rustls** | MIT/Apache-2.0 | TLS 1.3 implementation |
| VPN | **WireGuard** | GPL-2.0 | Mesh networking |

### 15.2 Compatibility Layer

| Component | Project | License | Role in SpiritROS |
|---|---|---|---|
| Linux compat | **gVisor** (adapted) | Apache-2.0 | Syscall translation layer |
| Windows compat | **Wine** | LGPL-2.1 | Win32 API implementation |
| DirectX compat | **DXVK** / **VKD3D-Proton** | Zlib | DX→Vulkan translation |
| Android compat | **Waydroid** | GPL-3.0 | Android container runtime |
| Init system | **dinit** | Apache-2.0 | Lightweight service manager |

### 15.3 Monitoring & Observability

| Component | Project | License | Role in SpiritROS |
|---|---|---|---|
| Metrics | **Prometheus** | Apache-2.0 | System and inference metrics |
| Dashboards | **Grafana** | AGPL-3.0 | Visualization and alerting |
| Tracing | **Jaeger** | Apache-2.0 | Distributed inference tracing |
| Logging | **Vector** | MPL-2.0 | Log aggregation and routing |

### 15.4 Developer Tools

| Component | Project | License | Role in SpiritROS |
|---|---|---|---|
| Build system | **just** | CC0-1.0 | Task runner (Makefile alternative) |
| CI/CD | **Forgejo Actions** / **Dagger** | MIT/Apache-2.0 | Build and test automation |
| Documentation | **mdBook** | MPL-2.0 | Documentation site generator |
| API docs | **OpenAPI** + **Scalar** | Various | API documentation and playground |

---

## 16. Roadmap & Milestones

### Phase 1 — Virtual Proof of Concept (Months 1-6)

**Goal:** SpiritROS running as a Docker container with measurable performance improvements over stock Linux + Ollama.

| Milestone | Target Date | Deliverables |
|---|---|---|
| M1.1: Foundation | Month 1 | Docker image, Spirit Shell prototype, Ollama integration |
| M1.2: Memory Manager | Month 2 | cgroups-based zone management, hugepage model pinning |
| M1.3: Scheduler Shim | Month 3 | CPU pinning, priority-based inference scheduling |
| M1.4: SpiritFS v0.1 | Month 4 | FUSE-based semantic filesystem with embedding index |
| M1.5: SDK & API | Month 5 | Rust/Python/TS SDK, gRPC inference API |
| M1.6: Benchmarks | Month 6 | spirit-bench suite, published performance comparison |

### Phase 2 — Kernel Development (Months 7-18)

**Goal:** SpiritKernel boots on real hardware, runs inference natively.

| Milestone | Target Date | Deliverables |
|---|---|---|
| M2.1: Bootloader | Month 7-8 | UEFI boot, basic console output |
| M2.2: Memory Manager | Month 9-10 | Physical + virtual memory with tensor zones |
| M2.3: Scheduler | Month 11-12 | Multi-queue scheduler with inference phases |
| M2.4: GPU Driver | Month 13-15 | NVIDIA/AMD GPU initialization, VRAM management |
| M2.5: Inference Syscalls | Month 16-17 | Full spirit_infer_* syscall implementation |
| M2.6: Self-Hosting | Month 18 | SpiritROS runs Ollama natively on custom kernel |

### Phase 3 — Production Ready (Months 19-30)

**Goal:** Complete OS with compatibility layers, security hardening, and distributed inference.

| Milestone | Target Date | Deliverables |
|---|---|---|
| M3.1: Linux Compat | Month 19-21 | Full Linux binary compatibility |
| M3.2: Security | Month 22-24 | Capability system, Spirit Guard, model signing |
| M3.3: Distributed | Month 25-27 | Multi-node inference, Spirit Mesh |
| M3.4: Windows/Android | Month 28-29 | Wine and Waydroid integration |
| M3.5: v1.0 Release | Month 30 | Stable release, documentation, community launch |

### Phase 4 — Ecosystem (Months 30+)

- Spirit App Store (curated model + plugin registry)
- Hardware partner program (pre-installed SpiritROS on inference servers)
- SpiritROS for robotics (ROS 2 integration — the *other* ROS)
- SpiritROS for edge (ARM/RISC-V support, quantized models on embedded hardware)
- Formal verification of core kernel security properties

---

## 17. Appendices

### Appendix A: Glossary

| Term | Definition |
|---|---|
| **Inference** | The process of generating output from a trained neural network given an input |
| **KV-cache** | Key-Value cache storing attention states from previously processed tokens |
| **Prefill** | The phase where the model processes the entire input prompt (compute-bound) |
| **Decode** | The phase where the model generates tokens one at a time (memory-bandwidth-bound) |
| **PagedAttention** | Technique for managing KV-cache in non-contiguous memory pages (from vLLM) |
| **GGUF** | File format for quantized LLM weights (used by llama.cpp) |
| **Quantization** | Reducing model weight precision (e.g., float16 → int4) to save memory |
| **Tensor parallelism** | Splitting a model's layers across multiple GPUs |
| **Zero-copy** | Transferring data between components without copying it in memory |
| **Capability** | An unforgeable token granting a specific permission in the security model |

### Appendix B: Hardware Requirements

**Minimum (CPU inference, 7B models):**
- 8-core x86_64 or ARM64 CPU
- 16 GB RAM
- 50 GB NVMe storage
- No GPU required

**Recommended (GPU inference, 7-70B models):**
- 16-core CPU
- 64 GB RAM
- NVIDIA RTX 4090 (24 GB VRAM) or AMD MI300X
- 500 GB NVMe storage

**Enterprise (Multi-GPU, 70-405B models):**
- 64-core CPU (EPYC / Xeon)
- 256 GB+ RAM
- 4-8x NVIDIA H100/H200 (80-141 GB VRAM each)
- NVLink interconnect
- 2 TB+ NVMe (RAID 0)
- InfiniBand for multi-node

### Appendix C: Comparison with Existing Approaches

| Feature | Linux + Docker + Ollama | NVIDIA TensorRT Inference Server | SpiritROS |
|---|---|---|---|
| OS designed for inference | No | No (runs on Linux) | **Yes** |
| Kernel-level KV-cache | No | No | **Yes** |
| Inference as syscall | No | No | **Yes** |
| Model weight memory pinning | Manual (hugepages) | Partial | **Automatic** |
| Multi-model GPU sharing | CUDA MPS (manual) | Triton scheduler | **Kernel-managed** |
| Semantic filesystem | No | No | **Yes** |
| Linux binary compat | Native | Native | **Compatibility layer** |
| Windows binary compat | Wine (manual) | No | **Integrated** |
| Android app compat | No | No | **Integrated** |
| Natural language shell | No | No | **Yes** |
| Distributed inference | Manual (vLLM cluster) | Triton ensemble | **Spirit Mesh** |
| Security model | DAC/MAC (generic) | No built-in | **Capability + inference sandbox** |

### Appendix D: Contributing

SpiritROS is open source under the **Apache 2.0 License** (kernel and runtime) with **GPL-2.0** for kernel modules that interface with Linux-derived code.

Repository structure:
```
spiritros/
├── kernel/              # SpiritKernel (Rust)
│   ├── src/
│   │   ├── boot/        # Bootloader, init
│   │   ├── mem/         # TensorFirst memory manager
│   │   ├── sched/       # Inference-aware scheduler
│   │   ├── gpu/         # GPU driver interface
│   │   ├── ipc/         # Capability-based IPC
│   │   └── syscall/     # Syscall table (including inference domain)
│   └── Cargo.toml
├── runtime/             # Spirit Runtime (userspace)
│   ├── shell/           # Spirit Shell
│   ├── spiritfs/        # Semantic filesystem
│   ├── inference/       # Inference coordinator
│   ├── compat/          # Linux/Windows/Android compat
│   ├── security/        # Spirit Guard, capability manager
│   └── networking/      # Network stack, mesh
├── sdk/                 # Spirit SDK
│   ├── rust/
│   ├── python/
│   └── typescript/
├── docker/              # Phase 1 container deployment
├── docs/                # mdBook documentation
├── benchmarks/          # spirit-bench
└── contrib/             # Community plugins and extensions
```

---

## Closing Statement

SpiritROS represents a fundamental rethinking of what an operating system should be in the age of large language models. By treating inference as a first-class kernel primitive rather than a userspace application, we can unlock performance improvements that are impossible within the constraints of a general-purpose OS — while maintaining the compatibility and security that production deployments demand.

The path from Docker container to custom kernel is long, but each phase delivers tangible, measurable improvements. Phase 1 alone — achievable in six months with a small team — can demonstrate 30-60% performance gains over stock Linux deployments through intelligent memory management, scheduling, and system architecture.

The future of computing is not general-purpose. It is *purpose-built*. SpiritROS is the first operating system built for that future.

---

*"The best way to predict the future is to build the operating system it runs on."*

**— The SpiritROS Project**
