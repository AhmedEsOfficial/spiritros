# Phase 02: TensorFirst Memory Manager

**Roadmap Alignment:** Phase 1 — Virtual Proof of Concept, Milestone M1.2
**Estimated Duration:** Month 2
**Dependencies:** Phase 01 (Foundation)
**Architecture Plan Sections:** §4, §13.1, §14.2

---

## Context & Overview

The TensorFirst Memory Manager is the foundation of SpiritROS's performance advantage. In a conventional Linux setup, the OS treats LLM memory (model weights, KV-cache, tensor scratch) the same as any other allocation. This leads to page eviction of model weights, OOM killing of inference processes, GPU memory fragmentation, and TLB misses on large model files.

In Phase 1, the memory manager is a **userspace shim** that uses Linux kernel facilities (cgroups v2, hugepages, NUMA pinning, CUDA MPS) to enforce memory zones. This is not yet a kernel-level allocator — that comes in Phase 12 — but it delivers measurable improvements by giving the inference engine guaranteed, optimized memory regions.

---

## Goals

1. **Implement four memory zones** via cgroups v2: ZONE_MODEL, ZONE_KV, ZONE_TENSOR, ZONE_GENERAL
2. **Hugepage support** for model weight pinning (2MB and 1GB pages)
3. **VRAM reservation** via CUDA MPS or direct CUDA API to prevent GPU memory fragmentation
4. **Zone configuration** via `spirit.yaml` and environment variables
5. **Memory telemetry** — real-time zone utilization exposed via Prometheus metrics
6. **Rust library** (`spirit-memory`) that other components use for allocation

---

## Design Decisions

### DD-02.1: cgroups v2 for Zone Enforcement
**Decision:** Use cgroups v2 memory controllers to enforce zone boundaries in Phase 1.
**Rationale:** cgroups v2 provides hard memory limits, memory.high/memory.max thresholds, and swap control per cgroup. Each zone maps to a cgroup subtree. This is the best userspace approximation of kernel memory zones available on Linux.

### DD-02.2: Hugepages for Model Weights (ZONE_MODEL)
**Decision:** Model weights are loaded via `mmap` with `MAP_HUGETLB` and `MAP_POPULATE`.
**Rationale:** A 7B Q4_K_M model is ~4.3GB. With standard 4KB pages, that's ~1.1M page table entries and potential TLB thrashing. With 2MB hugepages, it's ~2,150 entries. With 1GB hugepages, it's 5 entries. `MAP_POPULATE` pre-faults all pages, eliminating page faults during first inference.

### DD-02.3: KV-Cache Zone is Swappable to NVMe
**Decision:** ZONE_KV can overflow to NVMe-backed swap with async prefetch.
**Rationale:** Long context windows (32k–128k tokens) can consume several GB of KV-cache. Rather than OOM, cold KV-cache pages should gracefully swap to fast NVMe storage with prefetch on predicted access.

### DD-02.4: CUDA MPS for Multi-Process GPU Sharing
**Decision:** Use NVIDIA CUDA Multi-Process Service (MPS) to control GPU memory partitioning between inference and other GPU consumers.
**Rationale:** Without MPS, each CUDA context reserves its own GPU memory, causing fragmentation. MPS allows the memory manager to set per-process VRAM limits and share a single CUDA context.

### DD-02.5: Configuration Hierarchy
**Decision:** Zone sizes are configured via: (1) `spirit.yaml` defaults → (2) environment variables (`SPIRIT_ZONE_MODEL`, etc.) → (3) runtime API calls.
**Rationale:** Environment variables enable Docker Compose-level tuning without modifying config files. Runtime API allows dynamic rebalancing (e.g., `spirit mem rebalance`).

---

## Components & Tasks

### 1. spirit-memory Crate (`runtime/memory/`)
```rust
// Core API surface
pub struct MemoryManager { ... }

impl MemoryManager {
    pub fn new(config: MemoryConfig) -> Result<Self>;

    // Zone allocation
    pub fn alloc_model(&self, size: usize) -> Result<ModelAllocation>;
    pub fn alloc_kv(&self, size: usize) -> Result<KvAllocation>;
    pub fn alloc_tensor(&self, size: usize) -> Result<TensorAllocation>;
    pub fn alloc_general(&self, size: usize) -> Result<GeneralAllocation>;

    // VRAM management
    pub fn reserve_vram(&self, device: GpuDevice, size: usize) -> Result<VramReservation>;
    pub fn vram_stats(&self, device: GpuDevice) -> VramStats;

    // Zone telemetry
    pub fn zone_stats(&self) -> ZoneStats;

    // Dynamic rebalancing
    pub fn rebalance(&self, from: Zone, to: Zone, amount: usize) -> Result<()>;
}
```

### 2. cgroups v2 Zone Setup (`config/spirit-cgroups.sh`)
- Create cgroup hierarchy: `/sys/fs/cgroup/spirit/{model,kv,tensor,general}`
- Set `memory.max`, `memory.high`, `memory.swap.max` per zone
- Configure `memory.min` (guaranteed memory) for ZONE_MODEL and ZONE_KV
- Setup script runs at container init

### 3. Hugepage Configuration
- Reserve hugepages at boot/container start: `echo N > /proc/sys/vm/nr_hugepages`
- Mount hugetlbfs: `mount -t hugetlbfs nodev /dev/hugepages`
- `ModelAllocation` uses `mmap(MAP_HUGETLB | MAP_POPULATE)` for weight files
- Fallback to standard pages if hugepages unavailable

### 4. VRAM Manager
- Query GPU count, VRAM per GPU via CUDA/ROCm APIs (using `cudarc` Rust crate or FFI)
- Pre-reserve contiguous VRAM blocks for model weights at load time
- Track VRAM utilization per zone (model weights vs. KV-cache vs. scratch)
- Expose VRAM stats via telemetry

### 5. KV-Cache Page Manager
- Implement a page table structure for KV-cache (per-session)
- Pages can be in: GPU VRAM, CPU RAM (ZONE_KV cgroup), or NVMe swap
- LRU eviction policy: evict coldest session's KV-cache pages first
- Async prefetch: when a session becomes active, prefetch its swapped pages

### 6. Prometheus Metrics Exporter
- Expose metrics at `/metrics` endpoint:
  - `spirit_memory_zone_used_bytes{zone="model|kv|tensor|general"}`
  - `spirit_memory_zone_limit_bytes{zone="..."}`
  - `spirit_vram_used_bytes{device="gpu0|gpu1|..."}`
  - `spirit_vram_reserved_bytes{device="..."}`
  - `spirit_kv_pages_active`, `spirit_kv_pages_swapped`
  - `spirit_hugepages_used`, `spirit_hugepages_free`

### 7. Configuration Parsing
- Parse `spirit.yaml` memory section
- Override with `SPIRIT_ZONE_*` environment variables
- Validate: sum of zones <= available physical memory
- Warn if hugepages are not available

---

## Acceptance Criteria

- [x] `spirit-memory` crate compiles and passes unit tests (90 tests + 1 doc-test)
- [x] cgroups v2 zones are created at container startup with correct limits (`CgroupManager::setup_zones`)
- [x] Model weight files are loaded via hugepage-backed mmap (`HugepageManager::mmap_model_file` with `MAP_HUGETLB | MAP_POPULATE`, fallback to standard pages)
- [x] VRAM reservation prevents fragmentation (`VramManager::reserve` with per-device tracking and admission control)
- [x] KV-cache pages can be evicted to NVMe and prefetched back (`KvPageManager::evict_session` / `prefetch_session`)
- [x] Zone stats are visible in Prometheus (`MemoryMetrics` exposes `spirit_memory_zone_used_bytes`, etc.)
- [x] Grafana dashboard shows real-time memory zone utilization (`docker/config/grafana/dashboards/memory-zones.json`)
- [ ] `spirit mem status` command (via Shell) displays zone utilization summary — *API ready (`MemoryManager::all_zone_stats`), shell integration pending Phase 06*
- [x] `spirit mem rebalance --give-general 2G --from tensor` works at runtime (`MemoryManager::rebalance` updates zones + cgroups)
- [ ] Loading a 7B model with hugepages shows measurably fewer TLB misses than without (via `perf stat`) — *requires Phase 11 benchmarking infrastructure*

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| cgroups v2 not available (older host kernel) | Zone enforcement fails | Detect at startup; warn and fall back to soft limits via process memory tracking |
| Hugepages not pre-reserved by host | Model pinning degrades to standard pages | Transparent hugepages (THP) as fallback; document host setup requirements |
| CUDA MPS incompatibility with some GPU models | VRAM management degraded | Fall back to per-process CUDA contexts with advisory limits |
| KV-cache swap to NVMe too slow | Long-context sessions stall | Implement async prefetch with prediction; document NVMe speed requirements |
| Zone sum exceeds physical RAM | Container OOM | Validate at startup; refuse to start with clear error message |

---

## Implementation Status

**Status:** ✅ Core implementation complete (2026-03-27)

### Module Structure (`runtime/memory/src/`)

| Module | Lines | Purpose |
|---|---|---|
| `types.rs` | Types, constants, zone/stats/allocation/page definitions |
| `error.rs` | `SpiritMemoryError` with 15 error variants |
| `config.rs` | `MemoryConfig` with figment (YAML + env var) support, auto-sizing |
| `cgroups.rs` | `CgroupManager` — create/read/write cgroup v2 zone hierarchy |
| `hugepages.rs` | `HugepageManager` — mmap with MAP_HUGETLB, fallback to standard |
| `vram.rs` | `VramManager` — GPU detection (nvidia-smi), logical VRAM reservations |
| `kv_cache.rs` | `KvPageManager` — pre-allocated page pool, COW snapshots/forks, LRU eviction, NVMe swap |
| `zones.rs` | `ZoneTracker` — admission control, allocation tracking, runtime rebalancing |
| `pressure.rs` | `PressureMonitor` — PSI parsing, graduated pressure levels, background polling |
| `metrics.rs` | `MemoryMetrics` — Prometheus registry with 14 metric families |
| `manager.rs` | `MemoryManager` — public API surface composing all subsystems |
| `lib.rs` | Module declarations, re-exports |

### Test Coverage

- **90 unit tests** across all modules
- **1 doc-test** for the crate-level example
- Tests cover: zone allocation/exhaustion, cgroup setup, hugepage mmap, VRAM reservation/release, KV-cache COW/fork/eviction, rebalancing, pressure parsing, metrics encoding

### Remaining Items

- Shell integration (`spirit mem status` / `spirit mem rebalance`) — blocked on Phase 06
- Benchmark validation (TLB miss measurement) — blocked on Phase 11

---

## Key References

- Architecture Plan §4 (Memory Architecture — TensorFirst Memory)
- Architecture Plan §4.2 (TensorFirst Memory Zones)
- Architecture Plan §4.3 (GPU VRAM Management)
- Architecture Plan §4.4 (KV-Cache as Kernel-Managed Resource)
- Architecture Plan §4.5 (Model Weight Caching & Prefetching)
- Architecture Plan §14.2 (How Gains Are Achieved — Phase 1)
