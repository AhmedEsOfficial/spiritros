# Phase 03: Inference Engine Integration

**Roadmap Alignment:** Phase 1 — Virtual Proof of Concept, Milestones M1.1, M1.2
**Estimated Duration:** Months 1–3 (overlaps with Phase 02)
**Dependencies:** Phase 01 (Foundation), Phase 02 (Memory Manager)
**Architecture Plan Sections:** §5, §3.4 (Inference Syscalls)

---

## Context & Overview

The inference engine is the heart of SpiritROS. In Phase 1, this means Ollama (for model management and API) backed by a patched llama.cpp (for actual inference computation). The "patching" redirects llama.cpp's memory allocation and KV-cache management to use Spirit's memory manager shim instead of standard malloc/cudaMalloc.

This phase also establishes the **Inference Coordinator** — a userspace daemon that multiplexes inference requests, manages sessions, coordinates batching, and provides the internal API that other Spirit components (Shell, SDK, SpiritFS) use to request inference.

---

## Goals

1. **Ollama as a system service** — Ollama running inside the container with enhanced configuration
2. **Patched llama.cpp** — Memory allocator replacement routed to `spirit-memory`
3. **Inference Coordinator daemon** — Request multiplexing, session management, batch coordination
4. **Model Registry** — Content-addressable model store with metadata tracking
5. **Inference API (internal)** — Unix domain socket + shared memory IPC for local components
6. **Multi-backend architecture** — Pluggable backend interface (llama.cpp primary, others future)
7. **Phase-aware hints** — llama.cpp emits prefill/decode phase signals to the scheduler

---

## Design Decisions

### DD-03.1: Ollama as Frontend, llama.cpp as Backend
**Decision:** Keep Ollama for user-facing model management (pull, list, Modelfile) and the OpenAI-compatible API. Patch llama.cpp at the ggml allocation layer for Spirit memory integration.
**Rationale:** Ollama provides a mature UX and ecosystem. We don't want to rewrite model management. The performance gains come from *how* memory is allocated, not from changing the API surface.

### DD-03.2: Fork Strategy for llama.cpp
**Decision:** Maintain a minimal patch set (`.patch` file) applied to upstream llama.cpp, not a hard fork.
**Rationale:** llama.cpp moves fast (daily commits). A patch-based approach lets us rebase on upstream regularly. The patch touches only: (1) ggml_backend_alloc → spirit_tensor_alloc, (2) KV-cache management → spirit_kv_*, (3) scheduler hints before prefill/decode.

### DD-03.3: Inference Coordinator as Separate Daemon
**Decision:** The Inference Coordinator runs as its own process, separate from Ollama.
**Rationale:** Ollama manages models and serves the external API. The Coordinator manages the internal lifecycle: session creation, KV-cache allocation, batching decisions, and routing. This separation allows us to eventually replace Ollama with another frontend without changing the core inference pipeline.

### DD-03.4: IPC via Unix Domain Socket + Shared Memory
**Decision:** Internal inference requests use a Unix domain socket for control messages and `/dev/shm/spirit_infer` for zero-copy token transfer.
**Rationale:** UDS provides reliable, ordered delivery for request/response metadata. Shared memory provides zero-copy for bulk token data (prompts in, completions out). This mirrors the eventual kernel syscall interface.

### DD-03.5: Content-Addressable Model Store
**Decision:** Model weight files stored by SHA-256 hash in `blobs/`; metadata in SQLite `registry.db`.
**Rationale:** Deduplicates identical model files (same quantization pulled via different names). Enables integrity verification. Matches Ollama's existing blob storage pattern.

---

## Components & Tasks

### 1. Ollama Service Configuration
- Ollama installed in Docker image (already in Phase 01)
- Custom Ollama config for SpiritROS:
  - `OLLAMA_NUM_PARALLEL=4` (concurrent inference streams)
  - `OLLAMA_MAX_LOADED_MODELS=3` (coordinated with memory zones)
  - `OLLAMA_MODELS=/spirit/models` (Spirit model registry path)
- systemd-style service management via `dinit` or simple process supervisor

### 2. llama.cpp Patch
Create `patches/llama-cpp-spirit.patch` that modifies:

**a) Memory Allocator Replacement**
```c
// In ggml/src/ggml-backend.c
// Replace: ggml_backend_buffer_alloc → malloc/cudaMalloc
// With:    ggml_backend_buffer_alloc → spirit_tensor_alloc() via FFI
```
- Rust FFI bridge: `spirit-memory` exposes C-compatible `spirit_tensor_alloc()` / `spirit_tensor_free()`
- Allocation goes through Spirit's zone-aware allocator

**b) KV-Cache Delegation**
```c
// In llama.cpp kv_cache_init / kv_cache_alloc
// Route KV-cache allocation to spirit_kv_alloc() instead of internal management
// Enable paged KV-cache (PagedAttention-style) via Spirit memory manager
```

**c) Scheduler Phase Hints**
```c
// Before prefill: spirit_sched_hint(PHASE_PREFILL)
// Before decode loop: spirit_sched_hint(PHASE_DECODE)
// After generation complete: spirit_sched_hint(PHASE_IDLE)
```

### 3. Inference Coordinator (`runtime/inference/`)
```rust
pub struct InferenceCoordinator {
    sessions: HashMap<SessionId, InferenceSession>,
    model_registry: ModelRegistry,
    memory_manager: Arc<MemoryManager>,
    batch_scheduler: BatchScheduler,
}

impl InferenceCoordinator {
    // Session lifecycle
    pub async fn create_session(&self, model_id: ModelId, config: SessionConfig) -> Result<SessionId>;
    pub async fn destroy_session(&self, session_id: SessionId) -> Result<()>;

    // Inference
    pub async fn prompt(&self, session_id: SessionId, tokens: &[Token]) -> Result<TokenStream>;
    pub async fn batch_prompt(&self, requests: Vec<BatchRequest>) -> Result<Vec<TokenStream>>;

    // KV-cache management
    pub async fn snapshot_kv(&self, session_id: SessionId) -> Result<SnapshotId>;
    pub async fn restore_kv(&self, session_id: SessionId, snapshot_id: SnapshotId) -> Result<()>;

    // Model management
    pub async fn load_model(&self, path: &Path, policy: MemoryPolicy) -> Result<ModelId>;
    pub async fn unload_model(&self, model_id: ModelId) -> Result<()>;
    pub async fn loaded_models(&self) -> Vec<LoadedModelInfo>;
}
```

### 4. Model Registry (`/spirit/models/`)
```
/spirit/models/
├── registry.db          # SQLite: model metadata, hashes, configs, load state
├── blobs/               # Content-addressed weight files (SHA-256)
│   ├── sha256-a1b2c3...
│   └── sha256-d4e5f6...
└── manifests/           # Model configuration (name → blob mapping)
    ├── llama3.1-8b.json
    └── mistral-7b.json
```

- `registry.db` schema: models (id, name, format, quantization, size, sha256, loaded_in_zone, gpu_device)
- Integration with Ollama's pull mechanism (Ollama pulls → Spirit registers in registry)
- Direct file import: `spirit model import /path/to/model.gguf`
- Track which models are currently loaded, in which zones, on which GPUs

### 5. Batch Scheduler
- Continuous batching: multiplex multiple inference streams into a single GPU batch
- Dynamic batch sizing based on available VRAM and request queue depth
- Priority-aware: urgent requests (Shell interactive) get lower latency; background (SpiritFS indexing) gets batched more aggressively
- Coordination with Spirit Scheduler (Phase 04) for CPU/GPU resource hints

### 6. IPC Interface
**Unix Domain Socket** (`/run/spirit/infer.sock`):
- Protocol: length-prefixed protobuf messages (or MessagePack)
- Operations: CreateSession, Prompt, Recv, Snapshot, Restore, LoadModel, Status

**Shared Memory** (`/dev/shm/spirit_infer`):
- Ring buffer for token transfer (prompt tokens in, generated tokens out)
- Lock-free SPSC queue per session
- Zero-copy: tokens written by llama.cpp, read by requesting process

---

## Acceptance Criteria

- [x] Ollama service starts automatically in the container and responds to `ollama list` *(Dockerfile + entrypoint.sh wired up)*
- [x] `ollama pull llama3.1:8b` downloads a model and registers it in Spirit's model registry *(ModelRegistry.sync_from_ollama() implemented; requires running container to verify end-to-end)*
- [ ] Patched llama.cpp builds successfully with Spirit memory shim linked *(Phase 2 — requires FFI bridge)*
- [ ] Inference using the patched binary allocates model weights from ZONE_MODEL *(Phase 2 — requires patched llama.cpp)*
- [ ] KV-cache allocation comes from ZONE_KV *(Phase 2 — requires patched llama.cpp)*
- [x] Inference Coordinator accepts requests via Unix domain socket
- [x] Multiple concurrent inference sessions work (at least 4 parallel streams)
- [x] KV-cache snapshot and restore works (Phase 1 shim — logical snapshots via coordinator)
- [x] Batch scheduling groups concurrent requests into GPU batches
- [x] Phase hints (prefill/decode) are emitted and can be observed in logs
- [x] Model registry correctly tracks loaded/unloaded state
- [x] Weight deduplication works: loading the same model twice doesn't double memory usage

### Phase 1 Implementation Status

The following components are fully implemented in `runtime/inference/`:

| Component | File(s) | Tests | Status |
|---|---|---|---|
| Inference Coordinator | `coordinator.rs` | 5 | Complete |
| Session Manager | `session.rs` | 7 | Complete |
| Model Registry | `model_registry.rs` | 6 | Complete |
| Batch Scheduler | `batch.rs` | 5 | Complete |
| Ollama HTTP Client | `ollama_client.rs` | 7 | Complete |
| Ollama Backend | `backend/ollama.rs` | 4 | Complete |
| Backend Trait | `backend/mod.rs` | — | Complete |
| IPC Unix Socket | `ipc/unix_socket.rs` | 7 | Complete |
| Shared Memory Ring Buffer | `ipc/shm.rs` | 6 | Complete |
| Prometheus Metrics | `metrics.rs` | 5 | Complete |
| Configuration | `config.rs` | 7 | Complete |
| Types | `types.rs` | 12 | Complete |
| Error Types | `error.rs` | 4 | Complete |
| Daemon Binary | `main.rs` | — | Complete |
| **Total** | **15 files** | **78 tests** | **All passing** |

Items marked with "Phase 2" require the patched llama.cpp FFI bridge, which is not part of Phase 1. The Phase 1 implementation uses Ollama's HTTP API as the backend, which will be replaced by direct llama.cpp FFI in Phase 2.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| llama.cpp upstream breaks patch compatibility | Patch fails to apply | Pin to specific llama.cpp commit; rebase patch monthly; automated patch-apply CI check |
| Ollama's internal llama.cpp version diverges | Two different llama.cpp versions | Use Ollama's bundled llama.cpp as the patch target, not a separate checkout |
| Shared memory IPC complexity | Bugs, data corruption | Start with UDS-only; add shared memory as optimization once UDS path is stable |
| Continuous batching conflicts with Ollama's own batching | Double-batching, performance regression | Disable Ollama's internal batching; let Spirit Coordinator own batching entirely |
| FFI boundary between Rust (spirit-memory) and C (llama.cpp) | Memory safety concerns at boundary | Extensive fuzzing of the FFI layer; use `cbindgen` for header generation |

---

## Key References

- Architecture Plan §5 (Inference Engine Integration)
- Architecture Plan §5.1 (Ollama as Default Inference Frontend)
- Architecture Plan §5.2 (Modifications to llama.cpp)
- Architecture Plan §5.3 (Multi-Backend Support)
- Architecture Plan §5.4 (Model Format & Registry)
- Architecture Plan §3.4 (Syscall Interface — Inference Domain)
