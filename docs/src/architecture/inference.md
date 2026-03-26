# Inference Engine

The inference engine is the heart of SpiritROS. It manages model loading, session lifecycle, token streaming, and integration with the Ollama backend.

See [Architecture Overview](overview.md) for a high-level introduction to the inference subsystem.

## Architecture

```text
Client Request (HTTP / gRPC / Unix Socket / Shared Memory)
        |
        v
+---------------------------------------+
|     InferenceCoordinator              |  <- Rust daemon (spirit-infer)
|  +----------------------------------+ |
|  |  SessionManager                  | |  Tracks active sessions
|  |  +- session pool                 | |
|  |  +- KV-cache tracker            | |  Manages COW snapshots
|  |  +- pre-allocated buffers       | |  Zero-alloc hot path
|  +----------------------------------+ |
|  +----------------------------------+ |
|  |  BatchScheduler                  | |  Continuous batching
|  |  +- priority queue              | |  Interactive > Background
|  |  +- model grouping              | |  Same-model requests batched
|  +----------------------------------+ |
|  +----------------------------------+ |
|  |  ModelRegistry                   | |  Content-addressable store
|  |  +- name index                  | |  name -> ModelId
|  |  +- digest index                | |  SHA-256 dedup
|  +----------------------------------+ |
|  +----------------------------------+ |
|  |  InferenceBackend (trait)        | |  Pluggable engine interface
|  |  +- OllamaBackend (Phase 1)     | |  HTTP API delegation
|  +----------------------------------+ |
|  +----------------------------------+ |
|  |  IPC Layer                       | |
|  |  +- Unix Domain Socket          | |  Control messages (JSON)
|  |  +- Shared Memory Ring Buffer   | |  Zero-copy token streaming
|  +----------------------------------+ |
|  +----------------------------------+ |
|  |  InferenceMetrics                | |  20 Prometheus metrics
|  +----------------------------------+ |
+---------------------------------------+
        |
        v
    Ollama / llama-server
        |
        v
    GPU / CPU (via CUDA / ROCm / Vulkan)
```

## Key Components

### Inference Coordinator (`coordinator.rs`)

The central orchestration point. Composes all subsystems and provides the public API for:

- **Session management** - create, prompt, destroy sessions
- **Model management** - load, unload, list models
- **KV-cache operations** - snapshot, restore (COW semantics)
- **Metrics and status** - Prometheus metrics, health checks

### Session Manager (`session.rs`)

Each inference session owns pre-allocated buffers for token streaming. Buffers are allocated once at session creation and reused across generations. This ensures the hot path (token receive loop) never allocates.

Sessions follow a state machine:

```text
Pending -> Ready -> Prefilling -> Decoding -> Idle -> (next prompt) -> Prefilling
                                                 +-> Terminated
```

### Batch Scheduler (`batch.rs`)

Groups concurrent inference requests into GPU batches for maximum throughput. Priority-aware:

| Priority | Use Case | Behavior |
|---|---|---|
| Interactive | Spirit Shell | Lowest latency, dispatched immediately |
| High | SDK API calls | Preferential scheduling |
| Normal | Default | Standard batching |
| Background | SpiritFS indexing | Batched aggressively |

### Model Registry (`model_registry.rs`)

Maintains a catalog of available models with three index types:
- **Name index** - friendly name to ModelId
- **Digest index** - SHA-256 for deduplication
- **Load state tracking** - Unloaded, Loading, Loaded, Failed

Syncs with Ollama's model list via the HTTP API.

### Ollama Backend (`backend/ollama.rs`)

Phase 1 backend that delegates to a local Ollama instance. Implements the `InferenceBackend` trait:

- `load_model` / `unload_model` - model lifecycle
- `generate` - complete generation (non-streaming)
- `stream` - streaming token generation
- `kv_snapshot` / `kv_restore` - KV-cache operations (Phase 1 shim)

### IPC Layer

Two transport mechanisms:

**Unix Domain Socket** (`/run/spirit/infer.sock`):
- Length-prefixed JSON protocol (4-byte BE length + JSON payload)
- Operations: CreateSession, Prompt, Status, Snapshot, Restore, LoadModel, ListModels

**Shared Memory Ring Buffer** (`/dev/shm/spirit_infer`):
- Lock-free SPSC queue per session
- 16-byte `TokenEntry` structs (cache-line friendly)
- Zero-copy: producer writes, consumer reads with no syscalls

## Configuration

The inference subsystem is configured via `spirit.yaml`:

```yaml
inference:
  backend: "ollama"
  ollama:
    url: "http://localhost:11434"
    max_sessions: 8
    default_model: "llama3.2:3b"
    timeout_secs: 300
  coordinator:
    socket_path: "/run/spirit/infer.sock"
    max_queue_depth: 256
    token_buffer_size: 4096
    realtime_scheduling: true
    realtime_priority: 80
  kv_cache:
    eviction_policy: "lru"
    snapshots_enabled: true
  gpu:
    offload: "auto"
    mps_enabled: true
    memory_fraction: 0.9
```

Environment variable overrides use the `SPIRIT_INFERENCE_` prefix with double underscores for nesting (e.g., `SPIRIT_INFERENCE_OLLAMA__URL`).

## Metrics

The inference subsystem exposes 20 Prometheus metrics under the `spirit_inference_` prefix:

| Metric | Type | Description |
|---|---|---|
| `tokens_generated_total` | Counter | Total tokens generated |
| `prompt_tokens_total` | Counter | Total prompt tokens processed |
| `first_token_latency_ms` | Histogram | Time to first token |
| `decode_latency_ms` | Histogram | Per-token decode latency |
| `generation_total_ms` | Histogram | Total generation time |
| `tokens_per_second` | Gauge | Current throughput |
| `active_sessions` | Gauge | Active session count |
| `loaded_models` | Gauge | Loaded model count |
| `queue_depth` | Gauge | Batch queue depth |
| `batch_size` | Histogram | Requests per batch |
| `kv_snapshots_total` | Counter | KV-cache snapshots taken |
| `errors_total` | Counter | Total inference errors |

## Running the Daemon

```bash
spirit-infer                                   # Use defaults
spirit-infer --config /etc/spirit/spirit.yaml  # Custom config
spirit-infer --log-level debug                 # Verbose logging
spirit-infer --socket-path /tmp/test.sock      # Override socket path
```

## Hot-Path Contract

The token streaming path must **never** allocate heap memory:

1. Session buffers are pre-allocated at creation time
2. The shared memory ring buffer uses lock-free atomics
3. Token entries are fixed-size (16 bytes, cache-line friendly)
4. No serialization on the shared memory path
