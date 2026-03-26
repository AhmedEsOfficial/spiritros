# Phase 08: Networking & API Layer

**Roadmap Alignment:** Phase 1 — Milestone M1.5 (API), Phase 3 — M3.3 (distributed)
**Estimated Duration:** Month 5 (API), Months 25–27 (distributed inference)
**Dependencies:** Phase 01 (Foundation), Phase 03 (Inference Engine)
**Architecture Plan Sections:** §8

---

## Context & Overview

SpiritROS exposes inference over the network via multiple protocols, enabling both local and remote access. The networking layer handles everything from the OpenAI-compatible REST API (drop-in replacement for any OpenAI client) to low-latency gRPC streaming, WebSocket connections for web UIs, and the fastest paths: Unix domain sockets and shared memory IPC for local processes.

For multi-node deployments, SpiritROS supports **tensor parallelism across nodes** (splitting large models across machines) and **Spirit Mesh** — a self-organizing federation of SpiritROS instances for workload distribution.

Phase 1 focuses on the API endpoints and local IPC. Distributed inference and Spirit Mesh are Phase 3 deliverables.

---

## Goals

### Phase 1 Goals (Month 5)
1. **OpenAI-compatible REST API** — Drop-in replacement on `:11434/v1/chat/completions`
2. **gRPC inference API** — Low-latency, typed, streaming on `:50051`
3. **WebSocket endpoint** — Real-time streaming for web UIs on `:11434/ws`
4. **Unix domain socket** — Local IPC on `/run/spirit/infer.sock`
5. **Shared memory IPC** — Zero-copy tokens on `/dev/shm/spirit_infer`
6. **API authentication** — mTLS and API key support
7. **Rate limiting** — Configurable per-endpoint rate limits

### Phase 3 Goals (Months 25–27)
8. **Multi-node tensor parallelism** — Split models across machines via NCCL/Gloo
9. **Spirit Mesh** — Self-organizing federation with mDNS discovery
10. **Mesh load balancer** — Route requests to node with model already loaded
11. **WireGuard mesh networking** — Encrypted inter-node communication

---

## Design Decisions

### DD-08.1: Userspace Network Stack (smoltcp) for Phase 12+, Linux Stack for Phase 1
**Decision:** In Phase 1, use the host Linux network stack (standard sockets). The custom smoltcp-based stack is a Phase 12 deliverable for the native kernel.
**Rationale:** The custom network stack is unnecessary overhead for containerized deployment. Linux's networking is mature and well-tested. smoltcp makes sense only when running on the bare SpiritKernel.

### DD-08.2: OpenAI API Compatibility as Primary External Interface
**Decision:** The REST API is fully compatible with the OpenAI `/v1/chat/completions` specification, including streaming (SSE), function calling, and embeddings endpoints.
**Rationale:** Maximum ecosystem compatibility. Any tool built for the OpenAI API (LangChain, LlamaIndex, Cursor, Continue, etc.) works with SpiritROS without modification.

### DD-08.3: gRPC for Internal and High-Performance Clients
**Decision:** Expose a gRPC API using `tonic` (Rust) with protobuf message definitions for typed, streaming inference.
**Rationale:** gRPC provides lower overhead than REST for streaming, strong typing, and bidirectional communication. Ideal for SDK clients and inter-service communication.

### DD-08.4: etcd for Cluster Coordination
**Decision:** Use etcd for distributed state: cluster membership, leader election, model placement metadata.
**Rationale:** etcd is battle-tested for exactly this use case (Kubernetes uses it). It provides strong consistency, watch notifications, and distributed locking.

### DD-08.5: NCCL/Gloo for GPU-to-GPU Transfers
**Decision:** Use NVIDIA NCCL for NVIDIA multi-GPU/multi-node tensor transfers; Gloo as the cross-platform fallback.
**Rationale:** NCCL is optimized for NVLink and InfiniBand, delivering near-hardware-limit bandwidth. Gloo works across vendors but is slower. Auto-detect available backends.

---

## Components & Tasks

### 1. REST API Server (`runtime/networking/rest.rs`)
```rust
// Using axum (Rust web framework)
pub fn api_router() -> Router {
    Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/models", get(list_models))

        // Spirit-native endpoints
        .route("/spirit/v1/sessions", post(create_session))
        .route("/spirit/v1/sessions/:id/snapshot", post(snapshot_kv))
        .route("/spirit/v1/sessions/:id/restore", post(restore_kv))
        .route("/spirit/v1/memory/status", get(memory_status))
        .route("/spirit/v1/scheduler/status", get(scheduler_status))

        // Health & metrics
        .route("/health", get(health_check))
        .route("/metrics", get(prometheus_metrics))
}
```

### 2. gRPC Service (`runtime/networking/grpc.rs`)
```protobuf
// proto/spirit_inference.proto
syntax = "proto3";
package spirit;

service Inference {
    rpc CreateSession(CreateSessionRequest) returns (CreateSessionResponse);
    rpc Prompt(PromptRequest) returns (stream TokenResponse);
    rpc BatchPrompt(BatchPromptRequest) returns (stream BatchTokenResponse);
    rpc SnapshotKv(SnapshotRequest) returns (SnapshotResponse);
    rpc RestoreKv(RestoreRequest) returns (RestoreResponse);
    rpc LoadModel(LoadModelRequest) returns (LoadModelResponse);
    rpc UnloadModel(UnloadModelRequest) returns (UnloadModelResponse);
    rpc ListModels(ListModelsRequest) returns (ListModelsResponse);
}

message PromptRequest {
    string session_id = 1;
    repeated int32 tokens = 2;
    InferenceConfig config = 3;
}

message TokenResponse {
    int32 token = 1;
    string text = 2;
    bool is_final = 3;
    TokenMetadata metadata = 4;
}
```

### 3. WebSocket Endpoint (`runtime/networking/websocket.rs`)
- Upgrade path: `GET /ws` → WebSocket connection
- Message format: JSON (compatible with common web chat UIs)
- Supports session management, streaming tokens, and system notifications
- Heartbeat/keepalive for connection health

### 4. Unix Domain Socket Server
- Listen on `/run/spirit/infer.sock`
- Protocol: length-prefixed MessagePack or protobuf
- Used by: Spirit Shell, SDK clients, SpiritFS indexer
- Zero network overhead for local processes

### 5. Shared Memory IPC (`/dev/shm/spirit_infer`)
- Ring buffer per session for zero-copy token transfer
- SPSC (single-producer, single-consumer) lock-free queues
- Memory-mapped segments created on session creation, cleaned up on session destroy
- Fastest path: ~100ns per token transfer vs. ~10μs for UDS

### 6. API Authentication & Authorization
- **API keys:** Bearer token in `Authorization` header, keys stored in `/etc/spirit/api-keys.toml`
- **mTLS:** Client certificates verified against Spirit CA
- **Capability checking:** Each API request mapped to required capabilities; rejected if not granted
- **Rate limiting:** Token bucket per API key, configurable limits per endpoint

### 7. Distributed Inference (Phase 3)
```
Node 0 (Coordinator)              Node 1 (Worker)
┌─────────────────┐              ┌─────────────────┐
│ Layers 0-15     │  ←─RDMA──►  │ Layers 16-31    │
│ GPU 0           │              │ GPU 0           │
│ KV-cache (L0-15)│              │ KV-cache (L16+) │
└─────────────────┘              └─────────────────┘
```

- **Cluster discovery:** etcd for membership, mDNS for local network
- **Model sharding:** Coordinator decides how to split model across nodes based on available VRAM
- **Tensor transfer:** NCCL over NVLink/InfiniBand, Gloo over TCP fallback
- **Spirit Wire Protocol:** Custom protocol for inference coordination (prompt routing, KV-cache sync, token aggregation)

### 8. Spirit Mesh (Phase 3)
- Self-organizing federation of SpiritROS nodes
- WireGuard tunnels for encrypted communication
- Mesh-aware load balancer: route to node with model already loaded
- Model pinning: "deepseek-r1-671b runs only on Node 3 with 8xA100"
- Health checks and automatic failover

---

## Acceptance Criteria

### Phase 1
- [ ] REST API serves `/v1/chat/completions` compatible with OpenAI Python client
- [ ] `curl` to REST API with streaming produces SSE token stream
- [ ] gRPC service is callable via `grpcurl` and returns streamed tokens
- [ ] WebSocket endpoint connects and streams tokens to a browser client
- [ ] Unix domain socket accepts connections and processes inference requests
- [ ] Shared memory IPC demonstrates <1μs token transfer latency
- [ ] API key authentication rejects unauthorized requests
- [ ] Rate limiting returns 429 when exceeded
- [ ] Prometheus metrics endpoint exposes request counts, latency histograms, error rates
- [ ] API latency overhead (REST vs. direct inference) is <5ms for non-streaming requests

### Phase 3
- [ ] Two-node tensor parallelism splits a 70B model and produces correct output
- [ ] Spirit Mesh discovers peer nodes via mDNS within 5 seconds
- [ ] Load balancer routes requests to node with model loaded (avoids cold start)
- [ ] WireGuard tunnel established automatically between mesh nodes
- [ ] Failover works: if a worker node goes down, coordinator falls back to single-node inference

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| OpenAI API spec changes break compatibility | Client libraries fail | Pin to a specific API version; test against `openai` Python package in CI |
| gRPC code generation complexity | Developer friction | Pre-generate protobuf Rust code; include in repo; document regeneration |
| Shared memory IPC bugs (race conditions) | Data corruption | Extensive testing; use proven SPSC queue implementations (e.g., `rtrb` crate) |
| Distributed inference coordination failures | Multi-node unusable | Extensive integration tests; fallback to single-node; circuit breaker pattern |
| WireGuard kernel module not available in container | Mesh networking broken | Use `wireguard-go` (userspace implementation) as fallback |

---

## Key References

- Architecture Plan §8 (Networking & Distributed Inference)
- Architecture Plan §8.1 (Network Stack)
- Architecture Plan §8.2 (Inference API Endpoints)
- Architecture Plan §8.3 (Distributed Inference — Multi-Node)
- Architecture Plan §8.4 (Federation — Spirit Mesh)
- Architecture Plan §10.6 (Network Security)
