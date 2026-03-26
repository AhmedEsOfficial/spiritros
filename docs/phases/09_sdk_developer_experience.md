# Phase 09: SDK & Developer Experience

**Roadmap Alignment:** Phase 1 — Milestone M1.5
**Estimated Duration:** Month 5
**Dependencies:** Phases 01–08 (all core subsystems needed for SDK to expose)
**Architecture Plan Sections:** §12

---

## Context & Overview

The Spirit SDK is the developer-facing surface of SpiritROS. It provides native language bindings (Rust, Python, TypeScript) that wrap the inference syscalls (or their Phase 1 IPC equivalents) into ergonomic, idiomatic APIs. The SDK is how application developers interact with SpiritROS's unique capabilities: kernel-level inference, KV-cache snapshots, semantic filesystem operations, and more.

This phase also delivers the **Spirit Plugin System** (WebAssembly-based sandboxed extensions) and **Spirit Compose** (declarative YAML system configuration).

---

## Goals

1. **Rust SDK** — Primary, zero-overhead bindings to all Spirit APIs
2. **Python SDK** — PyO3-based bindings for the data science and ML ecosystem
3. **TypeScript/JavaScript SDK** — Neon-based bindings for web and Node.js developers
4. **Spirit Plugin System** — Sandboxed WebAssembly plugins with capability-limited access
5. **Spirit Compose** — Declarative YAML for system configuration (models, services, memory, security)
6. **API documentation** — OpenAPI spec for REST, protobuf docs for gRPC, mdBook for SDK guides
7. **Example applications** — Reference implementations demonstrating SDK usage

---

## Design Decisions

### DD-09.1: Rust as Primary SDK, Others as Bindings
**Decision:** The Rust SDK (`spirit-sdk`) is the canonical implementation. Python and TypeScript SDKs are generated bindings wrapping the Rust core.
**Rationale:** Single source of truth prevents API divergence. Rust provides zero-overhead access to Spirit IPC. PyO3 and Neon generate native extensions with minimal overhead.

### DD-09.2: PyO3 for Python Bindings
**Decision:** Use PyO3 to expose the Rust SDK to Python as a native extension module.
**Rationale:** PyO3 is the standard for Rust→Python bindings. It supports async Python (via `pyo3-asyncio`), which is essential for streaming inference. The Python ML ecosystem (LangChain, transformers, etc.) expects Python-native packages.

### DD-09.3: Neon for TypeScript/JavaScript Bindings
**Decision:** Use Neon (Node.js native module toolkit for Rust) for the TypeScript SDK.
**Rationale:** Neon compiles Rust to a Node.js native addon. This is more performant than a pure-JS client hitting the REST API, and supports streaming via Node.js async iterators.

### DD-09.4: WebAssembly Plugins via Wasmtime
**Decision:** Third-party plugins are sandboxed WASM modules executed by Wasmtime with capability-limited host functions.
**Rationale:** WASM provides portable, sandboxed execution. Plugins can be written in any language that compiles to WASM (Rust, C, Go, AssemblyScript). Wasmtime is the most mature WASM runtime for server-side use.

### DD-09.5: Spirit Compose is YAML, Not Code
**Decision:** System configuration is declarative YAML, not imperative scripts.
**Rationale:** Declarative config is diffable, version-controllable, and reproducible. It clearly separates "what" (configuration) from "how" (implementation). It also enables validation tooling.

---

## Components & Tasks

### 1. Rust SDK (`sdk/rust/`)
```rust
// spirit-sdk crate
use spirit_sdk::{InferenceSession, Model, Config, Spirit};

// Initialize Spirit connection
let spirit = Spirit::connect()?;  // Connects via UDS or shared memory

// Load model
let model = spirit.load_model("llama-3.1-8b-instruct")?;

// Create session with configuration
let session = InferenceSession::new(&model, Config {
    context_length: 8192,
    temperature: 0.7,
    top_p: 0.9,
    ..Default::default()
})?;

// Streaming inference
let stream = session.prompt("Explain quantum computing")?;
for token in stream {
    print!("{}", token?);
}

// KV-cache snapshot for conversation branching
let snapshot = session.snapshot()?;
let branch = session.restore(&snapshot)?;
let alt = branch.prompt("Now explain it to a 5-year-old")?;

// Semantic filesystem
let results = spirit.fs().search("quarterly revenue report Q3")?;
let similar = spirit.fs().similar("./architecture.md")?;
let summary = spirit.fs().summarize("/var/log/")?;

// Memory info
let zones = spirit.memory().zone_stats()?;
let vram = spirit.memory().vram_stats(GpuDevice::Gpu(0))?;
```

**SDK modules:**
- `spirit_sdk::inference` — Session management, prompting, streaming, KV-cache
- `spirit_sdk::model` — Model loading, listing, registry
- `spirit_sdk::memory` — Zone stats, VRAM stats, rebalancing
- `spirit_sdk::scheduler` — Phase info, priority management
- `spirit_sdk::fs` — Semantic search, similar, summarize, tags
- `spirit_sdk::security` — Capability management, sandbox info
- `spirit_sdk::config` — System configuration reading

### 2. Python SDK (`sdk/python/`)
```python
import spirit

# Connect to Spirit
s = spirit.connect()

# Load and infer
model = s.load_model("llama-3.1-8b-instruct")
session = spirit.InferenceSession(model, context_length=8192, temperature=0.7)

# Streaming
for token in session.prompt("Explain quantum computing"):
    print(token, end="", flush=True)

# KV-cache branching
snapshot = session.snapshot()
branch = session.restore(snapshot)

# Async support
async for token in session.prompt_async("Hello"):
    print(token, end="")

# Semantic filesystem
results = s.fs.search("quarterly revenue report")
similar = s.fs.similar("./architecture.md")

# Memory
zones = s.memory.zone_stats()
print(f"Model zone: {zones.model.used_gb:.1f} / {zones.model.limit_gb:.1f} GB")
```

**Build:** PyO3 + maturin for package building and distribution.

### 3. TypeScript/JavaScript SDK (`sdk/typescript/`)
```typescript
import { Spirit } from '@spirit-os/sdk';

const spirit = await Spirit.connect();

const model = await spirit.loadModel('llama-3.1-8b-instruct');
const session = model.createSession({ contextLength: 8192, temperature: 0.7 });

// Async iterator streaming
for await (const token of session.prompt('Explain quantum computing')) {
    process.stdout.write(token);
}

// KV-cache branching
const snapshot = await session.snapshot();
const branch = await session.restore(snapshot);

// Semantic filesystem
const results = await spirit.fs.search('quarterly revenue report');
```

**Build:** Neon for native bindings, published as npm package.

### 4. Spirit Plugin System
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

**Plugin host interface:**
```rust
pub struct PluginHost {
    runtime: wasmtime::Engine,
    plugins: Vec<LoadedPlugin>,
}

impl PluginHost {
    pub fn load_plugin(&mut self, manifest: &Path) -> Result<PluginId>;
    pub fn unload_plugin(&mut self, id: PluginId) -> Result<()>;

    // Hook dispatch
    pub fn on_infer_pre(&self, session: &SessionId, prompt: &mut String) -> Result<()>;
    pub fn on_infer_post(&self, session: &SessionId, output: &mut String) -> Result<()>;
    pub fn on_file_write(&self, path: &Path) -> Result<()>;
}
```

- Plugins communicate with the host via WASM imports/exports
- Host functions are capability-checked before execution
- Plugin memory is isolated (WASM linear memory)
- Plugins can be written in Rust, C, Go, or any WASM-targeting language

### 5. Spirit Compose
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
    gpu: [0, 1]

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
  compat_layer_limit: 4_cores
  gpu_exclusive: true

memory:
  zone_model: 40G
  zone_kv: 12G
  zone_tensor: 4G
  zone_general: 8G

security:
  model_signing: required
  audit_log: /var/log/spirit/audit.log
  sandbox:
    max_output_tokens: 8192
    blocked_patterns: ["API_KEY", "SECRET", "password"]
```

**Compose CLI:**
```bash
spirit compose validate spirit-compose.yaml     # Validate config
spirit compose apply spirit-compose.yaml        # Apply configuration
spirit compose diff spirit-compose.yaml         # Show changes vs. running config
spirit compose export > current-config.yaml     # Export current running config
```

### 6. Documentation
- **OpenAPI spec** for REST API (auto-generated via `utoipa` crate)
- **Protobuf docs** for gRPC API
- **mdBook** SDK guides with examples for each language
- **API playground** via Scalar (interactive API explorer)

### 7. Example Applications
- `examples/chat-cli/` — Simple chat CLI using Rust SDK
- `examples/rag-python/` — RAG pipeline using Python SDK + SpiritFS
- `examples/web-chat/` — Web chat UI using TypeScript SDK + WebSocket
- `examples/multi-model/` — Model routing/switching using SDK
- `examples/kv-branching/` — Conversation branching demonstration

---

## Acceptance Criteria

- [ ] Rust SDK compiles and all public APIs are documented with `cargo doc`
- [ ] Python SDK installs via `pip install spirit-sdk` and basic inference works
- [ ] TypeScript SDK installs via `npm install @spirit-os/sdk` and basic inference works
- [ ] All three SDKs demonstrate streaming inference, KV-cache snapshot/restore, and semantic search
- [ ] Plugin system loads a sample WASM plugin and executes `on_infer_pre` hook
- [ ] Plugin capabilities are enforced (plugin requesting filesystem outside granted path is denied)
- [ ] `spirit compose validate` catches configuration errors
- [ ] `spirit compose apply` configures the system from YAML
- [ ] OpenAPI spec is generated and accessible at `/docs`
- [ ] mdBook documentation builds and covers all SDK features
- [ ] All example applications run successfully

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| PyO3 async support is complex | Python streaming broken | Test thoroughly; provide sync fallback API |
| WASM plugin ecosystem adoption is slow | Plugin system unused | Ship useful first-party plugins; make plugin development easy with templates |
| SDK API churn as subsystems evolve | Breaking changes for developers | Semver strictly; stabilize core APIs first; deprecation warnings before removal |
| Spirit Compose schema grows unwieldy | Config becomes intimidating | Sensible defaults for everything; minimal config for simple setups; validation with helpful errors |
| TypeScript SDK Neon build complexity | npm install fails on some platforms | Provide pre-built binaries for Linux/macOS/Windows; fallback to pure-JS REST client |

---

## Key References

- Architecture Plan §12 (Developer Experience & SDK)
- Architecture Plan §12.1 (Spirit SDK)
- Architecture Plan §12.2 (Spirit Plugin System)
- Architecture Plan §12.3 (Spirit Compose — Declarative System Configuration)
