# SpiritROS

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Status](https://img.shields.io/badge/status-experimental-orange)
![Rust](https://img.shields.io/badge/rust-1.87%2B-informational)

SpiritROS is an experiment in operating-system support for local LLM inference: treating model weights, KV-cache and inference sessions as managed system resources rather than ordinary process memory. The long-term vision (see [the architecture plan](docs/SpiritROS_Architecture_Plan.md)) is a purpose-built microkernel. **Today the project is a Rust workspace of userspace daemons and libraries, running on stock Linux and using [Ollama](https://ollama.com) as the inference backend**, that approximate those kernel-level ideas with cgroups v2, `SCHED_FIFO`/cpuset, FUSE and Unix-socket IPC.

> **Status: early-stage / experimental.** No performance claims are made. The repository contains no benchmark results; the `spirit-bench` suites are placeholders (see below).

## What works today vs. roadmap

| Component | Crate | Status |
|---|---|---|
| Spirit Shell: interactive CLI (reedline) that resolves input as a Unix command or a natural-language query sent to Ollama; pipes, completion, highlighting, config | `runtime/shell` | Implemented; unit-tested. Needs a running Ollama for the LLM path. `-c "echo hello"` works without it. |
| Inference coordinator (`spirit-infer`): session manager, priority batch scheduler, model registry, Ollama backend, Unix-socket and shared-memory (SPSC ring buffer) IPC, Prometheus metrics | `runtime/inference` | Implemented; unit-tested. End-to-end behaviour against a live Ollama is not covered by automated tests. |
| Memory zones (model / KV / tensor / general): zone accounting, cgroups v2 writer, hugepage and VRAM tracking, KV-page manager, pressure monitor | `runtime/memory` | Library implemented; unit-tested against a fake cgroup tree. Not wired into a daemon; real cgroup enforcement is unverified. |
| Scheduler daemon (`spirit-scheduler`): phase hints (prefill/decode/...), cpuset/topology, `SCHED_FIFO`, starvation watchdog, IPC | `runtime/scheduler` | Implemented; unit-tested. Real-time scheduling needs `CAP_SYS_NICE`. |
| SpiritFS: content-addressed store, metadata, embedding pipeline, semantic index (Qdrant), FUSE layer | `runtime/spiritfs` | Implemented; unit-tested. FUSE mount is behind the `fuse` feature (needs `libfuse3-dev`) and is not exercised in CI. |
| Spirit Guard (capabilities / audit) | `runtime/security` | Types and policy logic only. The `spirit-guard` daemon is a stub that prints a message and exits. |
| Networking (HTTP/gRPC API, mesh) | `runtime/networking` | Config and type definitions only. |
| Linux compat layer | `runtime/compat` | Type definitions only. |
| Rust SDK | `sdk/rust` | Types and config; `infer` returns a "not implemented" error. |
| Benchmarks (`spirit-bench`) | `benchmarks` | CLI skeleton; suites log "not yet implemented". No results exist. |
| SpiritKernel (`no_std`) | `kernel` | Placeholder crate (version constant and boot-info type). Planned for a later phase. |
| Python / TypeScript SDKs, mdBook docs content | - | Planned. Several mdBook pages are still stubs. |

The roadmap is in [`docs/phases/`](docs/phases/00_PHASE_INDEX.md); the plan's performance targets are goals, not measurements.

## Architecture

```
                 spirit-shell  (NL + Unix commands)       SDK / API clients
                       │                                        │
                       ▼                                        ▼
              ┌──────────────────────────────────────────────────────┐
              │ spirit-infer  — inference coordinator                │
              │  sessions · priority batching · model registry       │
              │  IPC: Unix socket (control) + shm ring (tokens)      │
              └───────┬───────────────────┬──────────────────────────┘
                      │ HTTP              │ hints / registration
                      ▼                   ▼
                   Ollama         spirit-scheduler   (cpuset, SCHED_FIFO)
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
  spirit-memory (zones via        SpiritFS (FUSE + CAS +
  cgroups v2, hugepages, VRAM)    embeddings → Qdrant)

  Phase 1 deployment: all of the above in one Docker container on a Linux host,
  next to Qdrant, etcd, Prometheus and Grafana (docker/docker-compose.yml).
```

## Repository layout

```
kernel/              SpiritKernel placeholder (no_std)
runtime/shell/       spirit-shell binary + library
runtime/inference/   spirit-infer daemon + library
runtime/memory/      memory-zone manager (library)
runtime/scheduler/   spirit-scheduler daemon + library
runtime/spiritfs/    spiritfs binary (feature "fuse") + library
runtime/security/    capability types + spirit-guard stub
runtime/networking/  API/mesh types
runtime/compat/      compat-layer types
sdk/rust/            Rust client SDK (skeleton)
benchmarks/          spirit-bench (skeleton) and criterion bench stubs
docker/              Dockerfile(s), compose stack, Prometheus/Grafana config
docs/                architecture plan, per-phase plans, mdBook sources
```

## Build, test, run

Requires Rust 1.87+ (`rustup`). Everything below was run on Linux with Rust 1.91.

```bash
git clone https://github.com/AhmedEsOfficial/spiritros.git
cd spiritros

cargo build --workspace
cargo test --workspace                                # all unit/doc tests pass
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings  # clean

cargo run -p spirit-bench -- list                      # lists the (placeholder) suites
cargo run -p spirit-shell -- -c "echo hello"           # non-interactive
cargo run -p spirit-shell                              # interactive; set OLLAMA_HOST for the LLM path
```

`just build | test | lint` wrap the same commands (see the [justfile](justfile)).

Not verified here: the FUSE build (`just spiritfs-build`, needs `libfuse3-dev`), the GPU path, and the Docker stack (`just docker-up`; needs Docker, and `nvidia-container-toolkit` for GPU, otherwise CPU only). Copy [`.env.example`](.env.example) to `docker/.env` and change `SPIRIT_API_KEY` and the Grafana password before running the stack anywhere other than localhost. Details in [`docs/debug.md`](docs/debug.md).

## Design decisions worth highlighting

- **Userspace shims first, kernel later.** Each kernel-level idea (memory zones, inference-aware scheduling, semantic FS) is first built as a daemon over Linux primitives, so the userspace tooling carries over to a later kernel. Shims are marked `PHASE1-SHIM` in the code.
- **Pluggable inference backend.** The coordinator talks to engines through a trait; Ollama is the first implementation.
- **Zero-allocation token hot path (design contract).** Session buffers are pre-allocated, and tokens cross processes through a lock-free SPSC ring of fixed 16-byte entries.
- **Priority classes in batching and scheduling**, so interactive decode is not starved by background work.
- **Capability-based security as an invariant**: no ambient authority, loopback-only binding by default. Currently enforced only at the type/policy level.
- **Strict lints**: `clippy::pedantic` is enabled per crate and CI runs with `-D warnings`.

## Documentation

[Architecture plan](docs/SpiritROS_Architecture_Plan.md) (vision document), [phase plans](docs/phases/00_PHASE_INDEX.md), and mdBook sources in `docs/src` (`just docs`, requires `mdbook`).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Licensed under the [Apache License 2.0](LICENSE).
