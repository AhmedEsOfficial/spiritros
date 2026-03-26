# SpiritROS

**The first operating system designed from the ground up for LLM inference.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

SpiritROS treats model weights, KV-caches, and inference sessions as first-class operating system resources. Instead of running inference workloads on a general-purpose OS that was designed for web servers and desktop applications, SpiritROS provides inference-aware memory management, scheduling, and security from the ground up. The result is measurably faster inference with stronger isolation guarantees.

## Feature Highlights

- **TensorFirst Memory Zones** -- Dedicated memory zones for model weights, KV-cache, compute scratch, and general processes. Inference memory is protected from OOM eviction.
- **Inference-Aware Scheduler** -- Decode threads get `SCHED_FIFO` priority on dedicated CPU cores. Background tasks never steal cycles from token generation.
- **KV-Cache as a Resource** -- Copy-on-write KV-cache forking, page-granular eviction, and cross-session sharing managed at the OS level.
- **Semantic Filesystem (SpiritFS)** -- FUSE-backed filesystem with automatic embedding and meaning-based search. Find files by what they contain, not what they are named.
- **Spirit Shell** -- LLM-native CLI that understands natural language alongside traditional commands.
- **Capability-Based Security** -- No ambient authority. GPU access, inference sessions, and agentic operations each require explicit capabilities.

## Quick Start

```bash
git clone https://github.com/spiritros/spiritros.git
cd spiritros
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml exec spiritros spirit-shell
```

## Project Structure

```
spiritros/
├── kernel/          # SpiritKernel (Phase 2+, Rust no_std)
├── runtime/         # Spirit Runtime (userspace components)
│   ├── shell/       # Spirit Shell (LLM-native CLI)
│   ├── spiritfs/    # Semantic filesystem (FUSE)
│   ├── inference/   # Inference coordinator + Ollama integration
│   ├── memory/      # Memory zone manager (cgroups shim)
│   ├── scheduler/   # Scheduler shim (cpuset/nice)
│   ├── security/    # Spirit Guard (capability manager)
│   └── ...
├── sdk/             # Spirit SDK (Rust, Python, TypeScript)
├── docker/          # Phase 1 Docker deployment
├── benchmarks/      # spirit-bench performance suite
├── docs/            # mdBook documentation
└── tests/           # Integration and end-to-end tests
```

## Development Setup

```bash
# Install Rust toolchain
rustup default stable
rustup component add rustfmt clippy

# Install development tools
cargo install cargo-watch cargo-nextest cargo-criterion just

# Build
just build

# Run tests
just test

# Lint
just lint

# Start Docker environment
just docker-up
```

See the [justfile](justfile) for all available recipes.

## Phase 1 Status

SpiritROS is currently in **Phase 1** (Virtualized Deployment). The system runs inside a Docker container on a Linux host, using Ollama as the inference backend. Kernel-level features are approximated with userspace shims (cgroups v2, FUSE, cpuset). The goal is to prove 30-60% inference performance improvement over stock Linux + Ollama while building the userspace tooling that persists into later phases.

## Documentation

Build and serve the documentation locally:

```bash
just docs-serve
```

Or read the [architecture plan](docs/SpiritROS_Architecture_Plan.md) directly.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to SpiritROS.

## License

SpiritROS is licensed under the [Apache License 2.0](LICENSE).
