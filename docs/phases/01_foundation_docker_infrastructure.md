# Phase 01: Project Foundation & Docker Infrastructure

**Roadmap Alignment:** Phase 1 — Virtual Proof of Concept, Milestone M1.1
**Estimated Duration:** Month 1
**Dependencies:** None (this is the starting phase)
**Architecture Plan Sections:** §2, §13, §15, §16, Appendix D

---

## Context & Overview

This is the bootstrapping phase. Before any SpiritROS subsystem can be built, we need a working project structure, build system, CI/CD pipeline, and a Docker-based deployment environment that will serve as the foundation for all Phase 1 (Virtual PoC) development.

SpiritROS Phase 1 runs entirely inside a Docker container on a Linux host. The container provides GPU passthrough (via NVIDIA Container Toolkit or ROCm Docker), a controlled environment for the Spirit userspace shims, and a reproducible build for all subsequent phases.

This phase establishes the "empty shell" that every other phase fills in.

---

## Goals

1. **Repository scaffolding** — Establish the canonical directory structure from Appendix D
2. **Build system** — `just` (task runner) + Cargo workspace for all Rust components
3. **Docker image** — Base `Dockerfile.spiritros-phase1` that builds and runs
4. **Docker Compose stack** — Full stack with Qdrant, etcd, Prometheus, Grafana
5. **CI/CD pipeline** — Automated build, test, lint on every push
6. **Basic Spirit Shell stub** — Entrypoint binary that starts and accepts input
7. **Ollama pre-installed** — Ollama available inside the container with GPU access
8. **Documentation scaffolding** — mdBook structure for project docs

---

## Design Decisions

### DD-01.1: Rust Workspace Structure
**Decision:** Use a single Cargo workspace at the repo root with member crates for each subsystem (kernel, runtime/shell, runtime/spiritfs, runtime/inference, sdk/rust, benchmarks).
**Rationale:** A unified workspace ensures consistent dependency versions, enables cross-crate testing, and simplifies CI. Each crate remains independently compilable.

### DD-01.2: Task Runner — `just` over Make
**Decision:** Use `just` as the primary task runner.
**Rationale:** `just` has cleaner syntax than Make, supports arguments natively, and is Rust-native (aligns with project language). Recipes will wrap cargo, docker, and test commands.

### DD-01.3: Base Docker Image — nvidia/cuda:12.8-devel-ubuntu24.04
**Decision:** Use NVIDIA's CUDA development image based on Ubuntu 24.04 LTS.
**Rationale:** This provides CUDA toolkit, GPU drivers, and a stable Ubuntu base. ROCm support will be an alternative Dockerfile target for AMD GPUs.

### DD-01.4: CI/CD — Forgejo Actions (self-hosted) with GitHub Actions fallback
**Decision:** Primary CI on Forgejo Actions; mirror config for GitHub Actions.
**Rationale:** Keeps CI close to the project's self-hosted infrastructure while maintaining compatibility with GitHub for community contributors.

### DD-01.5: Container Privileges — Minimal Required Set
**Decision:** Use `SYS_ADMIN` (for FUSE), `NET_ADMIN` (for WireGuard), and `seccomp:unconfined` (for syscall shim). No `--privileged`.
**Rationale:** Balances functionality requirements with security. Full `--privileged` mode is avoided to practice the security-first philosophy even in development.

---

## Components & Tasks

### 1. Repository Structure
```
spiritros/
├── kernel/                  # SpiritKernel (Phase 12)
│   ├── src/
│   └── Cargo.toml
├── runtime/                 # Spirit Runtime (userspace)
│   ├── shell/               # Spirit Shell
│   ├── spiritfs/            # Semantic filesystem
│   ├── inference/           # Inference coordinator
│   ├── compat/              # Compatibility layers
│   ├── security/            # Spirit Guard, capabilities
│   ├── networking/          # Network stack
│   └── memory/              # Memory manager shim
├── sdk/                     # Spirit SDK
│   ├── rust/
│   ├── python/
│   └── typescript/
├── docker/                  # Dockerfiles and compose
│   ├── Dockerfile.spiritros-phase1
│   ├── Dockerfile.rocm      # AMD GPU variant
│   └── docker-compose.yml
├── config/                  # System configuration
│   ├── spirit.yaml           # Default Spirit config
│   ├── spirit-cgroups.sh     # cgroups setup script
│   └── prometheus.yml        # Monitoring config
├── patches/                 # Third-party patches
│   └── llama-cpp-spirit.patch
├── docs/                    # mdBook documentation
│   ├── book.toml
│   └── src/
├── benchmarks/              # spirit-bench
│   └── Cargo.toml
├── contrib/                 # Community plugins
├── Cargo.toml               # Workspace root
├── justfile                 # Task runner recipes
├── .forgejo/workflows/      # CI/CD
├── .github/workflows/       # GitHub Actions mirror
└── README.md
```

### 2. Cargo Workspace Setup
- Root `Cargo.toml` defining workspace members
- Stub `lib.rs` / `main.rs` for each crate (compiles but does nothing yet)
- Shared workspace dependency versions for: `tokio`, `serde`, `tracing`, `clap`, `tonic` (gRPC)

### 3. Docker Image Build
- Write `Dockerfile.spiritros-phase1` per §13.2
- Install system deps, Rust toolchain, Ollama
- Stub Spirit Shell binary as entrypoint
- Verify GPU passthrough works (`nvidia-smi` inside container)

### 4. Docker Compose Stack
- `spiritros` main container (GPU-enabled)
- `qdrant` for vector search (SpiritFS)
- `etcd` for cluster coordination
- `prometheus` + `grafana` for monitoring
- Named volumes for models, data, logs

### 5. CI/CD Pipeline
- Build all Rust crates
- Run `cargo clippy` and `cargo fmt --check`
- Run unit tests (`cargo test`)
- Build Docker image
- (Future: integration test stage)

### 6. Spirit Shell Stub
- Minimal Rust binary using `rustyline` or `reedline` for line editing
- Prints welcome banner with version
- Accepts input, echoes it back (placeholder for Phase 06)
- Serves as container entrypoint

### 7. justfile Recipes
```just
build:           cargo build --release
test:            cargo test
lint:            cargo clippy -- -D warnings && cargo fmt --check
docker-build:    docker compose build
docker-up:       docker compose up -d
docker-shell:    docker compose exec spiritros spirit-shell
bench:           cargo run --release -p spirit-bench
docs:            mdbook build docs/
```

---

## Acceptance Criteria

- [ ] `just build` compiles all workspace crates without errors or warnings
- [ ] `just test` runs and all stub tests pass
- [ ] `just lint` passes clippy and fmt checks
- [ ] `just docker-build` produces a working Docker image
- [ ] `just docker-up` starts the full compose stack (spiritros + qdrant + etcd + monitoring)
- [ ] `docker compose exec spiritros nvidia-smi` shows GPU(s) inside container
- [ ] `docker compose exec spiritros ollama --version` returns successfully
- [ ] `docker compose exec spiritros spirit-shell` starts the shell stub
- [ ] CI pipeline runs on push and reports pass/fail
- [ ] `mdbook build docs/` produces a documentation site skeleton
- [ ] Repository has LICENSE (Apache-2.0), README.md, and CONTRIBUTING.md

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| NVIDIA Container Toolkit setup issues | Blocks GPU testing | Document manual GPU setup; provide CPU-only fallback Dockerfile |
| Cargo workspace compilation conflicts | Slows all development | Pin shared deps early; use `workspace.dependencies` in root Cargo.toml |
| Docker image size too large (>10GB) | Slow CI, slow pulls | Multi-stage build; separate dev and runtime images |
| ROCm Docker support lagging | AMD GPU users blocked | AMD variant is a stretch goal; focus NVIDIA first |

---

## Key References

- Architecture Plan §2 (System Architecture Overview)
- Architecture Plan §13 (Virtualized Deployment)
- Architecture Plan §15 (Open Source Stack & Dependencies)
- Architecture Plan §16 (Roadmap — M1.1)
- Architecture Plan Appendix D (Contributing / Repo Structure)
