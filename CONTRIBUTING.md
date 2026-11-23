# Contributing to SpiritROS

Thank you for your interest in contributing to SpiritROS. This document covers the process and conventions for contributing to the project.

## Getting Started

1. Fork the repository and clone your fork.
2. Create a feature branch from `main` using the naming convention below.
3. Make your changes, following the code style and testing requirements.
4. Submit a pull request against `main`.

## Development Setup

```bash
# Rust toolchain
rustup default stable
rustup component add rustfmt clippy rust-src
cargo install cargo-watch cargo-nextest cargo-criterion cargo-deny just

# Python toolchain (for SDK bindings and tooling)
pip install maturin pyo3 pytest ruff black pydantic

# System tools
# Docker with nvidia-container-toolkit, Ollama, hyperfine
```

## Branch Naming

Use the format `<type>/<short-description>`:

- `feat/spiritfs-embedding-index`
- `fix/memory-zone-overflow`
- `perf/kv-cache-cow`
- `refactor/inference-coordinator-cleanup`
- `docs/architecture-update`
- `test/scheduler-integration`

## Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:** `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`, `ci`

**Scopes:** `kernel`, `runtime`, `shell`, `spiritfs`, `inference`, `memory`, `scheduler`, `security`, `sdk`, `docker`, `bench`

**Examples:**

```
feat(spiritfs): add semantic search via embedding similarity
fix(memory): prevent ZONE_MODEL eviction under OOM pressure
perf(inference): eliminate allocations in token decode loop
docs(architecture): update memory zone diagram for Phase 1
```

## Code Style

### Rust

- Run `cargo fmt --workspace` before committing.
- Run `cargo clippy --workspace --all-targets -- -D warnings` and fix all warnings.
- No `unwrap()` in library code. Use `?` propagation or explicit error handling. `unwrap()` is allowed in tests.
- Use `thiserror` for library errors, `anyhow` for binary errors.
- Prefer `&str` over `String` in function signatures when ownership is not needed.
- Use `Arc<T>` for shared ownership across threads, never `Rc<T>`.

### Python

- Python 3.11+ with type hints everywhere.
- Format with `black`, lint with `ruff`.
- Test with `pytest`.

### Documentation

- Every public function, struct, and module must have a doc comment (`///`).
- Architecture decisions are recorded in `docs/src/appendices/adr.md`.

Architecture and phase notes live in [`docs/`](docs/).

## Testing Requirements

### Unit Tests

Every module should have `#[cfg(test)]` tests. Target 80% coverage on core logic.

### Test Naming

```rust
#[test]
fn test_<module>_<scenario>_<expected_outcome>() { }
```

For example: `test_memory_zone_allocation_respects_model_limit()`

### Integration Tests

Place integration tests in each crate's `tests/` directory. These test cross-module interactions.

### Benchmarks

If your change touches the inference path, include before/after benchmark numbers in the PR description. Use `just bench` to run the benchmark suite. Benchmark numbers must be reproducible -- include hardware spec, model used, quantization, batch size, and the exact command to reproduce.

## Pull Request Process

1. Ensure all tests pass: `just test`
2. Ensure linting passes: `just lint`
3. If the change is performance-sensitive, include benchmark numbers.
4. If the change modifies the architecture, update `docs/SpiritROS_Architecture_Plan.md` in the same PR.
5. Request review from a maintainer.

## Phase Awareness

SpiritROS is currently in Phase 1. This means:

- Do not write kernel code unless explicitly working on Phase 2 features.
- Everything runs inside Docker on a Linux host.
- Kernel-level features are approximated with userspace shims (cgroups, FUSE, cpuset).
- Mark Phase 1 simplifications with `// PHASE1-SHIM: Replace with kernel-native implementation in Phase 2`.

## License

By contributing to SpiritROS, you agree that your contributions will be licensed under the Apache License 2.0.
