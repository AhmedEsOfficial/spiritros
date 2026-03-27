# SpiritROS — LLM-Native Robot Operating System
# Task runner recipes for development, testing, and deployment.

# Default recipe
default: build

# Build all crates
build:
    cargo build --workspace

# Build release
build-release:
    cargo build --workspace --release

# Run all tests
test:
    cargo test --workspace

# Run specific crate tests
test-crate crate:
    cargo test -p {{crate}}

# Lint
lint:
    cargo clippy --workspace -- -D warnings
    cargo fmt --all --check

# Format code
fmt:
    cargo fmt --all

# Docker build
docker-build:
    docker compose -f docker/docker-compose.yml build

# Docker up
docker-up:
    docker compose -f docker/docker-compose.yml up -d

# Docker down
docker-down:
    docker compose -f docker/docker-compose.yml down

# Docker shell
docker-shell:
    docker compose -f docker/docker-compose.yml exec spiritros spirit-shell

# Docker logs
docker-logs:
    docker compose -f docker/docker-compose.yml logs -f spiritros

# Build SpiritFS with FUSE support
spiritfs-build:
    cargo build -p spirit-spiritfs --features fuse

# Test SpiritFS
spiritfs-test:
    cargo test -p spirit-spiritfs

# Lint SpiritFS
spiritfs-lint:
    cargo clippy -p spirit-spiritfs -- -D warnings

# Docker SpiritFS logs
docker-spiritfs-logs:
    docker compose -f docker/docker-compose.yml exec spiritros cat /var/log/spiritros/spiritfs.log

# Run benchmarks
bench:
    cargo run --release -p spirit-bench

# Build docs
docs:
    mdbook build docs/

# Serve docs
docs-serve:
    mdbook serve docs/

# Clean
clean:
    cargo clean

# Check (fast compile check without codegen)
check:
    cargo check --workspace

# Dev watch mode
watch:
    cargo watch -x 'check --workspace' -x 'test --workspace'
