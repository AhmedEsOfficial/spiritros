# Installation

SpiritROS Phase 1 runs as a Docker container on a Linux host. This page covers the prerequisites and installation steps.

## Prerequisites

- **Docker** >= 24.0 with Docker Compose v2
- **NVIDIA GPU** with driver >= 535 (for GPU-accelerated inference)
- **nvidia-container-toolkit** installed and configured
- **Linux host** with cgroups v2 enabled (default on most modern distributions)

### Optional Tools

- **Rust** >= 1.75 (stable) -- required for building from source
- **just** -- task runner for development workflows
- **Ollama** -- for local testing outside Docker

## Docker Installation

1. Clone the repository:

```bash
git clone https://github.com/AhmedEsOfficial/spiritros.git
cd spiritros
```

2. Build and start the container:

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d
```

3. Verify the installation:

```bash
docker compose -f docker/docker-compose.yml exec spiritros spirit-shell --version
```

## Building from Source

If you prefer to build from source for development:

```bash
# Install Rust toolchain
rustup default stable
rustup component add rustfmt clippy

# Install development tools
cargo install cargo-watch cargo-nextest just

# Build all crates
just build

# Run tests
just test
```

## Verifying GPU Access

After installation, confirm that the container can access your GPU:

```bash
docker compose -f docker/docker-compose.yml exec spiritros nvidia-smi
```

You should see your GPU listed with available VRAM. If not, check that `nvidia-container-toolkit` is correctly installed and Docker is configured to use the NVIDIA runtime.
