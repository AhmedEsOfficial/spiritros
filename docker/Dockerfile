# SpiritROS Phase 1 — Production Docker Image
# Multi-stage build: Rust builder + NVIDIA CUDA runtime

# ==============================================================================
# Stage 1: Builder
# ==============================================================================
FROM rust:1.87-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libfuse3-dev \
    libssl-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy workspace manifests for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY kernel/Cargo.toml kernel/Cargo.toml
COPY runtime/shell/Cargo.toml runtime/shell/Cargo.toml
COPY runtime/spiritfs/Cargo.toml runtime/spiritfs/Cargo.toml
COPY runtime/inference/Cargo.toml runtime/inference/Cargo.toml
COPY runtime/memory/Cargo.toml runtime/memory/Cargo.toml
COPY runtime/scheduler/Cargo.toml runtime/scheduler/Cargo.toml
COPY runtime/compat/Cargo.toml runtime/compat/Cargo.toml
COPY runtime/security/Cargo.toml runtime/security/Cargo.toml
COPY runtime/networking/Cargo.toml runtime/networking/Cargo.toml
COPY sdk/rust/Cargo.toml sdk/rust/Cargo.toml
COPY benchmarks/Cargo.toml benchmarks/Cargo.toml

# Create dummy source files so cargo can resolve the workspace and cache deps
RUN mkdir -p kernel/src \
             runtime/shell/src \
             runtime/spiritfs/src \
             runtime/inference/src \
             runtime/memory/src \
             runtime/scheduler/src \
             runtime/compat/src \
             runtime/security/src \
             runtime/networking/src \
             sdk/rust/src \
             benchmarks/src \
    && for d in kernel runtime/memory runtime/scheduler runtime/compat \
                runtime/security runtime/networking runtime/inference \
                runtime/spiritfs sdk/rust; do \
         echo '//! stub' > "$d/src/lib.rs"; \
       done \
    && echo 'fn main() {}' > runtime/shell/src/main.rs \
    && echo 'fn main() {}' > runtime/inference/src/main.rs \
    && echo 'fn main() {}' > runtime/security/src/main.rs \
    && echo 'fn main() {}' > runtime/spiritfs/src/main.rs \
    && echo 'fn main() {}' > benchmarks/src/main.rs \
    && echo '//! stub' > benchmarks/src/lib.rs \
    && cargo build --release --workspace 2>/dev/null || true

# Remove dummy sources, copy real source code
RUN find kernel runtime sdk benchmarks -name "*.rs" -delete

COPY kernel/ kernel/
COPY runtime/ runtime/
COPY sdk/ sdk/
COPY benchmarks/ benchmarks/
COPY patches/ patches/

# Build release binaries — enable fuse feature since libfuse3-dev is installed
RUN cargo build --release --workspace --features spirit-spiritfs/fuse \
    && for bin in spirit-shell spirit-infer spirit-guard spirit-scheduler spiritfs spirit-bench; do \
         strip "target/release/$bin" 2>/dev/null || true; \
       done

# Stage binaries into a single directory so COPY in the runtime stage doesn't
# fail when an optional binary (e.g. spiritfs) wasn't produced.
RUN mkdir -p /build/dist \
    && for bin in spirit-shell spirit-infer spirit-guard spirit-scheduler spiritfs spirit-bench; do \
         cp "target/release/$bin" /build/dist/ 2>/dev/null || true; \
       done

# ==============================================================================
# Stage 2: Runtime
# ==============================================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS runtime

LABEL maintainer="SpiritROS Team"
LABEL description="SpiritROS Phase 1 — LLM-Native Operating System (Virtualized)"
LABEL version="0.1.0"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libfuse3-3 \
    fuse3 \
    curl \
    ca-certificates \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Create spiritros user and directories
RUN groupadd --force --gid 1000 spirit \
    && useradd --non-unique --uid 1000 --gid spirit --shell /bin/bash --create-home spirit \
    && mkdir -p /opt/spiritros/bin \
                /opt/spiritros/config \
                /opt/spiritros/scripts \
                /var/lib/spiritros/models \
                /var/lib/spiritros/data \
                /var/log/spiritros \
                /mnt/spiritfs \
    && chown -R spirit:spirit /opt/spiritros /var/lib/spiritros /var/log/spiritros /mnt/spiritfs

# Copy all produced binaries from the builder staging directory
COPY --from=builder /build/dist/ /opt/spiritros/bin/

# Copy configuration and scripts
COPY docker/config/ /opt/spiritros/config/
COPY docker/scripts/ /opt/spiritros/scripts/

RUN chmod +x /opt/spiritros/scripts/*.sh \
    && chmod +x /opt/spiritros/bin/* 2>/dev/null || true

# Add binaries to PATH
ENV PATH="/opt/spiritros/bin:${PATH}"
ENV SPIRIT_CONFIG="/opt/spiritros/config/spirit.yaml"
ENV SPIRIT_LOG_DIR="/var/log/spiritros"
ENV SPIRIT_DATA_DIR="/var/lib/spiritros/data"
ENV SPIRIT_MODEL_DIR="/var/lib/spiritros/models"
ENV OLLAMA_HOST="0.0.0.0:11434"
ENV OLLAMA_MODELS="/var/lib/spiritros/models"

# Ports:
#   11434 — Ollama API
#   50051 — Spirit Inference Coordinator (gRPC)
#   8080  — Spirit REST API
#   9090  — Metrics (Prometheus exposition)
EXPOSE 11434 50051 8080 9090

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD /opt/spiritros/scripts/health-check.sh

ENTRYPOINT ["/opt/spiritros/scripts/entrypoint.sh"]
CMD ["spirit-shell"]
