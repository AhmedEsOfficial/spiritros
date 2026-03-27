#!/bin/bash
set -euo pipefail

# =============================================================================
# SpiritROS Phase 1 — Container Entrypoint
# =============================================================================
# Initializes the SpiritROS runtime environment:
#   1. Detect GPU capabilities
#   2. Set up cgroups v2 memory zones
#   3. Start Spirit Scheduler (inference-aware CPU/GPU scheduling)
#   4. Start Ollama inference backend
#   5. Start Spirit Inference Coordinator
#   6. Mount SpiritFS (FUSE)
#   7. Start Spirit Guard (security)
#   8. Pull default model if not present
#   9. Execute CMD
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SPIRIT_LOG_DIR:-/var/log/spiritros}"
CONFIG="${SPIRIT_CONFIG:-/opt/spiritros/config/spirit.yaml}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [entrypoint] $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [entrypoint] ERROR: $*" >&2
}

cleanup() {
    log "Shutting down SpiritROS services..."

    # Stop services in reverse order
    if [[ -f /var/run/spirit-guard.pid ]]; then
        kill "$(cat /var/run/spirit-guard.pid)" 2>/dev/null || true
    fi
    if [[ -f /var/run/spiritfs.pid ]]; then
        fusermount3 -u /mnt/spiritfs 2>/dev/null || true
        kill "$(cat /var/run/spiritfs.pid)" 2>/dev/null || true
    fi
    if [[ -f /var/run/spirit-infer.pid ]]; then
        kill "$(cat /var/run/spirit-infer.pid)" 2>/dev/null || true
    fi
    if [[ -f /var/run/spirit-scheduler.pid ]]; then
        kill "$(cat /var/run/spirit-scheduler.pid)" 2>/dev/null || true
    fi
    if [[ -f /var/run/ollama.pid ]]; then
        kill "$(cat /var/run/ollama.pid)" 2>/dev/null || true
    fi

    log "Shutdown complete."
}
trap cleanup EXIT SIGTERM SIGINT

# =============================================================================
# Step 1: Detect GPU
# =============================================================================
log "Detecting GPU capabilities..."
source "${SCRIPT_DIR}/gpu-detect.sh"

if [[ "${SPIRIT_GPU_TYPE}" == "none" ]]; then
    log "WARNING: No GPU detected. Running in CPU-only mode."
    log "Performance will be significantly degraded."
else
    log "GPU detected: ${SPIRIT_GPU_TYPE} (${SPIRIT_GPU_COUNT} device(s), ${SPIRIT_GPU_VRAM_TOTAL_MB} MB total VRAM)"
fi

# =============================================================================
# Step 2: Set up cgroups v2 memory zones
# =============================================================================
log "Configuring cgroups v2 memory zones..."
if [[ -f /opt/spiritros/config/spirit-cgroups.sh ]]; then
    source /opt/spiritros/config/spirit-cgroups.sh || {
        log_error "Failed to configure cgroups. Memory zone enforcement will be disabled."
    }
else
    log "WARNING: spirit-cgroups.sh not found, skipping memory zone setup."
fi

# =============================================================================
# Step 3: Start Spirit Scheduler
# =============================================================================
log "Starting Spirit Scheduler..."
if command -v spirit-scheduler &> /dev/null; then
    mkdir -p /run/spirit
    spirit-scheduler \
        --config "${CONFIG}" \
        --log-level "${SPIRIT_LOG_LEVEL:-info}" \
        > "${LOG_DIR}/scheduler.log" 2>&1 &
    echo $! > /var/run/spirit-scheduler.pid
    log "Spirit Scheduler started (PID $(cat /var/run/spirit-scheduler.pid))."
else
    log "WARNING: spirit-scheduler not found, skipping."
fi

# =============================================================================
# Step 4: Start Ollama
# =============================================================================
log "Starting Ollama inference backend..."
ollama serve > "${LOG_DIR}/ollama.log" 2>&1 &
echo $! > /var/run/ollama.pid

# Wait for Ollama to be ready
OLLAMA_RETRIES=0
OLLAMA_MAX_RETRIES=30
until curl -sf http://localhost:11434/api/version > /dev/null 2>&1; do
    OLLAMA_RETRIES=$((OLLAMA_RETRIES + 1))
    if [[ ${OLLAMA_RETRIES} -ge ${OLLAMA_MAX_RETRIES} ]]; then
        log_error "Ollama failed to start after ${OLLAMA_MAX_RETRIES} attempts."
        exit 1
    fi
    sleep 1
done
log "Ollama is ready (took ${OLLAMA_RETRIES}s)."

# =============================================================================
# Step 5: Start Spirit Inference Coordinator
# =============================================================================
log "Starting Spirit Inference Coordinator..."
if command -v spirit-infer &> /dev/null; then
    spirit-infer \
        --config "${CONFIG}" \
        --log-level "${SPIRIT_LOG_LEVEL:-info}" \
        > "${LOG_DIR}/coordinator.log" 2>&1 &
    echo $! > /var/run/spirit-infer.pid
    log "Inference Coordinator started (PID $(cat /var/run/spirit-infer.pid))."
else
    log "WARNING: spirit-infer not found, skipping."
fi

# =============================================================================
# Step 6: Mount SpiritFS (FUSE)
# =============================================================================
log "Mounting SpiritFS at /mnt/spiritfs..."
if command -v spiritfs &> /dev/null; then
    # Ensure backing and CAS directories exist
    mkdir -p /var/lib/spiritros/data/spiritfs
    mkdir -p /var/lib/spiritros/cas
    mkdir -p /mnt/spiritfs

    spiritfs mount \
        --backing-dir /var/lib/spiritros/data/spiritfs \
        --mount-point /mnt/spiritfs \
        > "${LOG_DIR}/spiritfs.log" 2>&1 &
    echo $! > /var/run/spiritfs.pid
    log "SpiritFS mounted (PID $(cat /var/run/spiritfs.pid))."
else
    log "WARNING: spiritfs not found, skipping FUSE mount."
fi

# =============================================================================
# Step 7: Start Spirit Guard (Security)
# =============================================================================
log "Starting Spirit Guard..."
if command -v spirit-guard &> /dev/null; then
    spirit-guard \
        --config "${CONFIG}" \
        > "${LOG_DIR}/guard.log" 2>&1 &
    echo $! > /var/run/spirit-guard.pid
    log "Spirit Guard started (PID $(cat /var/run/spirit-guard.pid))."
else
    log "WARNING: spirit-guard not found, skipping."
fi

# =============================================================================
# Step 8: Pull default model if not already present
# =============================================================================
DEFAULT_MODEL="${SPIRIT_DEFAULT_MODEL:-llama3.2:3b}"
log "Ensuring default model '${DEFAULT_MODEL}' is available..."
if ! ollama list 2>/dev/null | grep -q "${DEFAULT_MODEL%%:*}"; then
    log "Pulling '${DEFAULT_MODEL}'... (this may take a while)"
    ollama pull "${DEFAULT_MODEL}" > "${LOG_DIR}/model-pull.log" 2>&1 || {
        log_error "Failed to pull default model '${DEFAULT_MODEL}'. Continuing anyway."
    }
else
    log "Model '${DEFAULT_MODEL}' already available."
fi

# =============================================================================
# Step 9: Execute CMD
# =============================================================================
log "SpiritROS Phase 1 initialization complete."

# If CMD is spirit-shell, check for a real interactive terminal.
# Docker's tty:true allocates a pseudo-TTY that passes [ -t 0 ] but can't
# answer cursor-position queries. We probe with the DSR escape sequence.
# The spirit-shell binary also does this check internally as a safety net.
if [ "$1" = "spirit-shell" ]; then
    if [ ! -t 0 ]; then
        log "No TTY detected — running in daemon mode (services only)."
        log "Attach interactively with: docker exec -it spiritros spirit-shell"
        exec sleep infinity
    fi

    # Probe: can the terminal answer a cursor position query?
    # Send \x1b[6n and wait 0.5s for a response.
    real_tty=false
    if command -v stty &>/dev/null; then
        old_stty=$(stty -g 2>/dev/null || true)
        stty raw -echo min 0 time 5 2>/dev/null || true
        printf '\033[6n' > /dev/tty 2>/dev/null || true
        if read -r -t 1 -d 'R' response < /dev/tty 2>/dev/null; then
            real_tty=true
        fi
        stty "$old_stty" 2>/dev/null || true
    fi

    if [ "$real_tty" = false ]; then
        log "Pseudo-TTY detected but no terminal emulator — running in daemon mode."
        log "Attach interactively with: docker exec -it spiritros spirit-shell"
        exec sleep infinity
    fi
fi

log "Executing: $*"
exec "$@"
