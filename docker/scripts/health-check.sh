#!/bin/bash
set -euo pipefail

# =============================================================================
# SpiritROS Phase 1 — Health Check Script
# =============================================================================
# Checks the health of core SpiritROS services.
# Returns 0 (healthy) or 1 (unhealthy).
# =============================================================================

HEALTHY=true

# Check Ollama is responding
if ! curl -sf --max-time 5 http://localhost:11434/api/version > /dev/null 2>&1; then
    echo "UNHEALTHY: Ollama is not responding on port 11434"
    HEALTHY=false
fi

# Check Spirit Inference Coordinator
if command -v spirit-infer &> /dev/null; then
    if [[ -f /var/run/spirit-infer.pid ]]; then
        PID=$(cat /var/run/spirit-infer.pid)
        if ! kill -0 "${PID}" 2>/dev/null; then
            echo "UNHEALTHY: Spirit Inference Coordinator (PID ${PID}) is not running"
            HEALTHY=false
        fi
    fi
fi

# Check SpiritFS mount
if command -v spiritfs &> /dev/null; then
    if mountpoint -q /mnt/spiritfs 2>/dev/null; then
        : # mounted, good
    elif [[ -f /var/run/spiritfs.pid ]]; then
        PID=$(cat /var/run/spiritfs.pid)
        if ! kill -0 "${PID}" 2>/dev/null; then
            echo "UNHEALTHY: SpiritFS (PID ${PID}) is not running"
            HEALTHY=false
        fi
    fi
fi

# Check disk space (warn if < 5GB free in model directory)
MODEL_DIR="${SPIRIT_MODEL_DIR:-/var/lib/spiritros/models}"
if [[ -d "${MODEL_DIR}" ]]; then
    AVAIL_KB=$(df --output=avail "${MODEL_DIR}" 2>/dev/null | tail -1 | tr -d ' ')
    if [[ -n "${AVAIL_KB}" ]] && [[ "${AVAIL_KB}" -lt 5242880 ]]; then
        echo "WARNING: Low disk space in model directory ($(( AVAIL_KB / 1024 )) MB remaining)"
    fi
fi

if [[ "${HEALTHY}" == "true" ]]; then
    exit 0
else
    exit 1
fi
