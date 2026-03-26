#!/bin/bash
set -euo pipefail

# =============================================================================
# SpiritROS Phase 1 — GPU Detection Script
# =============================================================================
# Detects GPU hardware and exports environment variables:
#   SPIRIT_GPU_TYPE       — "nvidia", "amd", or "none"
#   SPIRIT_GPU_COUNT      — Number of GPUs detected
#   SPIRIT_GPU_VRAM_TOTAL_MB — Total VRAM across all GPUs (MB)
#   SPIRIT_GPU_DEVICES    — Comma-separated list of device indices
#   SPIRIT_GPU_DRIVER     — Driver version string
#   SPIRIT_GPU_COMPUTE    — Compute capability (NVIDIA) or GFX version (AMD)
# =============================================================================

export SPIRIT_GPU_TYPE="none"
export SPIRIT_GPU_COUNT=0
export SPIRIT_GPU_VRAM_TOTAL_MB=0
export SPIRIT_GPU_DEVICES=""
export SPIRIT_GPU_DRIVER=""
export SPIRIT_GPU_COMPUTE=""

# ---------------------------------------------------------------------------
# NVIDIA Detection
# ---------------------------------------------------------------------------
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi > /dev/null 2>&1; then
        SPIRIT_GPU_TYPE="nvidia"

        # Count GPUs
        SPIRIT_GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1 | tr -d ' ')

        # Total VRAM (sum across all GPUs)
        VRAM_LIST=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
        TOTAL_VRAM=0
        while IFS= read -r vram; do
            vram=$(echo "${vram}" | tr -d ' ')
            TOTAL_VRAM=$((TOTAL_VRAM + vram))
        done <<< "${VRAM_LIST}"
        SPIRIT_GPU_VRAM_TOTAL_MB="${TOTAL_VRAM}"

        # Device indices
        SPIRIT_GPU_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr '\n' ',' | sed 's/,$//')

        # Driver version
        SPIRIT_GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1 | tr -d ' ')

        # Compute capability
        SPIRIT_GPU_COMPUTE=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1 | tr -d ' ')

        # Configure CUDA MPS if multiple GPUs
        if [[ "${SPIRIT_GPU_COUNT}" -gt 1 ]]; then
            export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
            export CUDA_MPS_LOG_DIRECTORY=/var/log/spiritros/nvidia-mps
            mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
        fi

        # Set Ollama GPU configuration
        export OLLAMA_NUM_GPU="${SPIRIT_GPU_COUNT}"

        export SPIRIT_GPU_TYPE SPIRIT_GPU_COUNT SPIRIT_GPU_VRAM_TOTAL_MB
        export SPIRIT_GPU_DEVICES SPIRIT_GPU_DRIVER SPIRIT_GPU_COMPUTE
        return 0 2>/dev/null || exit 0
    fi
fi

# ---------------------------------------------------------------------------
# AMD Detection (ROCm)
# ---------------------------------------------------------------------------
if command -v rocm-smi &> /dev/null; then
    if rocm-smi > /dev/null 2>&1; then
        SPIRIT_GPU_TYPE="amd"

        # Count GPUs
        SPIRIT_GPU_COUNT=$(rocm-smi --showid --csv 2>/dev/null | tail -n +2 | wc -l)
        if [[ "${SPIRIT_GPU_COUNT}" -eq 0 ]]; then
            SPIRIT_GPU_COUNT=1
        fi

        # Total VRAM
        VRAM_INFO=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n +2 || echo "")
        if [[ -n "${VRAM_INFO}" ]]; then
            TOTAL_VRAM=0
            while IFS=',' read -r _ total _; do
                total=$(echo "${total}" | tr -d ' ')
                TOTAL_VRAM=$((TOTAL_VRAM + total / 1048576))
            done <<< "${VRAM_INFO}"
            SPIRIT_GPU_VRAM_TOTAL_MB="${TOTAL_VRAM}"
        fi

        # Driver version
        SPIRIT_GPU_DRIVER=$(cat /sys/module/amdgpu/version 2>/dev/null || echo "unknown")

        export SPIRIT_GPU_TYPE SPIRIT_GPU_COUNT SPIRIT_GPU_VRAM_TOTAL_MB
        export SPIRIT_GPU_DEVICES SPIRIT_GPU_DRIVER SPIRIT_GPU_COMPUTE
        return 0 2>/dev/null || exit 0
    fi
fi

# ---------------------------------------------------------------------------
# No GPU — CPU-only mode
# ---------------------------------------------------------------------------
export SPIRIT_GPU_TYPE SPIRIT_GPU_COUNT SPIRIT_GPU_VRAM_TOTAL_MB
export SPIRIT_GPU_DEVICES SPIRIT_GPU_DRIVER SPIRIT_GPU_COMPUTE
