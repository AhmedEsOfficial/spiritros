#!/bin/bash
set -euo pipefail

# =============================================================================
# SpiritROS Phase 1 — cgroups v2 Memory Zone Setup
# =============================================================================
# PHASE1-SHIM: Replace with kernel-native TensorFirst memory zones in Phase 2.
#
# Creates cgroups v2 memory zones that approximate the SpiritROS memory
# architecture:
#   ZONE_MODEL   — 40% of memory, for model weights (read-only after load)
#   ZONE_KV      — 30% of memory, for KV-cache (inference state)
#   ZONE_TENSOR  — 20% of memory, for intermediate tensors (scratch)
#   ZONE_GENERAL — 10% of memory, for everything else
# =============================================================================

CGROUP_ROOT="/sys/fs/cgroup"
SPIRIT_CGROUP="${CGROUP_ROOT}/spiritros"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [cgroups] $*"
}

# Check if cgroups v2 is available
if [[ ! -f "${CGROUP_ROOT}/cgroup.controllers" ]]; then
    log "ERROR: cgroups v2 not available. Memory zone enforcement disabled."
    return 1 2>/dev/null || exit 1
fi

# Check if memory controller is available
if ! grep -q "memory" "${CGROUP_ROOT}/cgroup.controllers"; then
    log "ERROR: Memory controller not available in cgroups v2."
    return 1 2>/dev/null || exit 1
fi

# Detect total available memory (in bytes)
TOTAL_MEM_BYTES=$(grep MemTotal /proc/meminfo | awk '{print $2 * 1024}')
# Use 90% of total memory as the budget
BUDGET_BYTES=$(( TOTAL_MEM_BYTES * 90 / 100 ))

log "Total system memory: $(( TOTAL_MEM_BYTES / 1048576 )) MB"
log "SpiritROS memory budget (90%): $(( BUDGET_BYTES / 1048576 )) MB"

# Calculate zone sizes
ZONE_MODEL_BYTES=$((  BUDGET_BYTES * 40 / 100 ))
ZONE_KV_BYTES=$((     BUDGET_BYTES * 30 / 100 ))
ZONE_TENSOR_BYTES=$(( BUDGET_BYTES * 20 / 100 ))
ZONE_GENERAL_BYTES=$(( BUDGET_BYTES * 10 / 100 ))

# Enable memory controller delegation
if [[ -f "${CGROUP_ROOT}/cgroup.subtree_control" ]]; then
    echo "+memory" > "${CGROUP_ROOT}/cgroup.subtree_control" 2>/dev/null || true
fi

# Create parent cgroup
mkdir -p "${SPIRIT_CGROUP}"
echo "+memory" > "${SPIRIT_CGROUP}/cgroup.subtree_control" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Create memory zones
# ---------------------------------------------------------------------------

create_zone() {
    local name="$1"
    local limit_bytes="$2"
    local zone_path="${SPIRIT_CGROUP}/${name}"

    mkdir -p "${zone_path}"

    # Set memory limit
    echo "${limit_bytes}" > "${zone_path}/memory.max"

    # Set memory.high to 90% of max (soft limit, triggers reclaim)
    local high_bytes=$(( limit_bytes * 90 / 100 ))
    echo "${high_bytes}" > "${zone_path}/memory.high"

    # Disable swap for inference zones (swap kills latency)
    echo "0" > "${zone_path}/memory.swap.max" 2>/dev/null || true

    local limit_mb=$(( limit_bytes / 1048576 ))
    log "Zone '${name}' created: ${limit_mb} MB (max), $(( high_bytes / 1048576 )) MB (high)"
}

create_zone "spirit-zone-model"   "${ZONE_MODEL_BYTES}"
create_zone "spirit-zone-kv"      "${ZONE_KV_BYTES}"
create_zone "spirit-zone-tensor"  "${ZONE_TENSOR_BYTES}"
create_zone "spirit-zone-general" "${ZONE_GENERAL_BYTES}"

# ---------------------------------------------------------------------------
# Configure hugepages for model zone (reduces TLB misses during weight access)
# ---------------------------------------------------------------------------
HUGEPAGE_DIR="/mnt/hugepages"
if [[ -d /sys/kernel/mm/hugepages/hugepages-2048kB ]]; then
    AVAILABLE_HUGEPAGES=$(cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages)
    NEEDED_HUGEPAGES=$(( ZONE_MODEL_BYTES / 2097152 ))  # 2MB pages

    if [[ "${AVAILABLE_HUGEPAGES}" -lt "${NEEDED_HUGEPAGES}" ]]; then
        echo "${NEEDED_HUGEPAGES}" > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages 2>/dev/null || {
            log "WARNING: Could not allocate ${NEEDED_HUGEPAGES} hugepages. Using regular pages."
        }
    fi

    if [[ ! -d "${HUGEPAGE_DIR}" ]]; then
        mkdir -p "${HUGEPAGE_DIR}"
        mount -t hugetlbfs none "${HUGEPAGE_DIR}" 2>/dev/null || {
            log "WARNING: Could not mount hugetlbfs. Hugepages unavailable."
        }
    fi

    log "Hugepages: requested ${NEEDED_HUGEPAGES} x 2MB pages for model zone."
else
    log "WARNING: 2MB hugepages not available on this system."
fi

# ---------------------------------------------------------------------------
# Export zone paths for other scripts
# ---------------------------------------------------------------------------
export SPIRIT_CGROUP_MODEL="${SPIRIT_CGROUP}/spirit-zone-model"
export SPIRIT_CGROUP_KV="${SPIRIT_CGROUP}/spirit-zone-kv"
export SPIRIT_CGROUP_TENSOR="${SPIRIT_CGROUP}/spirit-zone-tensor"
export SPIRIT_CGROUP_GENERAL="${SPIRIT_CGROUP}/spirit-zone-general"

log "Memory zones configured successfully."
