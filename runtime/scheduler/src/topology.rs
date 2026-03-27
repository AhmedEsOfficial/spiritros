//! CPU topology discovery.
//!
//! Parses `/sys/devices/system/cpu/` for core count and NUMA topology,
//! and `/sys/bus/pci/devices/*/numa_node` for GPU NUMA affinity.
//!
//! // PHASE1-SHIM: In Phase 2, topology is discovered by the kernel directly.

use std::fs;
use std::path::Path;

use tracing::{debug, info};

use crate::error::{Result, SchedulerError};
use crate::types::{CpuId, CpuTopology, GpuDevice, NumaNode};

/// Discover the CPU/GPU topology of the current system.
///
/// # Errors
///
/// Returns `SchedulerError::TopologyError` if CPU count or NUMA topology
/// cannot be determined from sysfs.
pub fn discover() -> Result<CpuTopology> {
    let total_cores = discover_online_cpus()?;
    let numa_nodes = discover_numa_nodes(total_cores)?;
    let gpus = discover_gpus();

    let topo = CpuTopology {
        total_cores,
        numa_nodes,
        gpus,
    };

    info!(
        cores = topo.total_cores,
        numa_nodes = topo.numa_nodes.len(),
        gpus = topo.gpus.len(),
        "CPU topology discovered"
    );

    for node in &topo.numa_nodes {
        debug!(
            numa_node = node.id,
            cpus = ?node.cpus,
            "NUMA node"
        );
    }
    for gpu in &topo.gpus {
        debug!(
            gpu_index = gpu.index,
            pci = %gpu.pci_address,
            numa = ?gpu.numa_node,
            "GPU device"
        );
    }

    Ok(topo)
}

/// Count online CPUs from /sys/devices/system/cpu/online.
fn discover_online_cpus() -> Result<usize> {
    let online_path = "/sys/devices/system/cpu/online";

    if !Path::new(online_path).exists() {
        // Fallback: count cpu* directories
        return count_cpu_directories();
    }

    let content = fs::read_to_string(online_path)
        .map_err(|e| SchedulerError::TopologyError(format!("read {online_path}: {e}")))?;

    let cpus = parse_cpu_list(content.trim());
    if cpus.is_empty() {
        return Err(SchedulerError::TopologyError(
            "No online CPUs found".to_string(),
        ));
    }

    Ok(cpus.len())
}

/// Fallback: count /sys/devices/system/cpu/cpu* directories.
fn count_cpu_directories() -> Result<usize> {
    let cpu_dir = "/sys/devices/system/cpu";
    if !Path::new(cpu_dir).exists() {
        // Last resort: use libc sysconf
        let count = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
        return Ok(if count > 0 {
            usize::try_from(count).unwrap_or(1)
        } else {
            1
        });
    }

    let mut count = 0;
    let entries = fs::read_dir(cpu_dir)
        .map_err(|e| SchedulerError::TopologyError(format!("readdir {cpu_dir}: {e}")))?;

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with("cpu") && name_str[3..].parse::<usize>().is_ok() {
            count += 1;
        }
    }

    if count == 0 {
        count = 1; // At least one CPU
    }
    Ok(count)
}

/// Discover NUMA nodes from /sys/devices/system/node/.
fn discover_numa_nodes(total_cores: usize) -> Result<Vec<NumaNode>> {
    let node_dir = "/sys/devices/system/node";

    if !Path::new(node_dir).exists() {
        // Non-NUMA system: single node with all CPUs
        info!("No NUMA topology found, treating as single-node system");
        return Ok(vec![NumaNode {
            id: 0,
            cpus: (0..total_cores).collect(),
        }]);
    }

    let mut nodes = Vec::new();

    let entries = fs::read_dir(node_dir)
        .map_err(|e| SchedulerError::TopologyError(format!("readdir {node_dir}: {e}")))?;

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if let Some(id_str) = name_str.strip_prefix("node") {
            if let Ok(id) = id_str.parse::<usize>() {
                let cpulist_path = format!("{node_dir}/{name_str}/cpulist");
                let cpus = if let Ok(content) = fs::read_to_string(&cpulist_path) {
                    parse_cpu_list(content.trim())
                } else {
                    Vec::new()
                };

                if !cpus.is_empty() {
                    nodes.push(NumaNode { id, cpus });
                }
            }
        }
    }

    nodes.sort_by_key(|n| n.id);

    if nodes.is_empty() {
        // Fallback
        nodes.push(NumaNode {
            id: 0,
            cpus: (0..total_cores).collect(),
        });
    }

    Ok(nodes)
}

/// Discover NVIDIA GPUs and their NUMA affinity.
fn discover_gpus() -> Vec<GpuDevice> {
    let mut gpus = Vec::new();

    // Try to find NVIDIA GPUs via /sys/bus/pci/devices/
    let pci_dir = "/sys/bus/pci/devices";
    if let Ok(entries) = fs::read_dir(pci_dir) {
        let mut gpu_index = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            let class_path = path.join("class");

            // GPU class is 0x030000 (VGA compatible) or 0x030200 (3D controller)
            if let Ok(class) = fs::read_to_string(&class_path) {
                let class = class.trim();
                if class.starts_with("0x0302") || class.starts_with("0x0300") {
                    let pci_address = entry.file_name().to_string_lossy().to_string();

                    let numa_node = path
                        .join("numa_node")
                        .pipe_read()
                        .and_then(|s| s.trim().parse::<i32>().ok())
                        .and_then(|n| usize::try_from(n).ok());

                    gpus.push(GpuDevice {
                        index: gpu_index,
                        pci_address,
                        numa_node,
                    });
                    gpu_index += 1;
                }
            }
        }
    }

    if gpus.is_empty() {
        debug!("No GPUs detected via sysfs");
    }

    gpus
}

/// Parse a CPU list string like "0-3,5,7-9" into individual CPU IDs.
fn parse_cpu_list(s: &str) -> Vec<CpuId> {
    let mut result = Vec::new();
    if s.is_empty() {
        return result;
    }

    for part in s.split(',') {
        let part = part.trim();
        if let Some((start, end)) = part.split_once('-') {
            if let (Ok(s), Ok(e)) = (start.trim().parse::<usize>(), end.trim().parse::<usize>()) {
                result.extend(s..=e);
            }
        } else if let Ok(cpu) = part.parse::<usize>() {
            result.push(cpu);
        }
    }

    result
}

/// Extension trait for reading a file path to a String.
trait PipeRead {
    fn pipe_read(&self) -> Option<String>;
}

impl PipeRead for std::path::PathBuf {
    fn pipe_read(&self) -> Option<String> {
        fs::read_to_string(self).ok()
    }
}

/// Partition available cores into inference and system sets.
///
/// When NUMA-aware, inference cores are selected from the NUMA node
/// closest to the primary GPU. A minimum of `reserved_system` cores
/// are always kept for system services.
#[must_use]
pub fn partition_cores(
    topo: &CpuTopology,
    inference_ratio: f64,
    reserved_system: usize,
    numa_aware: bool,
) -> (Vec<CpuId>, Vec<CpuId>) {
    let all_cpus = if numa_aware {
        topo.gpu_local_cpus()
    } else {
        topo.all_cpus()
    };

    let total = all_cpus.len();
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let inference_count = ((total as f64 * inference_ratio).round() as usize)
        .min(total.saturating_sub(reserved_system));

    let inference_cores: Vec<CpuId> = all_cpus[..inference_count].to_vec();
    let system_cores: Vec<CpuId> = all_cpus[inference_count..].to_vec();

    (inference_cores, system_cores)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cpu_list_range() {
        assert_eq!(parse_cpu_list("0-3"), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_parse_cpu_list_mixed() {
        assert_eq!(parse_cpu_list("0-2,5,7-9"), vec![0, 1, 2, 5, 7, 8, 9]);
    }

    #[test]
    fn test_parse_cpu_list_single() {
        assert_eq!(parse_cpu_list("4"), vec![4]);
    }

    #[test]
    fn test_parse_cpu_list_empty() {
        assert_eq!(parse_cpu_list(""), Vec::<CpuId>::new());
    }

    #[test]
    fn test_partition_cores_75_percent() {
        let topo = CpuTopology {
            total_cores: 8,
            numa_nodes: vec![NumaNode {
                id: 0,
                cpus: vec![0, 1, 2, 3, 4, 5, 6, 7],
            }],
            gpus: vec![],
        };

        let (inference, system) = partition_cores(&topo, 0.75, 1, false);
        assert_eq!(inference.len(), 6);
        assert_eq!(system.len(), 2);
    }

    #[test]
    fn test_partition_cores_respects_reserved() {
        let topo = CpuTopology {
            total_cores: 4,
            numa_nodes: vec![NumaNode {
                id: 0,
                cpus: vec![0, 1, 2, 3],
            }],
            gpus: vec![],
        };

        let (inference, system) = partition_cores(&topo, 1.0, 2, false);
        assert_eq!(inference.len(), 2);
        assert_eq!(system.len(), 2);
    }

    #[test]
    fn test_partition_cores_numa_aware() {
        let topo = CpuTopology {
            total_cores: 8,
            numa_nodes: vec![
                NumaNode {
                    id: 0,
                    cpus: vec![0, 1, 2, 3],
                },
                NumaNode {
                    id: 1,
                    cpus: vec![4, 5, 6, 7],
                },
            ],
            gpus: vec![GpuDevice {
                index: 0,
                pci_address: "0000:01:00.0".to_string(),
                numa_node: Some(1),
            }],
        };

        let (inference, _system) = partition_cores(&topo, 0.75, 1, true);
        // GPU is on NUMA node 1 (CPUs 4-7), so inference cores come from there
        assert!(inference.iter().all(|c| *c >= 4));
    }
}
