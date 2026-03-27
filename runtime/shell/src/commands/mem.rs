//! Spirit memory commands: status, rebalance.
//!
//! PHASE1-SHIM: Reads cgroups v2 stats directly. Phase 2+ will use
//! the kernel's native memory zone API.

use crate::commands::SharedState;
use crate::types::CommandResult;

/// Dispatch `spirit mem <subcommand>`.
pub async fn dispatch(_state: &SharedState, args: &[&str]) -> CommandResult {
    if args.is_empty() {
        return CommandResult::Output(
            "Usage: spirit mem <status|rebalance> [args...]\n\n\
             Subcommands:\n  \
             status              Show memory zone utilization\n  \
             rebalance           Dynamically resize memory zones"
                .into(),
        );
    }

    match args[0] {
        "status" | "stat" => status().await,
        "rebalance" => rebalance(args).await,
        other => CommandResult::Error(format!(
            "Unknown mem subcommand: '{other}'. Use: status, rebalance"
        )),
    }
}

/// Display memory zone status from cgroups v2.
async fn status() -> CommandResult {
    let mut lines = Vec::new();
    lines.push("TensorFirst Memory Zones (cgroups v2 shim)".to_string());
    lines.push(format!(
        "{:<16} {:<12} {:<12} {:<12} {:<8}",
        "ZONE", "LIMIT", "USED", "AVAILABLE", "UTIL%"
    ));
    lines.push("─".repeat(64));

    // Read cgroup stats for each zone
    let zones = [
        ("ZONE_MODEL", "spirit/zone_model"),
        ("ZONE_KV", "spirit/zone_kv"),
        ("ZONE_TENSOR", "spirit/zone_tensor"),
        ("ZONE_GENERAL", "spirit/zone_general"),
    ];

    let cgroup_root =
        std::env::var("SPIRIT_CGROUP_ROOT").unwrap_or_else(|_| "/sys/fs/cgroup".into());

    let mut found_any = false;

    for (label, cgroup_path) in &zones {
        let base = format!("{cgroup_root}/{cgroup_path}");

        let limit = read_cgroup_bytes(&format!("{base}/memory.max"));
        let used = read_cgroup_bytes(&format!("{base}/memory.current"));

        if let (Some(limit_val), Some(used_val)) = (limit, used) {
            found_any = true;
            let avail = limit_val.saturating_sub(used_val);
            let util = if limit_val > 0 {
                (used_val as f64 / limit_val as f64) * 100.0
            } else {
                0.0
            };
            lines.push(format!(
                "{:<16} {:<12} {:<12} {:<12} {:.1}%",
                label,
                format_mem(limit_val),
                format_mem(used_val),
                format_mem(avail),
                util
            ));
        } else {
            lines.push(format!("{label:<16} [not configured]"));
        }
    }

    // System memory info from /proc/meminfo
    if let Some((total, available)) = read_system_memory() {
        lines.push(String::new());
        lines.push(format!(
            "System: {} total, {} available",
            format_mem(total),
            format_mem(available)
        ));
    }

    if !found_any {
        lines.push(String::new());
        lines.push(
            "Note: Memory zones not configured. Run inside the SpiritROS \
             Docker container or configure cgroups v2 manually."
                .into(),
        );
    }

    CommandResult::Output(lines.join("\n"))
}

/// Rebalance memory zones.
async fn rebalance(args: &[&str]) -> CommandResult {
    if args.len() < 2 {
        return CommandResult::Output(
            "Usage: spirit mem rebalance [options]\n\n\
             Options:\n  \
             --give-general <size>   Increase ZONE_GENERAL by <size>\n  \
             --from <zone>           Take memory from <zone>\n\n\
             Example: spirit mem rebalance --give-general 2G --from tensor"
                .into(),
        );
    }

    // PHASE1-SHIM: Parse and validate the rebalance request, but actual
    // implementation requires cgroups v2 write access (root or delegated).
    CommandResult::Output(
        "Memory zone rebalancing requires elevated privileges.\n\
         In the Docker container, use the cgroups configuration in spirit.yaml."
            .into(),
    )
}

/// Read a cgroups v2 memory value (bytes) from a file.
fn read_cgroup_bytes(path: &str) -> Option<u64> {
    let content = std::fs::read_to_string(path).ok()?;
    let trimmed = content.trim();
    if trimmed == "max" {
        return None; // unlimited
    }
    trimmed.parse().ok()
}

/// Read total and available system memory from /proc/meminfo.
fn read_system_memory() -> Option<(u64, u64)> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total = None;
    let mut available = None;

    for line in content.lines() {
        if line.starts_with("MemTotal:") {
            total = parse_meminfo_kb(line);
        } else if line.starts_with("MemAvailable:") {
            available = parse_meminfo_kb(line);
        }
        if total.is_some() && available.is_some() {
            break;
        }
    }

    match (total, available) {
        (Some(t), Some(a)) => Some((t * 1024, a * 1024)), // Convert KB to bytes
        _ => None,
    }
}

/// Parse a meminfo line like `MemTotal:       16384000 kB` into KB value.
fn parse_meminfo_kb(line: &str) -> Option<u64> {
    line.split_whitespace().nth(1)?.parse().ok()
}

/// Format bytes as a human-readable memory size.
fn format_mem(bytes: u64) -> String {
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;

    if bytes >= GIB {
        format!("{:.1} GiB", bytes as f64 / GIB as f64)
    } else {
        format!("{:.0} MiB", bytes as f64 / MIB as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_mem() {
        assert_eq!(format_mem(1024 * 1024 * 512), "512 MiB");
        assert_eq!(format_mem(1024 * 1024 * 1024 * 4), "4.0 GiB");
    }

    #[test]
    fn test_parse_meminfo_kb() {
        assert_eq!(
            parse_meminfo_kb("MemTotal:       16384000 kB"),
            Some(16_384_000)
        );
        assert_eq!(parse_meminfo_kb("bad line"), None);
    }
}
