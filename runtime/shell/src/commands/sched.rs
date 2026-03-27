//! Spirit scheduler commands: status.
//!
//! PHASE1-SHIM: Reads scheduler state from the IPC socket or falls back
//! to direct sysfs/cgroup reads. Phase 2+ uses the kernel scheduler API.

use crate::types::CommandResult;

/// Dispatch `spirit sched <subcommand>`.
pub fn dispatch(args: &[&str]) -> CommandResult {
    if args.is_empty() {
        return CommandResult::Output(
            "Usage: spirit sched <status>\n\n\
             Subcommands:\n  \
             status    Show scheduler state and phase"
                .into(),
        );
    }

    match args[0] {
        "status" | "stat" => status(),
        other => CommandResult::Error(format!("Unknown sched subcommand: '{other}'. Use: status")),
    }
}

/// Display scheduler status.
fn status() -> CommandResult {
    let mut lines = Vec::new();
    lines.push("Spirit Scheduler Status (Phase 1 — cgroups v2 shim)".to_string());
    lines.push("─".repeat(52));

    // Try to read from the scheduler IPC socket
    let socket_path = std::env::var("SPIRIT_SCHED_SOCKET")
        .unwrap_or_else(|_| "/run/spirit/scheduler.sock".into());

    if std::path::Path::new(&socket_path).exists() {
        lines.push(format!("  IPC Socket:    {socket_path} (connected)"));
    } else {
        lines.push(format!("  IPC Socket:    {socket_path} (not running)"));
    }

    // Read CPU topology
    if let Ok(cpus) = std::fs::read_to_string("/proc/cpuinfo") {
        let cpu_count = cpus.matches("processor\t:").count();
        lines.push(format!("  CPU Cores:     {cpu_count}"));
    }

    // Check for NUMA nodes
    let numa_dir = std::path::Path::new("/sys/devices/system/node");
    if numa_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(numa_dir) {
            let nodes: usize = entries
                .filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().starts_with("node"))
                .count();
            lines.push(format!("  NUMA Nodes:    {nodes}"));
        }
    }

    // Check GPU
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=name,memory.total")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        if output.status.success() {
            let gpus = String::from_utf8_lossy(&output.stdout);
            for (i, line) in gpus.trim().lines().enumerate() {
                lines.push(format!("  GPU {i}:         {}", line.trim()));
            }
        }
    } else {
        lines.push("  GPU:           not detected".into());
    }

    // Priority tiers
    lines.push(String::new());
    lines.push("Priority Tiers:".to_string());
    lines.push("  P0: InferenceCritical  SCHED_FIFO 99, dedicated cores".into());
    lines.push("  P1: InferenceBatch     SCHED_FIFO 50".into());
    lines.push("  P2: SystemServices     SCHED_OTHER nice -10".into());
    lines.push("  P3: Compatibility      SCHED_OTHER nice 0".into());
    lines.push("  P4: IdleMaintenance    SCHED_IDLE nice 19".into());

    // Inference phase
    lines.push(String::new());
    lines.push("  Inference Phase: idle".into());

    CommandResult::Output(lines.join("\n"))
}
