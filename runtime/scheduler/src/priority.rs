//! Thread priority management via `SCHED_FIFO` and nice levels.
//!
//! Sets real-time scheduling policies for inference-critical threads
//! and adjusts nice values for background workloads.
//!
//! Requires `CAP_SYS_NICE` capability for `SCHED_FIFO`. Falls back to
//! nice levels if real-time scheduling is unavailable.
//!
//! // PHASE1-SHIM: Replace with kernel-native priority classes in Phase 2.

use nix::sched::CpuSet;
use nix::unistd::Pid;
use tracing::{debug, info, warn};

use crate::error::{Result, SchedulerError};
use crate::types::PriorityTier;

/// Whether real-time scheduling (`SCHED_FIFO`) is available.
///
/// Checked once at startup by attempting a no-op `sched_setscheduler`.
#[must_use]
pub fn check_realtime_capability() -> bool {
    // Try to get our current scheduling policy - if we can set SCHED_FIFO
    // on ourselves momentarily, we have CAP_SYS_NICE
    let pid = nix::unistd::Pid::from_raw(0); // current thread

    // Try setting SCHED_FIFO with lowest priority
    let param = libc::sched_param { sched_priority: 1 };

    let result =
        unsafe { libc::sched_setscheduler(pid.as_raw(), libc::SCHED_FIFO, &raw const param) };

    if result == 0 {
        // Restore to SCHED_OTHER
        let restore_param = libc::sched_param { sched_priority: 0 };
        unsafe {
            libc::sched_setscheduler(pid.as_raw(), libc::SCHED_OTHER, &raw const restore_param);
        }
        info!("Real-time scheduling (SCHED_FIFO) is available");
        true
    } else {
        warn!("`SCHED_FIFO` not available (missing `CAP_SYS_NICE`), using nice fallback");
        false
    }
}

/// Set the scheduling policy and priority for a process.
///
/// For P0/P1 tiers, uses `SCHED_FIFO` if available, otherwise falls back
/// to aggressive nice values.
///
/// # Errors
///
/// Returns `SchedulerError::PolicyError` if the scheduling policy or nice
/// value cannot be set for the given process.
#[allow(clippy::too_many_arguments)]
pub fn set_process_priority(
    pid: u32,
    tier: PriorityTier,
    p0_fifo: u32,
    p1_fifo: u32,
    p2_nice: i32,
    p3_nice: i32,
    p4_nice: i32,
    realtime_available: bool,
) -> Result<()> {
    match tier {
        PriorityTier::InferenceCritical => {
            if realtime_available {
                set_sched_fifo(pid, p0_fifo)?;
            } else {
                set_nice(pid, -20)?;
            }
        }
        PriorityTier::InferenceBatch => {
            if realtime_available {
                set_sched_fifo(pid, p1_fifo)?;
            } else {
                set_nice(pid, -15)?;
            }
        }
        PriorityTier::SystemServices => {
            ensure_sched_other(pid);
            set_nice(pid, p2_nice)?;
        }
        PriorityTier::Compatibility => {
            ensure_sched_other(pid);
            set_nice(pid, p3_nice)?;
        }
        PriorityTier::IdleMaintenance => {
            if realtime_available {
                set_sched_idle(pid)?;
            } else {
                set_nice(pid, p4_nice)?;
            }
        }
    }

    debug!(pid, tier = %tier, "Process priority set");
    Ok(())
}

/// Set a process to `SCHED_FIFO` with the given priority (1-99).
fn set_sched_fifo(pid: u32, priority: u32) -> Result<()> {
    validate_fifo_priority(priority)?;

    // Safety: priority is validated to 1-99, always fits in i32.
    // pid is a Linux PID which always fits in i32.
    #[allow(clippy::cast_possible_wrap)]
    let sched_priority = priority as i32;
    let param = libc::sched_param { sched_priority };

    #[allow(clippy::cast_possible_wrap)]
    let result =
        unsafe { libc::sched_setscheduler(pid as i32, libc::SCHED_FIFO, &raw const param) };

    if result != 0 {
        let err = std::io::Error::last_os_error();
        return Err(SchedulerError::PolicyError {
            pid,
            reason: format!("sched_setscheduler(SCHED_FIFO, {priority}): {err}"),
        });
    }

    Ok(())
}

/// Set a process to `SCHED_IDLE` (lowest priority batch scheduling).
fn set_sched_idle(pid: u32) -> Result<()> {
    let param = libc::sched_param { sched_priority: 0 };

    #[allow(clippy::cast_possible_wrap)]
    let result =
        unsafe { libc::sched_setscheduler(pid as i32, libc::SCHED_IDLE, &raw const param) };

    if result != 0 {
        let err = std::io::Error::last_os_error();
        // Fall back to nice 19 if SCHED_IDLE fails
        warn!(
            pid,
            error = %err,
            "SCHED_IDLE failed, falling back to nice 19"
        );
        return set_nice(pid, 19);
    }

    Ok(())
}

/// Ensure a process is on `SCHED_OTHER` (CFS).
fn ensure_sched_other(pid: u32) {
    let param = libc::sched_param { sched_priority: 0 };

    #[allow(clippy::cast_possible_wrap)]
    let result =
        unsafe { libc::sched_setscheduler(pid as i32, libc::SCHED_OTHER, &raw const param) };

    if result != 0 {
        // May already be SCHED_OTHER; not a fatal error
        debug!(
            pid,
            "sched_setscheduler(SCHED_OTHER) returned error (may be expected)"
        );
    }
}

/// Set the nice value for a process.
fn set_nice(pid: u32, nice: i32) -> Result<()> {
    let nice = nice.clamp(-20, 19);

    let result = unsafe { libc::setpriority(libc::PRIO_PROCESS, pid, nice) };

    if result != 0 {
        let err = std::io::Error::last_os_error();
        return Err(SchedulerError::PolicyError {
            pid,
            reason: format!("setpriority(nice={nice}): {err}"),
        });
    }

    Ok(())
}

/// Set CPU affinity for a process to the specified cores.
///
/// # Errors
///
/// Returns `SchedulerError::CpusetError` if the affinity mask cannot be
/// applied to the process.
pub fn set_cpu_affinity(pid: u32, cores: &[usize]) -> Result<()> {
    if cores.is_empty() {
        return Ok(());
    }

    let mut cpuset = CpuSet::new();
    for &core in cores {
        cpuset.set(core).map_err(|e| SchedulerError::CpusetError {
            pid,
            reason: format!("CpuSet::set({core}): {e}"),
        })?;
    }

    #[allow(clippy::cast_possible_wrap)]
    let raw_pid = pid as i32;
    nix::sched::sched_setaffinity(Pid::from_raw(raw_pid), &cpuset).map_err(|e| {
        SchedulerError::CpusetError {
            pid,
            reason: format!("sched_setaffinity: {e}"),
        }
    })?;

    debug!(pid, cores = ?cores, "CPU affinity set");
    Ok(())
}

/// Get the current CPU affinity mask for a process.
///
/// # Errors
///
/// Returns `SchedulerError::CpusetError` if the affinity mask cannot be read.
pub fn get_cpu_affinity(pid: u32) -> Result<Vec<usize>> {
    #[allow(clippy::cast_possible_wrap)]
    let raw_pid = pid as i32;
    let cpuset = nix::sched::sched_getaffinity(Pid::from_raw(raw_pid)).map_err(|e| {
        SchedulerError::CpusetError {
            pid,
            reason: format!("sched_getaffinity: {e}"),
        }
    })?;

    let mut cores = Vec::new();
    for i in 0..CpuSet::count() {
        if cpuset.is_set(i).unwrap_or(false) {
            cores.push(i);
        }
    }

    Ok(cores)
}

/// Validates that a FIFO priority value is in the allowed range (1-99).
///
/// # Errors
///
/// Returns `SchedulerError::InvalidPriority` if `priority` is outside 1-99.
pub fn validate_fifo_priority(priority: u32) -> Result<()> {
    if (1..=99).contains(&priority) {
        Ok(())
    } else {
        Err(SchedulerError::InvalidPriority(priority))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_fifo_priority_valid() {
        assert!(validate_fifo_priority(1).is_ok());
        assert!(validate_fifo_priority(50).is_ok());
        assert!(validate_fifo_priority(99).is_ok());
    }

    #[test]
    fn test_validate_fifo_priority_invalid() {
        assert!(validate_fifo_priority(0).is_err());
        assert!(validate_fifo_priority(100).is_err());
    }
}
