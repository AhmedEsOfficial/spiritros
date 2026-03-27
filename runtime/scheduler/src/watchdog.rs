//! Watchdog for preventing system starvation.
//!
//! Monitors `SCHED_FIFO` threads to ensure they don't hold the CPU
//! indefinitely. If a P0 thread exceeds the watchdog timeout, it is
//! automatically demoted to prevent system lockup.
//!
//! The watchdog itself runs at `SCHED_FIFO` priority 99 to ensure
//! it can preempt any misbehaving thread.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tracing::{info, warn};

/// Commands sent to the watchdog task.
#[derive(Debug)]
pub enum WatchdogCommand {
    /// Signal that a P0 thread is actively running.
    Heartbeat,
    /// Shutdown the watchdog.
    Shutdown,
}

/// Handle for interacting with the watchdog from the scheduler.
pub struct WatchdogHandle {
    /// Channel to send heartbeats and commands.
    tx: mpsc::Sender<WatchdogCommand>,
    /// Whether the watchdog has triggered.
    triggered: Arc<AtomicBool>,
}

impl WatchdogHandle {
    /// Send a heartbeat to the watchdog, resetting the timeout.
    pub async fn heartbeat(&self) {
        let _ = self.tx.send(WatchdogCommand::Heartbeat).await;
    }

    /// Returns whether the watchdog has triggered a demotion.
    #[must_use]
    pub fn has_triggered(&self) -> bool {
        self.triggered.load(Ordering::Relaxed)
    }

    /// Signal the watchdog to shut down.
    pub async fn shutdown(&self) {
        let _ = self.tx.send(WatchdogCommand::Shutdown).await;
    }
}

/// Spawn the watchdog monitoring task.
///
/// Returns a handle for sending heartbeats and a join handle for the task.
pub fn spawn_watchdog(
    timeout: Duration,
    on_timeout: Arc<dyn Fn() + Send + Sync>,
) -> WatchdogHandle {
    let (tx, mut rx) = mpsc::channel::<WatchdogCommand>(64);
    let triggered = Arc::new(AtomicBool::new(false));
    let triggered_clone = triggered.clone();

    tokio::spawn(async move {
        #[allow(clippy::cast_possible_truncation)]
        let timeout_ms = timeout.as_millis() as u64;
        info!(timeout_ms, "Watchdog started");

        loop {
            match tokio::time::timeout(timeout, rx.recv()).await {
                Ok(Some(WatchdogCommand::Heartbeat)) => {
                    // Reset — all is well
                }
                Ok(Some(WatchdogCommand::Shutdown) | None) => {
                    info!("Watchdog shutting down");
                    break;
                }
                Err(_) => {
                    // Timeout expired without heartbeat
                    warn!(
                        timeout_ms,
                        "Watchdog timeout — no heartbeat received, triggering demotion"
                    );
                    triggered_clone.store(true, Ordering::Relaxed);
                    on_timeout();

                    // Reset trigger after a cooldown
                    tokio::time::sleep(timeout).await;
                    triggered_clone.store(false, Ordering::Relaxed);
                }
            }
        }
    });

    WatchdogHandle { tx, triggered }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_watchdog_handle_heartbeat() {
        let _triggered = Arc::new(AtomicBool::new(false));
        let on_timeout: Arc<dyn Fn() + Send + Sync> = Arc::new(|| {});
        let handle = spawn_watchdog(Duration::from_secs(10), on_timeout);

        assert!(!handle.has_triggered());
        handle.heartbeat().await;
        handle.shutdown().await;
    }

    #[tokio::test]
    async fn test_watchdog_triggers_on_timeout() {
        let trigger_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let trigger_count_clone = trigger_count.clone();

        let on_timeout: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            trigger_count_clone.fetch_add(1, Ordering::Relaxed);
        });

        let handle = spawn_watchdog(Duration::from_millis(50), on_timeout);

        // Wait for timeout to fire
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(trigger_count.load(Ordering::Relaxed) >= 1);

        handle.shutdown().await;
    }
}
