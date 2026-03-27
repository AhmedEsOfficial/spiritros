//! Spirit Scheduler daemon binary.
//!
//! Runs the inference-aware scheduler as a long-lived daemon process.
//! Listens for phase hints and process registration via Unix socket IPC.
//!
//! Usage:
//!   spirit-scheduler [OPTIONS]
//!
//! Options:
//!   --config <PATH>     Path to scheduler config YAML file
//!   --socket <PATH>     Override IPC socket path
//!   --no-realtime       Disable SCHED_FIFO even if CAP_SYS_NICE is available
//!   --no-gpu            Disable GPU exclusive mode management
//!   --no-watchdog       Disable the starvation watchdog

use anyhow::Context;
use clap::Parser;
use tracing_subscriber::EnvFilter;

/// Spirit Scheduler — Inference-Aware Process Scheduling Daemon
#[derive(Parser, Debug)]
#[command(name = "spirit-scheduler")]
#[command(about = "Inference-aware scheduler for SpiritROS")]
struct Args {
    /// Path to the scheduler configuration YAML file.
    #[arg(short, long)]
    config: Option<String>,

    /// Override the IPC socket path.
    #[arg(long)]
    socket: Option<String>,

    /// Disable real-time scheduling (SCHED_FIFO).
    #[arg(long, default_value_t = false)]
    no_realtime: bool,

    /// Disable GPU exclusive compute mode.
    #[arg(long, default_value_t = false)]
    no_gpu: bool,

    /// Disable the starvation watchdog.
    #[arg(long, default_value_t = false)]
    no_watchdog: bool,

    /// Log level filter (e.g., "info", "debug", "spirit_scheduler=trace").
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&args.log_level)),
        )
        .with_target(true)
        .with_thread_ids(true)
        .init();

    tracing::info!("Starting Spirit Scheduler daemon");

    // Load configuration
    let mut config = spirit_scheduler::SchedulerConfig::load(args.config.as_deref())
        .context("Failed to load scheduler configuration")?;

    // Apply CLI overrides
    if let Some(socket) = args.socket {
        config.socket_path = socket;
    }
    if args.no_realtime {
        config.enable_realtime = false;
    }
    if args.no_gpu {
        config.gpu_exclusive_mode = false;
    }
    if args.no_watchdog {
        config.watchdog_enabled = false;
    }

    tracing::info!(config = ?config, "Configuration loaded");

    // Create and run scheduler
    let mut scheduler =
        spirit_scheduler::SpiritScheduler::new(config).context("Failed to initialize scheduler")?;

    // Install signal handlers
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .context("Failed to install SIGTERM handler")?;
    let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
        .context("Failed to install SIGINT handler")?;

    tokio::select! {
        result = scheduler.run() => {
            result.map_err(|e| anyhow::anyhow!("Scheduler error: {e}"))?;
        }
        _ = sigterm.recv() => {
            tracing::info!("Received SIGTERM, shutting down");
        }
        _ = sigint.recv() => {
            tracing::info!("Received SIGINT, shutting down");
        }
    }

    tracing::info!("Spirit Scheduler daemon stopped");
    Ok(())
}
