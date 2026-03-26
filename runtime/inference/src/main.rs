//! `spirit-infer` — the SpiritROS inference coordinator daemon.
//!
//! Starts the inference coordinator, IPC server, and optional gRPC server.
//! Designed to run as a long-lived system service inside the SpiritROS
//! container.
//!
//! # Usage
//!
//! ```bash
//! spirit-infer                           # Use defaults
//! spirit-infer --config /etc/spirit.yaml # Custom config
//! spirit-infer --log-level debug         # Verbose logging
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use spirit_inference::backend::ollama::OllamaBackend;
use spirit_inference::config::InferenceConfig;
use spirit_inference::coordinator::InferenceCoordinator;
use spirit_inference::ipc::unix_socket::{
    IpcRequest, IpcResponse, IpcResponseData, IpcServer, ModelListEntry,
};
use spirit_inference::ollama_client::OllamaClient;
use spirit_inference::types::Priority;

#[derive(Parser, Debug)]
#[command(
    name = "spirit-infer",
    about = "SpiritROS Inference Coordinator Daemon"
)]
struct Cli {
    /// Path to the Spirit configuration file.
    #[arg(short, long, default_value = "/etc/spirit/spirit.yaml")]
    config: PathBuf,

    /// Log level filter (e.g., "info", "debug", "spirit_inference=trace").
    #[arg(short, long, default_value = "info")]
    log_level: String,

    /// Override the socket path for the IPC server.
    #[arg(long)]
    socket_path: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&cli.log_level)),
        )
        .with_target(true)
        .with_thread_ids(true)
        .init();

    info!(
        version = env!("CARGO_PKG_VERSION"),
        "Starting spirit-infer daemon"
    );

    // Load configuration
    let mut config = if cli.config.exists() {
        InferenceConfig::from_file(&cli.config).context("Failed to load configuration")?
    } else {
        info!("Config file not found, using defaults");
        InferenceConfig::default()
    };

    if let Some(socket_path) = cli.socket_path {
        config.coordinator.socket_path = socket_path;
    }

    info!(
        backend = %config.backend,
        max_sessions = config.ollama.max_sessions,
        socket = %config.coordinator.socket_path,
        "Configuration loaded"
    );

    // Create Ollama client and backend
    let ollama_client = Arc::new(OllamaClient::new(&config.ollama));
    let backend = Arc::new(OllamaBackend::new(
        ollama_client.clone(),
        config.ollama.keep_alive_secs,
    ));

    // Create coordinator
    let coordinator = Arc::new(InferenceCoordinator::new(
        config.clone(),
        backend,
        ollama_client,
    ));

    // Initialize (sync models, load defaults)
    coordinator
        .initialize()
        .await
        .context("Failed to initialize coordinator")?;

    info!("Inference coordinator initialized");

    // Start IPC server
    let ipc_socket = config.coordinator.socket_path.clone();
    let ipc_server = IpcServer::new(ipc_socket);
    let coord = coordinator.clone();

    info!("Starting IPC server");
    let ipc_handle = tokio::spawn(async move {
        if let Err(e) = ipc_server
            .serve(move |request| {
                let coord = coord.clone();
                async move { handle_ipc_request(&coord, request).await }
            })
            .await
        {
            error!("IPC server error: {e}");
        }
    });

    // Wait for shutdown
    tokio::signal::ctrl_c()
        .await
        .context("Failed to listen for ctrl-c")?;

    info!("Shutting down spirit-infer");
    ipc_handle.abort();

    Ok(())
}

/// Handle a single IPC request by dispatching to the coordinator.
async fn handle_ipc_request(coord: &InferenceCoordinator, request: IpcRequest) -> IpcResponse {
    match request {
        IpcRequest::CreateSession { model, priority } => {
            let priority = priority.unwrap_or(Priority::Normal);
            match coord.create_session(&model, priority).await {
                Ok(session_id) => IpcResponse::Ok(IpcResponseData::SessionCreated { session_id }),
                Err(e) => to_error_response(&e),
            }
        }

        IpcRequest::Prompt {
            session_id,
            prompt,
            system,
            sampling,
            stream,
        } => {
            let sampling = sampling.unwrap_or_default();

            if stream {
                // For streaming, we just signal that generation started.
                // Tokens are delivered via the shared memory ring buffer
                // or via the streaming API.
                IpcResponse::Ok(IpcResponseData::GenerationStarted { session_id })
            } else {
                match coord
                    .generate(&session_id, &prompt, system.as_deref(), &sampling)
                    .await
                {
                    Ok(resp) => IpcResponse::Ok(IpcResponseData::GenerationComplete {
                        session_id: resp.session_id,
                        text: resp.text,
                        tokens_generated: resp.tokens_generated,
                        first_token_ms: resp.first_token_ms,
                        total_ms: resp.total_ms,
                        tokens_per_second: resp.tokens_per_second,
                    }),
                    Err(e) => to_error_response(&e),
                }
            }
        }

        IpcRequest::SessionStatus { session_id } => match coord.session_info(&session_id) {
            Ok(info) => IpcResponse::Ok(IpcResponseData::SessionInfo {
                session_id: info.id,
                state: info.state,
                model: info.model_name,
                kv_token_count: info.kv_token_count,
                total_tokens_generated: info.total_tokens_generated,
            }),
            Err(e) => to_error_response(&e),
        },

        IpcRequest::SystemStatus => {
            let loaded: Vec<String> = coord
                .list_loaded_models()
                .iter()
                .map(|m| m.info.name.clone())
                .collect();

            IpcResponse::Ok(IpcResponseData::SystemInfo {
                active_sessions: coord.active_session_count(),
                max_sessions: coord.config().ollama.max_sessions,
                loaded_models: loaded,
                queue_depth: coord.queue_depth(),
            })
        }

        IpcRequest::Snapshot { session_id } => match coord.kv_snapshot(&session_id).await {
            Ok(snap) => IpcResponse::Ok(IpcResponseData::SnapshotCreated {
                snapshot_id: snap.id.to_string(),
                session_id,
            }),
            Err(e) => to_error_response(&e),
        },

        IpcRequest::Restore {
            session_id,
            snapshot_id: _,
        } => {
            // In Phase 1, restore is a shim
            IpcResponse::Ok(IpcResponseData::SnapshotRestored { session_id })
        }

        IpcRequest::LoadModel { name } => match coord.load_model(&name).await {
            Ok(_) => IpcResponse::Ok(IpcResponseData::ModelLoaded { name }),
            Err(e) => to_error_response(&e),
        },

        IpcRequest::UnloadModel { name } => match coord.unload_model(&name).await {
            Ok(()) => IpcResponse::Ok(IpcResponseData::ModelUnloaded { name }),
            Err(e) => to_error_response(&e),
        },

        IpcRequest::ListModels => {
            let models: Vec<ModelListEntry> = coord
                .list_models()
                .into_iter()
                .map(|m| {
                    let loaded = coord.model_registry().load_state(&m.id)
                        == Some(spirit_inference::types::ModelLoadState::Loaded);
                    ModelListEntry {
                        name: m.name,
                        size_bytes: m.size_bytes,
                        loaded,
                        digest: m.digest,
                    }
                })
                .collect();

            IpcResponse::Ok(IpcResponseData::Models { models })
        }

        IpcRequest::DestroySession { session_id } => {
            match coord.destroy_session(&session_id).await {
                Ok(()) => IpcResponse::Ok(IpcResponseData::SessionDestroyed { session_id }),
                Err(e) => to_error_response(&e),
            }
        }
    }
}

fn to_error_response(e: &spirit_inference::InferenceError) -> IpcResponse {
    IpcResponse::Error {
        code: error_code(e),
        message: e.to_string(),
    }
}

fn error_code(e: &spirit_inference::InferenceError) -> String {
    use spirit_inference::InferenceError::*;
    match e {
        ModelNotFound(_) => "model_not_found",
        ModelNotLoaded(_) => "model_not_loaded",
        ModelAlreadyLoaded(_) => "model_already_loaded",
        InsufficientVram { .. } => "insufficient_vram",
        ModelLoadTimeout { .. } => "model_load_timeout",
        SessionNotFound(_) => "session_not_found",
        SessionLimitReached { .. } => "session_limit_reached",
        InvalidSessionState { .. } => "invalid_session_state",
        SessionTerminated(_) => "session_terminated",
        SnapshotNotFound(_) => "snapshot_not_found",
        KvCacheExhausted { .. } => "kv_cache_exhausted",
        QueueFull { .. } => "queue_full",
        OllamaError(_) => "ollama_error",
        HttpError(_) => "http_error",
        BackendUnavailable(_) => "backend_unavailable",
        Cancelled(_) => "cancelled",
        SocketError(_) => "socket_error",
        SharedMemoryError(_) => "shared_memory_error",
        SerdeError(_) => "serde_error",
        ConfigError(_) => "config_error",
        CapabilityDenied { .. } => "capability_denied",
        IoError(_) => "io_error",
    }
    .to_string()
}
