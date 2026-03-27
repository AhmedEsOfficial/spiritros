//! SpiritFS — FUSE daemon entry point.
//!
//! Mounts the SpiritFS semantic filesystem, starts the indexing pipeline
//! and file watcher, and serves until signaled to stop.
//!
//! Usage:
//!   spiritfs mount [--config <path>] [--backing-dir <path>] [--mount-point <path>]
//!   spiritfs search <query> [--limit <n>]
//!   spiritfs similar <file> [--limit <n>]
//!   spiritfs summarize <directory>
//!   spiritfs index status
//!   spiritfs index rebuild

#![cfg(feature = "fuse")]

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use prometheus_client::registry::Registry;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use spirit_spiritfs::cache::ModelCache;
use spirit_spiritfs::cas::ContentAddressableStore;
use spirit_spiritfs::config::SpiritFsConfig;
use spirit_spiritfs::embedding::EmbeddingClient;
use spirit_spiritfs::fuse::SpiritFuse;
use spirit_spiritfs::index::{FileWatcher, IndexPipeline};
use spirit_spiritfs::metadata::MetadataStore;
use spirit_spiritfs::metrics::SpiritFsMetrics;
use spirit_spiritfs::qdrant::VectorStore;
use spirit_spiritfs::semantic::SemanticEngine;

/// SpiritFS — Semantic FUSE filesystem for SpiritROS
#[derive(Parser)]
#[command(name = "spiritfs", about = "Semantic FUSE filesystem")]
struct Cli {
    /// Path to YAML configuration file
    #[arg(short, long)]
    config: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Mount the SpiritFS filesystem
    Mount {
        /// Backing directory (overrides config)
        #[arg(long)]
        backing_dir: Option<PathBuf>,

        /// Mount point (overrides config)
        #[arg(long)]
        mount_point: Option<PathBuf>,

        /// Allow other users to access the mount
        #[arg(long)]
        allow_other: bool,
    },

    /// Search files by semantic meaning
    Search {
        /// Search query
        query: String,

        /// Maximum number of results
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },

    /// Find files similar to a given file
    Similar {
        /// Path to the reference file
        file: PathBuf,

        /// Maximum number of results
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },

    /// Summarize directory contents
    Summarize {
        /// Directory to summarize
        directory: PathBuf,
    },

    /// Index management commands
    Index {
        #[command(subcommand)]
        action: IndexAction,
    },
}

#[derive(Subcommand)]
enum IndexAction {
    /// Show index status
    Status,
    /// Rebuild the entire index
    Rebuild,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    // Load configuration
    let mut config = SpiritFsConfig::load(cli.config.as_deref())?;

    match cli.command {
        Commands::Mount {
            backing_dir,
            mount_point,
            allow_other,
        } => {
            // Apply CLI overrides
            if let Some(dir) = backing_dir {
                config.backing_dir = dir;
            }
            if let Some(mp) = mount_point {
                config.mount_point = mp;
            }
            if allow_other {
                config.allow_other = true;
            }

            run_daemon(config).await
        }

        Commands::Search { query, limit } => {
            let engine = create_semantic_engine(&config).await?;
            let results = engine.search(&query, limit).await?;

            if results.is_empty() {
                println!("No results found for: {query}");
            } else {
                println!("Results for: {query}\n");
                for (i, result) in results.iter().enumerate() {
                    println!(
                        "  {}. {} (score: {:.4})",
                        i + 1,
                        result.path.display(),
                        result.score
                    );
                    if let Some(text) = &result.chunk_text {
                        let preview: String = text.chars().take(120).collect();
                        println!("     {preview}...");
                    }
                }
            }
            Ok(())
        }

        Commands::Similar { file, limit } => {
            let engine = create_semantic_engine(&config).await?;
            let results = engine.similar(&file, limit).await?;

            if results.is_empty() {
                println!("No similar files found for: {}", file.display());
            } else {
                println!("Files similar to: {}\n", file.display());
                for (i, result) in results.iter().enumerate() {
                    println!(
                        "  {}. {} (score: {:.4})",
                        i + 1,
                        result.path.display(),
                        result.score
                    );
                }
            }
            Ok(())
        }

        Commands::Summarize { directory } => {
            let engine = create_semantic_engine(&config).await?;
            let summary = engine.summarize(&directory).await?;
            println!("{summary}");
            Ok(())
        }

        Commands::Index { action } => match action {
            IndexAction::Status => {
                let (metadata, _, _, metrics, _) = create_subsystems(&config).await?;
                let pipeline = create_index_pipeline(&config, &metadata, &metrics).await?;
                let stats = pipeline.stats()?;

                println!("SpiritFS Index Status");
                println!("  Total files:   {}", stats.total_files);
                println!("  Indexed files: {}", stats.indexed_files);
                println!("  Pending:       {}", stats.pending_files);
                println!("  Queue depth:   {}", stats.queue_depth);
                Ok(())
            }
            IndexAction::Rebuild => {
                info!("rebuilding index...");
                let (metadata, vector_store, cas, metrics, _) = create_subsystems(&config).await?;

                let embedding_client = EmbeddingClient::new(&config);
                let pipeline = IndexPipeline::new(
                    config.clone(),
                    embedding_client,
                    vector_store,
                    metadata,
                    cas,
                    metrics,
                );

                pipeline.submit(spirit_spiritfs::IndexJob::RebuildAll);

                let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
                let pipeline_handle = tokio::spawn(async move {
                    pipeline.run(shutdown_rx).await;
                });

                // Give it time to process, then shut down
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                let _ = shutdown_tx.send(true);
                pipeline_handle.await?;

                println!("Index rebuild complete.");
                Ok(())
            }
        },
    }
}

/// Run the FUSE daemon (mount + index pipeline + file watcher).
async fn run_daemon(config: SpiritFsConfig) -> anyhow::Result<()> {
    info!(
        mount_point = %config.mount_point.display(),
        backing_dir = %config.backing_dir.display(),
        "starting SpiritFS daemon"
    );

    // Ensure directories exist
    std::fs::create_dir_all(&config.backing_dir)?;
    std::fs::create_dir_all(&config.mount_point)?;

    // Create subsystems
    let (metadata, vector_store, cas, metrics_arc, registry) = create_subsystems(&config).await?;

    let embedding_client = EmbeddingClient::new(&config);

    // Create index pipeline
    let pipeline = IndexPipeline::new(
        config.clone(),
        embedding_client,
        vector_store.clone(),
        metadata.clone(),
        cas.clone(),
        metrics_arc.clone(),
    );

    let index_tx = pipeline.job_sender();

    // Start index pipeline
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let pipeline_handle = tokio::spawn(async move {
        pipeline.run(shutdown_rx).await;
    });

    // Start file watcher
    let _watcher = if config.enable_watcher {
        match FileWatcher::start(&config.backing_dir, index_tx.clone(), &config.mount_point) {
            Ok(w) => Some(w),
            Err(e) => {
                error!(error = %e, "failed to start file watcher, continuing without");
                None
            }
        }
    } else {
        None
    };

    // Start Prometheus metrics HTTP server
    let metrics_bind = config.metrics_bind.clone();
    let metrics_registry = registry.clone();
    tokio::spawn(async move {
        if let Err(e) = serve_metrics(&metrics_bind, metrics_registry).await {
            error!(error = %e, "metrics server failed");
        }
    });

    // Set up signal handling for graceful shutdown
    let shutdown_tx_clone = shutdown_tx.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install CTRL+C handler");
        info!("received CTRL+C, shutting down...");
        let _ = shutdown_tx_clone.send(true);
    });

    // Mount FUSE filesystem (this blocks until unmounted)
    let fuse_mount_point = config.mount_point.clone();
    let fuse_backing_dir = config.backing_dir.clone();
    let allow_other = config.allow_other;

    let fuse_fs = SpiritFuse::new(
        fuse_backing_dir,
        fuse_mount_point.clone(),
        index_tx,
        metrics_arc,
    );

    info!("mounting SpiritFS at {}", fuse_mount_point.display());

    // Run FUSE mount in a blocking thread
    let fuse_handle =
        tokio::task::spawn_blocking(move || fuse_fs.mount(&fuse_mount_point, allow_other));

    // Wait for FUSE to finish (unmount or error)
    match fuse_handle.await? {
        Ok(()) => info!("SpiritFS unmounted cleanly"),
        Err(e) => error!(error = %e, "FUSE mount failed"),
    }

    // Shutdown pipeline
    let _ = shutdown_tx.send(true);
    pipeline_handle.await?;

    info!("SpiritFS daemon stopped");
    Ok(())
}

/// Create shared subsystems (metadata, vector store, CAS, metrics + registry).
async fn create_subsystems(
    config: &SpiritFsConfig,
) -> anyhow::Result<(
    Arc<MetadataStore>,
    Arc<VectorStore>,
    Arc<ContentAddressableStore>,
    Arc<SpiritFsMetrics>,
    Arc<parking_lot::Mutex<Registry>>,
)> {
    let metadata = Arc::new(MetadataStore::open(&config.metadata_db)?);
    let vector_store = Arc::new(VectorStore::connect(config).await?);
    let cas = Arc::new(ContentAddressableStore::new(&config.cas_dir)?);

    let mut registry = Registry::default();
    let metrics = Arc::new(SpiritFsMetrics::new(&mut registry));
    let registry = Arc::new(parking_lot::Mutex::new(registry));

    Ok((metadata, vector_store, cas, metrics, registry))
}

/// Create a semantic engine for CLI queries.
async fn create_semantic_engine(config: &SpiritFsConfig) -> anyhow::Result<SemanticEngine> {
    let metadata = Arc::new(MetadataStore::open(&config.metadata_db)?);
    let vector_store = Arc::new(VectorStore::connect(config).await?);
    let cas = Arc::new(ContentAddressableStore::new(&config.cas_dir)?);
    let embedding_client = EmbeddingClient::new(config);

    Ok(SemanticEngine::new(
        config.clone(),
        embedding_client,
        vector_store,
        metadata,
        cas,
    ))
}

/// Create an index pipeline for CLI commands.
async fn create_index_pipeline(
    config: &SpiritFsConfig,
    metadata: &Arc<MetadataStore>,
    metrics: &Arc<SpiritFsMetrics>,
) -> anyhow::Result<IndexPipeline> {
    let vector_store = Arc::new(VectorStore::connect(config).await?);
    let cas = Arc::new(ContentAddressableStore::new(&config.cas_dir)?);
    let embedding_client = EmbeddingClient::new(config);

    Ok(IndexPipeline::new(
        config.clone(),
        embedding_client,
        vector_store,
        metadata.clone(),
        cas,
        metrics.clone(),
    ))
}

/// Serve Prometheus metrics on a simple HTTP endpoint.
async fn serve_metrics(
    bind_addr: &str,
    registry: Arc<parking_lot::Mutex<Registry>>,
) -> anyhow::Result<()> {
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    let listener = TcpListener::bind(bind_addr).await?;
    info!(bind = bind_addr, "metrics server listening");

    loop {
        let (mut stream, _) = listener.accept().await?;

        // Encode metrics from the shared registry
        let body = {
            let reg = registry.lock();
            let mut buf = String::new();
            prometheus_client::encoding::text::encode(&mut buf, &reg).unwrap_or_else(|e| {
                buf = format!("# encoding error: {e}\n");
            });
            buf
        };

        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4; charset=utf-8\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = stream.write_all(response.as_bytes()).await;
    }
}
