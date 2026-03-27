//! Async indexing pipeline for SpiritFS.
//!
//! Watches for file changes (via inotify/notify), generates embeddings,
//! and upserts them into the vector store. All indexing is asynchronous
//! and runs at P4 (idle) priority — it never blocks POSIX file operations.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender, TrySendError};
use tokio::sync::Semaphore;
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

use crate::cas::ContentAddressableStore;
use crate::config::SpiritFsConfig;
use crate::embedding::{chunk_text, detect_mime_type, ChunkConfig, EmbeddingClient};
use crate::error::{Result, SpiritFsError};
use crate::metadata::MetadataStore;
use crate::metrics::SpiritFsMetrics;
use crate::qdrant::{VectorPayload, VectorStore};
use crate::types::{should_index, IndexJob, IndexStats};

/// The asynchronous indexing pipeline.
pub struct IndexPipeline {
    config: SpiritFsConfig,
    embedding_client: EmbeddingClient,
    vector_store: Arc<VectorStore>,
    metadata: Arc<MetadataStore>,
    #[allow(dead_code)]
    cas: Arc<ContentAddressableStore>,
    metrics: Arc<SpiritFsMetrics>,
    job_tx: Sender<IndexJob>,
    job_rx: Receiver<IndexJob>,
    concurrency_semaphore: Arc<Semaphore>,
}

impl IndexPipeline {
    /// Create a new indexing pipeline.
    pub fn new(
        config: SpiritFsConfig,
        embedding_client: EmbeddingClient,
        vector_store: Arc<VectorStore>,
        metadata: Arc<MetadataStore>,
        cas: Arc<ContentAddressableStore>,
        metrics: Arc<SpiritFsMetrics>,
    ) -> Self {
        let (job_tx, job_rx) = crossbeam_channel::bounded(config.index_queue_capacity);
        let concurrency_semaphore = Arc::new(Semaphore::new(config.max_concurrent_embeddings));

        Self {
            config,
            embedding_client,
            vector_store,
            metadata,
            cas,
            metrics,
            job_tx,
            job_rx,
            concurrency_semaphore,
        }
    }

    /// Get a sender handle for submitting index jobs.
    pub fn job_sender(&self) -> Sender<IndexJob> {
        self.job_tx.clone()
    }

    /// Submit an index job (non-blocking).
    pub fn submit(&self, job: IndexJob) {
        match self.job_tx.try_send(job) {
            Ok(()) => {
                self.metrics.queue_depth.inc();
            }
            Err(TrySendError::Full(job)) => {
                warn!(?job, "index queue full, dropping job");
                self.metrics.jobs_dropped.inc();
            }
            Err(TrySendError::Disconnected(_)) => {
                warn!("index queue disconnected");
            }
        }
    }

    /// Run the indexing pipeline loop. Call this in a spawned tokio task.
    pub async fn run(&self, shutdown: tokio::sync::watch::Receiver<bool>) {
        info!("index pipeline started");

        loop {
            // Check for shutdown signal
            if *shutdown.borrow() {
                info!("index pipeline shutting down");
                break;
            }

            // Try to receive a job (non-blocking with short timeout)
            let job = {
                let rx = self.job_rx.clone();
                tokio::task::spawn_blocking(move || rx.recv_timeout(Duration::from_millis(200)))
                    .await
            };

            match job {
                Ok(Ok(job)) => {
                    self.metrics.queue_depth.dec();
                    self.process_job(job).await;
                }
                Ok(Err(crossbeam_channel::RecvTimeoutError::Timeout)) => {
                    // Check shutdown again
                    if shutdown.has_changed().unwrap_or(false) && *shutdown.borrow() {
                        break;
                    }
                }
                Ok(Err(crossbeam_channel::RecvTimeoutError::Disconnected)) | Err(_) => break,
            }
        }

        info!("index pipeline stopped");
    }

    /// Process a single index job.
    #[instrument(skip(self))]
    async fn process_job(&self, job: IndexJob) {
        match job {
            IndexJob::Update(path) => {
                if let Err(e) = self.index_file(&path).await {
                    warn!(path = %path.display(), error = %e, "failed to index file");
                    self.metrics.errors_total.inc();
                }
            }
            IndexJob::Remove(path) => {
                if let Err(e) = self.remove_file(&path).await {
                    warn!(path = %path.display(), error = %e, "failed to remove from index");
                    self.metrics.errors_total.inc();
                }
            }
            IndexJob::RebuildAll => {
                if let Err(e) = self.rebuild_all().await {
                    error!(error = %e, "failed to rebuild index");
                    self.metrics.errors_total.inc();
                }
            }
        }
    }

    /// Index a single file: read content, chunk, embed, upsert.
    #[instrument(skip(self))]
    async fn index_file(&self, path: &Path) -> Result<()> {
        let _permit = self
            .concurrency_semaphore
            .acquire()
            .await
            .map_err(|e| SpiritFsError::Channel(e.to_string()))?;

        // Resolve to backing path
        let backing_path = self
            .config
            .backing_dir
            .join(path.strip_prefix(&self.config.mount_point).unwrap_or(path));

        // Check file exists and is indexable
        let file_meta = match std::fs::metadata(&backing_path) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                debug!(path = %path.display(), "file no longer exists, skipping");
                return Ok(());
            }
            Err(e) => return Err(e.into()),
        };

        if !should_index(path, file_meta.len(), self.config.max_embedding_file_size) {
            debug!(path = %path.display(), "file not eligible for indexing");
            return Ok(());
        }

        // Read file content
        let content = std::fs::read_to_string(&backing_path).map_err(|e| {
            SpiritFsError::Embedding(format!("failed to read {}: {e}", backing_path.display()))
        })?;

        if content.trim().is_empty() {
            return Ok(());
        }

        // Delete existing vectors for this path
        self.vector_store.delete_by_path(path).await?;

        // Chunk the text
        let chunk_config = ChunkConfig {
            max_tokens: self.config.chunk_max_tokens,
            overlap_tokens: self.config.chunk_overlap_tokens,
        };
        let chunks = chunk_text(path, &content, &chunk_config);

        if chunks.is_empty() {
            return Ok(());
        }

        // Generate embeddings and upsert
        let mime_type = detect_mime_type(path).unwrap_or_else(|| "text/plain".to_string());
        let mut first_point_id = None;

        for chunk in &chunks {
            let embedding = match self.embedding_client.embed_text(&chunk.text).await {
                Ok(e) => e,
                Err(e) => {
                    warn!(
                        path = %path.display(),
                        chunk_index = chunk.chunk_index,
                        error = %e,
                        "embedding failed for chunk"
                    );
                    continue;
                }
            };

            let point_id = Uuid::new_v4();
            if first_point_id.is_none() {
                first_point_id = Some(point_id);
            }

            let payload = VectorPayload {
                path: path.to_string_lossy().to_string(),
                name: path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default(),
                chunk_text: chunk.text.clone(),
                chunk_index: chunk.chunk_index,
                mime_type: mime_type.clone(),
                tags: String::new(),
            };

            self.vector_store
                .upsert(point_id, embedding, payload)
                .await?;
            self.metrics.embeddings_generated.inc();
        }

        // Update metadata with the first chunk's embedding ID
        if let Some(point_id) = first_point_id {
            if let Ok(Some(meta)) = self.metadata.get_by_path(path) {
                let _ = self.metadata.set_embedding_id(meta.inode, point_id);
            }
        }

        self.metrics.files_indexed.inc();
        debug!(
            path = %path.display(),
            chunks = chunks.len(),
            "indexed file"
        );
        Ok(())
    }

    /// Remove a file from the vector index.
    async fn remove_file(&self, path: &Path) -> Result<()> {
        let deleted = self.vector_store.delete_by_path(path).await?;
        debug!(path = %path.display(), deleted, "removed from index");
        Ok(())
    }

    /// Rebuild the entire index from scratch.
    async fn rebuild_all(&self) -> Result<()> {
        info!("rebuilding entire vector index");
        self.vector_store.rebuild_collection().await?;

        // Get all files from metadata that need indexing
        let files = self.metadata.files_needing_indexing(100_000)?;
        info!(files = files.len(), "re-indexing files");

        for meta in &files {
            if let Err(e) = self.index_file(&meta.path).await {
                warn!(path = %meta.path.display(), error = %e, "rebuild: failed to index");
            }
        }

        info!("index rebuild complete");
        Ok(())
    }

    /// Get current index statistics.
    pub fn stats(&self) -> Result<IndexStats> {
        let total_files = self.metadata.file_count()?;
        let indexed_files = self.metadata.indexed_file_count()?;

        Ok(IndexStats {
            total_files,
            indexed_files,
            pending_files: total_files.saturating_sub(indexed_files),
            total_vectors: 0, // Filled asynchronously
            queue_depth: self.job_rx.len(),
        })
    }
}

/// File system watcher that feeds the indexing pipeline.
pub struct FileWatcher {
    _watcher: notify::RecommendedWatcher,
}

impl FileWatcher {
    /// Start watching a directory for changes and feed them to the index pipeline.
    pub fn start(watch_dir: &Path, job_tx: Sender<IndexJob>, mount_point: &Path) -> Result<Self> {
        use notify::{EventKind, Watcher};

        let mp = mount_point.to_path_buf();
        let mut watcher =
            notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
                let event = match event {
                    Ok(e) => e,
                    Err(e) => {
                        warn!(error = %e, "file watcher error");
                        return;
                    }
                };

                for path in &event.paths {
                    // Convert backing path to mount-relative path
                    let relative = path
                        .strip_prefix(&mp)
                        .map_or_else(|_| path.clone(), |p| mp.join(p));

                    let job = match event.kind {
                        EventKind::Create(_) | EventKind::Modify(_) => IndexJob::Update(relative),
                        EventKind::Remove(_) => IndexJob::Remove(relative),
                        _ => continue,
                    };

                    let _ = job_tx.try_send(job);
                }
            })
            .map_err(|e| SpiritFsError::Config(format!("failed to create watcher: {e}")))?;

        watcher
            .watch(watch_dir, notify::RecursiveMode::Recursive)
            .map_err(|e| {
                SpiritFsError::Config(format!("failed to watch {}: {e}", watch_dir.display()))
            })?;

        info!(dir = %watch_dir.display(), "file watcher started");
        Ok(Self { _watcher: watcher })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_index_job_variants() {
        let update = IndexJob::Update(PathBuf::from("/test.rs"));
        let remove = IndexJob::Remove(PathBuf::from("/test.rs"));
        let rebuild = IndexJob::RebuildAll;

        assert!(matches!(update, IndexJob::Update(_)));
        assert!(matches!(remove, IndexJob::Remove(_)));
        assert!(matches!(rebuild, IndexJob::RebuildAll));
    }
}
