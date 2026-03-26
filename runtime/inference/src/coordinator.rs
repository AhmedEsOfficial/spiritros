//! The inference coordinator — central dispatch for all inference operations.
//!
//! The coordinator owns and composes:
//! - [`SessionManager`] — session lifecycle
//! - [`ModelRegistry`] — model catalog and load state
//! - [`BatchScheduler`] — request queuing and prioritization
//! - [`InferenceBackend`] — the actual inference engine (Ollama)
//! - [`InferenceMetrics`] — Prometheus metrics
//!
//! All public methods are async and thread-safe. The coordinator is designed
//! to be shared via `Arc` across multiple IPC handlers and API servers.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use futures::Stream;
use parking_lot::RwLock;
use prometheus_client::registry::Registry;
use tokio_stream::StreamExt;
use tracing::{info, instrument, warn};

use crate::backend::{InferenceBackend, ModelConfig, ModelHandle};
use crate::batch::{BatchScheduler, BatchSchedulerConfig};
use crate::config::InferenceConfig;
use crate::error::{InferenceError, Result};
use crate::metrics::InferenceMetrics;
use crate::model_registry::ModelRegistry;
use crate::ollama_client::OllamaClient;
use crate::session::SessionManager;
use crate::types::{
    InferenceRequest, InferenceResponse, KvSnapshot, ModelId, Priority, SamplingConfig, SessionId,
    SessionState, StopReason, Token,
};

/// The inference coordinator — the central orchestration point.
///
/// Composes all inference subsystems and provides the public API for:
/// - Session management (create, prompt, destroy)
/// - Model management (load, unload, list)
/// - KV-cache operations (snapshot, restore)
/// - Metrics and status queries
pub struct InferenceCoordinator {
    /// Configuration.
    config: InferenceConfig,
    /// Session manager.
    sessions: SessionManager,
    /// Model registry.
    registry: Arc<ModelRegistry>,
    /// Batch scheduler.
    batch_scheduler: Arc<BatchScheduler>,
    /// Active backend.
    backend: Arc<dyn InferenceBackend>,
    /// Model handles for loaded models (model name → handle).
    model_handles: RwLock<std::collections::HashMap<String, ModelHandle>>,
    /// Prometheus metrics.
    metrics: Arc<InferenceMetrics>,
    /// Metrics registry (for encoding).
    metrics_registry: Arc<RwLock<Registry>>,
}

impl InferenceCoordinator {
    /// Create a new coordinator with the given configuration.
    pub fn new(
        config: InferenceConfig,
        backend: Arc<dyn InferenceBackend>,
        ollama_client: Arc<OllamaClient>,
    ) -> Self {
        let mut registry = Registry::default();
        let metrics = Arc::new(InferenceMetrics::new(&mut registry));

        let sessions = SessionManager::new(
            config.ollama.max_sessions,
            config.coordinator.token_buffer_size,
        );

        let batch_config = BatchSchedulerConfig {
            max_batch_size: config.ollama.max_sessions,
            batch_timeout: std::time::Duration::from_millis(50),
            max_queue_depth: config.coordinator.max_queue_depth,
        };

        Self {
            sessions,
            registry: Arc::new(ModelRegistry::new(ollama_client)),
            batch_scheduler: Arc::new(BatchScheduler::new(batch_config)),
            backend,
            model_handles: RwLock::new(std::collections::HashMap::new()),
            metrics,
            metrics_registry: Arc::new(RwLock::new(registry)),
            config,
        }
    }

    // ── Initialization ──────────────────────────────────────────────────

    /// Initialize the coordinator: sync models, load defaults.
    #[instrument(skip(self))]
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing inference coordinator");

        // Check backend health
        if !self.backend.health_check().await {
            warn!(
                "Backend '{}' is not healthy — will retry on first request",
                self.backend.name()
            );
        } else {
            info!("Backend '{}' is healthy", self.backend.name());
        }

        // Sync model registry
        match self.registry.sync_from_ollama().await {
            Ok(count) => {
                info!(count, "Synced models from Ollama");
            }
            Err(e) => {
                warn!("Failed to sync models from Ollama: {e}");
            }
        }

        // Load default model if configured
        if let Some(ref default_model) = self.config.ollama.default_model {
            info!(model = %default_model, "Loading default model");
            match self.load_model(default_model).await {
                Ok(_) => info!(model = %default_model, "Default model loaded"),
                Err(e) => warn!(
                    model = %default_model,
                    error = %e,
                    "Failed to load default model"
                ),
            }
        }

        Ok(())
    }

    // ── Session Management ──────────────────────────────────────────────

    /// Create a new inference session for a model.
    #[instrument(skip(self))]
    pub async fn create_session(&self, model_name: &str, priority: Priority) -> Result<SessionId> {
        // Resolve model
        let model_id = self
            .registry
            .resolve_name(model_name)
            .ok_or_else(|| InferenceError::ModelNotFound(model_name.into()))?;

        // Ensure model is loaded (check without holding guard across await)
        let is_loaded = self.model_handles.read().contains_key(model_name);
        if !is_loaded {
            self.load_model(model_name).await?;
        }

        // Create session
        let session_id =
            self.sessions
                .create_session(model_id, model_name.to_string(), priority)?;

        // Mark session ready (model is loaded)
        self.sessions
            .with_session_mut(&session_id, |s| s.mark_ready())?
            .map_err(|e| {
                // Clean up on failure
                let _ = self.sessions.destroy_session(&session_id);
                e
            })?;

        // Track in registry
        self.registry.increment_sessions(&model_id)?;

        // Update metrics
        self.metrics.sessions_created_total.inc();
        self.metrics
            .active_sessions
            .set(self.sessions.active_count() as i64);

        Ok(session_id)
    }

    /// Destroy an inference session.
    #[instrument(skip(self))]
    pub async fn destroy_session(&self, session_id: &SessionId) -> Result<()> {
        // Get model ID before destroying
        let model_id = self.sessions.with_session(session_id, |s| s.model_id)?;

        self.sessions.destroy_session(session_id)?;

        // Update registry session count
        let _ = self.registry.decrement_sessions(&model_id);

        // Update metrics
        self.metrics.sessions_destroyed_total.inc();
        self.metrics
            .active_sessions
            .set(self.sessions.active_count() as i64);

        Ok(())
    }

    // ── Inference ────────────────────────────────────────────────────────

    /// Run inference (non-streaming) for a session.
    #[instrument(skip(self, prompt, system, sampling))]
    pub async fn generate(
        &self,
        session_id: &SessionId,
        prompt: &str,
        system: Option<&str>,
        sampling: &SamplingConfig,
    ) -> Result<InferenceResponse> {
        // Get model handle
        let model_name = self
            .sessions
            .with_session(session_id, |s| s.model_name.clone())?;

        let handle = {
            let handles = self.model_handles.read();
            handles
                .get(&model_name)
                .cloned()
                .ok_or_else(|| InferenceError::ModelNotLoaded(model_name.clone()))?
        };

        // Transition session to generating
        self.sessions
            .with_session_mut(session_id, |s| s.mark_prefilling())?
            .map_err(|e| {
                self.metrics.errors_total.inc();
                e
            })?;

        // Delegate to backend
        let result = self
            .backend
            .generate(&handle, prompt, system, sampling)
            .await;

        match result {
            Ok((text, stats)) => {
                // Transition through decoding → idle
                let _ = self.sessions.with_session_mut(session_id, |s| {
                    let _ = s.mark_decoding();
                    s.mark_idle()
                });

                // Record metrics
                self.metrics.record_generation(
                    stats.prompt_tokens,
                    stats.generated_tokens,
                    stats.first_token_ms,
                    stats.total_ms,
                    stats.tokens_per_second,
                );

                Ok(InferenceResponse {
                    session_id: *session_id,
                    text,
                    tokens: Vec::new(),
                    first_token_ms: stats.first_token_ms,
                    total_ms: stats.total_ms,
                    tokens_generated: stats.generated_tokens,
                    tokens_per_second: stats.tokens_per_second,
                    stop_reason: StopReason::EndOfSequence,
                })
            }
            Err(e) => {
                // Return session to idle on error
                let _ = self
                    .sessions
                    .with_session_mut(session_id, |s| s.mark_idle());
                self.metrics.errors_total.inc();
                Err(e)
            }
        }
    }

    /// Run inference (streaming) for a session.
    #[instrument(skip(self, prompt, system, sampling))]
    pub async fn stream(
        &self,
        session_id: &SessionId,
        prompt: &str,
        system: Option<&str>,
        sampling: &SamplingConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Token>> + Send>>> {
        let model_name = self
            .sessions
            .with_session(session_id, |s| s.model_name.clone())?;

        let handle = {
            let handles = self.model_handles.read();
            handles
                .get(&model_name)
                .cloned()
                .ok_or_else(|| InferenceError::ModelNotLoaded(model_name.clone()))?
        };

        // Transition to generating
        self.sessions
            .with_session_mut(session_id, |s| s.mark_prefilling())?
            .map_err(|e| {
                self.metrics.errors_total.inc();
                e
            })?;

        let stream = self
            .backend
            .stream(&handle, prompt, system, sampling)
            .await?;

        // Mark as decoding once stream starts
        let _ = self
            .sessions
            .with_session_mut(session_id, |s| s.mark_decoding());

        let metrics = self.metrics.clone();
        let start = Instant::now();
        let mut first_token = true;

        let instrumented = stream.map(move |token_result| {
            match &token_result {
                Ok(token) => {
                    if first_token {
                        let elapsed = start.elapsed().as_millis() as u64;
                        metrics.first_token_latency_ms.observe(elapsed as f64);
                        first_token = false;
                    }
                    if token.done {
                        let total = start.elapsed().as_millis() as u64;
                        metrics.generation_total_ms.observe(total as f64);
                    }
                    metrics.tokens_generated_total.inc();
                }
                Err(_) => {
                    metrics.errors_total.inc();
                }
            }
            token_result
        });

        Ok(Box::pin(instrumented))
    }

    /// Submit a request through the batch scheduler.
    pub async fn enqueue_request(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        // Create a temporary session if needed
        let session_id = self
            .create_session(&request.model, request.priority)
            .await?;

        let result = self
            .generate(
                &session_id,
                &request.prompt,
                request.system.as_deref(),
                &request.sampling,
            )
            .await;

        // Clean up ephemeral session on error
        if result.is_err() {
            let _ = self.destroy_session(&session_id).await;
        }

        result
    }

    // ── Model Management ────────────────────────────────────────────────

    /// Load a model into the backend.
    #[instrument(skip(self))]
    pub async fn load_model(&self, name: &str) -> Result<ModelId> {
        // Check if already loaded
        if self.model_handles.read().contains_key(name) {
            return Err(InferenceError::ModelAlreadyLoaded(name.into()));
        }

        // Resolve or discover the model
        let model_id = if let Some(id) = self.registry.resolve_name(name) {
            id
        } else {
            // Try to sync from Ollama
            self.registry.sync_from_ollama().await?;
            self.registry
                .resolve_name(name)
                .ok_or_else(|| InferenceError::ModelNotFound(name.into()))?
        };

        self.registry.set_loading(&model_id)?;

        let config = ModelConfig {
            name: name.to_string(),
            gpu_offload: self.config.gpu.offload.clone(),
            keep_alive_secs: self.config.ollama.keep_alive_secs,
        };

        match self.backend.load_model(&config).await {
            Ok(handle) => {
                self.registry.set_loaded(&model_id, vec![0], 0, 0)?;
                self.model_handles.write().insert(name.to_string(), handle);

                self.metrics.model_loads_total.inc();
                self.metrics
                    .loaded_models
                    .set(self.registry.loaded_count() as i64);

                Ok(model_id)
            }
            Err(e) => {
                self.registry.set_failed(&model_id)?;
                self.metrics.errors_total.inc();
                Err(e)
            }
        }
    }

    /// Unload a model from the backend.
    #[instrument(skip(self))]
    pub async fn unload_model(&self, name: &str) -> Result<()> {
        let handle = self
            .model_handles
            .write()
            .remove(name)
            .ok_or_else(|| InferenceError::ModelNotLoaded(name.into()))?;

        self.backend.unload_model(&handle).await?;

        if let Some(model_id) = self.registry.resolve_name(name) {
            self.registry.set_unloaded(&model_id)?;
        }

        self.metrics.model_unloads_total.inc();
        self.metrics
            .loaded_models
            .set(self.registry.loaded_count() as i64);

        Ok(())
    }

    /// List all available models.
    pub fn list_models(&self) -> Vec<crate::types::ModelInfo> {
        self.registry.list()
    }

    /// List loaded models.
    pub fn list_loaded_models(&self) -> Vec<crate::types::LoadedModelInfo> {
        self.registry.list_loaded()
    }

    // ── KV-Cache Operations ─────────────────────────────────────────────

    /// Take a KV-cache snapshot for a session.
    #[instrument(skip(self))]
    pub async fn kv_snapshot(&self, session_id: &SessionId) -> Result<KvSnapshot> {
        // Verify session exists and is idle/ready
        self.sessions.with_session(session_id, |s| {
            if !s.state().can_prompt() && s.state() != SessionState::Pending {
                Err(InferenceError::InvalidSessionState {
                    session_id: *session_id,
                    state: s.state(),
                    expected: vec![SessionState::Ready, SessionState::Idle],
                })
            } else {
                Ok(())
            }
        })??;

        let snapshot = self.backend.kv_snapshot(session_id).await?;

        // Store in session
        self.sessions.with_session_mut(session_id, |s| {
            s.snapshots.push(snapshot.clone());
        })?;

        self.metrics.kv_snapshots_total.inc();
        Ok(snapshot)
    }

    /// Restore a KV-cache snapshot for a session.
    #[instrument(skip(self))]
    pub async fn kv_restore(&self, session_id: &SessionId, snapshot: &KvSnapshot) -> Result<()> {
        self.backend.kv_restore(session_id, snapshot).await?;

        self.metrics.kv_restores_total.inc();
        Ok(())
    }

    // ── Status & Metrics ────────────────────────────────────────────────

    /// Get the number of active sessions.
    #[must_use]
    pub fn active_session_count(&self) -> usize {
        self.sessions.active_count()
    }

    /// Get all session IDs.
    pub fn session_ids(&self) -> Vec<SessionId> {
        self.sessions.session_ids()
    }

    /// Query session state.
    pub fn session_state(&self, session_id: &SessionId) -> Result<SessionState> {
        self.sessions.with_session(session_id, |s| s.state())
    }

    /// Get session info.
    pub fn session_info(&self, session_id: &SessionId) -> Result<SessionInfo> {
        self.sessions.with_session(session_id, |s| SessionInfo {
            id: s.id,
            model_name: s.model_name.clone(),
            state: s.state(),
            kv_token_count: s.kv_token_count,
            total_tokens_generated: s.total_tokens_generated,
            priority: s.priority,
        })
    }

    /// Get the batch queue depth.
    #[must_use]
    pub fn queue_depth(&self) -> usize {
        self.batch_scheduler.queue_depth()
    }

    /// Get the model registry.
    #[must_use]
    pub fn model_registry(&self) -> &Arc<ModelRegistry> {
        &self.registry
    }

    /// Check if the backend is healthy.
    pub async fn health_check(&self) -> bool {
        self.backend.health_check().await
    }

    /// Get the backend resource usage report.
    pub fn resource_usage(&self) -> crate::types::ResourceReport {
        self.backend.resource_usage()
    }

    /// Encode metrics in Prometheus text format.
    pub fn encode_metrics(&self) -> String {
        let registry = self.metrics_registry.read();
        self.metrics.encode(&registry)
    }

    /// Get a reference to the metrics.
    pub fn metrics(&self) -> &Arc<InferenceMetrics> {
        &self.metrics
    }

    /// Get the configuration.
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }
}

/// Public session information.
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub id: SessionId,
    pub model_name: String,
    pub state: SessionState,
    pub kv_token_count: usize,
    pub total_tokens_generated: u64,
    pub priority: Priority,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ollama::OllamaBackend;

    fn test_coordinator() -> InferenceCoordinator {
        let config = InferenceConfig::for_test();
        let ollama_client = Arc::new(OllamaClient::new(&config.ollama));
        let backend = Arc::new(OllamaBackend::new(
            ollama_client.clone(),
            config.ollama.keep_alive_secs,
        ));
        InferenceCoordinator::new(config, backend, ollama_client)
    }

    #[test]
    fn test_coordinator_creation() {
        let coord = test_coordinator();
        assert_eq!(coord.active_session_count(), 0);
        assert!(coord.session_ids().is_empty());
    }

    #[test]
    fn test_coordinator_queue_depth() {
        let coord = test_coordinator();
        assert_eq!(coord.queue_depth(), 0);
    }

    #[test]
    fn test_coordinator_metrics_encode() {
        let coord = test_coordinator();
        let output = coord.encode_metrics();
        assert!(output.contains("spirit_inference"));
    }

    #[test]
    fn test_coordinator_config_access() {
        let coord = test_coordinator();
        assert_eq!(coord.config().backend, "ollama");
    }

    #[test]
    fn test_coordinator_list_models_empty() {
        let coord = test_coordinator();
        assert!(coord.list_models().is_empty());
    }
}
