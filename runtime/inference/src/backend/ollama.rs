//! Ollama backend implementation.
//!
//! Delegates inference to a local Ollama instance via its HTTP API.
//! This is the primary backend for Phase 1.
//!
//! // PHASE1-SHIM: Uses Ollama HTTP API; Phase 2+ uses patched llama.cpp FFI.

use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use futures::Stream;
use tokio_stream::StreamExt;
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

use super::{GenerationStats, InferenceBackend, ModelConfig, ModelHandle};
use crate::error::InferenceError;
use crate::ollama_client::{GenerateOptions, GenerateRequest, OllamaClient};
use crate::types::{KvSnapshot, ModelId, ResourceReport, SamplingConfig, SessionId, Token};

/// Ollama-backed inference engine.
///
/// Wraps the [`OllamaClient`] HTTP client and provides the
/// [`InferenceBackend`] trait implementation.
pub struct OllamaBackend {
    client: Arc<OllamaClient>,
    /// Track active generation count for resource reporting.
    active_generations: Arc<AtomicU64>,
    /// Total tokens generated (for metrics).
    total_tokens: Arc<AtomicU64>,
    /// Default keep-alive for models.
    keep_alive_secs: u64,
}

impl OllamaBackend {
    /// Create a new Ollama backend.
    pub fn new(client: Arc<OllamaClient>, keep_alive_secs: u64) -> Self {
        Self {
            client,
            active_generations: Arc::new(AtomicU64::new(0)),
            total_tokens: Arc::new(AtomicU64::new(0)),
            keep_alive_secs,
        }
    }

    /// Convert Spirit sampling config to Ollama generate options.
    fn to_ollama_options(sampling: &SamplingConfig) -> GenerateOptions {
        GenerateOptions {
            temperature: Some(sampling.temperature),
            top_p: Some(sampling.top_p),
            top_k: if sampling.top_k > 0 {
                Some(sampling.top_k)
            } else {
                None
            },
            repeat_penalty: Some(sampling.repeat_penalty),
            seed: if sampling.seed > 0 {
                Some(sampling.seed)
            } else {
                None
            },
            num_predict: if sampling.max_tokens > 0 {
                Some(sampling.max_tokens)
            } else {
                None
            },
            stop: if sampling.stop.is_empty() {
                None
            } else {
                Some(sampling.stop.clone())
            },
        }
    }

    fn format_keep_alive(&self) -> String {
        format!("{}s", self.keep_alive_secs)
    }
}

#[async_trait]
impl InferenceBackend for OllamaBackend {
    fn name(&self) -> &str {
        "ollama"
    }

    #[instrument(skip(self))]
    async fn health_check(&self) -> bool {
        self.client.health_check().await.unwrap_or(false)
    }

    #[instrument(skip(self))]
    async fn load_model(&self, config: &ModelConfig) -> Result<ModelHandle, InferenceError> {
        // Ollama loads models on first use. We issue a trivial generate
        // to trigger model loading and verify it's available.
        info!(model = %config.name, "Loading model via Ollama");

        // First, verify the model exists
        self.client.show_model(&config.name).await?;

        // Issue a minimal generate to trigger model loading
        let request = GenerateRequest {
            model: config.name.clone(),
            prompt: "".into(),
            system: None,
            stream: false,
            options: Some(GenerateOptions {
                num_predict: Some(1),
                temperature: Some(0.0),
                ..Default::default()
            }),
            keep_alive: Some(self.format_keep_alive()),
        };

        self.client.generate(&request).await.map_err(|e| {
            InferenceError::OllamaError(format!("Failed to load model '{}': {e}", config.name))
        })?;

        info!(model = %config.name, "Model loaded successfully");

        Ok(ModelHandle {
            model_id: ModelId::new(),
            backend_name: config.name.clone(),
        })
    }

    #[instrument(skip(self))]
    async fn unload_model(&self, handle: &ModelHandle) -> Result<(), InferenceError> {
        // Ollama unloads models by setting keep_alive to 0
        let request = GenerateRequest {
            model: handle.backend_name.clone(),
            prompt: "".into(),
            system: None,
            stream: false,
            options: None,
            keep_alive: Some("0".into()),
        };

        // Best effort — if Ollama is down, the model is already "unloaded"
        match self.client.generate(&request).await {
            Ok(_) => {
                info!(model = %handle.backend_name, "Model unloaded");
                Ok(())
            }
            Err(e) => {
                warn!(
                    model = %handle.backend_name,
                    error = %e,
                    "Failed to unload model (may already be unloaded)"
                );
                Ok(())
            }
        }
    }

    #[instrument(skip(self))]
    async fn list_models(&self) -> Result<Vec<String>, InferenceError> {
        let models = self.client.list_models().await?;
        Ok(models.into_iter().map(|m| m.name).collect())
    }

    #[instrument(skip(self, prompt, system, sampling))]
    async fn generate(
        &self,
        handle: &ModelHandle,
        prompt: &str,
        system: Option<&str>,
        sampling: &SamplingConfig,
    ) -> Result<(String, GenerationStats), InferenceError> {
        self.active_generations.fetch_add(1, Ordering::Relaxed);

        let request = GenerateRequest {
            model: handle.backend_name.clone(),
            prompt: prompt.to_string(),
            system: system.map(String::from),
            stream: false,
            options: Some(Self::to_ollama_options(sampling)),
            keep_alive: Some(self.format_keep_alive()),
        };

        let result = self.client.generate(&request).await;
        self.active_generations.fetch_sub(1, Ordering::Relaxed);

        let resp = result?;

        let eval_count = resp.eval_count.unwrap_or(0);
        let eval_duration_ns = resp.eval_duration.unwrap_or(1);
        let prompt_eval_count = resp.prompt_eval_count.unwrap_or(0);
        let total_duration_ns = resp.total_duration.unwrap_or(0);
        let load_duration_ns = resp.load_duration.unwrap_or(0);

        // Calculate time to first token (total - eval - load ≈ prefill)
        let first_token_ms =
            (total_duration_ns.saturating_sub(eval_duration_ns + load_duration_ns)) / 1_000_000;
        let total_ms = total_duration_ns / 1_000_000;
        let tps = if eval_duration_ns > 0 {
            (eval_count as f64 / eval_duration_ns as f64 * 1_000_000_000.0) as f32
        } else {
            0.0
        };

        self.total_tokens
            .fetch_add(u64::from(eval_count), Ordering::Relaxed);

        let stats = GenerationStats {
            first_token_ms,
            total_ms,
            prompt_tokens: prompt_eval_count,
            generated_tokens: eval_count,
            tokens_per_second: tps,
        };

        debug!(
            model = %handle.backend_name,
            tokens = eval_count,
            tps = format!("{tps:.1}"),
            total_ms,
            "Generation complete"
        );

        Ok((resp.response, stats))
    }

    #[instrument(skip(self, prompt, system, sampling))]
    async fn stream(
        &self,
        handle: &ModelHandle,
        prompt: &str,
        system: Option<&str>,
        sampling: &SamplingConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Token, InferenceError>> + Send>>, InferenceError>
    {
        self.active_generations.fetch_add(1, Ordering::Relaxed);

        let request = GenerateRequest {
            model: handle.backend_name.clone(),
            prompt: prompt.to_string(),
            system: system.map(String::from),
            stream: true,
            options: Some(Self::to_ollama_options(sampling)),
            keep_alive: Some(self.format_keep_alive()),
        };

        let chunk_stream = self.client.generate_stream(&request).await?;

        // Clone Arcs for the stream closure
        let active_gens = self.active_generations.clone();
        let total_tokens = self.total_tokens.clone();

        let token_stream = chunk_stream.map(move |chunk_result| {
            match chunk_result {
                Ok(chunk) => {
                    if chunk.done {
                        active_gens.fetch_sub(1, Ordering::Relaxed);
                        if let Some(eval_count) = chunk.eval_count {
                            total_tokens.fetch_add(u64::from(eval_count), Ordering::Relaxed);
                        }
                    }

                    Ok(Token {
                        text: chunk.response,
                        token_id: 0, // Ollama doesn't expose token IDs
                        logprob: None,
                        done: chunk.done,
                    })
                }
                Err(e) => Err(e),
            }
        });

        Ok(Box::pin(token_stream))
    }

    async fn kv_snapshot(&self, session_id: &SessionId) -> Result<KvSnapshot, InferenceError> {
        // PHASE1-SHIM: Ollama doesn't expose KV-cache snapshots directly.
        // In Phase 1, we create a logical snapshot. Actual KV-cache management
        // is delegated to the memory manager's KV page system.
        info!(session = %session_id, "Creating logical KV snapshot (Phase 1 shim)");
        Ok(KvSnapshot {
            id: Uuid::new_v4(),
            session_id: *session_id,
            token_count: 0,
            created_at: Utc::now(),
        })
    }

    async fn kv_restore(
        &self,
        session_id: &SessionId,
        snapshot: &KvSnapshot,
    ) -> Result<(), InferenceError> {
        // PHASE1-SHIM: Ollama doesn't support KV-cache restore.
        // In Phase 1, restoring a snapshot means re-sending the original
        // prompt to recreate the KV-cache state.
        info!(
            session = %session_id,
            snapshot = %snapshot.id,
            "Restoring KV snapshot (Phase 1 shim — re-prompt required)"
        );
        Ok(())
    }

    fn resource_usage(&self) -> ResourceReport {
        ResourceReport {
            loaded_models: 0, // Updated by coordinator
            active_sessions: self.active_generations.load(Ordering::Relaxed) as usize,
            vram_used_bytes: 0,  // Not directly queryable from Ollama
            vram_total_bytes: 0, // Not directly queryable
            ram_used_bytes: 0,   // Not directly queryable
            aggregate_tokens_per_sec: 0.0,
        }
    }
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            temperature: None,
            top_p: None,
            top_k: None,
            repeat_penalty: None,
            seed: None,
            num_predict: None,
            stop: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OllamaConfig;

    #[test]
    fn test_ollama_options_conversion() {
        let sampling = SamplingConfig {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 50,
            repeat_penalty: 1.2,
            max_tokens: 512,
            stop: vec!["###".into()],
            seed: 42,
        };

        let opts = OllamaBackend::to_ollama_options(&sampling);
        assert_eq!(opts.temperature, Some(0.8));
        assert_eq!(opts.top_p, Some(0.95));
        assert_eq!(opts.top_k, Some(50));
        assert_eq!(opts.seed, Some(42));
        assert_eq!(opts.num_predict, Some(512));
        assert!(opts.stop.is_some());
    }

    #[test]
    fn test_ollama_options_zero_seed_is_none() {
        let sampling = SamplingConfig {
            seed: 0,
            ..SamplingConfig::default()
        };

        let opts = OllamaBackend::to_ollama_options(&sampling);
        assert_eq!(opts.seed, None);
    }

    #[test]
    fn test_backend_name() {
        let client = Arc::new(OllamaClient::new(&OllamaConfig::default()));
        let backend = OllamaBackend::new(client, 3600);
        assert_eq!(backend.name(), "ollama");
    }

    #[test]
    fn test_resource_usage_initial() {
        let client = Arc::new(OllamaClient::new(&OllamaConfig::default()));
        let backend = OllamaBackend::new(client, 3600);
        let report = backend.resource_usage();
        assert_eq!(report.active_sessions, 0);
    }
}
