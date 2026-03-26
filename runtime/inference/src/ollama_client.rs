//! Ollama HTTP API client.
//!
//! Async client for the Ollama REST API. Handles model management
//! (`/api/tags`, `/api/pull`, `/api/show`), generation (`/api/generate`,
//! `/api/chat`), and embeddings (`/api/embeddings`).
//!
//! Token streaming uses chunked HTTP responses with newline-delimited JSON.
//!
//! // PHASE1-SHIM: Direct Ollama HTTP; Phase 2+ uses patched llama.cpp via FFI.

use std::pin::Pin;
use std::time::Duration;

use futures::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;
use tracing::{debug, instrument, warn};

use crate::config::OllamaConfig;
use crate::error::{InferenceError, Result};

/// Async client for the Ollama REST API.
#[derive(Debug, Clone)]
pub struct OllamaClient {
    client: Client,
    base_url: String,
    timeout: Duration,
}

// ── Ollama API request/response types ───────────────────────────────────────

/// Request body for `/api/generate`.
#[derive(Debug, Clone, Serialize)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<GenerateOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

/// Model-level options for generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

/// A single chunk in the streamed generate response.
#[derive(Debug, Clone, Deserialize)]
pub struct GenerateChunk {
    pub model: String,
    pub response: String,
    pub done: bool,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u32>,
    #[serde(default)]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default)]
    pub eval_count: Option<u32>,
    #[serde(default)]
    pub eval_duration: Option<u64>,
}

/// Response body for `/api/generate` (non-streaming).
#[derive(Debug, Clone, Deserialize)]
pub struct GenerateResponse {
    pub model: String,
    pub response: String,
    pub done: bool,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u32>,
    #[serde(default)]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default)]
    pub eval_count: Option<u32>,
    #[serde(default)]
    pub eval_duration: Option<u64>,
}

/// Request body for `/api/embeddings`.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub prompt: String,
}

/// Response body for `/api/embeddings`.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingResponse {
    pub embedding: Vec<f32>,
}

/// A model listed by `/api/tags`.
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModel {
    pub name: String,
    pub model: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub digest: String,
    #[serde(default)]
    pub details: Option<OllamaModelDetails>,
    #[serde(default)]
    pub modified_at: Option<String>,
}

/// Detailed model info from `/api/tags` or `/api/show`.
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModelDetails {
    #[serde(default)]
    pub parent_model: Option<String>,
    #[serde(default)]
    pub format: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub parameter_size: String,
    #[serde(default)]
    pub quantization_level: String,
}

/// Response body for `/api/tags`.
#[derive(Debug, Clone, Deserialize)]
pub struct TagsResponse {
    pub models: Vec<OllamaModel>,
}

/// Response body for `/api/show`.
#[derive(Debug, Clone, Deserialize)]
pub struct ShowResponse {
    #[serde(default)]
    pub modelfile: String,
    #[serde(default)]
    pub parameters: String,
    #[serde(default)]
    pub template: String,
    #[serde(default)]
    pub details: Option<OllamaModelDetails>,
    #[serde(default)]
    pub model_info: Option<serde_json::Value>,
}

/// Request body for `/api/pull`.
#[derive(Debug, Clone, Serialize)]
pub struct PullRequest {
    pub name: String,
    pub stream: bool,
}

/// A chunk in the pull stream.
#[derive(Debug, Clone, Deserialize)]
pub struct PullChunk {
    pub status: String,
    #[serde(default)]
    pub digest: Option<String>,
    #[serde(default)]
    pub total: Option<u64>,
    #[serde(default)]
    pub completed: Option<u64>,
}

/// Ollama API error response.
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaErrorResponse {
    pub error: String,
}

// ── Client Implementation ───────────────────────────────────────────────────

impl OllamaClient {
    /// Create a new Ollama client from configuration.
    pub fn new(config: &OllamaConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .pool_max_idle_per_host(4)
            .build()
            .expect("Failed to build HTTP client");

        Self {
            client,
            base_url: config.url.trim_end_matches('/').to_string(),
            timeout: Duration::from_secs(config.timeout_secs),
        }
    }

    /// Check if the Ollama server is reachable.
    #[instrument(skip(self))]
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/api/tags", self.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// List all available models.
    #[instrument(skip(self))]
    pub async fn list_models(&self) -> Result<Vec<OllamaModel>> {
        let url = format!("{}/api/tags", self.base_url);
        let resp = self.client.get(&url).send().await?;

        if !resp.status().is_success() {
            let err = self.extract_error(resp).await;
            return Err(InferenceError::OllamaError(err));
        }

        let tags: TagsResponse = resp.json().await?;
        Ok(tags.models)
    }

    /// Get detailed information about a model.
    #[instrument(skip(self))]
    pub async fn show_model(&self, name: &str) -> Result<ShowResponse> {
        let url = format!("{}/api/show", self.base_url);
        let body = serde_json::json!({ "name": name });
        let resp = self.client.post(&url).json(&body).send().await?;

        if !resp.status().is_success() {
            if resp.status().as_u16() == 404 {
                return Err(InferenceError::ModelNotFound(name.to_string()));
            }
            let err = self.extract_error(resp).await;
            return Err(InferenceError::OllamaError(err));
        }

        Ok(resp.json().await?)
    }

    /// Pull (download) a model. Returns a stream of progress chunks.
    #[instrument(skip(self))]
    pub async fn pull_model(
        &self,
        name: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<PullChunk>> + Send>>> {
        let url = format!("{}/api/pull", self.base_url);
        let body = PullRequest {
            name: name.to_string(),
            stream: true,
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .timeout(Duration::from_secs(3600)) // model pulls can be slow
            .send()
            .await?;

        if !resp.status().is_success() {
            let err = self.extract_error(resp).await;
            return Err(InferenceError::OllamaError(err));
        }

        let stream = resp.bytes_stream();
        let parsed = stream.map(|chunk_result| {
            let chunk = chunk_result.map_err(InferenceError::HttpError)?;
            // Ollama sends newline-delimited JSON
            let text = String::from_utf8_lossy(&chunk);
            for line in text.lines() {
                if !line.trim().is_empty() {
                    match serde_json::from_str::<PullChunk>(line) {
                        Ok(parsed) => return Ok(parsed),
                        Err(e) => {
                            warn!("Failed to parse pull chunk: {e}");
                        }
                    }
                }
            }
            Ok(PullChunk {
                status: "unknown".into(),
                digest: None,
                total: None,
                completed: None,
            })
        });

        Ok(Box::pin(parsed))
    }

    /// Generate a completion (non-streaming).
    #[instrument(skip(self, request))]
    pub async fn generate(&self, request: &GenerateRequest) -> Result<GenerateResponse> {
        let url = format!("{}/api/generate", self.base_url);
        let mut req = request.clone();
        req.stream = false;

        let resp = self.client.post(&url).json(&req).send().await?;

        if !resp.status().is_success() {
            let err = self.extract_error(resp).await;
            return Err(InferenceError::OllamaError(err));
        }

        Ok(resp.json().await?)
    }

    /// Generate a streaming completion. Returns a stream of token chunks.
    #[instrument(skip(self, request))]
    pub async fn generate_stream(
        &self,
        request: &GenerateRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GenerateChunk>> + Send>>> {
        let url = format!("{}/api/generate", self.base_url);
        let mut req = request.clone();
        req.stream = true;

        let resp = self
            .client
            .post(&url)
            .json(&req)
            .timeout(self.timeout)
            .send()
            .await?;

        if !resp.status().is_success() {
            let err = self.extract_error(resp).await;
            return Err(InferenceError::OllamaError(err));
        }

        let byte_stream = resp.bytes_stream();
        // Buffer for handling partial JSON lines across chunk boundaries
        let stream = PartialLineStream::new(byte_stream);

        Ok(Box::pin(stream))
    }

    /// Generate embeddings for text.
    #[instrument(skip(self))]
    pub async fn embeddings(&self, model: &str, prompt: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embeddings", self.base_url);
        let body = EmbeddingRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
        };

        let resp = self.client.post(&url).json(&body).send().await?;

        if !resp.status().is_success() {
            let err = self.extract_error(resp).await;
            return Err(InferenceError::OllamaError(err));
        }

        let embed: EmbeddingResponse = resp.json().await?;
        Ok(embed.embedding)
    }

    /// Get the base URL this client is configured for.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Extract an error message from a non-success response.
    async fn extract_error(&self, resp: reqwest::Response) -> String {
        let status = resp.status();
        match resp.json::<OllamaErrorResponse>().await {
            Ok(err_resp) => format!("{status}: {}", err_resp.error),
            Err(_) => format!("HTTP {status}"),
        }
    }
}

// ── Streaming Helper ────────────────────────────────────────────────────────

/// Wraps a byte stream and handles partial JSON lines across chunk boundaries.
///
/// Ollama sends newline-delimited JSON. Chunks from the HTTP response may
/// split a JSON line across boundaries. This stream buffers partial lines
/// and emits parsed `GenerateChunk` objects.
struct PartialLineStream;

impl PartialLineStream {
    fn new(
        inner: impl Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
    ) -> impl Stream<Item = Result<GenerateChunk>> + Send {
        let state = PartialLineState {
            inner: Box::pin(inner),
            buffer: String::with_capacity(4096),
        };

        futures::stream::unfold(state, |mut state| async move {
            loop {
                // Try to parse a complete line from the buffer first
                if let Some(newline_pos) = state.buffer.find('\n') {
                    let line = state.buffer[..newline_pos].trim().to_string();
                    state.buffer = state.buffer[newline_pos + 1..].to_string();
                    if line.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<GenerateChunk>(&line) {
                        Ok(chunk) => return Some((Ok(chunk), state)),
                        Err(e) => {
                            debug!("Skipping unparseable line: {e}");
                            continue;
                        }
                    }
                }

                // Need more data from the inner stream
                match state.inner.next().await {
                    Some(Ok(bytes)) => {
                        state.buffer.push_str(&String::from_utf8_lossy(&bytes));
                    }
                    Some(Err(e)) => {
                        return Some((Err(InferenceError::HttpError(e)), state));
                    }
                    None => {
                        // Stream ended. Try to parse any remaining buffer.
                        let remaining = state.buffer.trim().to_string();
                        state.buffer.clear();
                        if remaining.is_empty() {
                            return None;
                        }
                        match serde_json::from_str::<GenerateChunk>(&remaining) {
                            Ok(chunk) => return Some((Ok(chunk), state)),
                            Err(_) => return None,
                        }
                    }
                }
            }
        })
    }
}

struct PartialLineState {
    inner: Pin<Box<dyn Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>> + Send>>,
    buffer: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_request_serialization() {
        let req = GenerateRequest {
            model: "llama3.1:8b".into(),
            prompt: "Hello".into(),
            system: None,
            stream: true,
            options: Some(GenerateOptions {
                temperature: Some(0.7),
                top_p: Some(0.9),
                top_k: None,
                repeat_penalty: None,
                seed: None,
                num_predict: Some(100),
                stop: None,
            }),
            keep_alive: None,
        };
        let json = serde_json::to_string(&req).expect("serialize");
        assert!(json.contains("llama3.1:8b"));
        assert!(json.contains("temperature"));
        // system is None, should not appear
        assert!(!json.contains("system"));
    }

    #[test]
    fn test_generate_chunk_deserialization() {
        let json = r#"{"model":"llama3.1:8b","response":"Hello","done":false}"#;
        let chunk: GenerateChunk = serde_json::from_str(json).expect("deserialize");
        assert_eq!(chunk.response, "Hello");
        assert!(!chunk.done);
    }

    #[test]
    fn test_generate_chunk_final_deserialization() {
        let json = r#"{
            "model": "llama3.1:8b",
            "response": "",
            "done": true,
            "total_duration": 5000000000,
            "eval_count": 42,
            "eval_duration": 3000000000
        }"#;
        let chunk: GenerateChunk = serde_json::from_str(json).expect("deserialize");
        assert!(chunk.done);
        assert_eq!(chunk.eval_count, Some(42));
    }

    #[test]
    fn test_tags_response_deserialization() {
        let json = r#"{
            "models": [{
                "name": "llama3.1:8b",
                "model": "llama3.1:8b",
                "size": 4661211808,
                "digest": "sha256:abc123"
            }]
        }"#;
        let tags: TagsResponse = serde_json::from_str(json).expect("deserialize");
        assert_eq!(tags.models.len(), 1);
        assert_eq!(tags.models[0].name, "llama3.1:8b");
    }

    #[test]
    fn test_embedding_response_deserialization() {
        let json = r#"{"embedding": [0.1, 0.2, 0.3, 0.4]}"#;
        let resp: EmbeddingResponse = serde_json::from_str(json).expect("deserialize");
        assert_eq!(resp.embedding.len(), 4);
    }

    #[test]
    fn test_client_creation() {
        let config = OllamaConfig::default();
        let client = OllamaClient::new(&config);
        assert!(client.base_url().contains("127.0.0.1"));
    }

    #[test]
    fn test_pull_chunk_deserialization() {
        let json = r#"{"status":"downloading","digest":"sha256:abc","total":1000,"completed":500}"#;
        let chunk: PullChunk = serde_json::from_str(json).expect("deserialize");
        assert_eq!(chunk.status, "downloading");
        assert_eq!(chunk.total, Some(1000));
    }
}
