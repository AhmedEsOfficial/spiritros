//! Inference client for the Spirit Shell.
//!
//! Provides a lightweight async HTTP client that talks directly to Ollama's
//! REST API for natural language processing, pipe-to-LLM, and `spirit infer`.
//!
//! PHASE1-SHIM: In Phase 2+ this will use the inference coordinator's IPC/gRPC
//! interface instead of direct Ollama HTTP calls.

use std::io::Write;
use std::time::{Duration, Instant};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;
use tracing::debug;

use crate::config::ShellConfig;
use crate::error::{self, ShellError};
use crate::types::GenerationStats;

/// Async client for inference operations from the shell.
#[derive(Debug, Clone)]
pub struct InferenceClient {
    client: Client,
    base_url: String,
    model: String,
    system_prompt: String,
    timeout: Duration,
}

/// Request body for Ollama `/api/generate`.
#[derive(Debug, Serialize)]
struct GenerateRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<GenerateOptions>,
}

/// Generation options.
#[derive(Debug, Serialize)]
struct GenerateOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
}

/// A chunk in the streamed generate response.
#[derive(Debug, Deserialize)]
struct GenerateChunk {
    response: String,
    done: bool,
    #[serde(default)]
    #[allow(dead_code)]
    total_duration: Option<u64>,
    #[serde(default)]
    eval_count: Option<u32>,
    #[serde(default)]
    #[allow(dead_code)]
    eval_duration: Option<u64>,
    #[serde(default)]
    #[allow(dead_code)]
    prompt_eval_duration: Option<u64>,
}

/// Model listed by `/api/tags`.
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModel {
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub digest: String,
    #[serde(default)]
    pub details: Option<ModelDetails>,
}

/// Model details from Ollama.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelDetails {
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub parameter_size: String,
    #[serde(default)]
    pub quantization_level: String,
}

#[derive(Debug, Deserialize)]
struct TagsResponse {
    models: Vec<OllamaModel>,
}

/// Pull progress chunk.
#[derive(Debug, Deserialize)]
pub struct PullChunk {
    pub status: String,
    #[serde(default)]
    pub total: Option<u64>,
    #[serde(default)]
    pub completed: Option<u64>,
}

/// Ollama error response body.
#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: String,
}

impl InferenceClient {
    /// Create a new inference client from shell configuration.
    pub fn new(config: &ShellConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.inference.timeout_secs))
            .pool_max_idle_per_host(2)
            .build()
            .expect("Failed to build HTTP client");

        Self {
            client,
            base_url: config
                .inference
                .ollama_url
                .trim_end_matches('/')
                .to_string(),
            model: config.llm.model.clone(),
            system_prompt: config.llm.system_prompt.clone(),
            timeout: Duration::from_secs(config.inference.timeout_secs),
        }
    }

    /// Check if Ollama is reachable.
    pub async fn health_check(&self) -> bool {
        let url = format!("{}/api/tags", self.base_url);
        matches!(self.client.get(&url).send().await, Ok(r) if r.status().is_success())
    }

    /// Get the currently configured model name.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Set the model to use for inference.
    pub fn set_model(&mut self, model: String) {
        self.model = model;
    }

    /// List all available models from Ollama.
    pub async fn list_models(&self) -> error::Result<Vec<OllamaModel>> {
        let url = format!("{}/api/tags", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ShellError::InferenceUnavailable(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(ShellError::InferenceError(format!(
                "HTTP {}",
                resp.status()
            )));
        }

        let tags: TagsResponse = resp
            .json()
            .await
            .map_err(|e| ShellError::InferenceError(e.to_string()))?;
        Ok(tags.models)
    }

    /// Generate a streaming response and print tokens to stdout as they arrive.
    /// Returns generation statistics.
    pub async fn generate_streaming(
        &self,
        prompt: &str,
        system: Option<&str>,
    ) -> error::Result<GenerationStats> {
        let url = format!("{}/api/generate", self.base_url);
        let body = GenerateRequest {
            model: self.model.clone(),
            prompt: prompt.to_string(),
            system: system.map(String::from),
            stream: true,
            options: Some(GenerateOptions {
                temperature: Some(0.7),
                num_predict: Some(2048),
            }),
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| ShellError::InferenceUnavailable(e.to_string()))?;

        if !resp.status().is_success() {
            let err = extract_error(resp).await;
            return Err(ShellError::InferenceError(err));
        }

        let start = Instant::now();
        let mut first_token_ms = 0u64;
        let mut token_count = 0u32;
        let mut buffer = String::with_capacity(4096);
        let mut stream = resp.bytes_stream();

        while let Some(chunk_result) = stream.next().await {
            let bytes = chunk_result.map_err(|e| ShellError::InferenceError(e.to_string()))?;
            buffer.push_str(&String::from_utf8_lossy(&bytes));

            // Process complete lines (newline-delimited JSON)
            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim().to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                if line.is_empty() {
                    continue;
                }

                match serde_json::from_str::<GenerateChunk>(&line) {
                    Ok(chunk) => {
                        if !chunk.response.is_empty() {
                            if token_count == 0 {
                                first_token_ms = start.elapsed().as_millis() as u64;
                            }
                            token_count += 1;
                            print!("{}", chunk.response);
                            // Flush immediately for real-time streaming
                            let _ = std::io::stdout().flush();
                        }
                        if chunk.done {
                            println!();
                            let total_ms = start.elapsed().as_millis() as u64;
                            let tps = if total_ms > 0 {
                                token_count as f32 / (total_ms as f32 / 1000.0)
                            } else {
                                0.0
                            };
                            return Ok(GenerationStats {
                                first_token_ms,
                                total_ms,
                                tokens_generated: chunk.eval_count.unwrap_or(token_count),
                                tokens_per_second: tps,
                            });
                        }
                    }
                    Err(e) => {
                        debug!("Skipping unparseable chunk: {e}");
                    }
                }
            }
        }

        // Stream ended without done=true
        println!();
        let total_ms = start.elapsed().as_millis() as u64;
        Ok(GenerationStats {
            first_token_ms,
            total_ms,
            tokens_generated: token_count,
            tokens_per_second: if total_ms > 0 {
                token_count as f32 / (total_ms as f32 / 1000.0)
            } else {
                0.0
            },
        })
    }

    /// Generate with the shell system prompt (for natural language queries).
    pub async fn query(&self, input: &str) -> error::Result<GenerationStats> {
        self.generate_streaming(input, Some(&self.system_prompt))
            .await
    }

    /// Generate with context (for pipe-to-LLM).
    pub async fn query_with_context(
        &self,
        context: &str,
        command: &str,
        query: &str,
    ) -> error::Result<GenerationStats> {
        let prompt = format!(
            "The following is the output of the command `{command}`:\n\n```\n{context}\n```\n\n{query}"
        );
        self.generate_streaming(&prompt, Some(&self.system_prompt))
            .await
    }

    /// Load a model into Ollama (warm it up).
    pub async fn load_model(&self, name: &str) -> error::Result<()> {
        let url = format!("{}/api/generate", self.base_url);
        let body = GenerateRequest {
            model: name.to_string(),
            prompt: String::new(),
            system: None,
            stream: false,
            options: None,
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .timeout(Duration::from_secs(600))
            .send()
            .await
            .map_err(|e| ShellError::InferenceUnavailable(e.to_string()))?;

        if !resp.status().is_success() {
            let err = extract_error(resp).await;
            return Err(ShellError::InferenceError(err));
        }

        Ok(())
    }

    /// Unload a model from Ollama.
    pub async fn unload_model(&self, name: &str) -> error::Result<()> {
        let url = format!("{}/api/generate", self.base_url);
        // Ollama unloads a model when keep_alive is set to 0
        let body = serde_json::json!({
            "model": name,
            "prompt": "",
            "keep_alive": 0,
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| ShellError::InferenceUnavailable(e.to_string()))?;

        if !resp.status().is_success() {
            let err = extract_error(resp).await;
            return Err(ShellError::InferenceError(err));
        }
        Ok(())
    }

    /// Pull a model from the Ollama registry, streaming progress to stdout.
    pub async fn pull_model(&self, name: &str) -> error::Result<()> {
        let url = format!("{}/api/pull", self.base_url);
        let body = serde_json::json!({
            "name": name,
            "stream": true,
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .timeout(Duration::from_secs(3600))
            .send()
            .await
            .map_err(|e| ShellError::InferenceUnavailable(e.to_string()))?;

        if !resp.status().is_success() {
            let err = extract_error(resp).await;
            return Err(ShellError::InferenceError(err));
        }

        let mut stream = resp.bytes_stream();
        let mut buffer = String::with_capacity(4096);
        let mut last_status = String::new();

        while let Some(chunk_result) = stream.next().await {
            let bytes = chunk_result.map_err(|e| ShellError::InferenceError(e.to_string()))?;
            buffer.push_str(&String::from_utf8_lossy(&bytes));

            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim().to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                if line.is_empty() {
                    continue;
                }

                if let Ok(chunk) = serde_json::from_str::<PullChunk>(&line) {
                    if chunk.status != last_status {
                        if !last_status.is_empty() {
                            println!();
                        }
                        last_status.clone_from(&chunk.status);
                    }
                    if let (Some(total), Some(completed)) = (chunk.total, chunk.completed) {
                        let pct = if total > 0 {
                            (completed as f64 / total as f64) * 100.0
                        } else {
                            0.0
                        };
                        print!("\r  {}: {pct:.1}%", chunk.status);
                        let _ = std::io::stdout().flush();
                    } else {
                        print!("\r  {}", chunk.status);
                        let _ = std::io::stdout().flush();
                    }
                }
            }
        }

        println!("\n  Done.");
        Ok(())
    }
}

/// Extract an error message from a non-success HTTP response.
async fn extract_error(resp: reqwest::Response) -> String {
    let status = resp.status();
    match resp.json::<ErrorResponse>().await {
        Ok(err_resp) => format!("{status}: {}", err_resp.error),
        Err(_) => format!("HTTP {status}"),
    }
}

/// Format bytes as a human-readable string.
pub fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;

    if bytes >= GIB {
        format!("{:.1} GB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1024 * 1024 * 5), "5.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024 * 4), "4.0 GB");
    }

    #[test]
    fn test_inference_client_creation() {
        let config = ShellConfig::default();
        let client = InferenceClient::new(&config);
        assert_eq!(client.model(), "llama3.2:3b");
    }

    #[test]
    fn test_set_model() {
        let config = ShellConfig::default();
        let mut client = InferenceClient::new(&config);
        client.set_model("qwen3:8b".into());
        assert_eq!(client.model(), "qwen3:8b");
    }
}
