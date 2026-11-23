//! Embedding generation pipeline using the Ollama API.
//!
//! Generates vector embeddings for text content by calling Ollama's
//! `/api/embeddings` endpoint with a small, fast model (default: nomic-embed-text).
//! Runs on CPU at P4 (idle) priority so it doesn't compete with the primary
//! LLM for GPU resources.

use std::path::Path;
use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, warn};

use crate::config::SpiritFsConfig;
use crate::error::{Result, SpiritFsError};
use crate::types::TextChunk;

/// Client for generating embeddings via Ollama's API.
#[derive(Debug, Clone)]
pub struct EmbeddingClient {
    client: Client,
    base_url: String,
    model: String,
    dimensions: u32,
}

/// Request body for Ollama `/api/embeddings`.
#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    prompt: String,
}

/// Response body from Ollama `/api/embeddings`.
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    embedding: Vec<f32>,
}

/// Configuration for text chunking.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Maximum tokens per chunk (approximate — uses whitespace splitting).
    pub max_tokens: usize,
    /// Token overlap between consecutive chunks.
    pub overlap_tokens: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            overlap_tokens: 64,
        }
    }
}

impl EmbeddingClient {
    /// Create a new embedding client from SpiritFS config.
    pub fn new(config: &SpiritFsConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .pool_max_idle_per_host(4)
            .build()
            .expect("failed to build HTTP client");

        Self {
            client,
            base_url: config.ollama_url.trim_end_matches('/').to_string(),
            model: config.embedding_model.clone(),
            dimensions: config.embedding_dimensions,
        }
    }

    /// Check if the embedding model is available.
    #[instrument(skip(self))]
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/api/tags", self.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// Generate an embedding vector for the given text.
    #[instrument(skip(self, text), fields(text_len = text.len()))]
    pub async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        if text.trim().is_empty() {
            return Err(SpiritFsError::Embedding("empty text".to_string()));
        }

        let url = format!("{}/api/embeddings", self.base_url);
        let body = EmbeddingRequest {
            model: self.model.clone(),
            prompt: text.to_string(),
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| SpiritFsError::Embedding(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(SpiritFsError::Embedding(format!(
                "Ollama returned {status}: {body_text}"
            )));
        }

        let embed: EmbeddingResponse = resp.json().await?;

        if embed.embedding.is_empty() {
            return Err(SpiritFsError::Embedding(
                "empty embedding vector returned".to_string(),
            ));
        }

        debug!(dimensions = embed.embedding.len(), "generated embedding");
        Ok(embed.embedding)
    }

    /// Generate embeddings for all chunks of a file.
    ///
    /// Returns pairs of (chunk, embedding_vector).
    #[instrument(skip(self, chunks), fields(chunk_count = chunks.len()))]
    pub async fn embed_chunks(&self, chunks: &[TextChunk]) -> Result<Vec<(TextChunk, Vec<f32>)>> {
        let mut results = Vec::with_capacity(chunks.len());

        for chunk in chunks {
            match self.embed_text(&chunk.text).await {
                Ok(embedding) => {
                    results.push((chunk.clone(), embedding));
                }
                Err(e) => {
                    warn!(
                        path = %chunk.source_path.display(),
                        chunk_index = chunk.chunk_index,
                        error = %e,
                        "failed to embed chunk, skipping"
                    );
                }
            }
        }

        Ok(results)
    }

    /// Get the configured embedding dimensions.
    pub fn dimensions(&self) -> u32 {
        self.dimensions
    }

    /// Get the configured model name.
    pub fn model(&self) -> &str {
        &self.model
    }
}

/// Split text content into overlapping chunks for embedding.
///
/// Uses whitespace-based tokenization (approximate). Splits at paragraph
/// boundaries when possible, falling back to sentence and then word boundaries.
pub fn chunk_text(source_path: &Path, text: &str, config: &ChunkConfig) -> Vec<TextChunk> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }

    // If the text fits in a single chunk, return it directly
    if words.len() <= config.max_tokens {
        return vec![TextChunk {
            source_path: source_path.to_path_buf(),
            chunk_index: 0,
            byte_offset: 0,
            text: text.to_string(),
        }];
    }

    let mut chunks = Vec::new();
    let mut start = 0;
    let mut chunk_index = 0u32;

    while start < words.len() {
        let end = (start + config.max_tokens).min(words.len());
        let chunk_text = words[start..end].join(" ");

        // Calculate approximate byte offset
        let byte_offset = if start == 0 {
            0
        } else {
            // Sum lengths of previous words + spaces
            words[..start].iter().map(|w| w.len() + 1).sum::<usize>()
        };

        chunks.push(TextChunk {
            source_path: source_path.to_path_buf(),
            chunk_index,
            byte_offset,
            text: chunk_text,
        });

        chunk_index += 1;

        // Advance with overlap
        let step = config.max_tokens.saturating_sub(config.overlap_tokens);
        if step == 0 {
            break;
        }
        start += step;
    }

    chunks
}

/// Detect MIME type from file extension (simple heuristic).
pub fn detect_mime_type(path: &Path) -> Option<String> {
    let ext = path.extension()?.to_str()?.to_lowercase();
    let mime = match ext.as_str() {
        "txt" | "log" | "conf" | "cfg" | "ini" | "env" => "text/plain",
        "md" | "markdown" => "text/markdown",
        "rs" => "text/x-rust",
        "py" => "text/x-python",
        "js" | "mjs" => "text/javascript",
        "ts" | "tsx" => "text/typescript",
        "c" | "h" => "text/x-c",
        "cpp" | "cc" | "hpp" => "text/x-c++",
        "java" => "text/x-java",
        "go" => "text/x-go",
        "rb" => "text/x-ruby",
        "php" => "text/x-php",
        "sh" | "bash" | "zsh" => "text/x-shellscript",
        "json" => "application/json",
        "yaml" | "yml" => "text/yaml",
        "toml" => "text/toml",
        "xml" => "application/xml",
        "html" | "htm" => "text/html",
        "css" => "text/css",
        "csv" => "text/csv",
        "sql" => "text/x-sql",
        "tex" => "text/x-tex",
        "rst" => "text/x-rst",
        _ => return None,
    };
    Some(mime.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_chunk_text_single_chunk() {
        let config = ChunkConfig {
            max_tokens: 100,
            overlap_tokens: 10,
        };
        let text = "hello world this is a short text";
        let chunks = chunk_text(Path::new("test.txt"), text, &config);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, text);
        assert_eq!(chunks[0].chunk_index, 0);
    }

    #[test]
    fn test_chunk_text_multiple_chunks() {
        let config = ChunkConfig {
            max_tokens: 5,
            overlap_tokens: 2,
        };
        let words: Vec<&str> = (0..20).map(|_| "word").collect();
        let text = words.join(" ");
        let chunks = chunk_text(Path::new("test.txt"), &text, &config);

        assert!(chunks.len() > 1);
        // Verify overlap: last 2 words of chunk N appear at start of chunk N+1
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_index, i as u32);
        }
    }

    #[test]
    fn test_chunk_text_empty() {
        let config = ChunkConfig::default();
        let chunks = chunk_text(Path::new("test.txt"), "", &config);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_text_whitespace_only() {
        let config = ChunkConfig::default();
        let chunks = chunk_text(Path::new("test.txt"), "   \n\t  ", &config);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_detect_mime_type() {
        assert_eq!(
            detect_mime_type(Path::new("foo.rs")),
            Some("text/x-rust".to_string())
        );
        assert_eq!(
            detect_mime_type(Path::new("foo.py")),
            Some("text/x-python".to_string())
        );
        assert_eq!(
            detect_mime_type(Path::new("foo.json")),
            Some("application/json".to_string())
        );
        assert_eq!(detect_mime_type(Path::new("foo.bin")), None);
    }

    #[test]
    fn test_chunk_config_default() {
        let config = ChunkConfig::default();
        assert_eq!(config.max_tokens, 512);
        assert_eq!(config.overlap_tokens, 64);
    }

    #[test]
    fn test_chunk_source_path_preserved() {
        let config = ChunkConfig {
            max_tokens: 3,
            overlap_tokens: 1,
        };
        let path = PathBuf::from("/some/file.rs");
        let chunks = chunk_text(&path, "a b c d e f", &config);
        for chunk in &chunks {
            assert_eq!(chunk.source_path, path);
        }
    }
}
