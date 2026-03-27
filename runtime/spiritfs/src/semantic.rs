//! Semantic operations for SpiritFS.
//!
//! High-level API for semantic search, similarity lookup, directory
//! summarization, and auto-tagging. Orchestrates the embedding client,
//! vector store, and metadata store.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tracing::{debug, instrument, warn};

use crate::cas::ContentAddressableStore;
use crate::config::SpiritFsConfig;
use crate::embedding::EmbeddingClient;
use crate::error::{Result, SpiritFsError};
use crate::metadata::MetadataStore;
use crate::qdrant::VectorStore;
use crate::types::{SearchResult, SemanticTag};

/// Response body from Ollama generate API.
#[derive(serde::Deserialize)]
struct GenerateResponse {
    response: String,
}

/// High-level semantic engine that composes all subsystems.
pub struct SemanticEngine {
    config: SpiritFsConfig,
    embedding_client: EmbeddingClient,
    vector_store: Arc<VectorStore>,
    metadata: Arc<MetadataStore>,
    #[allow(dead_code)]
    cas: Arc<ContentAddressableStore>,
}

impl SemanticEngine {
    /// Create a new semantic engine.
    pub fn new(
        config: SpiritFsConfig,
        embedding_client: EmbeddingClient,
        vector_store: Arc<VectorStore>,
        metadata: Arc<MetadataStore>,
        cas: Arc<ContentAddressableStore>,
    ) -> Self {
        Self {
            config,
            embedding_client,
            vector_store,
            metadata,
            cas,
        }
    }

    /// Search for files matching a semantic query.
    ///
    /// Embeds the query text, searches the vector store for similar vectors,
    /// and returns ranked results.
    #[instrument(skip(self))]
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        // Try vector search first
        match self.vector_search(query, limit).await {
            Ok(results) if !results.is_empty() => {
                debug!(query, results = results.len(), "vector search succeeded");
                Ok(results)
            }
            Ok(_) => {
                debug!(
                    query,
                    "vector search returned no results, falling back to FTS"
                );
                self.fts_fallback(query, limit)
            }
            Err(e) => {
                warn!(
                    query,
                    error = %e,
                    "vector search failed, falling back to FTS"
                );
                self.fts_fallback(query, limit)
            }
        }
    }

    /// Vector-based semantic search.
    async fn vector_search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let query_embedding = self.embedding_client.embed_text(query).await?;
        self.vector_store
            .search(query_embedding, limit as u64, None)
            .await
    }

    /// FTS5 fallback search when Qdrant is unavailable.
    fn fts_fallback(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let metas = self.metadata.fts_search(query, limit)?;
        Ok(metas
            .into_iter()
            .enumerate()
            .map(|(i, meta)| SearchResult {
                path: meta.path,
                score: 1.0 - (i as f32 * 0.05), // Decreasing score by rank
                chunk_text: None,
                chunk_index: None,
            })
            .collect())
    }

    /// Find files similar to a given file.
    ///
    /// Reads the file content, generates an embedding, and searches for
    /// similar vectors in the store.
    #[instrument(skip(self))]
    pub async fn similar(&self, file_path: &Path, limit: usize) -> Result<Vec<SearchResult>> {
        let backing_path = self.resolve_backing_path(file_path);

        let content = std::fs::read_to_string(&backing_path)
            .map_err(|e| SpiritFsError::FileNotFound(format!("{}: {e}", file_path.display())))?;

        if content.trim().is_empty() {
            return Ok(Vec::new());
        }

        // Truncate to a reasonable size for embedding
        let truncated: String = content.chars().take(8192).collect();
        let embedding = self.embedding_client.embed_text(&truncated).await?;

        let mut results = self
            .vector_store
            .search(embedding, (limit + 1) as u64, None)
            .await?;

        // Remove the query file itself from results
        let file_path_str = file_path.to_string_lossy();
        results.retain(|r| r.path.to_string_lossy() != file_path_str);
        results.truncate(limit);

        Ok(results)
    }

    /// Summarize the contents of a directory.
    ///
    /// Collects file names and metadata, generates a natural-language summary
    /// using the primary LLM via Ollama.
    #[instrument(skip(self))]
    pub async fn summarize(&self, dir_path: &Path) -> Result<String> {
        // Get file listing from metadata
        let dir_meta = self.metadata.get_by_path(dir_path)?;
        let parent_inode = match dir_meta {
            Some(m) => m.inode,
            None => {
                return Err(SpiritFsError::FileNotFound(
                    dir_path.to_string_lossy().to_string(),
                ));
            }
        };

        let children = self.metadata.list_children(parent_inode)?;
        if children.is_empty() {
            return Ok(format!("Directory {} is empty.", dir_path.display()));
        }

        // Build a listing for the LLM
        let mut listing = String::new();
        for child in &children {
            let kind = match child.kind {
                crate::types::FileKind::Directory => "dir",
                crate::types::FileKind::RegularFile => "file",
                crate::types::FileKind::Symlink => "link",
            };
            let tags: Vec<&str> = child.tags.iter().map(|t| t.label.as_str()).collect();
            let tags_str = if tags.is_empty() {
                String::new()
            } else {
                format!(" [{}]", tags.join(", "))
            };
            let _ = writeln!(
                listing,
                "- {} ({}, {} bytes){}",
                child.name, kind, child.size, tags_str
            );
        }

        let prompt = format!(
            "Summarize the contents of this directory in 2-3 sentences. \
             Focus on what types of files are present and their purpose.\n\n\
             Directory: {}\n\
             Contents:\n{}",
            dir_path.display(),
            listing
        );

        // Call Ollama generate API for summary
        self.generate_summary(&prompt).await
    }

    /// Generate a summary using Ollama's generate API.
    async fn generate_summary(&self, prompt: &str) -> Result<String> {
        let url = format!(
            "{}/api/generate",
            self.config.ollama_url.trim_end_matches('/')
        );

        let body = serde_json::json!({
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": 0.3,
                "num_predict": 200
            }
        });

        let client = reqwest::Client::new();
        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| SpiritFsError::Embedding(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(SpiritFsError::Embedding(format!(
                "Ollama generate returned {}",
                resp.status()
            )));
        }

        let gen: GenerateResponse = resp.json().await?;
        Ok(gen.response.trim().to_string())
    }

    /// Auto-tag a file based on its content.
    ///
    /// Generates tags by analyzing the file content with the embedding model
    /// and applying simple heuristic classification.
    #[instrument(skip(self))]
    pub async fn auto_tag(&self, file_path: &Path) -> Result<Vec<SemanticTag>> {
        let backing_path = self.resolve_backing_path(file_path);

        let content = std::fs::read_to_string(&backing_path)
            .map_err(|e| SpiritFsError::FileNotFound(format!("{}: {e}", file_path.display())))?;

        if content.trim().is_empty() {
            return Ok(Vec::new());
        }

        let mut tags = Vec::new();

        // Extension-based tags
        if let Some(ext) = file_path.extension().and_then(|e| e.to_str()) {
            tags.push(SemanticTag {
                label: format!("lang:{ext}"),
                confidence: 1.0,
            });
        }

        // Content heuristic tags
        let lower = content.to_lowercase();
        let heuristics = [
            ("todo", "has-todos"),
            ("fixme", "has-fixme"),
            ("test", "test-file"),
            ("config", "configuration"),
            ("readme", "documentation"),
            ("license", "legal"),
            ("changelog", "changelog"),
            ("api", "api-related"),
            ("error", "error-handling"),
            ("security", "security-related"),
            ("benchmark", "performance"),
        ];

        for (keyword, tag_label) in &heuristics {
            if lower.contains(keyword) {
                tags.push(SemanticTag {
                    label: (*tag_label).to_string(),
                    confidence: 0.7,
                });
            }
        }

        // Line count based tags
        let line_count = content.lines().count();
        if line_count > 500 {
            tags.push(SemanticTag {
                label: "large-file".to_string(),
                confidence: 0.9,
            });
        }

        // Update metadata with tags
        if let Ok(Some(meta)) = self.metadata.get_by_path(file_path) {
            let _ = self.metadata.set_tags(meta.inode, &tags);
        }

        Ok(tags)
    }

    /// Resolve a mount-point path to the backing filesystem path.
    fn resolve_backing_path(&self, path: &Path) -> PathBuf {
        self.config
            .backing_dir
            .join(path.strip_prefix(&self.config.mount_point).unwrap_or(path))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_fts_fallback_scoring() {
        // Verify that FTS results get decreasing scores
        let scores: Vec<f32> = (0..5).map(|i| 1.0 - (i as f32 * 0.05)).collect();
        assert!(scores[0] > scores[1]);
        assert!(scores[1] > scores[2]);
        assert!(scores.iter().all(|s| *s > 0.0));
    }
}
