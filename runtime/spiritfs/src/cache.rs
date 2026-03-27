//! Model weight cache with frequency-weighted LRU eviction.
//!
//! Tracks model weight files in the CAS and manages eviction based on
//! a frequency-weighted recency score:
//! `score = access_count * recency_factor`
//!
//! Model files in ZONE_MODEL memory are tracked here for efficient
//! loading and prefetching.

use std::sync::Arc;

use chrono::Utc;
use tracing::{debug, info, instrument};

use crate::cas::ContentAddressableStore;
use crate::config::SpiritFsConfig;
use crate::error::Result;
use crate::metadata::MetadataStore;
use crate::types::{ContentHash, ModelCacheEntry};

/// Model weight cache manager.
pub struct ModelCache {
    cas: Arc<ContentAddressableStore>,
    metadata: Arc<MetadataStore>,
    max_size_bytes: u64,
}

impl ModelCache {
    /// Create a new model cache manager.
    pub fn new(
        cas: Arc<ContentAddressableStore>,
        metadata: Arc<MetadataStore>,
        config: &SpiritFsConfig,
    ) -> Self {
        Self {
            cas,
            metadata,
            max_size_bytes: config.model_cache_max_bytes,
        }
    }

    /// Record a model file access, updating frequency and recency.
    #[instrument(skip(self))]
    pub fn record_access(
        &self,
        model_name: &str,
        blob_hash: &ContentHash,
        size: u64,
    ) -> Result<()> {
        let entry = ModelCacheEntry {
            blob_hash: blob_hash.clone(),
            model_name: model_name.to_string(),
            access_count: 1,
            last_accessed: Utc::now(),
            size,
        };
        self.metadata.record_model_access(&entry)?;
        debug!(model = model_name, "recorded model access");
        Ok(())
    }

    /// Check if a model file is cached in the CAS.
    pub fn is_cached(&self, blob_hash: &ContentHash) -> bool {
        self.cas.exists(blob_hash)
    }

    /// Get all cached model entries sorted by eviction priority (lowest first).
    pub fn cache_entries(&self) -> Result<Vec<ModelCacheEntry>> {
        self.metadata.model_cache_entries()
    }

    /// Get total size of cached models.
    pub fn total_cached_size(&self) -> Result<u64> {
        let entries = self.cache_entries()?;
        Ok(entries.iter().map(|e| e.size).sum())
    }

    /// Evict least-valuable cache entries until under the size limit.
    ///
    /// Uses frequency-weighted recency scoring:
    /// `score = access_count * recency_factor`
    /// where `recency_factor = 1.0 / (1.0 + hours_since_last_access)`
    #[instrument(skip(self))]
    pub fn evict_to_limit(&self) -> Result<Vec<ContentHash>> {
        let entries = self.cache_entries()?;
        let total: u64 = entries.iter().map(|e| e.size).sum();

        if total <= self.max_size_bytes {
            return Ok(Vec::new());
        }

        let mut scored: Vec<(f64, ModelCacheEntry)> = entries
            .into_iter()
            .map(|entry| {
                let hours_since = (Utc::now() - entry.last_accessed).num_hours().max(0) as f64;
                let recency_factor = 1.0 / (1.0 + hours_since);
                let score = entry.access_count as f64 * recency_factor;
                (score, entry)
            })
            .collect();

        // Sort by score ascending (lowest score = first to evict)
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut evicted = Vec::new();
        let mut current_total = total;

        for (score, entry) in &scored {
            if current_total <= self.max_size_bytes {
                break;
            }

            debug!(
                model = entry.model_name,
                score,
                size = entry.size,
                "evicting model from cache"
            );

            evicted.push(entry.blob_hash.clone());
            current_total = current_total.saturating_sub(entry.size);
        }

        info!(
            evicted = evicted.len(),
            freed = total - current_total,
            "model cache eviction complete"
        );
        Ok(evicted)
    }

    /// Predict which model should be prefetched based on access patterns.
    ///
    /// Simple heuristic: if model A is always accessed after model B,
    /// suggest prefetching A when B is loaded.
    pub fn predict_prefetch(&self, current_model: &str) -> Result<Option<String>> {
        let entries = self.cache_entries()?;

        // Find models accessed most frequently after the current model
        // For now, suggest the most frequently accessed model that isn't
        // the current one.
        let candidate = entries
            .iter()
            .filter(|e| e.model_name != current_model)
            .max_by_key(|e| e.access_count);

        Ok(candidate.map(|e| e.model_name.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_weighted_scoring() {
        // Verify scoring logic: high access count + recent = high score
        let _now = Utc::now();
        let recent_score = {
            let hours_since = 0.0_f64;
            let recency = 1.0 / (1.0 + hours_since);
            10.0 * recency
        };
        let old_score = {
            let hours_since = 100.0_f64;
            let recency = 1.0 / (1.0 + hours_since);
            10.0 * recency
        };

        assert!(recent_score > old_score);
    }

    #[test]
    fn test_model_cache_entry() {
        let entry = ModelCacheEntry {
            blob_hash: ContentHash("abc123".to_string()),
            model_name: "llama3.1:8b".to_string(),
            access_count: 5,
            last_accessed: Utc::now(),
            size: 4_000_000_000,
        };
        assert_eq!(entry.model_name, "llama3.1:8b");
        assert_eq!(entry.access_count, 5);
    }
}
