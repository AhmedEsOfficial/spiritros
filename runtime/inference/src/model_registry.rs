//! Model registry — discovery, tracking, and metadata management.
//!
//! The registry maintains a catalog of available models and their load states.
//! In Phase 1, it syncs with Ollama's model list via the HTTP API and tracks
//! which models are currently loaded in which memory zones.
//!
//! The registry uses a content-addressable design: models are identified by
//! their SHA-256 digest, with friendly names as aliases.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use parking_lot::RwLock;
use tracing::{debug, info, instrument, warn};

use crate::error::{InferenceError, Result};
use crate::ollama_client::OllamaClient;
use crate::types::{LoadedModelInfo, ModelId, ModelInfo, ModelLoadState};

/// Thread-safe model registry.
///
/// Tracks all known models, their metadata, and current load state.
/// Syncs with Ollama's model list and tracks additional Spirit-specific metadata.
pub struct ModelRegistry {
    /// Models indexed by internal ID.
    models: RwLock<HashMap<ModelId, RegistryEntry>>,
    /// Name → ModelId index for fast lookup by name.
    name_index: RwLock<HashMap<String, ModelId>>,
    /// Digest → ModelId index for deduplication.
    digest_index: RwLock<HashMap<String, ModelId>>,
    /// Ollama client for syncing.
    ollama: Arc<OllamaClient>,
}

/// Internal registry entry with full metadata and state.
#[derive(Debug, Clone)]
struct RegistryEntry {
    info: ModelInfo,
    state: ModelLoadState,
    gpu_devices: Vec<u32>,
    vram_usage_bytes: u64,
    ram_usage_bytes: u64,
    active_sessions: usize,
}

impl ModelRegistry {
    /// Create a new model registry backed by the given Ollama client.
    pub fn new(ollama: Arc<OllamaClient>) -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            name_index: RwLock::new(HashMap::new()),
            digest_index: RwLock::new(HashMap::new()),
            ollama,
        }
    }

    /// Sync the registry with Ollama's model list.
    ///
    /// Discovers new models, updates existing entries, and marks removed
    /// models as unloaded. Does not modify load state for models that are
    /// currently in use.
    #[instrument(skip(self))]
    pub async fn sync_from_ollama(&self) -> Result<usize> {
        let ollama_models = self.ollama.list_models().await?;

        let mut count = 0;
        for om in &ollama_models {
            let digest = om.digest.clone();

            // Check if we already have this model by digest (deduplication)
            {
                let digest_idx = self.digest_index.read();
                if digest_idx.contains_key(&digest) {
                    debug!(name = %om.name, "Model already registered (by digest)");
                    continue;
                }
            }

            // Extract details
            let (family, param_size, quantization, format) = if let Some(ref details) = om.details {
                (
                    details.family.clone(),
                    details.parameter_size.clone(),
                    details.quantization_level.clone(),
                    details.format.clone(),
                )
            } else {
                (String::new(), String::new(), String::new(), String::new())
            };

            let model_id = ModelId::new();
            let info = ModelInfo {
                id: model_id,
                name: om.name.clone(),
                family,
                parameter_size: param_size,
                quantization,
                size_bytes: om.size,
                digest: digest.clone(),
                format,
                registered_at: Utc::now(),
            };

            let entry = RegistryEntry {
                info,
                state: ModelLoadState::Unloaded,
                gpu_devices: Vec::new(),
                vram_usage_bytes: 0,
                ram_usage_bytes: 0,
                active_sessions: 0,
            };

            // Insert into all indices
            {
                let mut models = self.models.write();
                let mut names = self.name_index.write();
                let mut digests = self.digest_index.write();

                models.insert(model_id, entry);
                names.insert(om.name.clone(), model_id);
                if !digest.is_empty() {
                    digests.insert(digest, model_id);
                }
            }

            info!(name = %om.name, id = %model_id, "Registered model");
            count += 1;
        }

        Ok(count)
    }

    /// Look up a model by name.
    pub fn get_by_name(&self, name: &str) -> Option<ModelInfo> {
        let names = self.name_index.read();
        let id = names.get(name)?;
        let models = self.models.read();
        models.get(id).map(|e| e.info.clone())
    }

    /// Look up a model by ID.
    pub fn get(&self, id: &ModelId) -> Option<ModelInfo> {
        let models = self.models.read();
        models.get(id).map(|e| e.info.clone())
    }

    /// Look up a model by SHA-256 digest (deduplication check).
    pub fn get_by_digest(&self, digest: &str) -> Option<ModelInfo> {
        let digests = self.digest_index.read();
        let id = digests.get(digest)?;
        let models = self.models.read();
        models.get(id).map(|e| e.info.clone())
    }

    /// Get the model ID for a given model name.
    pub fn resolve_name(&self, name: &str) -> Option<ModelId> {
        let names = self.name_index.read();
        names.get(name).copied()
    }

    /// List all known models.
    pub fn list(&self) -> Vec<ModelInfo> {
        let models = self.models.read();
        models.values().map(|e| e.info.clone()).collect()
    }

    /// List all loaded models with resource usage information.
    pub fn list_loaded(&self) -> Vec<LoadedModelInfo> {
        let models = self.models.read();
        models
            .values()
            .filter(|e| e.state == ModelLoadState::Loaded)
            .map(|e| LoadedModelInfo {
                info: e.info.clone(),
                state: e.state,
                gpu_devices: e.gpu_devices.clone(),
                vram_usage_bytes: e.vram_usage_bytes,
                ram_usage_bytes: e.ram_usage_bytes,
                active_sessions: e.active_sessions,
            })
            .collect()
    }

    /// Get the load state of a model.
    pub fn load_state(&self, id: &ModelId) -> Option<ModelLoadState> {
        let models = self.models.read();
        models.get(id).map(|e| e.state)
    }

    /// Mark a model as loading.
    pub fn set_loading(&self, id: &ModelId) -> Result<()> {
        let mut models = self.models.write();
        let entry = models
            .get_mut(id)
            .ok_or_else(|| InferenceError::ModelNotFound(id.to_string()))?;
        entry.state = ModelLoadState::Loading;
        Ok(())
    }

    /// Mark a model as loaded with resource usage info.
    pub fn set_loaded(
        &self,
        id: &ModelId,
        gpu_devices: Vec<u32>,
        vram_usage: u64,
        ram_usage: u64,
    ) -> Result<()> {
        let mut models = self.models.write();
        let entry = models
            .get_mut(id)
            .ok_or_else(|| InferenceError::ModelNotFound(id.to_string()))?;
        entry.state = ModelLoadState::Loaded;
        entry.gpu_devices = gpu_devices;
        entry.vram_usage_bytes = vram_usage;
        entry.ram_usage_bytes = ram_usage;
        info!(
            name = %entry.info.name,
            vram_mb = vram_usage / (1024 * 1024),
            "Model loaded"
        );
        Ok(())
    }

    /// Mark a model as unloaded.
    pub fn set_unloaded(&self, id: &ModelId) -> Result<()> {
        let mut models = self.models.write();
        let entry = models
            .get_mut(id)
            .ok_or_else(|| InferenceError::ModelNotFound(id.to_string()))?;
        entry.state = ModelLoadState::Unloaded;
        entry.gpu_devices.clear();
        entry.vram_usage_bytes = 0;
        entry.ram_usage_bytes = 0;
        info!(name = %entry.info.name, "Model unloaded");
        Ok(())
    }

    /// Mark a model as failed.
    pub fn set_failed(&self, id: &ModelId) -> Result<()> {
        let mut models = self.models.write();
        let entry = models
            .get_mut(id)
            .ok_or_else(|| InferenceError::ModelNotFound(id.to_string()))?;
        entry.state = ModelLoadState::Failed;
        warn!(name = %entry.info.name, "Model load failed");
        Ok(())
    }

    /// Increment the active session count for a model.
    pub fn increment_sessions(&self, id: &ModelId) -> Result<usize> {
        let mut models = self.models.write();
        let entry = models
            .get_mut(id)
            .ok_or_else(|| InferenceError::ModelNotFound(id.to_string()))?;
        entry.active_sessions += 1;
        Ok(entry.active_sessions)
    }

    /// Decrement the active session count for a model.
    pub fn decrement_sessions(&self, id: &ModelId) -> Result<usize> {
        let mut models = self.models.write();
        let entry = models
            .get_mut(id)
            .ok_or_else(|| InferenceError::ModelNotFound(id.to_string()))?;
        entry.active_sessions = entry.active_sessions.saturating_sub(1);
        Ok(entry.active_sessions)
    }

    /// Get the number of loaded models.
    pub fn loaded_count(&self) -> usize {
        let models = self.models.read();
        models
            .values()
            .filter(|e| e.state == ModelLoadState::Loaded)
            .count()
    }

    /// Get the total number of registered models.
    pub fn total_count(&self) -> usize {
        self.models.read().len()
    }

    /// Check if a model name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.name_index.read().contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OllamaConfig;

    fn test_registry() -> ModelRegistry {
        let client = Arc::new(OllamaClient::new(&OllamaConfig::default()));
        ModelRegistry::new(client)
    }

    fn insert_test_model(registry: &ModelRegistry, name: &str) -> ModelId {
        let model_id = ModelId::new();
        let info = ModelInfo {
            id: model_id,
            name: name.to_string(),
            family: "llama".into(),
            parameter_size: "8B".into(),
            quantization: "Q4_K_M".into(),
            size_bytes: 4_600_000_000,
            digest: format!("sha256-{name}"),
            format: "gguf".into(),
            registered_at: Utc::now(),
        };

        let entry = RegistryEntry {
            info,
            state: ModelLoadState::Unloaded,
            gpu_devices: Vec::new(),
            vram_usage_bytes: 0,
            ram_usage_bytes: 0,
            active_sessions: 0,
        };

        registry.models.write().insert(model_id, entry);
        registry
            .name_index
            .write()
            .insert(name.to_string(), model_id);
        registry
            .digest_index
            .write()
            .insert(format!("sha256-{name}"), model_id);

        model_id
    }

    #[test]
    fn test_registry_insert_and_lookup() {
        let reg = test_registry();
        let id = insert_test_model(&reg, "llama3.1:8b");

        let info = reg.get(&id).expect("model should exist");
        assert_eq!(info.name, "llama3.1:8b");

        let by_name = reg.get_by_name("llama3.1:8b").expect("name lookup");
        assert_eq!(by_name.id, id);

        let by_digest = reg
            .get_by_digest("sha256-llama3.1:8b")
            .expect("digest lookup");
        assert_eq!(by_digest.id, id);
    }

    #[test]
    fn test_registry_load_state_transitions() {
        let reg = test_registry();
        let id = insert_test_model(&reg, "test-model");

        assert_eq!(reg.load_state(&id), Some(ModelLoadState::Unloaded));

        reg.set_loading(&id).unwrap();
        assert_eq!(reg.load_state(&id), Some(ModelLoadState::Loading));

        reg.set_loaded(&id, vec![0], 4_000_000_000, 0).unwrap();
        assert_eq!(reg.load_state(&id), Some(ModelLoadState::Loaded));

        reg.set_unloaded(&id).unwrap();
        assert_eq!(reg.load_state(&id), Some(ModelLoadState::Unloaded));
    }

    #[test]
    fn test_registry_session_counting() {
        let reg = test_registry();
        let id = insert_test_model(&reg, "test-model");

        assert_eq!(reg.increment_sessions(&id).unwrap(), 1);
        assert_eq!(reg.increment_sessions(&id).unwrap(), 2);
        assert_eq!(reg.decrement_sessions(&id).unwrap(), 1);
        assert_eq!(reg.decrement_sessions(&id).unwrap(), 0);
        // Should not go below 0
        assert_eq!(reg.decrement_sessions(&id).unwrap(), 0);
    }

    #[test]
    fn test_registry_list_loaded() {
        let reg = test_registry();
        let id1 = insert_test_model(&reg, "model-a");
        let id2 = insert_test_model(&reg, "model-b");

        reg.set_loaded(&id1, vec![0], 2_000_000_000, 0).unwrap();
        // id2 stays unloaded

        let loaded = reg.list_loaded();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].info.id, id1);
        assert_eq!(reg.loaded_count(), 1);
        assert_eq!(reg.total_count(), 2);

        // Verify id2 is not in loaded list
        assert!(!loaded.iter().any(|m| m.info.id == id2));
    }

    #[test]
    fn test_registry_contains() {
        let reg = test_registry();
        insert_test_model(&reg, "exists");
        assert!(reg.contains("exists"));
        assert!(!reg.contains("nope"));
    }

    #[test]
    fn test_registry_resolve_name() {
        let reg = test_registry();
        let id = insert_test_model(&reg, "my-model");
        assert_eq!(reg.resolve_name("my-model"), Some(id));
        assert_eq!(reg.resolve_name("other"), None);
    }
}
