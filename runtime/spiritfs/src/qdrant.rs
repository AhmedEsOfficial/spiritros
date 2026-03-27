//! Qdrant vector store client for SpiritFS semantic search.
//!
//! Manages the `spiritfs` collection in Qdrant, supporting vector upsert,
//! search, and deletion. Uses the Qdrant gRPC API for performance.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter, PointId,
    PointStruct, ScrollPointsBuilder, SearchPointsBuilder, UpsertPointsBuilder,
    Value as QdrantValue, VectorParamsBuilder,
};
use qdrant_client::Qdrant;
use tracing::{debug, info, instrument};
use uuid::Uuid;

use crate::config::SpiritFsConfig;
use crate::error::{Result, SpiritFsError};
use crate::types::SearchResult;

/// Wrapper around the Qdrant gRPC client for SpiritFS.
pub struct VectorStore {
    client: Qdrant,
    collection: String,
    dimensions: u32,
}

/// Payload stored with each vector point in Qdrant.
#[derive(Debug, Clone)]
pub struct VectorPayload {
    /// File path relative to mount point.
    pub path: String,
    /// File name (basename).
    pub name: String,
    /// Chunk text that was embedded.
    pub chunk_text: String,
    /// Chunk index within the file.
    pub chunk_index: u32,
    /// MIME type of the source file.
    pub mime_type: String,
    /// Comma-separated tags.
    pub tags: String,
}

impl VectorStore {
    /// Connect to Qdrant and ensure the collection exists.
    pub async fn connect(config: &SpiritFsConfig) -> Result<Self> {
        let client = Qdrant::from_url(&config.qdrant_url)
            .build()
            .map_err(|e| SpiritFsError::Qdrant(e.to_string()))?;

        let store = Self {
            client,
            collection: config.qdrant_collection.clone(),
            dimensions: config.embedding_dimensions,
        };

        store.ensure_collection().await?;
        Ok(store)
    }

    /// Create the collection if it doesn't exist.
    async fn ensure_collection(&self) -> Result<()> {
        let exists = self
            .client
            .collection_exists(&self.collection)
            .await
            .map_err(|e| SpiritFsError::Qdrant(e.to_string()))?;

        if exists {
            debug!(collection = %self.collection, "collection already exists");
            return Ok(());
        }

        info!(
            collection = %self.collection,
            dimensions = self.dimensions,
            "creating Qdrant collection"
        );

        let params = VectorParamsBuilder::new(self.dimensions.into(), Distance::Cosine);

        self.client
            .create_collection(
                CreateCollectionBuilder::new(&self.collection).vectors_config(params),
            )
            .await
            .map_err(|e| SpiritFsError::Qdrant(e.to_string()))?;

        Ok(())
    }

    /// Upsert a vector with metadata payload.
    #[instrument(skip(self, embedding), fields(collection = %self.collection))]
    pub async fn upsert(
        &self,
        point_id: Uuid,
        embedding: Vec<f32>,
        payload: VectorPayload,
    ) -> Result<()> {
        let mut payload_map: HashMap<String, QdrantValue> = HashMap::new();
        payload_map.insert("path".to_string(), payload.path.into());
        payload_map.insert("name".to_string(), payload.name.into());
        payload_map.insert("chunk_text".to_string(), payload.chunk_text.into());
        payload_map.insert(
            "chunk_index".to_string(),
            (payload.chunk_index as i64).into(),
        );
        payload_map.insert("mime_type".to_string(), payload.mime_type.into());
        payload_map.insert("tags".to_string(), payload.tags.into());

        let point = PointStruct::new(point_id.to_string(), embedding, payload_map);

        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection, vec![point]).wait(true))
            .await
            .map_err(|e| SpiritFsError::Qdrant(e.to_string()))?;

        debug!(point_id = %point_id, "upserted vector");
        Ok(())
    }

    /// Search for similar vectors.
    #[instrument(skip(self, query_vector), fields(limit))]
    pub async fn search(
        &self,
        query_vector: Vec<f32>,
        limit: u64,
        path_filter: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let mut builder =
            SearchPointsBuilder::new(&self.collection, query_vector, limit).with_payload(true);

        if let Some(prefix) = path_filter {
            let filter = Filter::must([Condition::matches("path", prefix.to_string())]);
            builder = builder.filter(filter);
        }

        let response = self
            .client
            .search_points(builder)
            .await
            .map_err(|e| SpiritFsError::Qdrant(e.to_string()))?;

        let results = response
            .result
            .into_iter()
            .map(|point| {
                let path = point
                    .payload
                    .get("path")
                    .and_then(|v| v.as_str())
                    .map(PathBuf::from)
                    .unwrap_or_default();
                let chunk_text = point
                    .payload
                    .get("chunk_text")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let chunk_index = point
                    .payload
                    .get("chunk_index")
                    .and_then(QdrantValue::as_integer)
                    .map(|i| i as u32);

                SearchResult {
                    path,
                    score: point.score,
                    chunk_text,
                    chunk_index,
                }
            })
            .collect();

        Ok(results)
    }

    /// Delete all vectors associated with a file path.
    #[instrument(skip(self))]
    pub async fn delete_by_path(&self, path: &Path) -> Result<u64> {
        let path_str = path.to_string_lossy().to_string();

        // First, find all points with this path
        let filter = Filter::must([Condition::matches("path", path_str)]);

        let scroll_result = self
            .client
            .scroll(
                ScrollPointsBuilder::new(&self.collection)
                    .filter(filter.clone())
                    .limit(10000),
            )
            .await
            .map_err(|e| SpiritFsError::Qdrant(e.to_string()))?;

        let point_ids: Vec<PointId> = scroll_result
            .result
            .iter()
            .filter_map(|p| p.id.clone())
            .collect();

        let count = point_ids.len() as u64;
        if count == 0 {
            return Ok(0);
        }

        self.client
            .delete_points(DeletePointsBuilder::new(&self.collection).points(point_ids))
            .await
            .map_err(|e| SpiritFsError::Qdrant(e.to_string()))?;

        debug!(path = %path.display(), deleted = count, "deleted vectors for path");
        Ok(count)
    }

    /// Get the total number of vectors in the collection.
    pub async fn vector_count(&self) -> Result<u64> {
        let info = self
            .client
            .collection_info(&self.collection)
            .await
            .map_err(|e| SpiritFsError::Qdrant(e.to_string()))?;

        Ok(info.result.map_or(0, |r| r.points_count.unwrap_or(0)))
    }

    /// Delete the entire collection (for rebuild).
    pub async fn drop_collection(&self) -> Result<()> {
        let exists = self
            .client
            .collection_exists(&self.collection)
            .await
            .map_err(|e| SpiritFsError::Qdrant(e.to_string()))?;

        if exists {
            self.client
                .delete_collection(&self.collection)
                .await
                .map_err(|e| SpiritFsError::Qdrant(e.to_string()))?;

            info!(collection = %self.collection, "dropped collection");
        }
        Ok(())
    }

    /// Rebuild the collection (drop and recreate).
    pub async fn rebuild_collection(&self) -> Result<()> {
        self.drop_collection().await?;
        self.ensure_collection().await?;
        Ok(())
    }

    /// Check if Qdrant is reachable.
    pub async fn health_check(&self) -> Result<bool> {
        match self.client.health_check().await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_payload_creation() {
        let payload = VectorPayload {
            path: "/src/main.rs".to_string(),
            name: "main.rs".to_string(),
            chunk_text: "fn main() {}".to_string(),
            chunk_index: 0,
            mime_type: "text/x-rust".to_string(),
            tags: "rust,main".to_string(),
        };
        assert_eq!(payload.path, "/src/main.rs");
        assert_eq!(payload.chunk_index, 0);
    }
}
