//! SQLite metadata store with FTS5 for SpiritFS.
//!
//! Stores file metadata (inodes, paths, content hashes, tags, embedding IDs)
//! in SQLite with FTS5 full-text search as a fallback when Qdrant is unavailable.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension};
use tracing::{debug, instrument};
use uuid::Uuid;

use crate::error::{Result, SpiritFsError};
use crate::types::{ContentHash, FileKind, FileMeta, InodeId, ModelCacheEntry, SemanticTag};

/// SQLite-backed metadata store for SpiritFS.
pub struct MetadataStore {
    conn: Mutex<Connection>,
}

impl MetadataStore {
    /// Open or create the metadata database at the given path.
    pub fn open(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(db_path)?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.init_schema()?;
        Ok(store)
    }

    /// Open an in-memory database (for testing).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.init_schema()?;
        Ok(store)
    }

    /// Initialize the database schema.
    fn init_schema(&self) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS files (
                inode       INTEGER PRIMARY KEY,
                parent_inode INTEGER NOT NULL,
                name        TEXT NOT NULL,
                path        TEXT NOT NULL UNIQUE,
                kind        TEXT NOT NULL DEFAULT 'file',
                size        INTEGER NOT NULL DEFAULT 0,
                mime_type   TEXT,
                blob_hash   TEXT,
                embedding_id TEXT,
                tags        TEXT NOT NULL DEFAULT '[]',
                created_at  TEXT NOT NULL,
                modified_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_files_parent ON files(parent_inode);
            CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);
            CREATE INDEX IF NOT EXISTS idx_files_blob_hash ON files(blob_hash);

            CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
                name, tags, path, content=files, content_rowid=inode
            );

            -- Triggers to keep FTS5 in sync
            CREATE TRIGGER IF NOT EXISTS files_ai AFTER INSERT ON files BEGIN
                INSERT INTO files_fts(rowid, name, tags, path)
                VALUES (new.inode, new.name, new.tags, new.path);
            END;

            CREATE TRIGGER IF NOT EXISTS files_ad AFTER DELETE ON files BEGIN
                INSERT INTO files_fts(files_fts, rowid, name, tags, path)
                VALUES ('delete', old.inode, old.name, old.tags, old.path);
            END;

            CREATE TRIGGER IF NOT EXISTS files_au AFTER UPDATE ON files BEGIN
                INSERT INTO files_fts(files_fts, rowid, name, tags, path)
                VALUES ('delete', old.inode, old.name, old.tags, old.path);
                INSERT INTO files_fts(rowid, name, tags, path)
                VALUES (new.inode, new.name, new.tags, new.path);
            END;

            CREATE TABLE IF NOT EXISTS model_cache (
                blob_hash     TEXT PRIMARY KEY,
                model_name    TEXT NOT NULL,
                access_count  INTEGER NOT NULL DEFAULT 0,
                last_accessed TEXT NOT NULL,
                size          INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS index_state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            ",
        )?;
        Ok(())
    }

    /// Allocate the next available inode number.
    pub fn next_inode(&self) -> Result<InodeId> {
        let conn = self.conn.lock();
        let max: Option<u64> = conn
            .query_row("SELECT MAX(inode) FROM files", [], |row| row.get(0))
            .optional()?
            .flatten();
        Ok(InodeId(max.unwrap_or(0) + 1))
    }

    /// Insert a new file record.
    #[instrument(skip(self, meta), fields(inode = meta.inode.0, path = %meta.path.display()))]
    pub fn insert_file(&self, meta: &FileMeta) -> Result<()> {
        let conn = self.conn.lock();
        let kind = match meta.kind {
            FileKind::RegularFile => "file",
            FileKind::Directory => "directory",
            FileKind::Symlink => "symlink",
        };
        let tags_json = serde_json::to_string(&meta.tags)?;
        let blob_hash = meta.content_hash.as_ref().map(|h| &h.0);
        let embedding_id = meta.embedding_id.map(|id| id.to_string());

        conn.execute(
            "INSERT OR REPLACE INTO files
             (inode, parent_inode, name, path, kind, size, mime_type,
              blob_hash, embedding_id, tags, created_at, modified_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                meta.inode.0,
                meta.parent_inode.0,
                meta.name,
                meta.path.to_string_lossy().as_ref(),
                kind,
                meta.size,
                meta.mime_type,
                blob_hash,
                embedding_id,
                tags_json,
                meta.created_at.to_rfc3339(),
                meta.modified_at.to_rfc3339(),
            ],
        )?;
        debug!(inode = meta.inode.0, "inserted file metadata");
        Ok(())
    }

    /// Get file metadata by inode.
    pub fn get_by_inode(&self, inode: InodeId) -> Result<Option<FileMeta>> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT inode, parent_inode, name, path, kind, size, mime_type,
                    blob_hash, embedding_id, tags, created_at, modified_at
             FROM files WHERE inode = ?1",
            params![inode.0],
            |row| Ok(Self::row_to_meta(row)),
        )
        .optional()?
        .transpose()
    }

    /// Get file metadata by path.
    pub fn get_by_path(&self, path: &Path) -> Result<Option<FileMeta>> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT inode, parent_inode, name, path, kind, size, mime_type,
                    blob_hash, embedding_id, tags, created_at, modified_at
             FROM files WHERE path = ?1",
            params![path.to_string_lossy().as_ref()],
            |row| Ok(Self::row_to_meta(row)),
        )
        .optional()?
        .transpose()
    }

    /// List children of a directory inode.
    pub fn list_children(&self, parent_inode: InodeId) -> Result<Vec<FileMeta>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT inode, parent_inode, name, path, kind, size, mime_type,
                    blob_hash, embedding_id, tags, created_at, modified_at
             FROM files WHERE parent_inode = ?1 ORDER BY name",
        )?;

        let rows = stmt.query_map(params![parent_inode.0], |row| Ok(Self::row_to_meta(row)))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row??);
        }
        Ok(results)
    }

    /// Update the embedding ID for a file.
    pub fn set_embedding_id(&self, inode: InodeId, embedding_id: Uuid) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE files SET embedding_id = ?1 WHERE inode = ?2",
            params![embedding_id.to_string(), inode.0],
        )?;
        Ok(())
    }

    /// Update tags for a file.
    pub fn set_tags(&self, inode: InodeId, tags: &[SemanticTag]) -> Result<()> {
        let conn = self.conn.lock();
        let tags_json = serde_json::to_string(tags)?;
        conn.execute(
            "UPDATE files SET tags = ?1 WHERE inode = ?2",
            params![tags_json, inode.0],
        )?;
        Ok(())
    }

    /// Update the content hash and size for a file.
    pub fn update_content(&self, inode: InodeId, hash: &ContentHash, size: u64) -> Result<()> {
        let conn = self.conn.lock();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE files SET blob_hash = ?1, size = ?2, modified_at = ?3 WHERE inode = ?4",
            params![hash.0, size, now, inode.0],
        )?;
        Ok(())
    }

    /// Delete a file record by path.
    pub fn delete_by_path(&self, path: &Path) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "DELETE FROM files WHERE path = ?1",
            params![path.to_string_lossy().as_ref()],
        )?;
        Ok(())
    }

    /// Full-text search using SQLite FTS5 (fallback when Qdrant is down).
    pub fn fts_search(&self, query: &str, limit: usize) -> Result<Vec<FileMeta>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT f.inode, f.parent_inode, f.name, f.path, f.kind, f.size,
                    f.mime_type, f.blob_hash, f.embedding_id, f.tags,
                    f.created_at, f.modified_at
             FROM files f
             JOIN files_fts fts ON f.inode = fts.rowid
             WHERE files_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2",
        )?;

        let rows = stmt.query_map(params![query, limit as i64], |row| {
            Ok(Self::row_to_meta(row))
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row??);
        }
        Ok(results)
    }

    /// Get all content hashes referenced by files (for CAS GC).
    pub fn all_content_hashes(&self) -> Result<std::collections::HashSet<ContentHash>> {
        let conn = self.conn.lock();
        let mut stmt =
            conn.prepare("SELECT DISTINCT blob_hash FROM files WHERE blob_hash IS NOT NULL")?;
        let rows = stmt.query_map([], |row| {
            let hash: String = row.get(0)?;
            Ok(ContentHash(hash))
        })?;

        let mut hashes = std::collections::HashSet::new();
        for row in rows {
            hashes.insert(row?);
        }
        Ok(hashes)
    }

    /// Count total files tracked.
    pub fn file_count(&self) -> Result<u64> {
        let conn = self.conn.lock();
        let count: u64 = conn.query_row(
            "SELECT COUNT(*) FROM files WHERE kind = 'file'",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Count files with embeddings.
    pub fn indexed_file_count(&self) -> Result<u64> {
        let conn = self.conn.lock();
        let count: u64 = conn.query_row(
            "SELECT COUNT(*) FROM files WHERE embedding_id IS NOT NULL AND kind = 'file'",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Get all files that need indexing (no embedding_id, indexable type).
    pub fn files_needing_indexing(&self, limit: usize) -> Result<Vec<FileMeta>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT inode, parent_inode, name, path, kind, size, mime_type,
                    blob_hash, embedding_id, tags, created_at, modified_at
             FROM files
             WHERE embedding_id IS NULL AND kind = 'file' AND size > 0
             ORDER BY modified_at DESC
             LIMIT ?1",
        )?;

        let rows = stmt.query_map(params![limit as i64], |row| Ok(Self::row_to_meta(row)))?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row??);
        }
        Ok(results)
    }

    // ── Model cache ─────────────────────────────────────────────────────

    /// Record a model cache access.
    pub fn record_model_access(&self, entry: &ModelCacheEntry) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO model_cache (blob_hash, model_name, access_count, last_accessed, size)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(blob_hash) DO UPDATE SET
                access_count = access_count + 1,
                last_accessed = ?4",
            params![
                entry.blob_hash.0,
                entry.model_name,
                entry.access_count,
                entry.last_accessed.to_rfc3339(),
                entry.size,
            ],
        )?;
        Ok(())
    }

    /// Get model cache entries sorted by frequency-weighted recency.
    pub fn model_cache_entries(&self) -> Result<Vec<ModelCacheEntry>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT blob_hash, model_name, access_count, last_accessed, size
             FROM model_cache
             ORDER BY access_count DESC, last_accessed DESC",
        )?;

        let rows = stmt.query_map([], |row| {
            let hash: String = row.get(0)?;
            let name: String = row.get(1)?;
            let count: u64 = row.get(2)?;
            let accessed: String = row.get(3)?;
            let size: u64 = row.get(4)?;
            Ok(ModelCacheEntry {
                blob_hash: ContentHash(hash),
                model_name: name,
                access_count: count,
                last_accessed: DateTime::parse_from_rfc3339(&accessed)
                    .unwrap_or_else(|_| {
                        DateTime::parse_from_rfc3339("1970-01-01T00:00:00Z").unwrap()
                    })
                    .with_timezone(&Utc),
                size,
            })
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // ── Internal helpers ────────────────────────────────────────────────

    fn row_to_meta(row: &rusqlite::Row<'_>) -> Result<FileMeta> {
        let inode: u64 = row
            .get(0)
            .map_err(|e| SpiritFsError::Metadata(e.to_string()))?;
        let parent_inode: u64 = row
            .get(1)
            .map_err(|e| SpiritFsError::Metadata(e.to_string()))?;
        let name: String = row
            .get(2)
            .map_err(|e| SpiritFsError::Metadata(e.to_string()))?;
        let path: String = row
            .get(3)
            .map_err(|e| SpiritFsError::Metadata(e.to_string()))?;
        let kind_str: String = row
            .get(4)
            .map_err(|e| SpiritFsError::Metadata(e.to_string()))?;
        let size: u64 = row
            .get(5)
            .map_err(|e| SpiritFsError::Metadata(e.to_string()))?;
        let mime_type: Option<String> = row
            .get(6)
            .map_err(|e| SpiritFsError::Metadata(e.to_string()))?;
        let blob_hash: Option<String> = row
            .get(7)
            .map_err(|e| SpiritFsError::Metadata(e.to_string()))?;
        let embedding_id: Option<String> = row
            .get(8)
            .map_err(|e| SpiritFsError::Metadata(e.to_string()))?;
        let tags_json: String = row
            .get(9)
            .map_err(|e| SpiritFsError::Metadata(e.to_string()))?;
        let created_str: String = row
            .get(10)
            .map_err(|e| SpiritFsError::Metadata(e.to_string()))?;
        let modified_str: String = row
            .get(11)
            .map_err(|e| SpiritFsError::Metadata(e.to_string()))?;

        let kind = match kind_str.as_str() {
            "directory" => FileKind::Directory,
            "symlink" => FileKind::Symlink,
            _ => FileKind::RegularFile,
        };

        let tags: Vec<SemanticTag> = serde_json::from_str(&tags_json).unwrap_or_default();

        let default_time = Utc::now();
        let created_at = DateTime::parse_from_rfc3339(&created_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or(default_time);
        let modified_at = DateTime::parse_from_rfc3339(&modified_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or(default_time);

        Ok(FileMeta {
            inode: InodeId(inode),
            parent_inode: InodeId(parent_inode),
            name,
            path: PathBuf::from(path),
            kind,
            size,
            mime_type,
            content_hash: blob_hash.map(ContentHash),
            embedding_id: embedding_id.and_then(|s| Uuid::parse_str(&s).ok()),
            tags,
            created_at,
            modified_at,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_meta(inode: u64, name: &str, path: &str) -> FileMeta {
        let now = Utc::now();
        FileMeta {
            inode: InodeId(inode),
            parent_inode: InodeId(1),
            name: name.to_string(),
            path: PathBuf::from(path),
            kind: FileKind::RegularFile,
            size: 100,
            mime_type: Some("text/plain".to_string()),
            content_hash: Some(ContentHash("abc123".to_string())),
            embedding_id: None,
            tags: vec![SemanticTag {
                label: "test".to_string(),
                confidence: 0.9,
            }],
            created_at: now,
            modified_at: now,
        }
    }

    #[test]
    fn test_open_in_memory() {
        let store = MetadataStore::open_in_memory().unwrap();
        assert_eq!(store.file_count().unwrap(), 0);
    }

    #[test]
    fn test_insert_and_get_by_inode() {
        let store = MetadataStore::open_in_memory().unwrap();
        let meta = make_test_meta(10, "test.rs", "/src/test.rs");
        store.insert_file(&meta).unwrap();

        let retrieved = store.get_by_inode(InodeId(10)).unwrap().unwrap();
        assert_eq!(retrieved.name, "test.rs");
        assert_eq!(retrieved.size, 100);
    }

    #[test]
    fn test_get_by_path() {
        let store = MetadataStore::open_in_memory().unwrap();
        let meta = make_test_meta(10, "test.rs", "/src/test.rs");
        store.insert_file(&meta).unwrap();

        let retrieved = store
            .get_by_path(Path::new("/src/test.rs"))
            .unwrap()
            .unwrap();
        assert_eq!(retrieved.inode, InodeId(10));
    }

    #[test]
    fn test_list_children() {
        let store = MetadataStore::open_in_memory().unwrap();
        let now = Utc::now();

        // Insert parent directory
        store
            .insert_file(&FileMeta {
                inode: InodeId(1),
                parent_inode: InodeId(0),
                name: "src".to_string(),
                path: PathBuf::from("/src"),
                kind: FileKind::Directory,
                size: 0,
                mime_type: None,
                content_hash: None,
                embedding_id: None,
                tags: vec![],
                created_at: now,
                modified_at: now,
            })
            .unwrap();

        store
            .insert_file(&make_test_meta(10, "a.rs", "/src/a.rs"))
            .unwrap();
        store
            .insert_file(&make_test_meta(11, "b.rs", "/src/b.rs"))
            .unwrap();

        let children = store.list_children(InodeId(1)).unwrap();
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].name, "a.rs");
        assert_eq!(children[1].name, "b.rs");
    }

    #[test]
    fn test_set_embedding_id() {
        let store = MetadataStore::open_in_memory().unwrap();
        let meta = make_test_meta(10, "test.rs", "/src/test.rs");
        store.insert_file(&meta).unwrap();

        let id = Uuid::new_v4();
        store.set_embedding_id(InodeId(10), id).unwrap();

        let retrieved = store.get_by_inode(InodeId(10)).unwrap().unwrap();
        assert_eq!(retrieved.embedding_id, Some(id));
    }

    #[test]
    fn test_delete_by_path() {
        let store = MetadataStore::open_in_memory().unwrap();
        store
            .insert_file(&make_test_meta(10, "test.rs", "/src/test.rs"))
            .unwrap();
        store.delete_by_path(Path::new("/src/test.rs")).unwrap();

        assert!(store
            .get_by_path(Path::new("/src/test.rs"))
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_fts_search() {
        let store = MetadataStore::open_in_memory().unwrap();
        store
            .insert_file(&make_test_meta(10, "config.yaml", "/config.yaml"))
            .unwrap();
        store
            .insert_file(&make_test_meta(11, "README.md", "/README.md"))
            .unwrap();

        let results = store.fts_search("config", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "config.yaml");
    }

    #[test]
    fn test_file_count() {
        let store = MetadataStore::open_in_memory().unwrap();
        store
            .insert_file(&make_test_meta(10, "a.rs", "/a.rs"))
            .unwrap();
        store
            .insert_file(&make_test_meta(11, "b.rs", "/b.rs"))
            .unwrap();
        assert_eq!(store.file_count().unwrap(), 2);
    }

    #[test]
    fn test_indexed_file_count() {
        let store = MetadataStore::open_in_memory().unwrap();
        let mut meta = make_test_meta(10, "a.rs", "/a.rs");
        meta.embedding_id = Some(Uuid::new_v4());
        store.insert_file(&meta).unwrap();
        store
            .insert_file(&make_test_meta(11, "b.rs", "/b.rs"))
            .unwrap();

        assert_eq!(store.indexed_file_count().unwrap(), 1);
    }

    #[test]
    fn test_next_inode() {
        let store = MetadataStore::open_in_memory().unwrap();
        let inode1 = store.next_inode().unwrap();
        assert_eq!(inode1, InodeId(1));

        store
            .insert_file(&make_test_meta(5, "test.rs", "/test.rs"))
            .unwrap();
        let inode2 = store.next_inode().unwrap();
        assert_eq!(inode2, InodeId(6));
    }
}
