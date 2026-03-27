//! Content-Addressable Storage (CAS).
//!
//! SHA-256 keyed blob store for file contents. Provides deduplication:
//! identical files share the same blob. Blobs are immutable once written,
//! enabling safe concurrent reads.
//!
//! ```text
//! /var/lib/spiritfs/cas/
//! ├── ab/
//! │   └── cdef1234...  (first 2 hex chars as shard directory)
//! ├── 01/
//! │   └── 23456789...
//! └── ...
//! ```

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use tracing::{debug, instrument, warn};

use crate::error::{Result, SpiritFsError};
use crate::types::ContentHash;

/// Content-addressable blob store backed by the filesystem.
#[derive(Debug, Clone)]
pub struct ContentAddressableStore {
    /// Root directory for blob storage.
    blob_dir: PathBuf,
}

/// Statistics from a garbage collection run.
#[derive(Debug, Clone)]
pub struct GcStats {
    /// Number of blobs scanned.
    pub scanned: u64,
    /// Number of unreferenced blobs removed.
    pub removed: u64,
    /// Bytes reclaimed.
    pub bytes_reclaimed: u64,
}

impl ContentAddressableStore {
    /// Create a new CAS store at the given directory.
    ///
    /// Creates the directory if it doesn't exist.
    pub fn new(blob_dir: &Path) -> Result<Self> {
        fs::create_dir_all(blob_dir)?;
        Ok(Self {
            blob_dir: blob_dir.to_path_buf(),
        })
    }

    /// Compute the SHA-256 hash of data.
    pub fn hash(data: &[u8]) -> ContentHash {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        ContentHash(hex::encode(result))
    }

    /// Compute the SHA-256 hash of a file on disk.
    pub fn hash_file(path: &Path) -> Result<ContentHash> {
        let data = fs::read(path)?;
        Ok(Self::hash(&data))
    }

    /// Store data and return its content hash.
    ///
    /// If the blob already exists, this is a no-op (deduplication).
    #[instrument(skip(self, data), fields(size = data.len()))]
    pub fn store(&self, data: &[u8]) -> Result<ContentHash> {
        let hash = Self::hash(data);
        let blob_path = self.blob_path(&hash);

        if blob_path.exists() {
            debug!(hash = %hash, "blob already exists (dedup)");
            return Ok(hash);
        }

        // Create shard directory
        if let Some(parent) = blob_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Atomic write: write to temp file then rename
        let tmp_path = blob_path.with_extension("tmp");
        let mut file = fs::File::create(&tmp_path)?;
        file.write_all(data)?;
        file.sync_all()?;
        fs::rename(&tmp_path, &blob_path)?;

        debug!(hash = %hash, size = data.len(), "stored blob");
        Ok(hash)
    }

    /// Store a file by path and return its content hash.
    #[instrument(skip(self))]
    pub fn store_file(&self, path: &Path) -> Result<ContentHash> {
        let data = fs::read(path)?;
        self.store(&data)
    }

    /// Retrieve blob data by content hash.
    #[instrument(skip(self))]
    pub fn retrieve(&self, hash: &ContentHash) -> Result<Vec<u8>> {
        let blob_path = self.blob_path(hash);
        if !blob_path.exists() {
            return Err(SpiritFsError::Cas(format!("blob not found: {hash}")));
        }
        Ok(fs::read(&blob_path)?)
    }

    /// Check if a blob exists.
    pub fn exists(&self, hash: &ContentHash) -> bool {
        self.blob_path(hash).exists()
    }

    /// Get the on-disk size of a blob.
    pub fn blob_size(&self, hash: &ContentHash) -> Result<u64> {
        let path = self.blob_path(hash);
        Ok(fs::metadata(&path)?.len())
    }

    /// Run garbage collection, removing blobs not in the referenced set.
    ///
    /// `referenced` is the set of content hashes still in use. Any blob
    /// not in this set is deleted.
    #[instrument(skip(self, referenced), fields(referenced_count = referenced.len()))]
    pub fn gc(&self, referenced: &std::collections::HashSet<ContentHash>) -> Result<GcStats> {
        let mut stats = GcStats {
            scanned: 0,
            removed: 0,
            bytes_reclaimed: 0,
        };

        // Walk all shard directories
        let entries = match fs::read_dir(&self.blob_dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(stats),
            Err(e) => return Err(e.into()),
        };

        for shard_entry in entries {
            let shard_entry = shard_entry?;
            if !shard_entry.file_type()?.is_dir() {
                continue;
            }

            for blob_entry in fs::read_dir(shard_entry.path())? {
                let blob_entry = blob_entry?;
                stats.scanned += 1;

                let hash_str = format!(
                    "{}{}",
                    shard_entry.file_name().to_string_lossy(),
                    blob_entry.file_name().to_string_lossy()
                );
                let hash = ContentHash(hash_str);

                if !referenced.contains(&hash) {
                    let size = blob_entry.metadata()?.len();
                    if let Err(e) = fs::remove_file(blob_entry.path()) {
                        warn!(hash = %hash, error = %e, "failed to remove unreferenced blob");
                    } else {
                        stats.removed += 1;
                        stats.bytes_reclaimed += size;
                    }
                }
            }
        }

        debug!(
            scanned = stats.scanned,
            removed = stats.removed,
            reclaimed = stats.bytes_reclaimed,
            "CAS garbage collection complete"
        );
        Ok(stats)
    }

    /// List all blob hashes in the store.
    pub fn list_blobs(&self) -> Result<Vec<ContentHash>> {
        let mut hashes = Vec::new();
        let entries = match fs::read_dir(&self.blob_dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(hashes),
            Err(e) => return Err(e.into()),
        };

        for shard_entry in entries {
            let shard_entry = shard_entry?;
            if !shard_entry.file_type()?.is_dir() {
                continue;
            }
            let shard_name = shard_entry.file_name().to_string_lossy().to_string();

            for blob_entry in fs::read_dir(shard_entry.path())? {
                let blob_entry = blob_entry?;
                let blob_name = blob_entry.file_name().to_string_lossy().to_string();
                hashes.push(ContentHash(format!("{shard_name}{blob_name}")));
            }
        }
        Ok(hashes)
    }

    /// Build the filesystem path for a blob given its hash.
    ///
    /// Uses the first 2 hex characters as a shard directory to avoid
    /// having millions of files in a single directory.
    fn blob_path(&self, hash: &ContentHash) -> PathBuf {
        let hex = &hash.0;
        if hex.len() < 2 {
            return self.blob_dir.join(hex);
        }
        self.blob_dir.join(&hex[..2]).join(&hex[2..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_deterministic() {
        let data = b"hello world";
        let h1 = ContentAddressableStore::hash(data);
        let h2 = ContentAddressableStore::hash(data);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_different_data() {
        let h1 = ContentAddressableStore::hash(b"hello");
        let h2 = ContentAddressableStore::hash(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_store_and_retrieve() {
        let dir = tempfile::tempdir().unwrap();
        let cas = ContentAddressableStore::new(dir.path()).unwrap();

        let data = b"test content for CAS";
        let hash = cas.store(data).unwrap();

        assert!(cas.exists(&hash));
        let retrieved = cas.retrieve(&hash).unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_deduplication() {
        let dir = tempfile::tempdir().unwrap();
        let cas = ContentAddressableStore::new(dir.path()).unwrap();

        let data = b"duplicate content";
        let h1 = cas.store(data).unwrap();
        let h2 = cas.store(data).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_retrieve_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let cas = ContentAddressableStore::new(dir.path()).unwrap();
        let hash = ContentHash("nonexistent".to_string());
        assert!(cas.retrieve(&hash).is_err());
    }

    #[test]
    fn test_gc_removes_unreferenced() {
        let dir = tempfile::tempdir().unwrap();
        let cas = ContentAddressableStore::new(dir.path()).unwrap();

        let h1 = cas.store(b"keep this").unwrap();
        let h2 = cas.store(b"remove this").unwrap();

        let mut referenced = std::collections::HashSet::new();
        referenced.insert(h1.clone());

        let stats = cas.gc(&referenced).unwrap();
        assert_eq!(stats.removed, 1);
        assert!(cas.exists(&h1));
        assert!(!cas.exists(&h2));
    }

    #[test]
    fn test_list_blobs() {
        let dir = tempfile::tempdir().unwrap();
        let cas = ContentAddressableStore::new(dir.path()).unwrap();

        cas.store(b"blob one").unwrap();
        cas.store(b"blob two").unwrap();

        let blobs = cas.list_blobs().unwrap();
        assert_eq!(blobs.len(), 2);
    }

    #[test]
    fn test_blob_size() {
        let dir = tempfile::tempdir().unwrap();
        let cas = ContentAddressableStore::new(dir.path()).unwrap();

        let data = b"twelve chars";
        let hash = cas.store(data).unwrap();
        assert_eq!(cas.blob_size(&hash).unwrap(), data.len() as u64);
    }
}
