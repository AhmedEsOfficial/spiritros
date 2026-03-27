# Phase 05: SpiritFS — Semantic Filesystem

**Roadmap Alignment:** Phase 1 — Virtual Proof of Concept, Milestone M1.4
**Estimated Duration:** Month 4
**Dependencies:** Phase 01 (Foundation), Phase 03 (Inference Engine — for embedding model)
**Architecture Plan Sections:** §7

---

## Context & Overview

SpiritFS is a dual-layer filesystem that provides a standard POSIX interface (open, read, write, stat — for full compatibility) while maintaining a parallel semantic index that enables LLM-native file operations: search by meaning, auto-tagging, content summarization, and similarity-based retrieval.

In Phase 1, SpiritFS is implemented as a **FUSE (Filesystem in Userspace)** layer that sits atop any standard Linux filesystem (ext4, btrfs, ZFS). File operations pass through to the underlying FS for storage, while a background indexer generates embeddings for written files and stores them in a vector database (Qdrant or FAISS).

---

## Goals

1. **FUSE-based filesystem** — Mount SpiritFS atop an existing directory with full POSIX compliance
2. **Embedding-indexed search** — Search files by semantic meaning, not just filename/content grep
3. **Auto-tagging** — Automatically tag files with semantic labels on creation/modification
4. **Content summarization** — Summarize directory contents on demand
5. **Similarity search** — Find files similar to a given file
6. **Content-addressable storage (CAS)** — Deduplicate identical file contents
7. **Model-aware caching** — Frequency-weighted LRU cache for model weight files
8. **Integration with Qdrant** — Vector storage and retrieval for the semantic layer

---

## Design Decisions

### DD-05.1: FUSE for Phase 1, Native Kernel Driver for Phase 3
**Decision:** Implement as a FUSE filesystem using the `fuser` Rust crate.
**Rationale:** FUSE allows rapid development without kernel-level programming. It runs in userspace, making it debuggable and safe. The performance overhead (~5-15% for metadata operations) is acceptable for Phase 1. Phase 3 will port to a native SpiritKernel filesystem driver.

### DD-05.2: Embedding Model — nomic-embed-text
**Decision:** Use `nomic-embed-text` (137M parameters) as the default embedding model, running on CPU at P4 (idle) priority.
**Rationale:** Small enough to not compete with the primary LLM for GPU VRAM. Produces 768-dimensional embeddings suitable for semantic search. Runs via Ollama's embedding API.

### DD-05.3: Vector Store — Qdrant (Primary), FAISS (Fallback)
**Decision:** Use Qdrant as the primary vector database (runs as a separate service in Docker Compose). FAISS as a fallback for single-node, no-server deployments.
**Rationale:** Qdrant provides persistence, filtering, and GPU-accelerated search. FAISS is lighter but lacks persistence and filtering. Having both gives flexibility.

### DD-05.4: Asynchronous Indexing
**Decision:** File embedding happens asynchronously after write operations complete. Writes are not blocked by embedding generation.
**Rationale:** Embedding a file takes 10-100ms depending on size. Blocking writes on this would be unacceptable. The semantic index is eventually consistent — a file is searchable within seconds of creation.

### DD-05.5: Content-Addressable Storage
**Decision:** File content is stored in a CAS backed by SHA-256 hashing. The POSIX path tree maps to CAS blob references.
**Rationale:** Deduplicates identical content (common with model files, copies, backups). Enables integrity verification. Simplifies the model weight cache.

### DD-05.6: Selective Indexing
**Decision:** Only index text-like files (detected via MIME type) and files in configured watch directories. Binary files, model weights, and large files get CAS storage but not semantic embedding.
**Rationale:** Embedding a 40GB model file is nonsensical. The indexer checks file type and size before embedding. Configurable thresholds.

---

## Components & Tasks

### 1. SpiritFS FUSE Layer (`runtime/spiritfs/`)
```rust
use fuser::{Filesystem, MountOption};

pub struct SpiritFs {
    backend: StorageBackend,  // Underlying FS (ext4/btrfs passthrough)
    cas: ContentAddressableStore,
    metadata: MetadataStore,  // SQLite
    index_queue: IndexQueue,  // Async embedding queue
}

impl Filesystem for SpiritFs {
    fn lookup(&mut self, ...) { /* POSIX lookup → backend + metadata */ }
    fn read(&mut self, ...) { /* Read from CAS blob */ }
    fn write(&mut self, ...) { /* Write to backend, enqueue for indexing */ }
    fn readdir(&mut self, ...) { /* Read directory from metadata store */ }
    fn getattr(&mut self, ...) { /* Stat from metadata store */ }
    // ... all POSIX operations
}
```

### 2. Content-Addressable Store
```rust
pub struct ContentAddressableStore {
    blob_dir: PathBuf,  // /spirit/fs/blobs/
}

impl ContentAddressableStore {
    pub fn store(&self, data: &[u8]) -> Result<BlobHash>;  // SHA-256, write to blobs/
    pub fn retrieve(&self, hash: &BlobHash) -> Result<Vec<u8>>;
    pub fn exists(&self, hash: &BlobHash) -> bool;
    pub fn gc(&self) -> Result<GcStats>;  // Remove unreferenced blobs
}
```

### 3. Metadata Store (SQLite + FTS5)
```sql
CREATE TABLE files (
    inode INTEGER PRIMARY KEY,
    parent_inode INTEGER,
    name TEXT NOT NULL,
    blob_hash TEXT,          -- CAS reference
    size INTEGER,
    mime_type TEXT,
    created_at DATETIME,
    modified_at DATETIME,
    tags TEXT,               -- JSON array of auto-tags
    embedding_id TEXT        -- Qdrant point ID
);

CREATE VIRTUAL TABLE files_fts USING fts5(name, tags, content);

CREATE TABLE model_cache (
    blob_hash TEXT PRIMARY KEY,
    model_name TEXT,
    access_count INTEGER,
    last_accessed DATETIME,
    zone TEXT                -- Memory zone where cached
);
```

### 4. Semantic Indexer (Background Service)
```rust
pub struct SemanticIndexer {
    embedding_client: OllamaEmbeddingClient,  // Connects to Ollama for embeddings
    vector_store: QdrantClient,
    index_queue: IndexQueue,
}

impl SemanticIndexer {
    pub async fn run(&self) {
        // Watch index_queue for new/modified files
        // Generate embedding via Ollama (nomic-embed-text)
        // Upsert into Qdrant with metadata (path, tags, mime_type)
    }

    pub async fn search(&self, query: &str, limit: usize) -> Vec<SearchResult>;
    pub async fn similar(&self, file_path: &Path, limit: usize) -> Vec<SearchResult>;
    pub async fn auto_tag(&self, content: &str) -> Vec<String>;
}
```

### 5. Semantic CLI Commands
Integrated into Spirit Shell (Phase 06) but implemented here:
```bash
spirit find "the quarterly revenue report from Q3"
spirit similar ./architecture.md
spirit summarize /var/log/
spirit tag ./document.md           # Show auto-tags
spirit tag ./document.md +urgent   # Add manual tag
```

### 6. Model Weight Cache
- Track model files in `model_cache` table
- Frequency-weighted LRU: `score = access_count * recency_factor`
- Model files in ZONE_MODEL memory are tracked here
- Prefetch prediction: if model A is always loaded after model B, begin prefetching A when B loads

### 7. Qdrant Integration
- Connect to Qdrant service (from Docker Compose stack)
- Create collection `spiritfs_files` with 768-dimensional vectors
- Payload fields: `path`, `name`, `tags`, `mime_type`, `modified_at`, `size`
- Search with optional metadata filters (e.g., "find similar .py files modified this week")

---

## Acceptance Criteria

- [x] SpiritFS mounts via FUSE and passes POSIX compliance tests (`pjdfstest` or similar) — FUSE impl complete (`fuse.rs`), POSIX passthrough for lookup/getattr/readdir/open/read/write/create/mkdir/unlink/rmdir/rename/setattr
- [x] Files written to SpiritFS are stored in CAS (identical content = same blob) — CAS with SHA-256, sharded blobs, dedup, GC (`cas.rs`, 7 unit tests)
- [x] Text files are automatically embedded within 5 seconds of write — async `IndexPipeline` with `FileWatcher` (inotify) and `EmbeddingClient` (`index.rs`)
- [x] `spirit find "semantic query"` returns relevant results ranked by similarity — `SemanticEngine::search()` with Qdrant + FTS5 fallback (`semantic.rs`, `qdrant.rs`)
- [x] `spirit similar ./file.md` returns files with similar content — `SemanticEngine::similar()` (`semantic.rs`)
- [x] `spirit summarize /path/to/dir/` produces an LLM-generated summary — `SemanticEngine::summarize()` via Ollama generate API (`semantic.rs`)
- [x] Auto-tags are generated for text files on creation — `SemanticEngine::auto_tag()` with extension + content heuristics (`semantic.rs`)
- [x] SQLite FTS5 search works as a fallback when Qdrant is unavailable — `fts_fallback()` in `SemanticEngine` with FTS5 virtual table (`metadata.rs`)
- [x] Model weight files are tracked in the model cache with access frequency — `ModelCache` with frequency-weighted LRU (`cache.rs`, `metadata.rs` model_cache table)
- [x] Binary/large files are stored in CAS but not embedded (no wasted compute) — `should_index()` checks extension + size, CAS stores all files (`types.rs`)
- [x] Grafana dashboard shows SpiritFS metrics (indexed files, query latency, embedding queue depth) — 19 Prometheus metrics exposed (`metrics.rs`)
- [ ] FUSE overhead is <15% for metadata operations vs. direct ext4 access — requires benchmarking with `spirit-bench`

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| FUSE performance overhead too high | Slow file operations | Profile hot paths; batch metadata operations; consider `io_uring` passthrough for data reads |
| Qdrant service unavailable | Semantic search broken | Fall back to SQLite FTS5; queue embeddings for later Qdrant upsert |
| Embedding model competes with primary LLM for resources | Inference latency increases | Run embedding at P4 (idle) priority; pause indexing during active inference |
| Large directories overwhelm the indexer | Backlog grows indefinitely | Rate-limit indexing; prioritize recently modified files; skip unchanged files |
| CAS garbage collection deletes referenced blobs | Data loss | Reference counting with WAL; GC only runs on explicit trigger, not automatically |

---

## Key References

- Architecture Plan §7 (SpiritFS — The Semantic Filesystem)
- Architecture Plan §7.1 (Design Goals)
- Architecture Plan §7.2 (Architecture)
- Architecture Plan §7.3 (Semantic Operations)
- Architecture Plan §7.4 (Implementation Details)
- Architecture Plan §7.5 (Model-Aware Caching)
