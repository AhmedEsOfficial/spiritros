# Semantic Filesystem (SpiritFS)

SpiritFS is a dual-layer filesystem that provides a standard POSIX interface (open, read, write, stat) while maintaining a parallel semantic index for LLM-native file operations: search by meaning, auto-tagging, content summarization, and similarity-based retrieval.

## Architecture

```text
User / Application
        |
        +-- POSIX ops (open, read, write, stat) --> Passthrough to backing FS
        |
        +-- Semantic ops (search, similar, summarize) --> Vector index
                |
                v
        +-------------------------+
        |    SpiritFS Daemon       |  <-- FUSE userspace filesystem
        |    (runtime/spiritfs/)   |
        |                         |
        |  +-------------------+  |
        |  | POSIX Layer       |  |  Standard file ops, passthrough to ext4/btrfs
        |  | (fuse.rs)         |  |  via fuser crate, feature-gated
        |  +-------------------+  |
        |                         |
        |  +-------------------+  |
        |  | Index Pipeline    |  |  Watches for file changes, generates embeddings
        |  | (index.rs)        |  |  Async, never blocks POSIX operations
        |  +-------------------+  |
        |                         |
        |  +-------------------+  |
        |  | Semantic Engine   |  |  search, similar, summarize, auto_tag
        |  | (semantic.rs)     |  |  Falls back to SQLite FTS5 when Qdrant is down
        |  +-------------------+  |
        +-------------------------+
                |           |
                v           v
        +----------+ +----------+
        | Qdrant   | | SQLite   |
        | (vectors)| | (meta +  |
        |          | |  FTS5)   |
        +----------+ +----------+
```

## Components

### FUSE Layer (`fuse.rs`)

The FUSE filesystem uses the `fuser` crate and requires `libfuse3-dev` at build time. It is feature-gated behind the `fuse` Cargo feature.

All POSIX operations (lookup, getattr, readdir, open, read, write, create, mkdir, unlink, rmdir, rename, setattr) passthrough to the backing directory. Write operations also queue async index jobs via a crossbeam channel.

### Content-Addressable Store (`cas.rs`)

File contents are stored in a SHA-256 keyed blob store with sharded directories (`blobs/ab/cdef...`). Identical file contents share the same blob (deduplication). Atomic writes via temp file + rename ensure crash consistency.

### Metadata Store (`metadata.rs`)

SQLite database (WAL mode, bundled build) stores file metadata: inodes, paths, content hashes, MIME types, tags, and embedding IDs. An FTS5 virtual table provides full-text search as a fallback when Qdrant is unavailable. Triggers keep FTS5 automatically in sync.

### Embedding Pipeline (`embedding.rs`)

Text files are split into overlapping chunks (512 tokens, 64 overlap) and embedded via Ollama's `/api/embeddings` endpoint using `nomic-embed-text` (768 dimensions). Runs on CPU at idle priority to avoid competing with the primary LLM.

### Vector Store (`qdrant.rs`)

Qdrant is the primary vector database for semantic search. The client uses gRPC via `qdrant-client`. Each chunk gets its own vector point with metadata payload (path, name, chunk_text, chunk_index, mime_type, tags).

### Index Pipeline (`index.rs`)

A bounded crossbeam channel (default 10K capacity) receives index jobs. The pipeline processes them asynchronously with a configurable concurrency limit (default 4 concurrent embeddings). A file watcher based on the `notify` crate (inotify on Linux) detects file changes.

### Semantic Engine (`semantic.rs`)

High-level API for:
- **Search**: Embed query text, search Qdrant, fallback to FTS5
- **Similar**: Embed file content, find similar vectors
- **Summarize**: Collect directory listing, generate summary via Ollama
- **Auto-tag**: Extension-based and content heuristic tagging

### Model Cache (`cache.rs`)

Tracks model weight files with frequency-weighted LRU eviction. Score = `access_count * (1 / (1 + hours_since_last_access))`. Supports eviction to a configurable max size and prefetch prediction.

## CLI Commands

```bash
spiritfs mount [--config <path>] [--backing-dir <path>] [--mount-point <path>]
spiritfs search <query> [--limit <n>]
spiritfs similar <file> [--limit <n>]
spiritfs summarize <directory>
spiritfs index status
spiritfs index rebuild
```

## Configuration

Configuration loads from YAML file, environment variables (`SPIRITFS_*`), and CLI arguments via `figment`.

| Key | Default | Description |
|---|---|---|
| `mount_point` | `/mnt/spiritfs` | FUSE mount point |
| `backing_dir` | `/var/lib/spiritfs/data` | Actual file storage |
| `qdrant_url` | `http://127.0.0.1:6334` | Qdrant gRPC endpoint |
| `ollama_url` | `http://127.0.0.1:11434` | Ollama API endpoint |
| `embedding_model` | `nomic-embed-text` | Embedding model |
| `embedding_dimensions` | `768` | Vector dimensions |
| `max_embedding_file_size` | `10485760` | Max file size for indexing (10 MiB) |

## Metrics

19 Prometheus metrics under `spirit_fs_` prefix covering FUSE operations, indexing, search, CAS, and model cache.

## Phase 1 Limitations

- Runs as a FUSE filesystem (userspace, ~5-15% metadata overhead)
- Phase 2+ will port to a native SpiritKernel VFS driver
- Embedding model runs on CPU only (to preserve GPU for primary LLM)
