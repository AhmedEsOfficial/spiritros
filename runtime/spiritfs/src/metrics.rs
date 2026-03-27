//! Prometheus metrics for SpiritFS.
//!
//! Exposes filesystem, indexing, and semantic search metrics for monitoring.

use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{exponential_buckets, Histogram};
use prometheus_client::registry::Registry;

/// All SpiritFS metrics, collected for Prometheus export.
pub struct SpiritFsMetrics {
    // ── Filesystem metrics ──────────────────────────────────────────────
    /// Total FUSE read operations.
    pub fuse_reads_total: Counter,

    /// Total FUSE write operations.
    pub fuse_writes_total: Counter,

    /// Total FUSE lookup operations.
    pub fuse_lookups_total: Counter,

    /// FUSE operation latency in microseconds.
    pub fuse_op_latency_us: Histogram,

    // ── Index metrics ───────────────────────────────────────────────────
    /// Total files indexed (embedding generated).
    pub files_indexed: Counter,

    /// Total embeddings generated.
    pub embeddings_generated: Counter,

    /// Current index queue depth.
    pub queue_depth: Gauge,

    /// Total index jobs dropped (queue full).
    pub jobs_dropped: Counter,

    /// Total indexing errors.
    pub errors_total: Counter,

    // ── Semantic search metrics ─────────────────────────────────────────
    /// Total semantic search queries.
    pub search_queries_total: Counter,

    /// Total similarity queries.
    pub similar_queries_total: Counter,

    /// Total summarization requests.
    pub summarize_requests_total: Counter,

    /// Search query latency in milliseconds.
    pub search_latency_ms: Histogram,

    // ── CAS metrics ─────────────────────────────────────────────────────
    /// Total blobs stored in CAS.
    pub cas_blobs_total: Gauge,

    /// Total CAS deduplication hits.
    pub cas_dedup_hits: Counter,

    /// Total bytes stored in CAS.
    pub cas_bytes_total: Gauge,

    // ── Model cache metrics ─────────────────────────────────────────────
    /// Number of cached models.
    pub model_cache_entries: Gauge,

    /// Model cache hits.
    pub model_cache_hits: Counter,

    /// Model cache evictions.
    pub model_cache_evictions: Counter,
}

impl SpiritFsMetrics {
    /// Create and register all metrics.
    #[allow(clippy::too_many_lines)]
    pub fn new(registry: &mut Registry) -> Self {
        let sub = registry.sub_registry_with_prefix("spirit_fs");

        // FUSE metrics
        let fuse_reads_total = Counter::default();
        sub.register(
            "fuse_reads_total",
            "Total FUSE read operations",
            fuse_reads_total.clone(),
        );

        let fuse_writes_total = Counter::default();
        sub.register(
            "fuse_writes_total",
            "Total FUSE write operations",
            fuse_writes_total.clone(),
        );

        let fuse_lookups_total = Counter::default();
        sub.register(
            "fuse_lookups_total",
            "Total FUSE lookup operations",
            fuse_lookups_total.clone(),
        );

        let fuse_op_latency_us = Histogram::new(exponential_buckets(1.0, 2.0, 16));
        sub.register(
            "fuse_op_latency_us",
            "FUSE op latency in microseconds",
            fuse_op_latency_us.clone(),
        );

        // Index metrics
        let files_indexed = Counter::default();
        sub.register(
            "files_indexed",
            "Total files indexed",
            files_indexed.clone(),
        );

        let embeddings_generated = Counter::default();
        sub.register(
            "embeddings_generated",
            "Total embeddings generated",
            embeddings_generated.clone(),
        );

        let queue_depth = Gauge::default();
        sub.register(
            "queue_depth",
            "Current index queue depth",
            queue_depth.clone(),
        );

        let jobs_dropped = Counter::default();
        sub.register(
            "jobs_dropped",
            "Index jobs dropped (queue full)",
            jobs_dropped.clone(),
        );

        let errors_total = Counter::default();
        sub.register(
            "errors_total",
            "Total indexing errors",
            errors_total.clone(),
        );

        // Search metrics
        let search_queries_total = Counter::default();
        sub.register(
            "search_queries_total",
            "Total semantic search queries",
            search_queries_total.clone(),
        );

        let similar_queries_total = Counter::default();
        sub.register(
            "similar_queries_total",
            "Total similarity queries",
            similar_queries_total.clone(),
        );

        let summarize_requests_total = Counter::default();
        sub.register(
            "summarize_requests_total",
            "Total summarize requests",
            summarize_requests_total.clone(),
        );

        let search_latency_ms = Histogram::new(exponential_buckets(1.0, 2.0, 14));
        sub.register(
            "search_latency_ms",
            "Search query latency in ms",
            search_latency_ms.clone(),
        );

        // CAS metrics
        let cas_blobs_total = Gauge::default();
        sub.register(
            "cas_blobs_total",
            "Total blobs in CAS",
            cas_blobs_total.clone(),
        );

        let cas_dedup_hits = Counter::default();
        sub.register(
            "cas_dedup_hits",
            "CAS deduplication hits",
            cas_dedup_hits.clone(),
        );

        let cas_bytes_total = Gauge::default();
        sub.register(
            "cas_bytes_total",
            "Total bytes in CAS",
            cas_bytes_total.clone(),
        );

        // Model cache metrics
        let model_cache_entries = Gauge::default();
        sub.register(
            "model_cache_entries",
            "Cached models",
            model_cache_entries.clone(),
        );

        let model_cache_hits = Counter::default();
        sub.register(
            "model_cache_hits",
            "Model cache hits",
            model_cache_hits.clone(),
        );

        let model_cache_evictions = Counter::default();
        sub.register(
            "model_cache_evictions",
            "Model cache evictions",
            model_cache_evictions.clone(),
        );

        Self {
            fuse_reads_total,
            fuse_writes_total,
            fuse_lookups_total,
            fuse_op_latency_us,
            files_indexed,
            embeddings_generated,
            queue_depth,
            jobs_dropped,
            errors_total,
            search_queries_total,
            similar_queries_total,
            summarize_requests_total,
            search_latency_ms,
            cas_blobs_total,
            cas_dedup_hits,
            cas_bytes_total,
            model_cache_entries,
            model_cache_hits,
            model_cache_evictions,
        }
    }

    /// Encode all metrics in Prometheus text format.
    pub fn encode(&self, registry: &Registry) -> String {
        let mut buf = String::new();
        prometheus_client::encoding::text::encode(&mut buf, registry)
            .expect("prometheus encoding should not fail");
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let mut registry = Registry::default();
        let metrics = SpiritFsMetrics::new(&mut registry);
        assert_eq!(metrics.queue_depth.get(), 0);
    }

    #[test]
    fn test_metrics_increment() {
        let mut registry = Registry::default();
        let metrics = SpiritFsMetrics::new(&mut registry);

        metrics.files_indexed.inc();
        metrics.embeddings_generated.inc();
        metrics.queue_depth.set(5);
        metrics.fuse_reads_total.inc();

        assert_eq!(metrics.queue_depth.get(), 5);
    }

    #[test]
    fn test_metrics_encode() {
        let mut registry = Registry::default();
        let metrics = SpiritFsMetrics::new(&mut registry);

        metrics.files_indexed.inc();
        metrics.queue_depth.set(3);

        let output = metrics.encode(&registry);
        assert!(output.contains("spirit_fs"));
        assert!(output.contains("files_indexed"));
    }

    #[test]
    fn test_histogram_observation() {
        let mut registry = Registry::default();
        let metrics = SpiritFsMetrics::new(&mut registry);

        metrics.fuse_op_latency_us.observe(50.0);
        metrics.search_latency_ms.observe(15.0);

        let output = metrics.encode(&registry);
        assert!(output.contains("fuse_op_latency_us"));
        assert!(output.contains("search_latency_ms"));
    }
}
