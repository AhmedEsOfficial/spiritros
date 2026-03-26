//! Prometheus metrics for the inference subsystem.
//!
//! Exposes key performance and resource metrics for monitoring:
//! - Token generation throughput
//! - Inference latency (first token, per-token)
//! - Batch utilization
//! - Session and model state
//! - Queue depth
//! - KV-cache operations

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{exponential_buckets, Histogram};
use prometheus_client::registry::Registry;

/// Label set for per-model metrics.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ModelLabel {
    pub model: String,
}

/// Label set for per-priority metrics.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct PriorityLabel {
    pub priority: String,
}

/// All inference metrics, collected in a single struct for easy access.
pub struct InferenceMetrics {
    // ── Token metrics ───────────────────────────────────────────────────
    /// Total tokens generated across all sessions.
    pub tokens_generated_total: Counter,

    /// Total prompt tokens processed (prefill).
    pub prompt_tokens_total: Counter,

    // ── Latency metrics ─────────────────────────────────────────────────
    /// Time to first token in milliseconds (histogram).
    pub first_token_latency_ms: Histogram,

    /// Per-token decode latency in milliseconds (histogram).
    pub decode_latency_ms: Histogram,

    /// Total generation time in milliseconds (histogram).
    pub generation_total_ms: Histogram,

    // ── Throughput metrics ───────────────────────────────────────────────
    /// Current tokens per second (gauge, updated periodically).
    pub tokens_per_second: Gauge,

    // ── Session metrics ─────────────────────────────────────────────────
    /// Number of active sessions.
    pub active_sessions: Gauge,

    /// Total sessions created.
    pub sessions_created_total: Counter,

    /// Total sessions destroyed.
    pub sessions_destroyed_total: Counter,

    // ── Model metrics ───────────────────────────────────────────────────
    /// Number of loaded models.
    pub loaded_models: Gauge,

    /// Total model loads.
    pub model_loads_total: Counter,

    /// Total model unloads.
    pub model_unloads_total: Counter,

    // ── Batch metrics ───────────────────────────────────────────────────
    /// Current batch queue depth.
    pub queue_depth: Gauge,

    /// Total batches dispatched.
    pub batches_total: Counter,

    /// Batch size histogram.
    pub batch_size: Histogram,

    // ── KV-cache metrics ────────────────────────────────────────────────
    /// Total KV-cache snapshots taken.
    pub kv_snapshots_total: Counter,

    /// Total KV-cache restores.
    pub kv_restores_total: Counter,

    // ── Error metrics ───────────────────────────────────────────────────
    /// Total inference errors.
    pub errors_total: Counter,

    /// Total requests rejected (queue full, session limit, etc.).
    pub rejected_total: Counter,
}

impl InferenceMetrics {
    /// Create and register all metrics.
    pub fn new(registry: &mut Registry) -> Self {
        let sub = registry.sub_registry_with_prefix("spirit_inference");

        // Token metrics
        let tokens_generated_total = Counter::default();
        sub.register(
            "tokens_generated_total",
            "Total tokens generated across all sessions",
            tokens_generated_total.clone(),
        );

        let prompt_tokens_total = Counter::default();
        sub.register(
            "prompt_tokens_total",
            "Total prompt tokens processed (prefill)",
            prompt_tokens_total.clone(),
        );

        // Latency metrics (buckets: 5ms to 5000ms, exponential)
        let first_token_latency_ms = Histogram::new(exponential_buckets(5.0, 2.0, 12));
        sub.register(
            "first_token_latency_ms",
            "Time to first token in milliseconds",
            first_token_latency_ms.clone(),
        );

        let decode_latency_ms = Histogram::new(exponential_buckets(0.5, 2.0, 10));
        sub.register(
            "decode_latency_ms",
            "Per-token decode latency in milliseconds",
            decode_latency_ms.clone(),
        );

        let generation_total_ms = Histogram::new(exponential_buckets(5.0, 2.0, 12));
        sub.register(
            "generation_total_ms",
            "Total generation time in milliseconds",
            generation_total_ms.clone(),
        );

        // Throughput
        let tokens_per_second = Gauge::default();
        sub.register(
            "tokens_per_second",
            "Current tokens per second",
            tokens_per_second.clone(),
        );

        // Session metrics
        let active_sessions = Gauge::default();
        sub.register(
            "active_sessions",
            "Number of active inference sessions",
            active_sessions.clone(),
        );

        let sessions_created_total = Counter::default();
        sub.register(
            "sessions_created_total",
            "Total sessions created",
            sessions_created_total.clone(),
        );

        let sessions_destroyed_total = Counter::default();
        sub.register(
            "sessions_destroyed_total",
            "Total sessions destroyed",
            sessions_destroyed_total.clone(),
        );

        // Model metrics
        let loaded_models = Gauge::default();
        sub.register(
            "loaded_models",
            "Number of loaded models",
            loaded_models.clone(),
        );

        let model_loads_total = Counter::default();
        sub.register(
            "model_loads_total",
            "Total model loads",
            model_loads_total.clone(),
        );

        let model_unloads_total = Counter::default();
        sub.register(
            "model_unloads_total",
            "Total model unloads",
            model_unloads_total.clone(),
        );

        // Batch metrics
        let queue_depth = Gauge::default();
        sub.register(
            "queue_depth",
            "Current batch queue depth",
            queue_depth.clone(),
        );

        let batches_total = Counter::default();
        sub.register(
            "batches_total",
            "Total batches dispatched",
            batches_total.clone(),
        );

        let batch_size = Histogram::new(exponential_buckets(1.0, 2.0, 6));
        sub.register(
            "batch_size",
            "Number of requests per batch",
            batch_size.clone(),
        );

        // KV-cache metrics
        let kv_snapshots_total = Counter::default();
        sub.register(
            "kv_snapshots_total",
            "Total KV-cache snapshots taken",
            kv_snapshots_total.clone(),
        );

        let kv_restores_total = Counter::default();
        sub.register(
            "kv_restores_total",
            "Total KV-cache restores",
            kv_restores_total.clone(),
        );

        // Error metrics
        let errors_total = Counter::default();
        sub.register(
            "errors_total",
            "Total inference errors",
            errors_total.clone(),
        );

        let rejected_total = Counter::default();
        sub.register(
            "rejected_total",
            "Total requests rejected",
            rejected_total.clone(),
        );

        Self {
            tokens_generated_total,
            prompt_tokens_total,
            first_token_latency_ms,
            decode_latency_ms,
            generation_total_ms,
            tokens_per_second,
            active_sessions,
            sessions_created_total,
            sessions_destroyed_total,
            loaded_models,
            model_loads_total,
            model_unloads_total,
            queue_depth,
            batches_total,
            batch_size,
            kv_snapshots_total,
            kv_restores_total,
            errors_total,
            rejected_total,
        }
    }

    /// Record a completed generation with stats.
    pub fn record_generation(
        &self,
        prompt_tokens: u32,
        generated_tokens: u32,
        first_token_ms: u64,
        total_ms: u64,
        tokens_per_sec: f32,
    ) {
        self.prompt_tokens_total.inc_by(u64::from(prompt_tokens));
        self.tokens_generated_total
            .inc_by(u64::from(generated_tokens));
        self.first_token_latency_ms.observe(first_token_ms as f64);
        self.generation_total_ms.observe(total_ms as f64);
        self.tokens_per_second.set(tokens_per_sec as i64);

        if generated_tokens > 0 {
            let per_token_ms = total_ms as f64 / f64::from(generated_tokens);
            self.decode_latency_ms.observe(per_token_ms);
        }
    }

    /// Encode all metrics in Prometheus text format.
    pub fn encode(&self, registry: &Registry) -> String {
        let mut buf = String::new();
        prometheus_client::encoding::text::encode(&mut buf, registry)
            .expect("Prometheus encoding should not fail");
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let mut registry = Registry::default();
        let metrics = InferenceMetrics::new(&mut registry);

        // Verify initial values
        assert_eq!(metrics.active_sessions.get(), 0);
        assert_eq!(metrics.loaded_models.get(), 0);
    }

    #[test]
    fn test_metrics_record_generation() {
        let mut registry = Registry::default();
        let metrics = InferenceMetrics::new(&mut registry);

        metrics.record_generation(100, 50, 45, 2000, 25.0);

        assert_eq!(metrics.tokens_per_second.get(), 25);
    }

    #[test]
    fn test_metrics_session_tracking() {
        let mut registry = Registry::default();
        let metrics = InferenceMetrics::new(&mut registry);

        metrics.active_sessions.set(3);
        metrics.sessions_created_total.inc();
        metrics.sessions_created_total.inc();
        metrics.sessions_created_total.inc();

        assert_eq!(metrics.active_sessions.get(), 3);
    }

    #[test]
    fn test_metrics_encode() {
        let mut registry = Registry::default();
        let metrics = InferenceMetrics::new(&mut registry);

        metrics.active_sessions.set(2);
        metrics.tokens_generated_total.inc_by(100);

        let output = metrics.encode(&registry);
        assert!(output.contains("spirit_inference"));
        assert!(output.contains("tokens_generated_total"));
    }

    #[test]
    fn test_metrics_batch_tracking() {
        let mut registry = Registry::default();
        let metrics = InferenceMetrics::new(&mut registry);

        metrics.queue_depth.set(5);
        metrics.batches_total.inc();
        metrics.batch_size.observe(4.0);

        assert_eq!(metrics.queue_depth.get(), 5);
    }
}
