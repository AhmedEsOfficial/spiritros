//! Continuous batching scheduler for inference requests.
//!
//! Groups concurrent inference requests into GPU batches for maximum
//! throughput. Priority-aware: interactive requests (Spirit Shell) get
//! lower latency; background requests (SpiritFS indexing) are batched
//! more aggressively.
//!
//! # Design
//!
//! The batch scheduler runs a continuous loop:
//! 1. Collect requests from the queue (up to max batch size or timeout).
//! 2. Sort by priority (interactive first).
//! 3. Group by model (same-model requests can share a GPU batch).
//! 4. Dispatch each group to the backend.
//! 5. Deliver results back to waiting tasks.
//!
//! This is coordination-only — the actual GPU batching happens inside
//! Ollama/llama.cpp. The scheduler's job is to multiplex requests and
//! manage priority.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, TrySendError};
use parking_lot::Mutex;
use tracing::debug;

use crate::types::{BatchId, IdGenerator, Priority, SessionId};

/// A request enqueued for batch processing.
#[derive(Debug)]
pub struct QueuedRequest {
    /// Session ID this request belongs to.
    pub session_id: SessionId,
    /// Model name to use.
    pub model: String,
    /// Prompt text.
    pub prompt: String,
    /// Request priority.
    pub priority: Priority,
    /// When this request was enqueued.
    pub enqueued_at: Instant,
    /// Channel to send the result back.
    pub result_tx: tokio::sync::oneshot::Sender<BatchItemResult>,
}

/// Result of a single batched request.
#[derive(Debug)]
pub enum BatchItemResult {
    /// Generation started — tokens will stream via the session.
    Started,
    /// The request failed.
    Failed(String),
}

/// A group of requests for the same model, ready to be dispatched.
#[derive(Debug)]
pub struct BatchGroup {
    /// Batch identifier.
    pub batch_id: BatchId,
    /// Model name.
    pub model: String,
    /// Requests in this batch, sorted by priority (highest first).
    pub requests: Vec<QueuedRequest>,
}

/// Priority wrapper for the binary heap (max-heap on priority, then FIFO).
struct PrioritizedRequest {
    request: QueuedRequest,
    sequence: u64,
}

impl PartialEq for PrioritizedRequest {
    fn eq(&self, other: &Self) -> bool {
        self.request.priority == other.request.priority && self.sequence == other.sequence
    }
}

impl Eq for PrioritizedRequest {}

impl PartialOrd for PrioritizedRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then lower sequence (FIFO within same priority)
        self.request
            .priority
            .cmp(&other.request.priority)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

/// Configuration for the batch scheduler.
#[derive(Debug, Clone)]
pub struct BatchSchedulerConfig {
    /// Maximum requests per batch.
    pub max_batch_size: usize,
    /// Maximum time to wait for a full batch before dispatching a partial one.
    pub batch_timeout: Duration,
    /// Maximum queue depth before rejecting requests.
    pub max_queue_depth: usize,
}

impl Default for BatchSchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            batch_timeout: Duration::from_millis(50),
            max_queue_depth: 256,
        }
    }
}

/// The batch scheduler that groups and prioritizes inference requests.
///
/// Thread-safe. Requests are submitted via `enqueue()` and dispatched
/// by calling `collect_batch()` in a loop.
pub struct BatchScheduler {
    /// Inbound request channel (bounded for backpressure).
    request_tx: Sender<QueuedRequest>,
    request_rx: Receiver<QueuedRequest>,
    /// Priority queue for pending requests.
    queue: Mutex<BinaryHeap<PrioritizedRequest>>,
    /// Monotonic sequence counter for FIFO ordering.
    sequence: IdGenerator,
    /// Batch ID generator.
    batch_ids: IdGenerator,
    /// Configuration.
    config: BatchSchedulerConfig,
}

impl BatchScheduler {
    /// Create a new batch scheduler.
    #[must_use]
    pub fn new(config: BatchSchedulerConfig) -> Self {
        let (tx, rx) = crossbeam_channel::bounded(config.max_queue_depth);
        Self {
            request_tx: tx,
            request_rx: rx,
            queue: Mutex::new(BinaryHeap::new()),
            sequence: IdGenerator::new(),
            batch_ids: IdGenerator::new(),
            config,
        }
    }

    /// Enqueue a request for batch processing.
    ///
    /// Returns a oneshot receiver that will receive the result.
    pub fn enqueue(
        &self,
        session_id: SessionId,
        model: String,
        prompt: String,
        priority: Priority,
    ) -> Result<tokio::sync::oneshot::Receiver<BatchItemResult>, crate::error::InferenceError> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        let request = QueuedRequest {
            session_id,
            model,
            prompt,
            priority,
            enqueued_at: Instant::now(),
            result_tx: tx,
        };

        match self.request_tx.try_send(request) {
            Ok(()) => Ok(rx),
            Err(TrySendError::Full(_)) => Err(crate::error::InferenceError::QueueFull {
                queue_depth: self.config.max_queue_depth,
                max_depth: self.config.max_queue_depth,
            }),
            Err(TrySendError::Disconnected(_)) => {
                Err(crate::error::InferenceError::BackendUnavailable(
                    "batch scheduler channel closed".into(),
                ))
            }
        }
    }

    /// Drain the inbound channel into the priority queue.
    fn drain_channel(&self) {
        let mut queue = self.queue.lock();
        while let Ok(request) = self.request_rx.try_recv() {
            let seq = self.sequence.next_id();
            queue.push(PrioritizedRequest {
                request,
                sequence: seq,
            });
        }
    }

    /// Collect the next batch of requests, grouped by model.
    ///
    /// Blocks for up to `batch_timeout` to accumulate requests.
    /// Returns `None` if no requests are available.
    pub fn collect_batch(&self) -> Option<Vec<BatchGroup>> {
        // Wait for at least one request
        match self.request_rx.recv_timeout(self.config.batch_timeout) {
            Ok(first) => {
                let seq = self.sequence.next_id();
                let mut queue = self.queue.lock();
                queue.push(PrioritizedRequest {
                    request: first,
                    sequence: seq,
                });
                drop(queue);
            }
            Err(_) => {
                // Timeout — check if anything accumulated
                self.drain_channel();
                if self.queue.lock().is_empty() {
                    return None;
                }
            }
        }

        // Drain any additional requests that arrived
        self.drain_channel();

        // Pop up to max_batch_size requests
        let mut queue = self.queue.lock();
        let count = queue.len().min(self.config.max_batch_size);
        let mut requests: Vec<QueuedRequest> = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(pr) = queue.pop() {
                requests.push(pr.request);
            }
        }
        drop(queue);

        if requests.is_empty() {
            return None;
        }

        // Group by model
        let mut groups: HashMap<String, Vec<QueuedRequest>> = HashMap::new();
        for req in requests {
            groups.entry(req.model.clone()).or_default().push(req);
        }

        let batch_groups: Vec<BatchGroup> = groups
            .into_iter()
            .map(|(model, reqs)| {
                let batch_id = BatchId::new(self.batch_ids.next_id());
                debug!(
                    batch_id = batch_id.as_u64(),
                    model = %model,
                    count = reqs.len(),
                    "Assembled batch group"
                );
                BatchGroup {
                    batch_id,
                    model,
                    requests: reqs,
                }
            })
            .collect();

        Some(batch_groups)
    }

    /// Get the current queue depth.
    #[must_use]
    pub fn queue_depth(&self) -> usize {
        self.request_rx.len() + self.queue.lock().len()
    }

    /// Get the maximum queue depth.
    #[must_use]
    pub fn max_queue_depth(&self) -> usize {
        self.config.max_queue_depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> BatchSchedulerConfig {
        BatchSchedulerConfig {
            max_batch_size: 4,
            batch_timeout: Duration::from_millis(10),
            max_queue_depth: 16,
        }
    }

    #[tokio::test]
    async fn test_enqueue_and_collect() {
        let sched = BatchScheduler::new(make_config());

        let _rx = sched
            .enqueue(
                SessionId::new(),
                "llama3:8b".into(),
                "Hello".into(),
                Priority::Normal,
            )
            .unwrap();

        let groups = sched.collect_batch();
        assert!(groups.is_some());
        let groups = groups.unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].model, "llama3:8b");
        assert_eq!(groups[0].requests.len(), 1);
    }

    #[tokio::test]
    async fn test_batch_groups_by_model() {
        let sched = BatchScheduler::new(make_config());

        let _r1 = sched
            .enqueue(
                SessionId::new(),
                "model-a".into(),
                "p1".into(),
                Priority::Normal,
            )
            .unwrap();
        let _r2 = sched
            .enqueue(
                SessionId::new(),
                "model-b".into(),
                "p2".into(),
                Priority::Normal,
            )
            .unwrap();
        let _r3 = sched
            .enqueue(
                SessionId::new(),
                "model-a".into(),
                "p3".into(),
                Priority::Normal,
            )
            .unwrap();

        let groups = sched.collect_batch().unwrap();
        assert_eq!(groups.len(), 2);

        let total_requests: usize = groups.iter().map(|g| g.requests.len()).sum();
        assert_eq!(total_requests, 3);
    }

    #[tokio::test]
    async fn test_priority_ordering() {
        let sched = BatchScheduler::new(BatchSchedulerConfig {
            max_batch_size: 2,
            batch_timeout: Duration::from_millis(10),
            max_queue_depth: 16,
        });

        let _r1 = sched
            .enqueue(
                SessionId::new(),
                "model".into(),
                "low".into(),
                Priority::Background,
            )
            .unwrap();
        let _r2 = sched
            .enqueue(
                SessionId::new(),
                "model".into(),
                "high".into(),
                Priority::Interactive,
            )
            .unwrap();

        let groups = sched.collect_batch().unwrap();
        let group = &groups[0];
        // Interactive should come first (higher priority)
        assert_eq!(group.requests[0].priority, Priority::Interactive);
    }

    #[tokio::test]
    async fn test_queue_depth() {
        let sched = BatchScheduler::new(make_config());
        assert_eq!(sched.queue_depth(), 0);

        let _r1 = sched
            .enqueue(SessionId::new(), "m".into(), "p".into(), Priority::Normal)
            .unwrap();
        assert!(sched.queue_depth() > 0);
    }

    #[tokio::test]
    async fn test_empty_collect_returns_none() {
        let sched = BatchScheduler::new(BatchSchedulerConfig {
            max_batch_size: 4,
            batch_timeout: Duration::from_millis(5),
            max_queue_depth: 16,
        });

        let result = sched.collect_batch();
        assert!(result.is_none());
    }
}
