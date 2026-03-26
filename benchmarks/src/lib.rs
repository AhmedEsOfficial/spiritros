//! # spirit-bench
//!
//! The benchmarking suite for `SpiritROS`. Provides standardized benchmark suites for measuring
//! inference throughput, memory management efficiency, filesystem performance, and scheduler
//! behavior.
//!
//! All benchmark results include hardware specs, model configuration, and exact reproduction
//! commands as required by the project's benchmark reproducibility invariant.

#![warn(clippy::all, clippy::pedantic)]

pub mod filesystem;
pub mod inference;
pub mod memory;
pub mod scheduler;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// The result of a single benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Name of the benchmark suite (e.g., "inference", "memory").
    pub suite: String,
    /// Name of the specific benchmark within the suite.
    pub name: String,
    /// When the benchmark was executed.
    pub timestamp: DateTime<Utc>,
    /// Duration of the benchmark in milliseconds.
    pub duration_ms: f64,
    /// Primary metric value (interpretation depends on suite).
    pub value: f64,
    /// Unit of the primary metric (e.g., "tokens/s", "MB/s", "us").
    pub unit: String,
    /// Hardware and configuration metadata for reproducibility.
    pub metadata: BenchmarkMetadata,
}

/// Metadata required for benchmark reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetadata {
    /// Hardware description (e.g., "NVIDIA RTX 4090, 24GB VRAM").
    pub hardware: String,
    /// Model used (e.g., "llama3:8b-q4_K_M").
    pub model: Option<String>,
    /// Batch size if applicable.
    pub batch_size: Option<u32>,
    /// Exact command to reproduce this benchmark.
    pub reproduce_command: String,
}

/// Lists all available benchmark suites.
#[must_use]
pub fn available_suites() -> Vec<SuiteInfo> {
    vec![
        SuiteInfo {
            name: "inference".to_string(),
            description: "Token throughput, first-token latency, batch inference".to_string(),
        },
        SuiteInfo {
            name: "memory".to_string(),
            description: "Zone allocation, KV-cache COW, hugepage utilization".to_string(),
        },
        SuiteInfo {
            name: "filesystem".to_string(),
            description: "SpiritFS semantic search, POSIX path latency, embedding throughput"
                .to_string(),
        },
        SuiteInfo {
            name: "scheduler".to_string(),
            description: "Inference thread priority, context switch latency, CPU pinning"
                .to_string(),
        },
    ]
}

/// Information about an available benchmark suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteInfo {
    /// Suite identifier, used as the CLI argument.
    pub name: String,
    /// Human-readable description.
    pub description: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_available_suites_returns_four() {
        let suites = available_suites();
        assert_eq!(suites.len(), 4);
    }

    #[test]
    fn test_available_suites_names() {
        let suites = available_suites();
        let names: Vec<&str> = suites.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"inference"));
        assert!(names.contains(&"memory"));
        assert!(names.contains(&"filesystem"));
        assert!(names.contains(&"scheduler"));
    }

    #[test]
    fn test_benchmark_result_serialization() {
        let result = BenchmarkResult {
            suite: "inference".to_string(),
            name: "first_token_latency".to_string(),
            timestamp: Utc::now(),
            duration_ms: 1500.0,
            value: 42.5,
            unit: "ms".to_string(),
            metadata: BenchmarkMetadata {
                hardware: "test".to_string(),
                model: Some("llama3:8b".to_string()),
                batch_size: Some(1),
                reproduce_command: "spirit-bench run inference".to_string(),
            },
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: BenchmarkResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.suite, "inference");
        assert_eq!(deserialized.value, 42.5);
    }
}
