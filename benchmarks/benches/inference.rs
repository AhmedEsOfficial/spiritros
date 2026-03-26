//! Criterion benchmarks for the inference path.
//!
//! These micro-benchmarks measure the overhead of SpiritROS's inference coordination
//! layer. They do NOT measure actual LLM inference time — that is done by the
//! `spirit-bench run inference` system-level benchmarks.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_benchmark_result_serialization(c: &mut Criterion) {
    use spirit_bench::{BenchmarkMetadata, BenchmarkResult};

    let result = BenchmarkResult {
        suite: "inference".to_string(),
        name: "first_token_latency".to_string(),
        timestamp: chrono::Utc::now(),
        duration_ms: 1500.0,
        value: 42.5,
        unit: "ms".to_string(),
        metadata: BenchmarkMetadata {
            hardware: "NVIDIA RTX 4090, 24GB VRAM".to_string(),
            model: Some("llama3:8b-q4_K_M".to_string()),
            batch_size: Some(1),
            reproduce_command: "spirit-bench run inference".to_string(),
        },
    };

    c.bench_function("benchmark_result_serialize_json", |b| {
        b.iter(|| serde_json::to_string(black_box(&result)).unwrap());
    });

    let json = serde_json::to_string(&result).unwrap();
    c.bench_function("benchmark_result_deserialize_json", |b| {
        b.iter(|| {
            serde_json::from_str::<BenchmarkResult>(black_box(&json)).unwrap();
        });
    });
}

criterion_group!(benches, bench_benchmark_result_serialization);
criterion_main!(benches);
