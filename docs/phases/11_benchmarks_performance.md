# Phase 11: Benchmarks & Performance Validation

**Roadmap Alignment:** Phase 1 — Milestone M1.6
**Estimated Duration:** Month 6
**Dependencies:** Phases 01–10 (all Phase 1 subsystems complete)
**Architecture Plan Sections:** §14

---

## Context & Overview

Phase 11 is the validation phase. After all Phase 1 subsystems are built, we need to prove that SpiritROS delivers on its performance promises. The architecture plan claims 30–60% improvement over stock Linux + Ollama — this phase builds the benchmarking infrastructure to measure, validate, and publish those results.

This phase delivers `spirit-bench`, a comprehensive benchmarking tool that measures every performance-critical metric, compares against a baseline (stock Linux + Ollama), and produces reproducible results suitable for publication.

---

## Goals

1. **spirit-bench tool** — Comprehensive benchmarking suite with standardized test scenarios
2. **Baseline measurements** — Stock Linux + Ollama numbers for direct comparison
3. **Performance targets** — Validate against architecture plan targets (§14.1)
4. **Regression CI** — Automated benchmark runs to catch performance regressions
5. **Grafana dashboards** — Real-time performance monitoring during benchmarks
6. **Published results** — Reproducible benchmark report with methodology

---

## Design Decisions

### DD-11.1: Benchmark in Controlled, Reproducible Conditions
**Decision:** Benchmarks run in a standardized environment: specific Docker image, fixed model (llama-3.1-8b-q4_k_m), fixed hardware profile, system quiesced (no background load).
**Rationale:** Reproducibility is essential for credible results. Uncontrolled variables (background processes, thermal throttling, varying models) make comparisons meaningless.

### DD-11.2: A/B Comparison Against Stock Linux + Ollama
**Decision:** Every benchmark produces two numbers: SpiritROS and baseline (stock Ubuntu 24.04 + Ollama with identical model/hardware).
**Rationale:** Relative improvement is more meaningful than absolute numbers. The baseline runs on the same hardware with the same model to isolate the OS-level optimizations.

### DD-11.3: Warm-Up Runs Before Measurement
**Decision:** Each benchmark scenario runs 3 warm-up iterations (results discarded) before 10 measured iterations (results averaged with std dev).
**Rationale:** Warm-up eliminates cold-cache effects. Multiple iterations provide statistical confidence. Standard deviation shows consistency.

### DD-11.4: Benchmark Categories Aligned to Subsystems
**Decision:** Organize benchmarks by subsystem: memory, scheduling, inference, filesystem, API.
**Rationale:** When a regression occurs, categorical benchmarks pinpoint which subsystem degraded.

### DD-11.5: Score System for Quick Comparison
**Decision:** Aggregate results into a single score (0–1000) for quick comparison, with detailed per-metric breakdowns available.
**Rationale:** A single number is useful for headlines and CI gates. Detailed breakdowns are essential for debugging.

---

## Components & Tasks

### 1. spirit-bench CLI (`benchmarks/`)
```rust
// benchmarks/src/main.rs
use clap::Parser;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: BenchCommand,
}

enum BenchCommand {
    Run {
        #[arg(long, default_value = "full")]
        suite: String,  // "full", "quick", "memory", "inference", "scheduler", "fs", "api"

        #[arg(long, default_value = "10")]
        iterations: u32,

        #[arg(long)]
        baseline: bool,  // Run as baseline (stock Linux mode)

        #[arg(long)]
        output: Option<PathBuf>,  // JSON output file
    },
    Compare {
        spirit_results: PathBuf,
        baseline_results: PathBuf,
    },
    Report {
        results: PathBuf,
        format: ReportFormat,  // markdown, html, json
    },
}
```

### 2. Benchmark Suites

#### Memory Benchmarks
| Benchmark | Metric | Phase 1 Target | Baseline |
|---|---|---|---|
| Model cold load (7B, Q4) | Time to first inference-ready | 2.1s | 4.2s |
| Model hot swap | Time to switch loaded model | 0.3s | 3.8s |
| Hugepage TLB misses | TLB misses during inference | 90%+ reduction | Baseline |
| VRAM fragmentation | Wasted GPU memory after load/unload cycles | <8% | 15-20% |
| KV-cache fork | Time to fork 32k context | 0.02ms | N/A (full copy) |
| Zone enforcement | OOM behavior (inference survives when general OOMs) | Pass | Fail (OOM killer random) |

#### Inference Benchmarks
| Benchmark | Metric | Phase 1 Target | Baseline |
|---|---|---|---|
| First token latency (7B) | Time from prompt submit to first token | 95ms | 180ms |
| Decode throughput (single) | Tokens/second, single stream | 85 tok/s | 65 tok/s |
| Decode throughput (batch=4) | Tokens/second, 4 concurrent streams | 312 tok/s | ~240 tok/s |
| Decode throughput (batch=8) | Tokens/second, 8 concurrent streams | 486 tok/s | 320 tok/s |
| Inter-token latency (p99) | 99th percentile token-to-token time | <25ms | ~35ms |

#### Scheduler Benchmarks
| Benchmark | Metric | Phase 1 Target | Baseline |
|---|---|---|---|
| Context switch to inference | Time from ready to running | 2ms | 12ms |
| Prefill CPU utilization | CPU cores used during prefill | 90%+ | ~60% (shared) |
| Decode thread latency | Scheduling jitter on decode thread | <0.5ms | ~3ms |
| Inference under load | Throughput with background processes | <5% degradation | 15-25% degradation |

#### Filesystem Benchmarks
| Benchmark | Metric | Phase 1 Target | Baseline |
|---|---|---|---|
| FUSE overhead (metadata) | Stat/readdir latency vs. ext4 | <15% overhead | N/A |
| Semantic search latency | Time for embedding search query | <100ms | N/A |
| Auto-tag latency | Time to tag a new file | <500ms | N/A |
| Model cache hit rate | Weight loading from cache vs. disk | >80% for frequent models | 0% (no cache) |

#### API Benchmarks
| Benchmark | Metric | Phase 1 Target | Baseline |
|---|---|---|---|
| REST API overhead | Added latency vs. direct inference | <5ms | N/A |
| gRPC streaming overhead | Per-token delivery overhead | <0.1ms | N/A |
| Shared memory IPC | Token transfer latency | <1μs | N/A |
| Concurrent API connections | Max concurrent without degradation | 100+ | N/A |

### 3. Benchmark Runner
```rust
pub struct BenchmarkRunner {
    config: BenchConfig,
    memory_manager: Arc<MemoryManager>,
    inference: InferenceClient,
    scheduler: SchedulerClient,
}

impl BenchmarkRunner {
    pub async fn run_suite(&self, suite: &str) -> BenchResults {
        let benchmarks = self.get_suite(suite);
        let mut results = BenchResults::new();

        for bench in benchmarks {
            // Warm-up
            for _ in 0..self.config.warmup_iterations {
                bench.run().await;
            }

            // Measured runs
            let mut measurements = Vec::new();
            for _ in 0..self.config.measured_iterations {
                let start = Instant::now();
                let result = bench.run().await;
                let duration = start.elapsed();
                measurements.push(Measurement { duration, result });
            }

            results.add(bench.name(), measurements);
        }

        results
    }
}
```

### 4. Comparison Tool
```bash
$ spirit-bench compare spirit-results.json baseline-results.json

SpiritROS vs Baseline Comparison
═════════════════════════════════════════════════════════════════
Metric                       SpiritROS    Baseline    Change
─────────────────────────────────────────────────────────────────
Model cold load (7B)         2.14s        4.23s       ▲ 49.4%
Model hot swap               0.31s        3.81s       ▲ 91.9%
First token latency          94ms         181ms       ▲ 48.1%
Decode throughput (single)   87 tok/s     65 tok/s    ▲ 33.8%
Decode throughput (batch=8)  486 tok/s    321 tok/s   ▲ 51.4%
KV-cache fork                0.018ms      N/A         —
Context switch to inference  1.8ms        12.3ms      ▲ 85.4%
Memory overhead (OS)         198 MB       812 MB      ▲ 75.6%
GPU memory waste             6.2%         17.8%       ▲ 65.2%
─────────────────────────────────────────────────────────────────
Overall Score                847/1000     520/1000    ▲ 62.9%
```

### 5. Regression CI Pipeline
- Run `spirit-bench run --suite quick` on every PR merge
- Quick suite: first-token latency, single-stream decode, model load time
- Compare against last known-good results
- Gate: fail CI if any metric degrades by >5%
- Full suite runs nightly

### 6. Grafana Benchmark Dashboards
- Real-time metrics during benchmark runs
- Historical trend lines for each metric
- Regression alerts when metrics degrade
- Per-subsystem breakdown views

---

## Acceptance Criteria

- [ ] `spirit-bench run --suite full` completes without errors
- [ ] All 8 core benchmark scenarios produce valid measurements
- [ ] Baseline measurements are collected on stock Linux + Ollama
- [ ] SpiritROS shows measurable improvement in at least 6/8 core metrics
- [ ] Overall score is ≥700/1000 (Phase 1 target, representing 30%+ improvement)
- [ ] Results are reproducible: running twice on same hardware produces <5% variance
- [ ] `spirit-bench compare` produces a clear comparison report
- [ ] `spirit-bench report --format markdown` generates a publishable document
- [ ] Regression CI pipeline catches a deliberate performance degradation (tested by intentionally disabling an optimization)
- [ ] Benchmark results are published to Grafana with historical tracking

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Hardware variability between test runs | Noisy results | Run on dedicated hardware; thermal throttling detection; report std dev |
| Phase 1 gains are smaller than projected (30-60%) | Credibility issue | Identify which subsystems underperform; focus optimization effort there |
| Baseline setup doesn't match production Ollama | Unfair comparison | Use identical Ollama version, identical model, identical configuration |
| Benchmark suite takes too long | Developer won't run it | Quick suite (<5 min) for CI; full suite (~30 min) for nightly/release |
| Thermal throttling during long benchmarks | Results degrade over time | Monitor GPU/CPU temps; cool-down periods between heavy benchmarks |

---

## Key References

- Architecture Plan §14 (Performance Benchmarks & Targets)
- Architecture Plan §14.1 (Key Metrics)
- Architecture Plan §14.2 (How Gains Are Achieved)
- Architecture Plan §14.3 (Benchmark Suite)
