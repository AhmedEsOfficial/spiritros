# SpiritROS — Phase Breakdown Index

**Source:** [SpiritROS_Architecture_Plan.md](../SpiritROS_Architecture_Plan.md)
**Last Updated:** 2026-03-26

---

## Roadmap Alignment

| Roadmap Phase | Months | Component Phases |
|---|---|---|
| **Phase 1 — Virtual Proof of Concept** | 1–6 | Phases 01–11 |
| **Phase 2 — Kernel Development** | 7–18 | Phase 12 |
| **Phase 3 — Production Ready** | 19–30 | Phase 13 |
| **Phase 4 — Ecosystem** | 30+ | Phase 14 |

---

## Component Phases

| # | Phase | Primary Focus | Dependencies |
|---|---|---|---|
| [01](01_foundation_docker_infrastructure.md) | Project Foundation & Docker Infrastructure | Scaffolding, Docker image, CI/CD, repo structure | None |
| [02](02_tensorfirst_memory_manager.md) | TensorFirst Memory Manager | cgroups v2 zones, hugepages, VRAM reservation | Phase 01 |
| [03](03_inference_engine_integration.md) | Inference Engine Integration | Ollama service, patched llama.cpp, model registry | Phases 01, 02 |
| [04](04_spirit_scheduler.md) | Spirit Scheduler | Inference-aware CPU/GPU scheduling shim | Phases 01, 02, 03 |
| [05](05_spiritfs_semantic_filesystem.md) | SpiritFS — Semantic Filesystem | FUSE-based semantic FS, embedding index, CAS | Phases 01, 03 |
| [06](06_spirit_shell.md) | Spirit Shell — LLM-Native Interface | Natural language CLI, command resolution, agentic mode | Phases 01, 03, 05 |
| [07](07_security_architecture.md) | Security Architecture | Capabilities, inference sandboxing, Spirit Guard | Phases 01, 03 |
| [08](08_networking_api_layer.md) | Networking & API Layer | API endpoints, smoltcp, gRPC, Spirit Mesh groundwork | Phases 01, 03 |
| [09](09_sdk_developer_experience.md) | SDK & Developer Experience | Rust/Python/TS SDKs, plugin system, Spirit Compose | Phases 01–08 |
| [10](10_compatibility_layers.md) | Compatibility Layers | Linux compat, OCI containers, Wine, Waydroid | Phases 01, 02, 04 |
| [11](11_benchmarks_performance.md) | Benchmarks & Performance Validation | spirit-bench, performance targets, regression CI | Phases 01–10 |
| [12](12_spiritkernel_microkernel.md) | SpiritKernel — Hybrid Microkernel | Rust microkernel, boot, native memory/scheduler/GPU | Phase 1 complete (01–11) |
| [13](13_production_hardening_v1.md) | Production Hardening & v1.0 | Full compat, security hardening, distributed inference | Phases 12, 07, 08, 10 |
| [14](14_ecosystem_expansion.md) | Ecosystem & Expansion | App store, hardware partners, robotics, edge, formal verification | Phase 13 |

---

## How to Use These Documents

Each phase document contains:
- **Context & Overview** — What this phase is and why it matters
- **Goals** — Concrete objectives for the phase
- **Dependencies** — What must be completed before starting
- **Design Decisions** — Key architectural choices and rationale
- **Components & Tasks** — Detailed breakdown of work items
- **Acceptance Criteria** — How we know the phase is complete
- **Risks & Mitigations** — What could go wrong and how to handle it
- **Key References** — Relevant sections of the master architecture plan

Phases are designed to be worked on sequentially where dependencies exist, and in parallel where they don't. Phases 05–08 can largely proceed in parallel once Phases 01–03 are complete.
