# Phase 13: Production Hardening & v1.0 Release

**Roadmap Alignment:** Phase 3 — Production Ready (Months 19–30)
**Estimated Duration:** 12 months
**Dependencies:** Phase 12 (SpiritKernel), Phase 07 (Security), Phase 08 (Networking), Phase 10 (Compatibility)
**Architecture Plan Sections:** §9, §10, §8.3, §8.4, §16 (Phase 3)

---

## Context & Overview

Phase 13 transforms SpiritROS from a working kernel into a production-ready operating system. This means full Linux binary compatibility (not just the subset needed for Ollama), complete security hardening with kernel-enforced capabilities, distributed inference across multiple nodes via Spirit Mesh, Windows and Android compatibility layers, and the reliability expected of a system people depend on.

This phase ends with the **v1.0 stable release** — complete documentation, community infrastructure, and a public launch.

---

## Goals

1. **Full Linux binary compatibility** — 95%+ syscall coverage, Docker containers work seamlessly
2. **Kernel-enforced security** — Capability system enforced at syscall boundary, not just userspace
3. **Spirit Guard at kernel level** — Output filtering with kernel-backed enforcement
4. **Distributed inference** — Multi-node tensor parallelism, Spirit Mesh federation
5. **Windows compatibility** — Wine 9.x + Proton integrated and tested
6. **Android compatibility** — Waydroid + Android runtime
7. **Production reliability** — Crash recovery, watchdogs, graceful degradation
8. **v1.0 documentation** — Complete user guide, developer guide, API reference
9. **Community launch** — Open source release, contributor onboarding, governance

---

## Design Decisions

### DD-13.1: Kernel-Enforced Capabilities Replace Userspace Checks
**Decision:** All capability checks that were in userspace (Phase 1) are now enforced at the kernel syscall boundary. A process cannot invoke `spirit_infer_prompt()` without the kernel verifying `CAP_INFER_PROMPT`.
**Rationale:** Userspace enforcement is bypassable by a compromised process. Kernel enforcement is the only truly reliable security boundary. This is a hard requirement for production deployments.

### DD-13.2: Gradual Syscall Coverage Expansion
**Decision:** Start with syscalls needed for the Rust runtime + Go runtime + Python interpreter. Expand based on real-world binary testing (run the top 100 CLI tools and fix failures).
**Rationale:** Full POSIX compliance is thousands of syscalls. Prioritize by actual usage rather than trying to implement everything speculatively.

### DD-13.3: Spirit Mesh Uses Raft for Consensus
**Decision:** Replace etcd dependency with a built-in Raft consensus implementation for Spirit Mesh cluster coordination.
**Rationale:** Removes the external dependency on etcd. A built-in Raft implementation gives tighter integration with the kernel's networking and reduces operational complexity.

### DD-13.4: v1.0 Requires Passing a Certification Suite
**Decision:** Define a "SpiritROS Certification Suite" that must pass before v1.0: syscall compliance tests, security audit, performance benchmarks, 72-hour stability test.
**Rationale:** Concrete, measurable criteria for release readiness. Prevents premature release.

---

## Components & Tasks

### Milestone M3.1: Full Linux Compatibility (Months 19–21)

**Tasks:**
- Expand syscall translation to cover 500+ Linux syscalls
- Pass Linux Test Project (LTP) test suite (target: 95%+ pass rate)
- Docker/containerd runs natively on SpiritKernel
- systemd service files translate to dinit (for existing Linux services)
- eBPF basic support (for observability tools that depend on it)
- Test top 100 CLI tools: bash, python, node, go, cargo, git, curl, wget, ssh, docker, kubectl, etc.
- Test popular ML frameworks: PyTorch, TensorFlow, JAX, Hugging Face transformers

### Milestone M3.2: Security Hardening (Months 22–24)

**Tasks:**
- Kernel-enforced capability checks on every syscall
- Spirit Guard runs as a kernel-level filter (inline in inference output path)
- Audit log backed by kernel append-only enforcement
- Secure boot chain: UEFI Secure Boot → signed bootloader → signed kernel → signed userspace
- Model integrity verification at kernel level (signed models loaded into read-only ZONE_MODEL)
- Penetration testing program (internal + external security audit)
- CVE response process established

### Milestone M3.3: Distributed Inference (Months 25–27)

**Tasks:**
- Multi-node tensor parallelism (split models across 2–8 nodes)
- NCCL integration for GPU-to-GPU transfers over NVLink/InfiniBand
- Gloo fallback for TCP-based tensor transfer
- Spirit Wire Protocol for inference coordination
- Built-in Raft consensus for cluster membership
- Spirit Mesh self-organization via mDNS + WireGuard
- Mesh-aware load balancer with model affinity routing
- Model pinning to specific hardware
- Automatic failover when worker nodes go down
- Cross-node KV-cache synchronization

### Milestone M3.4: Windows & Android Compatibility (Months 28–29)

**Tasks:**
- Wine 9.x integration: PE loader, Win32 API, .NET/Mono
- DXVK/VKD3D-Proton for DirectX → Vulkan translation
- Waydroid: LXC container with Android stack
- GPU passthrough for both Wine and Waydroid
- `spirit-pkg` backends for Windows (.exe) and Android (.apk) installation
- Test common Windows ML tools and Android apps

### Milestone M3.5: v1.0 Release (Month 30)

**Tasks:**
- SpiritROS Certification Suite: all tests pass
- 72-hour stability test under continuous inference load
- Complete user documentation (mdBook):
  - Installation guide (bare metal, VM, Docker)
  - Getting started tutorial
  - Configuration reference
  - Spirit Shell user guide
  - SDK guides (Rust, Python, TypeScript)
  - Plugin development guide
  - Security hardening guide
  - Distributed deployment guide
- API reference (OpenAPI, protobuf, SDK docs)
- Release build pipeline (ISO image, Docker image, cloud images)
- Community infrastructure:
  - Public Git repository (Forgejo + GitHub mirror)
  - Issue tracker
  - Discussion forum
  - Contributing guide
  - Code of conduct
  - Governance model
- Launch announcement and blog post

---

## Acceptance Criteria

- [ ] M3.1: LTP test suite passes at 95%+ rate
- [ ] M3.1: Docker containers start and run on SpiritKernel
- [ ] M3.1: Python, Node.js, Go, Rust toolchains work without modification
- [ ] M3.2: Capability checks are enforced at kernel level (userspace bypass impossible)
- [ ] M3.2: Security audit completes with no critical findings
- [ ] M3.2: Secure boot chain works end-to-end
- [ ] M3.3: 70B model runs across 2 nodes with tensor parallelism
- [ ] M3.3: Spirit Mesh discovers and connects 3+ nodes automatically
- [ ] M3.3: Load balancer routes to correct node (model affinity)
- [ ] M3.3: Worker node failure triggers automatic failover
- [ ] M3.4: Wine runs Windows executables (basic Win32 apps)
- [ ] M3.4: Waydroid runs Android apps with GPU acceleration
- [ ] M3.5: Certification Suite passes 100%
- [ ] M3.5: 72-hour stability test completes without kernel panics or inference failures
- [ ] M3.5: Documentation is complete and published
- [ ] M3.5: ISO image boots on standard x86_64 hardware
- [ ] M3.5: Performance meets or exceeds Phase 2 benchmarks (2–3x over stock Linux)

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Linux syscall coverage gaps cause popular tools to break | Adoption blocker | Prioritize testing with real workloads; maintain a compatibility tracker; rapid fix cycle |
| Security audit reveals fundamental design issues | Major rework needed | Continuous security review during development, not just at the end |
| Distributed inference coordination bugs | Data corruption, incorrect output | Extensive integration testing; comparison of distributed vs. single-node output |
| Wine/Waydroid integration harder than expected | Phase 3 scope reduction | These are lower priority; can ship v1.0 with "experimental" status |
| v1.0 readiness criteria too stringent | Release delayed indefinitely | Define minimum viable criteria vs. aspirational criteria; release with known limitations documented |

---

## Key References

- Architecture Plan §16 (Roadmap — Phase 3)
- Architecture Plan §9 (Compatibility Layers)
- Architecture Plan §10 (Security Architecture)
- Architecture Plan §8.3 (Distributed Inference)
- Architecture Plan §8.4 (Federation — Spirit Mesh)
- Architecture Plan Appendix C (Comparison with Existing Approaches)
