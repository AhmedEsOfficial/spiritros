# Phase 10: Compatibility Layers — Universal Software Integration

**Roadmap Alignment:** Phase 1 — basic Linux compat, Phase 3 — M3.1 (full Linux), M3.4 (Windows/Android)
**Estimated Duration:** Month 2–4 (Phase 1 basics), Months 19–21 + 28–29 (Phase 3 full)
**Dependencies:** Phase 01 (Foundation), Phase 02 (Memory Manager), Phase 04 (Scheduler)
**Architecture Plan Sections:** §9, §3.5

---

## Context & Overview

SpiritROS does not require developers to rewrite their software. It brings the entire ecosystem of existing applications through a layered compatibility architecture:

- **Linux compatibility** (primary): Run any Linux binary, CLI tool, Docker container
- **Windows compatibility**: Wine/Proton for Windows executables
- **Android compatibility**: Waydroid for Android apps
- **OCI containers**: Native container runtime (youki) for Docker/Podman images

In Phase 1 (containerized deployment on a Linux host), "Linux compatibility" is largely native since we're running on Linux. The compatibility layer here is primarily about the syscall translation shim that will become essential in Phase 12 when the custom kernel replaces Linux. Building it now ensures applications continue to work seamlessly through the kernel transition.

---

## Goals

### Phase 1 Goals
1. **Linux compat shim prototype** — Syscall interception and translation layer (foundational work for Phase 12)
2. **OCI container runtime** — youki-based container execution with Spirit memory zone awareness
3. **Init system** — dinit or custom lightweight init for managing Spirit services
4. **Cross-platform package manager** — `spirit-pkg` wrapping apt/apk/native backends
5. **GPU driver shim** — CUDA/ROCm applications route through Spirit VRAM manager

### Phase 3 Goals
6. **Full Linux binary compatibility** — Complete syscall translation (500+ syscalls)
7. **Windows compat** — Wine 9.x + Proton + DXVK integration
8. **Android compat** — Waydroid + LXC container with GPU passthrough
9. **systemd service shim** — Map systemd unit files to dinit services

---

## Design Decisions

### DD-10.1: gVisor-Inspired Syscall Translation
**Decision:** Build the Linux compatibility layer as a modified gVisor-style sandbox that intercepts Linux syscalls and translates them to Spirit equivalents.
**Rationale:** gVisor (Apache-2.0) already implements a comprehensive Linux syscall surface in a safe language. Adapting its approach (ptrace + seccomp interception) gives us a head start. In Phase 1, the translation is Linux→Linux (identity in many cases); in Phase 12, it becomes Linux→SpiritKernel.

### DD-10.2: youki for Container Runtime
**Decision:** Use youki (Rust, Apache-2.0) as the OCI-compatible container runtime.
**Rationale:** youki is a Rust implementation of the OCI runtime spec. It's faster to start than runc, written in Rust (consistent with the project), and can be modified to integrate with Spirit's memory zones and scheduler.

### DD-10.3: dinit for Service Management
**Decision:** Use dinit (Apache-2.0) as the init/service manager instead of systemd.
**Rationale:** systemd is too heavy and assumes control over the entire system. dinit is lightweight, dependency-aware, and suitable for managing Spirit services within a container or minimal OS. It handles service ordering, restart policies, and logging.

### DD-10.4: Spirit-Pkg is a Thin Wrapper
**Decision:** `spirit-pkg` delegates to underlying package managers (apt for deb, apk for Alpine, Ollama for models, Wine for Windows apps) rather than implementing a new package format from scratch.
**Rationale:** Package management is a solved problem for each ecosystem. The value of `spirit-pkg` is the unified interface, not a new package format. Native Spirit packages (for Spirit-specific apps) use a Flatpak-like sandboxed format for Phase 3.

### DD-10.5: Windows/Android are Phase 3 Stretch Goals
**Decision:** Wine and Waydroid integration is deferred to Phase 3. Phase 1 focuses exclusively on Linux compatibility and containers.
**Rationale:** The primary use case (LLM inference, ML tooling, development) is Linux-centric. Windows and Android compatibility adds value for enterprise and edge deployments but is not critical for PoC.

---

## Components & Tasks

### 1. Linux Compat Shim (`runtime/compat/linux/`)
```rust
pub struct LinuxCompatServer {
    syscall_table: SyscallTranslationTable,
    process_table: ProcessTable,
}

impl LinuxCompatServer {
    /// Translate a Linux syscall to Spirit equivalent
    pub fn translate(&self, syscall: LinuxSyscall) -> Result<SpiritSyscall> {
        match syscall.number {
            // File operations
            SYS_OPEN => self.translate_open(syscall),
            SYS_READ => self.translate_read(syscall),
            SYS_WRITE => self.translate_write(syscall),
            SYS_CLOSE => self.translate_close(syscall),

            // Memory
            SYS_MMAP => self.translate_mmap(syscall),
            SYS_MUNMAP => self.translate_munmap(syscall),
            SYS_BRK => self.translate_brk(syscall),

            // Process
            SYS_FORK => self.translate_fork(syscall),
            SYS_EXEC => self.translate_exec(syscall),
            SYS_EXIT => self.translate_exit(syscall),

            // Network
            SYS_SOCKET => self.translate_socket(syscall),
            SYS_BIND => self.translate_bind(syscall),
            SYS_CONNECT => self.translate_connect(syscall),

            // ... 500+ syscalls
            _ => Err(SyscallError::Unsupported(syscall.number)),
        }
    }
}
```

**Phase 1 implementation:** In Phase 1 (running on Linux), the shim is a passthrough with instrumentation — it intercepts syscalls, logs them, translates the few that differ (memory allocation → zone-aware), and passes the rest through. This builds the infrastructure for Phase 12 where the translation becomes essential.

### 2. Container Runtime Integration (`runtime/compat/containers/`)
```rust
pub struct SpiritContainerRuntime {
    youki: YoukiRuntime,
    memory_manager: Arc<MemoryManager>,
    scheduler: Arc<SpiritScheduler>,
}

impl SpiritContainerRuntime {
    /// Create container with Spirit zone awareness
    pub fn create(&self, spec: OciSpec) -> Result<ContainerId> {
        // Modify OCI spec to enforce Spirit memory zones
        let mut spec = spec;
        spec.linux.resources.memory.limit = self.zone_general_limit();

        // Containers run in ZONE_GENERAL (no inference memory access)
        // Unless explicitly granted CAP_INFER_* capabilities
        spec.linux.resources.cgroup_path = "/spirit/general";

        // Set scheduler tier to P3 (COMPATIBILITY)
        // ...

        self.youki.create(spec)
    }

    /// Create container with GPU + inference access
    pub fn create_with_gpu(&self, spec: OciSpec, caps: CapabilitySet) -> Result<ContainerId>;
}
```

**CLI:**
```bash
spirit container run -it ubuntu:24.04 /bin/bash
spirit container run --gpu all nvidia/cuda:12.8-runtime python inference.py
spirit compose up -f docker-compose.yml
spirit container list
spirit container stop <id>
```

### 3. Service Manager (dinit Integration)
```
/etc/spirit/services/
├── ollama.service          # Ollama inference server
├── inference-coord.service # Inference Coordinator
├── spiritfs.service        # SpiritFS FUSE mount
├── scheduler.service       # Spirit Scheduler daemon
├── guard.service           # Spirit Guard output filter
├── networking.service      # API server (REST + gRPC + WebSocket)
├── qdrant.service          # Vector database
└── metrics.service         # Prometheus + metrics exporter
```

Each service file specifies:
- Dependencies (e.g., inference-coord depends on ollama and memory manager)
- Restart policy
- Environment variables
- Capability grants
- Log destination

### 4. Cross-Platform Package Manager (`spirit-pkg`)
```bash
spirit-pkg install python3          # → apt/apk backend
spirit-pkg install nodejs           # → apt/apk backend
spirit-pkg install ollama           # → native Spirit backend
spirit-pkg install notepad++        # → Wine + Windows installer (Phase 3)
spirit-pkg install termux           # → Waydroid + F-Droid (Phase 3)
spirit-pkg search "image editor"    # → searches all backends
spirit-pkg list                     # → shows installed from all backends
```

### 5. GPU Driver Shim
- Intercept CUDA API calls from Linux applications
- Route `cudaMalloc` → Spirit VRAM manager (prevents fragmentation)
- Report VRAM availability accounting for Spirit's zone reservations
- Support for: CUDA 12.x, ROCm 6.x, Vulkan
- Implementation: `LD_PRELOAD` library that wraps CUDA runtime

### 6. Native Spirit Package Format (Phase 3)
```toml
# spirit-package.toml
[package]
name = "my-inference-app"
version = "1.0.0"
runtime = "spirit-runtime-1.0"

[capabilities]
inference = true
network = "localhost"
filesystem = ["/data"]
gpu = true
```

- Flatpak-like sandboxed format
- Declarative capability requests
- Content-addressable storage for deduplication
- Signed packages with model integrity verification

### 7. Windows Compatibility (Phase 3)
- Wine 9.x + Proton integration
- DXVK/VKD3D-Proton for DirectX → Vulkan
- .NET support via Wine-Mono
- Managed via `spirit-pkg install <windows-app>`

### 8. Android Compatibility (Phase 3)
- Waydroid in LXC container
- GPU passthrough for OpenGL ES / Vulkan
- Android-native ML frameworks (TFLite, NNAPI)
- Managed via `spirit-pkg install <android-app>`

---

## Acceptance Criteria

### Phase 1
- [ ] Linux CLI tools (bash, python, node, git, cargo) run inside Spirit container
- [ ] `spirit container run` creates and runs OCI containers with Spirit zone enforcement
- [ ] Containers in ZONE_GENERAL cannot access ZONE_MODEL memory
- [ ] `spirit container run --gpu all` provides GPU access to containers
- [ ] dinit manages all Spirit services with correct dependency ordering
- [ ] `spirit-pkg install python3` installs Python via apt backend
- [ ] GPU driver shim routes cudaMalloc through Spirit VRAM manager
- [ ] Syscall translation shim logs all intercepted syscalls (instrumentation mode)
- [ ] Services restart automatically on failure (dinit restart policy)

### Phase 3
- [ ] Full Linux syscall table translated (95%+ coverage)
- [ ] Wine runs Windows executables (notepad.exe, simple Win32 apps)
- [ ] Waydroid runs Android apps with GPU acceleration
- [ ] `spirit-pkg` installs from all backends (Linux, Windows, Android, native)
- [ ] Native Spirit packages are sandboxed with enforced capabilities

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Syscall translation coverage gaps | Some Linux apps fail | Prioritize by usage; focus on syscalls used by LLM tooling first |
| youki modifications break OCI compliance | Docker images don't work | Run OCI compliance test suite in CI; modifications are additive, not breaking |
| GPU driver shim breaks CUDA applications | ML tools fail | Thorough testing with popular CUDA apps; fallback to native CUDA if shim causes issues |
| Wine/Waydroid integration complexity | Phase 3 schedule slip | These are stretch goals; core value is Linux + containers |
| dinit lacks features compared to systemd | Some service patterns unsupported | Implement missing features (socket activation, etc.) as needed |

---

## Key References

- Architecture Plan §9 (Compatibility Layers — Universal Software Integration)
- Architecture Plan §9.1 (Design Philosophy)
- Architecture Plan §9.2 (Linux Compatibility)
- Architecture Plan §9.3 (Windows Compatibility)
- Architecture Plan §9.4 (Android Compatibility)
- Architecture Plan §9.5 (Container Runtime)
- Architecture Plan §9.6 (Cross-Platform Package Management)
- Architecture Plan §3.5 (Linux Compatibility Shim)
