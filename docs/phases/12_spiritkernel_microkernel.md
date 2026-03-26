# Phase 12: SpiritKernel — Hybrid Microkernel

**Roadmap Alignment:** Phase 2 — Kernel Development (Months 7–18)
**Estimated Duration:** 12 months
**Dependencies:** Phase 1 complete (Phases 01–11) — the kernel replaces the userspace shims
**Architecture Plan Sections:** §3, §4, §6, §3.4

---

## Context & Overview

This is the most ambitious phase: building a custom operating system kernel from scratch. SpiritKernel is a **hybrid microkernel** written in Rust, inspired by seL4 (formal verification), Zircon/Fuchsia (capability-based security), and Linux (driver ecosystem compatibility).

The kernel is "hybrid" because it keeps the core microkernel minimal (<50,000 lines of Rust + assembly) while running higher-level services (filesystem, networking, device management, inference coordination) as userspace servers that communicate via capability-based IPC.

Everything built in Phase 1 as userspace shims (memory zones via cgroups, scheduler via sched_setaffinity, KV-cache via userspace page tables) becomes native kernel functionality in this phase.

---

## Goals

1. **Bootloader** — UEFI boot on x86_64 and ARM64, basic console output
2. **Physical memory manager** — TensorFirst zones as native kernel memory regions
3. **Virtual memory** — Page tables with tensor-aware mapping (hugepage support, VRAM mapping)
4. **Thread scheduler** — Native multi-queue priority scheduler with inference phase awareness
5. **IPC** — Zero-copy, async, capability-based inter-process communication
6. **GPU driver** — NVIDIA and AMD GPU initialization, VRAM management, compute dispatch
7. **Inference syscalls** — Full `spirit_infer_*` and `spirit_tensor_*` syscall implementation
8. **Device drivers** — NVMe, NIC (virtio-net, e1000e), serial console, timer
9. **Self-hosting milestone** — SpiritKernel boots and runs Ollama natively

---

## Design Decisions

### DD-12.1: Rust with #![no_std] and Minimal Assembly
**Decision:** Kernel written in Rust with `#![no_std]`, thin assembly stubs for boot, context switch, interrupt vectors.
**Rationale:** Rust provides memory safety without GC, zero-cost abstractions, and an ecosystem for OS development. Assembly is minimized to platform-specific operations that Rust can't express.

### DD-12.2: Microkernel Core < 50K Lines
**Decision:** The microkernel core handles only: capability IPC, physical memory management, thread scheduling, interrupt handling, hardware abstraction (CPU, GPU doorbell, DMA). Everything else is a userspace server.
**Rationale:** Small trusted computing base (TCB) for security and auditability. Filesystem bugs can't crash the kernel. Driver bugs are contained to their process.

### DD-12.3: Capability-Based IPC (seL4-Inspired)
**Decision:** All IPC uses unforgeable capability tokens. No process can communicate with another without holding the appropriate capability.
**Rationale:** Eliminates ambient authority. A compromised process can only access resources it already had capabilities for. Capabilities can be revoked without restarting the system.

### DD-12.4: GPU Driver in Kernel Space (Exception to Microkernel Rule)
**Decision:** The GPU driver runs in kernel space, not as a userspace server.
**Rationale:** GPU performance is critical for inference. The overhead of crossing the kernel-userspace boundary for every GPU operation (kernel launch, memory transfer, completion notification) would negate the performance gains. The GPU driver is the one exception to the microkernel principle.

### DD-12.5: Linux Driver Shim for Non-Critical Devices
**Decision:** For non-critical devices (USB, audio, input), provide a translation shim that allows running Linux kernel driver modules in a sandboxed userspace process.
**Rationale:** Writing drivers for every device from scratch is infeasible. Linux has decades of driver coverage. A compatibility shim (similar to User-Mode Linux or Rump Kernels) lets us leverage this ecosystem while keeping drivers out of the kernel TCB.

---

## Components & Tasks

### Milestone M2.1: Bootloader (Months 7–8)

#### Boot Sequence
```
UEFI Firmware → Spirit Bootloader → Kernel Entry Point
    │                │                    │
    ├─ Load kernel   ├─ Parse memory     ├─ Set up GDT/IDT
    │  binary from   │  map from UEFI    ├─ Initialize BSS
    │  ESP partition  ├─ Set up initial   ├─ Set up initial
    │                │  page tables       │  kernel heap
    └─               └─ Jump to kernel   └─ Print "SpiritKernel booting..."
                       entry              └─ Initialize subsystems
```

**Tasks:**
- UEFI bootloader using the `bootloader` Rust crate or custom UEFI app
- x86_64 GDT, IDT, TSS setup in assembly
- Initial identity-mapped page tables
- Console output via UEFI framebuffer or serial
- ACPI table parsing for hardware enumeration
- ARM64 support (device tree parsing, EL2→EL1 transition)

### Milestone M2.2: Memory Manager (Months 9–10)

#### TensorFirst Physical Memory
```rust
pub struct PhysicalMemoryManager {
    zone_model: BuddyAllocator,    // Pinned, read-only after load
    zone_kv: PagedAllocator,       // Dynamic, LRU evictable
    zone_tensor: ArenaAllocator,   // Scratch, recycled between inference steps
    zone_general: BuddyAllocator,  // Standard allocation
    vram_managers: Vec<VramManager>,
}

impl PhysicalMemoryManager {
    pub fn alloc_model(&mut self, pages: usize) -> Result<PhysFrame>;
    pub fn alloc_kv_page(&mut self) -> Result<KvPage>;
    pub fn alloc_tensor(&mut self, size: usize) -> Result<TensorBuffer>;
    pub fn alloc_general(&mut self, pages: usize) -> Result<PhysFrame>;
}
```

**Tasks:**
- Physical frame allocator with zone partitioning (configured at boot from UEFI memory map)
- Virtual memory manager with 4-level page tables (x86_64) and hugepage support (2MB, 1GB)
- Kernel heap allocator (linked list allocator → slab allocator)
- GPU VRAM mapping via PCI BAR regions
- KV-cache page table (kernel data structure, not just a userspace abstraction)
- Memory-mapped I/O for device registers

### Milestone M2.3: Scheduler (Months 11–12)

```rust
pub struct InferenceScheduler {
    queues: [RunQueue; 5],  // P0 through P4
    per_cpu: Vec<PerCpuState>,
    phase: AtomicInferencePhase,
}

impl InferenceScheduler {
    pub fn schedule(&mut self, cpu: CpuId) -> Option<ThreadId>;
    pub fn hint_phase(&mut self, phase: InferencePhase);
    pub fn create_thread(&mut self, entry: fn(), tier: PriorityTier) -> ThreadId;
    pub fn block(&mut self, thread: ThreadId, reason: BlockReason);
    pub fn wake(&mut self, thread: ThreadId);
}
```

**Tasks:**
- Per-CPU run queues with priority tiers
- Context switching (save/restore registers, FPU/SSE/AVX state, page tables)
- Timer interrupt handler for preemption
- Phase-aware core allocation (prefill: all cores; decode: dedicated cores)
- Thread creation, blocking, waking primitives
- IPC-aware scheduling (boost thread when IPC message arrives)

### Milestone M2.4: GPU Driver (Months 13–15)

**The hardest milestone.** GPU drivers are complex, proprietary, and poorly documented.

**NVIDIA approach:**
- Use NVIDIA's open-source kernel modules (GPL, released 2022+) as reference
- Implement minimal GPU initialization: PCIe BAR mapping, GPU reset, firmware loading
- VRAM allocation via the kernel memory manager
- Compute dispatch: submit work to GPU command queues
- Interrupt handling: completion notifications from GPU
- **Fallback:** Use NVIDIA's proprietary driver via a compatibility shim (run the Linux driver in a sandboxed process)

**AMD approach:**
- AMD's AMDGPU driver is fully open source
- Port relevant portions to SpiritKernel
- ROCm runtime support via userspace libraries

**Tasks:**
- PCI bus enumeration and GPU detection
- GPU initialization (firmware load, display engine disable, compute-only mode)
- VRAM allocator (contiguous blocks for model weights, paged for KV-cache)
- Compute command submission (GPU command queues / doorbell registers)
- DMA engine control (CPU↔GPU memory transfers)
- Multi-GPU support (NVLink detection, peer-to-peer memory access)
- GPUDirect Storage support (NVMe→GPU direct path)

### Milestone M2.5: Inference Syscalls (Months 16–17)

```rust
// Kernel syscall implementations
pub fn sys_spirit_infer_create(model_id: ModelId, config: &InferConfig) -> Result<SessionHandle>;
pub fn sys_spirit_infer_prompt(session: SessionHandle, tokens: &[Token], flags: u32) -> Result<RequestId>;
pub fn sys_spirit_infer_recv(request_id: RequestId, buf: &mut [Token], max: usize, flags: u32) -> Result<usize>;
pub fn sys_spirit_infer_kv_snapshot(session: SessionHandle) -> Result<SnapshotHandle>;
pub fn sys_spirit_infer_kv_restore(session: SessionHandle, snapshot: SnapshotHandle) -> Result<()>;
pub fn sys_spirit_infer_batch(requests: &[BatchRequest]) -> Result<BatchId>;
pub fn sys_spirit_model_load(path: &str, policy: MemoryPolicy) -> Result<ModelId>;
pub fn sys_spirit_model_unload(model_id: ModelId) -> Result<()>;
pub fn sys_spirit_tensor_alloc(size: usize, device: Device, flags: u32) -> Result<TensorHandle>;
pub fn sys_spirit_tensor_free(handle: TensorHandle) -> Result<()>;
```

**Tasks:**
- Syscall dispatch table (x86_64 `syscall` instruction, ARM64 `svc`)
- POSIX-compatible syscalls (spirit_read/write/open/close/mmap/fork/exec)
- Inference domain syscalls (above)
- Capability domain syscalls (grant, revoke, check)
- Userspace ↔ kernel buffer management (copy_from_user, copy_to_user)
- Integration with GPU driver for inference execution

### Milestone M2.6: Self-Hosting (Month 18)

**The culmination:** SpiritKernel boots on real hardware, starts userspace servers (SpiritFS, networking, inference coordinator), loads Ollama, and runs inference end-to-end.

**Tasks:**
- NVMe driver (for loading models from disk)
- Network driver (virtio-net for VMs, e1000e for bare metal)
- Userspace process loading (ELF loader)
- IPC infrastructure (capability-based message passing)
- Port all Phase 1 userspace components to run on SpiritKernel
- End-to-end test: boot → load model → run inference → output tokens

---

## Acceptance Criteria

- [ ] M2.1: Kernel boots via UEFI and prints to screen on x86_64 hardware/VM
- [ ] M2.1: ARM64 boot works on QEMU aarch64
- [ ] M2.2: Physical memory manager allocates frames from correct zones
- [ ] M2.2: Virtual memory with 4-level paging works (userspace processes have isolated address spaces)
- [ ] M2.2: 2MB and 1GB hugepages work for model weight mapping
- [ ] M2.3: Threads are scheduled according to priority tiers
- [ ] M2.3: Context switching preserves all register state (including AVX-512)
- [ ] M2.3: Phase hints cause core reallocation within <0.3ms
- [ ] M2.4: GPU detected via PCI enumeration
- [ ] M2.4: VRAM allocated and accessible from kernel
- [ ] M2.4: GPU compute dispatch executes a simple kernel (matrix multiply)
- [ ] M2.5: All `spirit_infer_*` syscalls implemented and callable from userspace
- [ ] M2.5: POSIX syscall subset sufficient for basic Linux binary compatibility
- [ ] M2.6: Full boot-to-inference pipeline works on real hardware
- [ ] M2.6: Ollama starts and serves inference requests on SpiritKernel
- [ ] M2.6: Performance exceeds Phase 1 (containerized) by at least 30%

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| GPU driver is the hardest problem in OS development | M2.4 blocks everything after | Start with NVIDIA open-source modules as reference; have fallback plan (Linux driver in userspace shim) |
| Kernel development takes longer than 12 months | Schedule overrun | Prioritize self-hosting milestone; defer non-essential features (ARM64, multi-GPU) |
| Subtle kernel bugs (memory corruption, race conditions) | Instability, data loss | Extensive use of Rust safety; QEMU testing with ASAN; integration tests at every milestone |
| NVMe driver complexity | Can't load models | Start with virtio-blk (simpler) for VM testing; NVMe for bare metal later |
| POSIX syscall compatibility gaps | Linux binaries crash | Prioritize syscalls used by Rust runtime, Ollama, and llama.cpp; fuzz with common binaries |

---

## Key References

- Architecture Plan §3 (SpiritKernel — The Hybrid Microkernel)
- Architecture Plan §3.1 (Why a New Kernel Architecture)
- Architecture Plan §3.2 (Architecture: Hybrid Microkernel)
- Architecture Plan §3.3 (Implementation Language: Rust)
- Architecture Plan §3.4 (Syscall Interface)
- Architecture Plan §3.5 (Linux Compatibility Shim)
- Architecture Plan §4 (Memory Architecture — now kernel-native)
- Architecture Plan §6 (Scheduler — now kernel-native)
- Architecture Plan Appendix B (Hardware Requirements)
