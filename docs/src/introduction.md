# Introduction

SpiritROS is an experiment in operating-system support for LLM inference. Rather than running inference workloads on top of a general-purpose OS that was designed for web servers and desktop applications, SpiritROS treats model weights, KV-caches, and inference sessions as first-class operating system resources.

## Why SpiritROS?

Modern LLM inference on stock Linux suffers from several fundamental mismatches:

- **Memory management** treats model weights the same as any other allocation. The OOM killer does not understand that evicting a 70B model from VRAM takes minutes to recover from.
- **Process scheduling** has no concept of inference priority. A background `apt update` can steal CPU cycles from token decoding.
- **Filesystem semantics** are purely syntactic. Finding "that document about quarterly revenue" requires knowing the exact filename.
- **Security models** grant ambient authority. Any process with GPU access can hijack inference sessions.

SpiritROS solves each of these by making inference a core OS concern rather than an afterthought.

## Project Status

SpiritROS is currently in **Phase 1** (Virtualized Deployment). The system runs inside a Docker container on a Linux host, using Ollama as the inference backend. Kernel-level features are approximated with userspace shims (cgroups, FUSE, cpuset). The goal of Phase 1 is to prove that the architecture delivers measurable performance gains -- targeting 30-60% improvement over stock Linux + Ollama -- while building all the userspace tooling that will survive into later phases.

## Key Components

- **Spirit Shell** -- An LLM-native command-line interface that understands natural language alongside traditional CLI commands.
- **SpiritFS** -- A semantic filesystem backed by FUSE that supports search by meaning, not just by filename.
- **Inference Coordinator** -- Manages model loading, session lifecycle, KV-cache sharing, and GPU resource allocation.
- **Memory Zone Manager** -- Enforces TensorFirst memory zones that protect inference workloads from general-purpose memory pressure.
- **Spirit Guard** -- Capability-based security system that governs access to inference, GPU, and agentic operations.
