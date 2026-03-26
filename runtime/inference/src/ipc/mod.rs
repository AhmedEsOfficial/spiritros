//! IPC layer for the inference coordinator.
//!
//! Provides two transport mechanisms:
//!
//! - **Unix domain socket** (`unix_socket`): Reliable, ordered delivery for
//!   control messages (create session, prompt, status). Uses length-prefixed
//!   JSON messages.
//!
//! - **Shared memory** (`shm`): Zero-copy token streaming via a lock-free
//!   ring buffer in `/dev/shm/spirit_infer`. Used for high-throughput
//!   token transfer between the coordinator and local consumers.
//!
//! In Phase 1, the Unix socket is the primary IPC mechanism. Shared memory
//! is available as an optimization for latency-sensitive local consumers.

pub mod shm;
pub mod unix_socket;
