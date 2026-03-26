//! # Spirit Security — Capability-Based Access Control
//!
//! This crate implements the `SpiritROS` security model (ADR-007), built on
//! capability-based access control with no ambient authority. Every IPC
//! message, resource access, and privileged operation requires an explicit
//! capability token.
//!
//! ## Key components
//!
//! - **Capabilities**: Fine-grained permission tokens (e.g., `CAP_GPU_COMPUTE`,
//!   `CAP_MODEL_LOAD`, `CAP_INFERENCE`, `CAP_AGENTIC`).
//! - **Spirit Guard**: The policy enforcement daemon that validates capability
//!   tokens on every privileged operation.
//! - **Audit trail**: Every security-relevant event is logged to an append-only
//!   audit log for forensic analysis.
//!
//! ## Invariants
//!
//! - No process can access GPU resources without `CAP_GPU_COMPUTE` (#5).
//! - Every IPC message uses the capability system (#4).
//! - The Spirit Shell never executes LLM-generated code without explicit
//!   user approval or an agent policy with `CAP_AGENTIC` (#7).
//! - All network-facing APIs require authentication by default (#8).

#![warn(clippy::all, clippy::pedantic)]

// ── Modules ──────────────────────────────────────────────────────────────────

/// Core type definitions for the security subsystem.
pub mod types {
    use serde::{Deserialize, Serialize};
    use uuid::Uuid;

    /// A capability token granting specific permissions.
    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct CapabilityToken {
        /// Unique token identifier.
        pub id: Uuid,
        /// The capability this token grants.
        pub capability: Capability,
        /// Process or session this token is bound to.
        pub principal: Principal,
        /// Whether this token can be delegated to child processes.
        pub delegatable: bool,
    }

    /// Well-known capabilities in `SpiritROS`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum Capability {
        /// Access GPU compute resources.
        GpuCompute,
        /// Load models into `ZONE_MODEL`.
        ModelLoad,
        /// Create and use inference sessions.
        Inference,
        /// Execute LLM-generated code (agent mode).
        Agentic,
        /// Access the network.
        Network,
        /// Mount or modify `SpiritFS`.
        Filesystem,
        /// Modify scheduling parameters.
        Scheduling,
        /// Access the audit log.
        AuditRead,
        /// Administer security policies.
        SecurityAdmin,
    }

    /// Identifies the entity holding a capability.
    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Principal {
        /// Unique principal identifier.
        pub id: Uuid,
        /// Human-readable name.
        pub name: String,
    }

    /// Outcome of a capability check.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum AccessDecision {
        /// Access granted.
        Allow,
        /// Access denied — capability missing or invalid.
        Deny,
    }
}

/// Error types for security operations.
pub mod error {
    use super::types::Capability;

    /// Errors that can occur during security operations.
    #[derive(Debug, thiserror::Error)]
    pub enum SecurityError {
        /// A required capability was not presented.
        #[error("Capability denied: {action} requires {capability:?}")]
        CapabilityDenied {
            action: String,
            capability: Capability,
        },

        /// The presented capability token is invalid or expired.
        #[error("Invalid capability token: {0}")]
        InvalidToken(String),

        /// Token delegation is not permitted.
        #[error("Token {token_id} is not delegatable")]
        DelegationDenied { token_id: uuid::Uuid },

        /// Audit log write failed.
        #[error("Audit log error: {0}")]
        AuditError(String),

        /// Configuration error.
        #[error("Security configuration error: {0}")]
        ConfigError(String),
    }
}

pub mod capabilities {
    //! Capability creation, validation, and management.
    //!
    //! Manages the lifecycle of capability tokens: creation, validation,
    //! delegation, and revocation.
    //!
    //! Tokens are cryptographically bound to their principal and cannot be
    //! forged. In Phase 1, tokens are HMAC-signed in-process. Phase 2+
    //! will use kernel-managed capabilities.
}

pub mod guard {
    //! Spirit Guard — the policy enforcement daemon.
    //!
    //! The Spirit Guard daemon intercepts all privileged operations and
    //! validates the presented capability tokens against the policy store.
    //!
    //! It runs as a separate process and communicates via Unix domain sockets.
    //! Every Spirit component must authenticate with the Guard before
    //! performing any capability-gated operation.
}

pub mod audit {
    //! Audit trail — append-only security event log.
    //!
    //! Records all security-relevant events: capability grants, denials,
    //! token delegations, policy changes, and security violations.
    //!
    //! The audit log is append-only and integrity-protected. In Phase 1 it
    //! writes to a local file; Phase 2+ stores events in the `SpiritKernel`'s
    //! secure audit ring buffer.

    use serde::{Deserialize, Serialize};
    use uuid::Uuid;

    /// A single entry in the security audit log.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AuditEvent {
        /// Unique event identifier.
        pub id: Uuid,
        /// Timestamp (Unix epoch, nanoseconds).
        pub timestamp_ns: u64,
        /// The principal involved.
        pub principal_id: Uuid,
        /// Human-readable description of the event.
        pub description: String,
        /// Whether the action was allowed or denied.
        pub outcome: super::types::AccessDecision,
    }
}

/// Configuration for the security subsystem.
pub mod config {
    use serde::{Deserialize, Serialize};

    /// Configuration for Spirit Guard and the security subsystem.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SecurityConfig {
        /// Path to the Unix domain socket for Spirit Guard IPC.
        pub guard_socket_path: String,
        /// Path to the audit log file.
        pub audit_log_path: String,
        /// Whether to enforce capabilities (false = permissive/audit-only mode).
        pub enforce: bool,
        /// Environment variable name holding the HMAC secret for token signing (Phase 1).
        pub token_secret_env: String,
    }

    impl Default for SecurityConfig {
        fn default() -> Self {
            Self {
                guard_socket_path: "/run/spirit/guard.sock".to_string(),
                audit_log_path: "/var/log/spirit/audit.log".to_string(),
                enforce: true,
                token_secret_env: "SPIRIT_TOKEN_SECRET".to_string(),
            }
        }
    }
}

// ── Re-exports ───────────────────────────────────────────────────────────────

pub use config::SecurityConfig;
pub use error::SecurityError;
pub use types::{AccessDecision, Capability, CapabilityToken, Principal};

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_capability_variants_are_distinct() {
        let caps = [
            Capability::GpuCompute,
            Capability::ModelLoad,
            Capability::Inference,
            Capability::Agentic,
            Capability::Network,
            Capability::Filesystem,
            Capability::Scheduling,
            Capability::AuditRead,
            Capability::SecurityAdmin,
        ];
        for (i, a) in caps.iter().enumerate() {
            for (j, b) in caps.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_capability_token_serialization_roundtrip() {
        let token = CapabilityToken {
            id: Uuid::new_v4(),
            capability: Capability::GpuCompute,
            principal: Principal {
                id: Uuid::new_v4(),
                name: "test-process".to_string(),
            },
            delegatable: false,
        };
        let json = serde_json::to_string(&token).expect("serialize");
        let back: CapabilityToken = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(token.id, back.id);
        assert_eq!(token.capability, back.capability);
    }

    #[test]
    fn test_security_config_default_enforces() {
        let cfg = SecurityConfig::default();
        assert!(cfg.enforce);
        assert!(cfg.guard_socket_path.contains("guard.sock"));
    }

    #[test]
    fn test_access_decision_equality() {
        assert_eq!(AccessDecision::Allow, AccessDecision::Allow);
        assert_ne!(AccessDecision::Allow, AccessDecision::Deny);
    }

    #[test]
    fn test_security_error_display_capability_denied() {
        let err = SecurityError::CapabilityDenied {
            action: "load_model".to_string(),
            capability: Capability::ModelLoad,
        };
        let msg = err.to_string();
        assert!(msg.contains("load_model"));
        assert!(msg.contains("ModelLoad"));
    }

    #[test]
    fn test_audit_event_construction() {
        let event = audit::AuditEvent {
            id: Uuid::new_v4(),
            timestamp_ns: 1_700_000_000_000_000_000,
            principal_id: Uuid::new_v4(),
            description: "Attempted GPU access".to_string(),
            outcome: AccessDecision::Deny,
        };
        assert_eq!(event.outcome, AccessDecision::Deny);
    }
}
