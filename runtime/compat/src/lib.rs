//! # Spirit Compat — Compatibility Layers
//!
//! This crate provides Linux binary compatibility for `SpiritROS`, inspired by
//! gVisor's syscall interception approach (ADR-010). In Phase 1 this is
//! minimal — since we run on a real Linux host, most compatibility is
//! automatic. The layer primarily:
//!
//! - Intercepts and audits syscalls from untrusted processes.
//! - Translates Spirit-specific filesystem paths (e.g., `/spirit/models/`)
//!   to their real host locations.
//! - Enforces capability restrictions on compatibility-mode processes.
//!
//! In Phase 2+, this becomes the primary Linux ABI compatibility layer
//! for the `SpiritKernel`, intercepting all POSIX syscalls and translating
//! them to Spirit-native operations.
//!
//! // PHASE1-SHIM: Thin audit/translation layer on real Linux.

#![warn(clippy::all, clippy::pedantic)]

// ── Modules ──────────────────────────────────────────────────────────────────

/// Core type definitions for the compatibility layer.
pub mod types {
    use serde::{Deserialize, Serialize};

    /// Represents a syscall intercepted by the compatibility layer.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InterceptedSyscall {
        /// Syscall number.
        pub number: i64,
        /// Human-readable syscall name (if known).
        pub name: Option<String>,
        /// PID of the calling process.
        pub pid: u32,
        /// Whether the syscall was allowed, denied, or translated.
        pub disposition: SyscallDisposition,
    }

    /// How the compatibility layer handled an intercepted syscall.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum SyscallDisposition {
        /// Passed through to the host kernel unchanged.
        Allowed,
        /// Blocked due to capability restrictions.
        Denied,
        /// Arguments were translated (e.g., path rewriting).
        Translated,
    }
}

/// Error types for compatibility operations.
pub mod error {
    /// Errors that can occur in the compatibility layer.
    #[derive(Debug, thiserror::Error)]
    pub enum CompatError {
        /// Syscall interception setup failed.
        #[error("Syscall interception error: {0}")]
        InterceptionError(String),

        /// Path translation failed.
        #[error("Path translation failed for '{path}': {reason}")]
        PathTranslationError { path: String, reason: String },

        /// A syscall was denied by capability policy.
        #[error("Syscall {syscall} denied for pid {pid}: insufficient capabilities")]
        SyscallDenied { syscall: String, pid: u32 },

        /// Configuration error.
        #[error("Compat configuration error: {0}")]
        ConfigError(String),

        /// Underlying OS error.
        #[error("OS error: {0}")]
        OsError(#[from] nix::Error),
    }
}

pub mod linux_compat {
    //! Linux compatibility — syscall translation and path mapping.
    //!
    //! Provides the core syscall interception and translation logic.
    //!
    //! In Phase 1, this primarily handles path rewrites for Spirit-specific
    //! virtual paths (e.g., `/spirit/models/<name>` -> actual model path).
    //!
    //! In Phase 2+, this will implement the full gVisor-inspired syscall
    //! translation layer inside the `SpiritKernel`.

    /// Well-known Spirit virtual path prefixes.
    pub const SPIRIT_MODELS_PREFIX: &str = "/spirit/models/";
    /// Virtual path for the KV-cache directory.
    pub const SPIRIT_CACHE_PREFIX: &str = "/spirit/cache/";
    /// Virtual path for `SpiritFS` mount.
    pub const SPIRIT_FS_PREFIX: &str = "/spirit/fs/";
}

/// Configuration for the compatibility layer.
pub mod config {
    use serde::{Deserialize, Serialize};

    /// Configuration for the compatibility layer.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CompatConfig {
        /// Whether syscall auditing is enabled.
        pub audit_syscalls: bool,
        /// Host path where models are stored.
        pub host_models_dir: String,
        /// Host path for the KV-cache.
        pub host_cache_dir: String,
        /// List of syscall numbers to block in compat mode.
        pub blocked_syscalls: Vec<i64>,
    }

    impl Default for CompatConfig {
        fn default() -> Self {
            Self {
                audit_syscalls: true,
                host_models_dir: "/var/lib/spirit/models".to_string(),
                host_cache_dir: "/var/lib/spirit/cache".to_string(),
                blocked_syscalls: Vec::new(),
            }
        }
    }
}

// ── Re-exports ───────────────────────────────────────────────────────────────

pub use config::CompatConfig;
pub use error::CompatError;
pub use types::{InterceptedSyscall, SyscallDisposition};

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compat_config_default_enables_auditing() {
        let cfg = CompatConfig::default();
        assert!(cfg.audit_syscalls);
        assert!(cfg.blocked_syscalls.is_empty());
    }

    #[test]
    fn test_syscall_disposition_equality() {
        assert_eq!(SyscallDisposition::Allowed, SyscallDisposition::Allowed);
        assert_ne!(SyscallDisposition::Allowed, SyscallDisposition::Denied);
    }

    #[test]
    fn test_spirit_virtual_paths_are_absolute() {
        assert!(linux_compat::SPIRIT_MODELS_PREFIX.starts_with('/'));
        assert!(linux_compat::SPIRIT_CACHE_PREFIX.starts_with('/'));
        assert!(linux_compat::SPIRIT_FS_PREFIX.starts_with('/'));
    }

    #[test]
    fn test_intercepted_syscall_serialization_roundtrip() {
        let syscall = InterceptedSyscall {
            number: 1,
            name: Some("write".to_string()),
            pid: 1234,
            disposition: SyscallDisposition::Allowed,
        };
        let json = serde_json::to_string(&syscall).expect("serialize");
        let back: InterceptedSyscall = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(syscall.number, back.number);
        assert_eq!(syscall.disposition, back.disposition);
    }

    #[test]
    fn test_compat_error_display() {
        let err = CompatError::SyscallDenied {
            syscall: "mount".to_string(),
            pid: 42,
        };
        let msg = err.to_string();
        assert!(msg.contains("mount"));
        assert!(msg.contains("42"));
    }
}
