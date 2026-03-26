//! # Spirit Networking — Network & API Layer
//!
//! This crate provides the network-facing API surface for `SpiritROS`. It exposes
//! both an HTTP REST API (OpenAI-compatible for inference) and a gRPC API for
//! internal component communication and mesh coordination.
//!
//! ## Security invariant
//!
//! All network-facing APIs require authentication by default (invariant #8).
//! The inference API binds to `127.0.0.1` unless explicitly configured for
//! network access with mTLS or API keys.
//!
//! ## Architecture
//!
//! - **API layer**: `hyper` + `tower` middleware stack for HTTP, `tonic` for gRPC.
//! - **Mesh**: Peer discovery and coordination for multi-node inference.
//! - **Auth**: API key validation, mTLS certificate checking.

#![warn(clippy::all, clippy::pedantic)]

// ── Modules ──────────────────────────────────────────────────────────────────

/// Core type definitions for networking.
pub mod types {
    use serde::{Deserialize, Serialize};

    /// Bind address specification for a network service.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BindAddress {
        /// IP address to bind to.
        pub host: String,
        /// Port number.
        pub port: u16,
    }

    impl BindAddress {
        /// Returns the address as a `host:port` string.
        #[must_use]
        pub fn to_socket_addr_string(&self) -> String {
            format!("{}:{}", self.host, self.port)
        }
    }

    /// Authentication method for API access.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum AuthMethod {
        /// No authentication (localhost only).
        None,
        /// API key in the `Authorization: Bearer <key>` header.
        ApiKey,
        /// Mutual TLS with client certificate validation.
        #[serde(rename = "mtls")]
        MutualTls,
    }

    /// Status of a mesh peer.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum PeerStatus {
        /// Peer is reachable and healthy.
        Healthy,
        /// Peer is reachable but degraded.
        Degraded,
        /// Peer is unreachable.
        Unreachable,
    }
}

/// Error types for networking operations.
pub mod error {
    /// Errors that can occur in the networking layer.
    #[derive(Debug, thiserror::Error)]
    pub enum NetworkError {
        /// Failed to bind to the specified address.
        #[error("Failed to bind to {address}: {reason}")]
        BindError { address: String, reason: String },

        /// Authentication failed.
        #[error("Authentication failed: {0}")]
        AuthError(String),

        /// Mesh peer communication failed.
        #[error("Mesh peer {peer} unreachable: {reason}")]
        PeerError { peer: String, reason: String },

        /// Request routing error.
        #[error("Request routing error: {0}")]
        RoutingError(String),

        /// Configuration error.
        #[error("Network configuration error: {0}")]
        ConfigError(String),

        /// Underlying I/O error.
        #[error("I/O error: {0}")]
        IoError(#[from] std::io::Error),

        /// HTTP protocol error.
        #[error("HTTP error: {0}")]
        HttpError(#[from] hyper::Error),
    }
}

pub mod api {
    //! HTTP and gRPC API endpoint definitions.
    //!
    //! Defines the public API surface for `SpiritROS`.
    //!
    //! - **`/v1/chat/completions`**: OpenAI-compatible chat API (proxied to
    //!   the inference coordinator).
    //! - **`/v1/embeddings`**: Embedding generation API.
    //! - **`/v1/models`**: Model listing and management.
    //! - **`/spirit/health`**: Health check endpoint.
    //! - **`/spirit/metrics`**: Prometheus-compatible metrics.
    //!
    //! gRPC services are defined in `.proto` files and generated at build
    //! time via `tonic-build`.

    /// Well-known API path constants.
    pub const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";
    /// Embeddings endpoint path.
    pub const EMBEDDINGS_PATH: &str = "/v1/embeddings";
    /// Model listing endpoint path.
    pub const MODELS_PATH: &str = "/v1/models";
    /// Health check endpoint path.
    pub const HEALTH_PATH: &str = "/spirit/health";
    /// Metrics endpoint path.
    pub const METRICS_PATH: &str = "/spirit/metrics";
}

pub mod mesh {
    //! Mesh coordination for multi-node inference.
    //!
    //! Peer discovery, health checking, and work distribution for
    //! multi-node inference deployments.
    //!
    //! In Phase 1, mesh is optional — single-node is the default.
    //! When enabled, peers coordinate via gRPC to distribute large
    //! model inference across multiple machines.
}

/// Configuration for the networking subsystem.
pub mod config {
    use super::types::{AuthMethod, BindAddress};
    use serde::{Deserialize, Serialize};

    /// Configuration for the Spirit networking layer.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct NetworkConfig {
        /// HTTP API bind address.
        pub http_bind: BindAddress,
        /// gRPC API bind address.
        pub grpc_bind: BindAddress,
        /// Authentication method for the HTTP API.
        pub auth_method: AuthMethod,
        /// Whether to enable mesh coordination.
        pub mesh_enabled: bool,
        /// Known mesh peer addresses.
        pub mesh_peers: Vec<String>,
    }

    impl Default for NetworkConfig {
        fn default() -> Self {
            Self {
                http_bind: BindAddress {
                    host: "127.0.0.1".to_string(),
                    port: 8080,
                },
                grpc_bind: BindAddress {
                    host: "127.0.0.1".to_string(),
                    port: 50051,
                },
                auth_method: AuthMethod::None,
                mesh_enabled: false,
                mesh_peers: Vec::new(),
            }
        }
    }
}

// ── Re-exports ───────────────────────────────────────────────────────────────

pub use config::NetworkConfig;
pub use error::NetworkError;
pub use types::{AuthMethod, BindAddress, PeerStatus};

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_config_default_binds_to_localhost() {
        let cfg = NetworkConfig::default();
        assert_eq!(cfg.http_bind.host, "127.0.0.1");
        assert_eq!(cfg.grpc_bind.host, "127.0.0.1");
    }

    #[test]
    fn test_bind_address_to_string() {
        let addr = BindAddress {
            host: "0.0.0.0".to_string(),
            port: 8080,
        };
        assert_eq!(addr.to_socket_addr_string(), "0.0.0.0:8080");
    }

    #[test]
    fn test_api_paths_are_absolute() {
        assert!(api::CHAT_COMPLETIONS_PATH.starts_with('/'));
        assert!(api::EMBEDDINGS_PATH.starts_with('/'));
        assert!(api::MODELS_PATH.starts_with('/'));
        assert!(api::HEALTH_PATH.starts_with('/'));
        assert!(api::METRICS_PATH.starts_with('/'));
    }

    #[test]
    fn test_peer_status_equality() {
        assert_eq!(PeerStatus::Healthy, PeerStatus::Healthy);
        assert_ne!(PeerStatus::Healthy, PeerStatus::Unreachable);
    }

    #[test]
    fn test_auth_method_serialization_roundtrip() {
        let method = AuthMethod::ApiKey;
        let json = serde_json::to_string(&method).expect("serialize");
        let back: AuthMethod = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(format!("{method:?}"), format!("{back:?}"));
    }

    #[test]
    fn test_network_error_display() {
        let err = NetworkError::BindError {
            address: "0.0.0.0:80".to_string(),
            reason: "permission denied".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("0.0.0.0:80"));
        assert!(msg.contains("permission denied"));
    }
}
