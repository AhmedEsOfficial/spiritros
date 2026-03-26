//! Error types for the Spirit SDK.

use thiserror::Error;

/// Errors returned by Spirit SDK operations.
#[derive(Debug, Error)]
pub enum Error {
    /// Failed to connect to the `SpiritROS` inference coordinator.
    #[error("connection failed: {0}")]
    ConnectionFailed(String),

    /// The requested model was not found in the registry.
    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// An inference session error occurred.
    #[error("inference error: {0}")]
    InferenceError(String),

    /// The request was denied due to insufficient capabilities.
    #[error("capability denied: {action} requires {capability}")]
    CapabilityDenied {
        /// The action that was attempted.
        action: String,
        /// The capability that was missing.
        capability: String,
    },

    /// A gRPC transport error.
    #[error("transport error: {0}")]
    Transport(#[from] tonic::transport::Error),

    /// A gRPC status error.
    #[error("rpc error: {0}")]
    Rpc(#[from] tonic::Status),

    /// An I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
