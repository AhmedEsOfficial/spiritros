//! Spirit SDK client for communicating with `SpiritROS` services.

use crate::config::Config;
use crate::error::Error;
use crate::types::{InferenceRequest, SessionId, SystemStatus, TokenResponse};

/// Client for interacting with a `SpiritROS` instance.
///
/// Provides async methods for inference, model management, and system monitoring.
/// All communication goes through the inference coordinator's gRPC API.
#[derive(Debug)]
pub struct SpiritClient {
    config: Config,
    // PHASE1-SHIM: In Phase 1 this will use tonic::Channel to connect to the
    // inference coordinator daemon over a Unix domain socket or TCP.
}

impl SpiritClient {
    /// Connect to a `SpiritROS` instance with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `Error::ConnectionFailed` if the inference coordinator is unreachable.
    #[allow(clippy::unused_async)]
    pub async fn connect(config: Config) -> Result<Self, Error> {
        tracing::info!(endpoint = %config.endpoint, "connecting to `SpiritROS`");
        // PHASE1-SHIM: Establish gRPC channel to inference coordinator
        Ok(Self { config })
    }

    /// Return the endpoint this client is connected to.
    #[must_use]
    pub fn endpoint(&self) -> &str {
        &self.config.endpoint
    }

    /// Request a completion from the inference engine.
    ///
    /// Returns a session ID that can be used to receive streamed tokens.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not loaded or the system is at capacity.
    #[allow(clippy::unused_async)]
    pub async fn infer(&self, _request: InferenceRequest) -> Result<SessionId, Error> {
        // PHASE1-SHIM: Send inference request via gRPC to coordinator
        Err(Error::InferenceError(
            "not yet implemented — Phase 1 stub".to_string(),
        ))
    }

    /// Poll for the next token in a streaming inference session.
    ///
    /// # Errors
    ///
    /// Returns an error if the session does not exist or has already completed.
    #[allow(clippy::unused_async)]
    pub async fn next_token(&self, _session_id: SessionId) -> Result<TokenResponse, Error> {
        // PHASE1-SHIM: Receive next token from gRPC stream
        Err(Error::InferenceError(
            "not yet implemented — Phase 1 stub".to_string(),
        ))
    }

    /// Query the system status of the connected `SpiritROS` instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the coordinator is unreachable.
    #[allow(clippy::unused_async)]
    pub async fn system_status(&self) -> Result<SystemStatus, Error> {
        // PHASE1-SHIM: Query coordinator status endpoint
        Err(Error::ConnectionFailed(
            "not yet implemented — Phase 1 stub".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_connect_succeeds() {
        let config = Config::default();
        let client = SpiritClient::connect(config).await;
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_client_endpoint_matches_config() {
        let config = Config {
            endpoint: "http://10.0.0.1:8420".to_string(),
            ..Config::default()
        };
        let client = SpiritClient::connect(config).await.unwrap();
        assert_eq!(client.endpoint(), "http://10.0.0.1:8420");
    }

    #[tokio::test]
    async fn test_infer_returns_stub_error() {
        let client = SpiritClient::connect(Config::default()).await.unwrap();
        let req = InferenceRequest {
            model: "test".to_string(),
            prompt: "hello".to_string(),
            max_tokens: 10,
            temperature: 0.0,
        };
        let result = client.infer(req).await;
        assert!(result.is_err());
    }
}
