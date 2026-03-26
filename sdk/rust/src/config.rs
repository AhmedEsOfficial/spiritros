//! Configuration for the Spirit SDK client.

use serde::{Deserialize, Serialize};

/// Configuration for connecting to a `SpiritROS` instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// The endpoint of the `SpiritROS` inference coordinator.
    /// Defaults to `http://127.0.0.1:8420`.
    pub endpoint: String,

    /// Connection timeout in milliseconds.
    pub connect_timeout_ms: u64,

    /// Request timeout in milliseconds.
    pub request_timeout_ms: u64,

    /// Optional API key for authenticated access.
    pub api_key: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:8420".to_string(),
            connect_timeout_ms: 5_000,
            request_timeout_ms: 30_000,
            api_key: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default_endpoint() {
        let config = Config::default();
        assert_eq!(config.endpoint, "http://127.0.0.1:8420");
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let config = Config {
            endpoint: "http://localhost:9000".to_string(),
            connect_timeout_ms: 1000,
            request_timeout_ms: 5000,
            api_key: Some("test-key".to_string()),
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.endpoint, config.endpoint);
        assert_eq!(deserialized.api_key, config.api_key);
    }
}
