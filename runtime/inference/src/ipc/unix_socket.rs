//! Unix domain socket IPC server for the inference coordinator.
//!
//! Provides a JSON-over-UDS protocol for local IPC between Spirit components
//! and the inference coordinator. Messages are length-prefixed (4-byte big-endian
//! length header followed by JSON payload).
//!
//! # Protocol
//!
//! ```text
//! Client → Server:
//!   [4 bytes: payload length (big-endian)] [JSON payload]
//!
//! Server → Client:
//!   [4 bytes: payload length (big-endian)] [JSON payload]
//! ```
//!
//! # Operations
//!
//! - `CreateSession` — Create a new inference session
//! - `Prompt` — Submit a prompt for generation
//! - `Status` — Query session or system status
//! - `Snapshot` — Take a KV-cache snapshot
//! - `Restore` — Restore a KV-cache snapshot
//! - `LoadModel` — Load a model
//! - `UnloadModel` — Unload a model
//! - `ListModels` — List available models
//! - `DestroySession` — Destroy a session

use serde::{Deserialize, Serialize};

use crate::types::{Priority, SamplingConfig, SessionId, SessionState};

/// Maximum message size (16 MiB).
pub const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024;

/// IPC request message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", content = "data")]
pub enum IpcRequest {
    /// Create a new inference session.
    CreateSession {
        model: String,
        priority: Option<Priority>,
    },

    /// Submit a prompt for generation.
    Prompt {
        session_id: SessionId,
        prompt: String,
        system: Option<String>,
        sampling: Option<SamplingConfig>,
        stream: bool,
    },

    /// Query session status.
    SessionStatus { session_id: SessionId },

    /// Query system status.
    SystemStatus,

    /// Take a KV-cache snapshot.
    Snapshot { session_id: SessionId },

    /// Restore a KV-cache snapshot.
    Restore {
        session_id: SessionId,
        snapshot_id: String,
    },

    /// Load a model.
    LoadModel { name: String },

    /// Unload a model.
    UnloadModel { name: String },

    /// List available models.
    ListModels,

    /// Destroy a session.
    DestroySession { session_id: SessionId },
}

/// IPC response message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", content = "data")]
pub enum IpcResponse {
    /// Operation succeeded.
    Ok(IpcResponseData),

    /// Operation failed.
    Error { code: String, message: String },
}

/// Response data payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum IpcResponseData {
    /// Session created.
    SessionCreated { session_id: SessionId },

    /// Generation started (tokens will stream).
    GenerationStarted { session_id: SessionId },

    /// Complete generation result (non-streaming).
    GenerationComplete {
        session_id: SessionId,
        text: String,
        tokens_generated: u32,
        first_token_ms: u64,
        total_ms: u64,
        tokens_per_second: f32,
    },

    /// Session status.
    SessionInfo {
        session_id: SessionId,
        state: SessionState,
        model: String,
        kv_token_count: usize,
        total_tokens_generated: u64,
    },

    /// System status.
    SystemInfo {
        active_sessions: usize,
        max_sessions: usize,
        loaded_models: Vec<String>,
        queue_depth: usize,
    },

    /// KV-cache snapshot taken.
    SnapshotCreated {
        snapshot_id: String,
        session_id: SessionId,
    },

    /// KV-cache restored.
    SnapshotRestored { session_id: SessionId },

    /// Model loaded.
    ModelLoaded { name: String },

    /// Model unloaded.
    ModelUnloaded { name: String },

    /// Model list.
    Models { models: Vec<ModelListEntry> },

    /// Session destroyed.
    SessionDestroyed { session_id: SessionId },

    /// Streaming token.
    Token {
        session_id: SessionId,
        text: String,
        done: bool,
    },
}

/// Entry in the model list response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListEntry {
    pub name: String,
    pub size_bytes: u64,
    pub loaded: bool,
    pub digest: String,
}

/// Encode a message for wire transport (length-prefixed JSON).
///
/// Returns the serialized bytes: 4-byte big-endian length + JSON payload.
pub fn encode_message<T: Serialize>(msg: &T) -> Result<Vec<u8>, serde_json::Error> {
    let json = serde_json::to_vec(msg)?;
    let len = json.len() as u32;
    let mut buf = Vec::with_capacity(4 + json.len());
    buf.extend_from_slice(&len.to_be_bytes());
    buf.extend_from_slice(&json);
    Ok(buf)
}

/// Decode a length-prefixed JSON message.
///
/// The input must start with a 4-byte big-endian length header.
/// Returns the parsed message and the number of bytes consumed.
pub fn decode_message<T: for<'de> Deserialize<'de>>(
    buf: &[u8],
) -> Result<Option<(T, usize)>, MessageError> {
    if buf.len() < 4 {
        return Ok(None); // Need more data
    }

    let len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;

    if len > MAX_MESSAGE_SIZE {
        return Err(MessageError::TooLarge(len));
    }

    if buf.len() < 4 + len {
        return Ok(None); // Need more data
    }

    let msg: T = serde_json::from_slice(&buf[4..4 + len]).map_err(MessageError::Json)?;
    Ok(Some((msg, 4 + len)))
}

/// Errors during message encoding/decoding.
#[derive(Debug, thiserror::Error)]
pub enum MessageError {
    #[error("Message too large: {0} bytes (max {MAX_MESSAGE_SIZE})")]
    TooLarge(usize),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Unix domain socket IPC server.
///
/// Listens on a Unix socket and dispatches incoming requests to the
/// coordinator via a channel.
pub struct IpcServer {
    /// Socket path.
    socket_path: String,
}

impl IpcServer {
    /// Create a new IPC server.
    #[must_use]
    pub fn new(socket_path: String) -> Self {
        Self { socket_path }
    }

    /// Get the socket path.
    #[must_use]
    pub fn socket_path(&self) -> &str {
        &self.socket_path
    }

    /// Start the IPC server.
    ///
    /// Accepts connections and dispatches requests. Each connection is
    /// handled in a separate task.
    ///
    /// The `handler` function is called for each request and must return
    /// a response.
    pub async fn serve<F, Fut>(&self, handler: F) -> Result<(), crate::error::InferenceError>
    where
        F: Fn(IpcRequest) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = IpcResponse> + Send + 'static,
    {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::UnixListener;
        use tracing::{debug, error, info};

        // Remove stale socket file
        let _ = std::fs::remove_file(&self.socket_path);

        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(&self.socket_path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let listener = UnixListener::bind(&self.socket_path).map_err(|e| {
            crate::error::InferenceError::SocketError(format!(
                "Failed to bind {}: {e}",
                self.socket_path
            ))
        })?;

        info!(path = %self.socket_path, "IPC server listening");

        let handler = std::sync::Arc::new(handler);

        loop {
            let (stream, _addr) = listener.accept().await.map_err(|e| {
                crate::error::InferenceError::SocketError(format!("Accept failed: {e}"))
            })?;

            debug!("New IPC connection");

            let handler = handler.clone();
            tokio::spawn(async move {
                let (mut reader, mut writer) = stream.into_split();
                let mut buf = vec![0u8; 64 * 1024];
                let mut data = Vec::new();

                loop {
                    match reader.read(&mut buf).await {
                        Ok(0) => break, // Connection closed
                        Ok(n) => {
                            data.extend_from_slice(&buf[..n]);

                            // Process all complete messages in the buffer
                            loop {
                                match decode_message::<IpcRequest>(&data) {
                                    Ok(Some((request, consumed))) => {
                                        data.drain(..consumed);

                                        let response = handler(request).await;

                                        match encode_message(&response) {
                                            Ok(encoded) => {
                                                if let Err(e) = writer.write_all(&encoded).await {
                                                    error!("Write error: {e}");
                                                    return;
                                                }
                                            }
                                            Err(e) => {
                                                error!("Encode error: {e}");
                                                return;
                                            }
                                        }
                                    }
                                    Ok(None) => break, // Need more data
                                    Err(e) => {
                                        error!("Decode error: {e}");
                                        let err_resp = IpcResponse::Error {
                                            code: "decode_error".into(),
                                            message: e.to_string(),
                                        };
                                        if let Ok(encoded) = encode_message(&err_resp) {
                                            let _ = writer.write_all(&encoded).await;
                                        }
                                        return;
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            error!("Read error: {e}");
                            break;
                        }
                    }
                }
            });
        }
    }
}

impl Drop for IpcServer {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let req = IpcRequest::CreateSession {
            model: "llama3:8b".into(),
            priority: Some(Priority::Normal),
        };

        let encoded = encode_message(&req).unwrap();
        let (decoded, consumed) = decode_message::<IpcRequest>(&encoded).unwrap().unwrap();

        assert_eq!(consumed, encoded.len());
        match decoded {
            IpcRequest::CreateSession { model, .. } => {
                assert_eq!(model, "llama3:8b");
            }
            _ => panic!("Wrong request type"),
        }
    }

    #[test]
    fn test_decode_partial_returns_none() {
        // Only 2 bytes — not enough for the length header
        let result = decode_message::<IpcRequest>(&[0, 0]).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_decode_incomplete_payload_returns_none() {
        // Length says 100 bytes but only 10 in buffer
        let mut buf = vec![0u8; 14];
        buf[..4].copy_from_slice(&100u32.to_be_bytes());
        let result = decode_message::<IpcRequest>(&buf).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_decode_too_large_returns_error() {
        let mut buf = vec![0u8; 8];
        let huge = (MAX_MESSAGE_SIZE as u32 + 1).to_be_bytes();
        buf[..4].copy_from_slice(&huge);
        let result = decode_message::<IpcRequest>(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_response_serialization() {
        let resp = IpcResponse::Ok(IpcResponseData::SessionCreated {
            session_id: SessionId::new(),
        });
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("Ok"));
    }

    #[test]
    fn test_error_response_serialization() {
        let resp = IpcResponse::Error {
            code: "not_found".into(),
            message: "Session not found".into(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("not_found"));
    }

    #[test]
    fn test_all_request_types_serialize() {
        let requests = vec![
            IpcRequest::SystemStatus,
            IpcRequest::ListModels,
            IpcRequest::CreateSession {
                model: "m".into(),
                priority: None,
            },
            IpcRequest::Prompt {
                session_id: SessionId::new(),
                prompt: "hello".into(),
                system: None,
                sampling: None,
                stream: false,
            },
            IpcRequest::DestroySession {
                session_id: SessionId::new(),
            },
        ];

        for req in &requests {
            let json = serde_json::to_string(req).unwrap();
            assert!(!json.is_empty());
        }
    }

    #[test]
    fn test_ipc_server_creation() {
        let server = IpcServer::new("/tmp/test-infer.sock".into());
        assert_eq!(server.socket_path(), "/tmp/test-infer.sock");
    }
}
