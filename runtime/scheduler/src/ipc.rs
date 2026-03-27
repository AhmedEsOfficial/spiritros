//! IPC server for receiving scheduler commands.
//!
//! Listens on a Unix domain socket at `/run/spirit/scheduler.sock` for
//! phase hints, process registration, and status queries. Uses
//! length-prefixed JSON messages for simplicity (sub-millisecond
//! latency on localhost).
//!
//! Protocol: `[4-byte LE length][JSON payload]`
//!
//! // PHASE1-SHIM: Phase 2 replaces this with kernel IPC (capability-gated).

use std::path::{Path, PathBuf};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;
use tokio::sync::mpsc;
use tracing::{debug, error, info};

use crate::error::{Result, SchedulerError};
use crate::types::{SchedulerMessage, SchedulerResponse};

/// Commands forwarded from the IPC server to the scheduler main loop.
pub struct IpcCommand {
    /// The parsed message.
    pub message: SchedulerMessage,
    /// Channel to send the response back to the client.
    pub reply_tx: tokio::sync::oneshot::Sender<SchedulerResponse>,
}

/// IPC server that listens for scheduler commands.
pub struct IpcServer {
    socket_path: PathBuf,
}

impl IpcServer {
    /// Create a new IPC server.
    #[must_use]
    pub fn new(socket_path: &str) -> Self {
        Self {
            socket_path: PathBuf::from(socket_path),
        }
    }

    /// Start listening for connections.
    ///
    /// Returns a receiver that yields incoming commands. Each command
    /// includes a oneshot channel for sending the response.
    ///
    /// # Errors
    ///
    /// Returns `SchedulerError::IpcError` if the socket directory cannot be
    /// created, a stale socket cannot be removed, or binding fails.
    pub async fn start(&self, command_tx: mpsc::Sender<IpcCommand>) -> Result<()> {
        // Ensure parent directory exists
        if let Some(parent) = self.socket_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                SchedulerError::IpcError(format!("create dir {}: {e}", parent.display()))
            })?;
        }

        // Remove stale socket
        if self.socket_path.exists() {
            tokio::fs::remove_file(&self.socket_path)
                .await
                .map_err(|e| {
                    SchedulerError::IpcError(format!(
                        "remove stale socket {}: {e}",
                        self.socket_path.display()
                    ))
                })?;
        }

        let listener = UnixListener::bind(&self.socket_path).map_err(|e| {
            SchedulerError::IpcError(format!("bind {}: {e}", self.socket_path.display()))
        })?;

        info!(
            path = %self.socket_path.display(),
            "Scheduler IPC server listening"
        );

        let socket_path = self.socket_path.clone();
        tokio::spawn(async move {
            Self::accept_loop(listener, command_tx).await;
            // Clean up socket on shutdown
            let _ = tokio::fs::remove_file(&socket_path).await;
        });

        Ok(())
    }

    async fn accept_loop(listener: UnixListener, command_tx: mpsc::Sender<IpcCommand>) {
        loop {
            match listener.accept().await {
                Ok((stream, _addr)) => {
                    let tx = command_tx.clone();
                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_connection(stream, tx).await {
                            debug!(error = %e, "IPC connection error");
                        }
                    });
                }
                Err(e) => {
                    error!(error = %e, "IPC accept error");
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            }
        }
    }

    async fn handle_connection(
        mut stream: tokio::net::UnixStream,
        command_tx: mpsc::Sender<IpcCommand>,
    ) -> Result<()> {
        // Read length-prefixed message
        let len = stream
            .read_u32_le()
            .await
            .map_err(|e| SchedulerError::IpcError(format!("read length: {e}")))?;

        if len > 65536 {
            return Err(SchedulerError::IpcError(format!(
                "message too large: {len} bytes"
            )));
        }

        let mut buf = vec![0u8; len as usize]; // len is bounded to <= 65536 above
        stream
            .read_exact(&mut buf)
            .await
            .map_err(|e| SchedulerError::IpcError(format!("read payload: {e}")))?;

        let message: SchedulerMessage = serde_json::from_slice(&buf)
            .map_err(|e| SchedulerError::IpcError(format!("parse message: {e}")))?;

        debug!(message = ?message, "IPC message received");

        // Forward to scheduler and wait for response
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        command_tx
            .send(IpcCommand { message, reply_tx })
            .await
            .map_err(|_| {
                SchedulerError::IpcError("scheduler command channel closed".to_string())
            })?;

        let response = reply_rx.await.map_err(|_| {
            SchedulerError::IpcError("scheduler response channel closed".to_string())
        })?;

        // Write response
        let response_bytes = serde_json::to_vec(&response)
            .map_err(|e| SchedulerError::IpcError(format!("serialize response: {e}")))?;

        #[allow(clippy::cast_possible_truncation)] // response is always small JSON
        let response_len = response_bytes.len() as u32;
        stream
            .write_u32_le(response_len)
            .await
            .map_err(|e| SchedulerError::IpcError(format!("write response length: {e}")))?;
        stream
            .write_all(&response_bytes)
            .await
            .map_err(|e| SchedulerError::IpcError(format!("write response: {e}")))?;

        Ok(())
    }

    /// Returns the socket path.
    #[must_use]
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }
}

/// Client for sending messages to the scheduler IPC server.
pub struct IpcClient {
    socket_path: PathBuf,
}

impl IpcClient {
    /// Create a new IPC client.
    #[must_use]
    pub fn new(socket_path: &str) -> Self {
        Self {
            socket_path: PathBuf::from(socket_path),
        }
    }

    /// Send a message to the scheduler and wait for a response.
    ///
    /// # Errors
    ///
    /// Returns `SchedulerError::IpcError` if connecting, sending, or
    /// receiving the response fails.
    pub async fn send(&self, message: &SchedulerMessage) -> Result<SchedulerResponse> {
        let mut stream = tokio::net::UnixStream::connect(&self.socket_path)
            .await
            .map_err(|e| {
                SchedulerError::IpcError(format!("connect to {}: {e}", self.socket_path.display()))
            })?;

        let payload = serde_json::to_vec(message)
            .map_err(|e| SchedulerError::IpcError(format!("serialize: {e}")))?;

        #[allow(clippy::cast_possible_truncation)] // payload is always small JSON
        let payload_len = payload.len() as u32;
        stream
            .write_u32_le(payload_len)
            .await
            .map_err(|e| SchedulerError::IpcError(format!("write length: {e}")))?;
        stream
            .write_all(&payload)
            .await
            .map_err(|e| SchedulerError::IpcError(format!("write payload: {e}")))?;

        // Read response
        let len = stream
            .read_u32_le()
            .await
            .map_err(|e| SchedulerError::IpcError(format!("read response length: {e}")))?;

        let mut response_buf = vec![0u8; len as usize];
        stream
            .read_exact(&mut response_buf)
            .await
            .map_err(|e| SchedulerError::IpcError(format!("read response: {e}")))?;

        serde_json::from_slice(&response_buf)
            .map_err(|e| SchedulerError::IpcError(format!("parse response: {e}")))
    }

    /// Send a phase hint (fire-and-forget convenience method).
    ///
    /// # Errors
    ///
    /// Returns `SchedulerError::IpcError` if the message cannot be sent.
    pub async fn hint_phase(
        &self,
        phase: crate::types::InferencePhase,
        session_id: Option<String>,
    ) -> Result<()> {
        let msg = SchedulerMessage::PhaseHint { phase, session_id };
        let _resp = self.send(&msg).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipc_server_construction() {
        let server = IpcServer::new("/tmp/test-sched.sock");
        assert_eq!(
            server.socket_path().to_str().unwrap(),
            "/tmp/test-sched.sock"
        );
    }

    #[test]
    fn test_ipc_client_construction() {
        let _client = IpcClient::new("/tmp/test-sched.sock");
    }

    #[tokio::test]
    async fn test_ipc_roundtrip() {
        let socket_path = format!("/tmp/spirit-scheduler-test-{}.sock", std::process::id());

        let (cmd_tx, mut cmd_rx) = mpsc::channel::<IpcCommand>(16);

        let server = IpcServer::new(&socket_path);
        server.start(cmd_tx).await.expect("start server");

        // Spawn a handler that responds to commands
        tokio::spawn(async move {
            while let Some(cmd) = cmd_rx.recv().await {
                let _ = cmd.reply_tx.send(SchedulerResponse::Ok);
            }
        });

        // Give server time to bind
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let client = IpcClient::new(&socket_path);
        let response = client
            .send(&SchedulerMessage::GetStats)
            .await
            .expect("send message");

        assert!(matches!(response, SchedulerResponse::Ok));

        // Cleanup
        let _ = tokio::fs::remove_file(&socket_path).await;
    }
}
