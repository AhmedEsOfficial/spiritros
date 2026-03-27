//! Error types for the Spirit Shell.

/// Errors that can occur during shell operation.
#[derive(Debug, thiserror::Error)]
pub enum ShellError {
    /// A command was not found in the built-in registry.
    #[error("Command not found: '{0}'")]
    CommandNotFound(String),

    /// Failed to parse user input.
    #[error("Parse error: {0}")]
    ParseError(String),

    /// The inference engine is not reachable.
    #[error("Inference engine unavailable: {0}")]
    InferenceUnavailable(String),

    /// An inference request failed.
    #[error("Inference error: {0}")]
    InferenceError(String),

    /// An I/O error occurred (e.g., history file access).
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// A command execution failed.
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    /// A Unix command failed with a non-zero exit code.
    #[error("Command '{command}' exited with code {code}")]
    UnixCommandFailed { command: String, code: i32 },

    /// An agent operation failed.
    #[error("Agent error: {0}")]
    AgentError(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// HTTP request error.
    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    /// Pipe-to-LLM: the Unix command portion failed.
    #[error("Pipe command failed: {0}")]
    PipeCommandFailed(String),

    /// The operation timed out.
    #[error("Operation timed out after {0}s")]
    Timeout(u64),
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, ShellError>;
