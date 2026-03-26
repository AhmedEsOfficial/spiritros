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
    #[error("Inference engine unavailable")]
    InferenceUnavailable,

    /// An I/O error occurred (e.g., history file access).
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// A command execution failed.
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
}
