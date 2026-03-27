//! Public types for the Spirit Shell.

use std::fmt;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The result of executing a built-in command.
#[derive(Debug)]
pub enum CommandResult {
    /// The command produced output to display.
    Output(String),
    /// The command requests a screen clear.
    Clear,
    /// The command requests the shell to exit.
    Exit,
    /// The input was not a built-in command and should be routed elsewhere.
    NotBuiltin,
    /// The command produced an error message.
    Error(String),
}

/// The result of command resolution — determines how input is dispatched.
#[derive(Debug)]
pub enum Resolution {
    /// A spirit built-in command (e.g., `spirit model list`).
    SpiritBuiltin(String),
    /// A standard Unix command found in PATH.
    UnixCommand(String),
    /// A natural language query to be sent to the LLM.
    NaturalLanguage(String),
    /// A pipe-to-LLM expression: Unix command piped to natural language.
    PipeToLlm { command: String, query: String },
    /// An alias expansion — the expanded command string.
    Alias(String),
}

// ── Agent Types ─────────────────────────────────────────────────────────────

/// Unique identifier for an agentic task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub Uuid);

impl AgentId {
    /// Create a new random agent ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Short ID for display (first 8 hex chars).
    #[must_use]
    pub fn short(&self) -> String {
        self.0.to_string()[..8].to_string()
    }
}

impl Default for AgentId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short())
    }
}

/// Status of an agentic task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentStatus {
    /// Agent is starting up.
    Starting,
    /// Agent is actively running.
    Running,
    /// Agent is waiting for user approval.
    AwaitingApproval,
    /// Agent completed successfully.
    Completed,
    /// Agent failed with an error.
    Failed,
    /// Agent was cancelled by the user.
    Cancelled,
}

impl fmt::Display for AgentStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Starting => write!(f, "starting"),
            Self::Running => write!(f, "running"),
            Self::AwaitingApproval => write!(f, "awaiting-approval"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Information about a running or completed agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    /// Unique agent identifier.
    pub id: AgentId,
    /// Description of the agent's task.
    pub task: String,
    /// Current status.
    pub status: AgentStatus,
    /// When the agent was started.
    pub started_at: DateTime<Utc>,
    /// Number of actions the agent has taken.
    pub actions_taken: u32,
    /// Path to the agent's log file.
    pub log_path: String,
}

/// Statistics returned by the LLM after a generation.
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    /// Time to first token in milliseconds.
    pub first_token_ms: u64,
    /// Total generation time in milliseconds.
    pub total_ms: u64,
    /// Number of tokens generated.
    pub tokens_generated: u32,
    /// Tokens per second throughput.
    pub tokens_per_second: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_id_short() {
        let id = AgentId::new();
        assert_eq!(id.short().len(), 8);
    }

    #[test]
    fn test_agent_id_display() {
        let id = AgentId::new();
        let display = format!("{id}");
        assert_eq!(display.len(), 8);
    }

    #[test]
    fn test_agent_status_display() {
        assert_eq!(AgentStatus::Running.to_string(), "running");
        assert_eq!(AgentStatus::Completed.to_string(), "completed");
        assert_eq!(
            AgentStatus::AwaitingApproval.to_string(),
            "awaiting-approval"
        );
    }

    #[test]
    fn test_resolution_variants() {
        let r = Resolution::NaturalLanguage("hello".into());
        assert!(matches!(r, Resolution::NaturalLanguage(_)));

        let r = Resolution::PipeToLlm {
            command: "ls".into(),
            query: "explain".into(),
        };
        assert!(matches!(r, Resolution::PipeToLlm { .. }));
    }
}
