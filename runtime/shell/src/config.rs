//! Configuration types for the Spirit Shell.

use std::path::PathBuf;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Operating mode for the shell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum ShellMode {
    /// Built-in commands are dispatched directly; unrecognized input goes to inference.
    #[default]
    Hybrid,
    /// All input is routed to the inference engine.
    Llm,
    /// Only built-in commands are accepted; no inference routing.
    Cli,
}

impl FromStr for ShellMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "hybrid" => Ok(Self::Hybrid),
            "llm" => Ok(Self::Llm),
            "cli" => Ok(Self::Cli),
            other => Err(format!(
                "Unknown shell mode: '{other}'. Expected hybrid, llm, or cli"
            )),
        }
    }
}

impl std::fmt::Display for ShellMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hybrid => write!(f, "hybrid"),
            Self::Llm => write!(f, "llm"),
            Self::Cli => write!(f, "cli"),
        }
    }
}

/// Configuration for the Spirit Shell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellConfig {
    /// The prompt style identifier.
    pub prompt_style: String,
    /// Path to the history file.
    pub history_file: PathBuf,
    /// The shell operating mode.
    pub shell_mode: ShellMode,
}

impl Default for ShellConfig {
    fn default() -> Self {
        let history_file = dirs_or_default();
        Self {
            prompt_style: "spirit".to_string(),
            history_file,
            shell_mode: ShellMode::default(),
        }
    }
}

/// Returns the default history file path (`~/.spirit/history.txt`),
/// falling back to `/tmp/.spirit/history.txt` if `HOME` is unset.
fn dirs_or_default() -> PathBuf {
    let base = std::env::var("HOME").map_or_else(|_| PathBuf::from("/tmp"), PathBuf::from);
    base.join(".spirit").join("history.txt")
}
