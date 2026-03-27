//! Configuration types for the Spirit Shell.
//!
//! Loads from `spirit.yaml` shell section with environment variable
//! overrides (prefixed `SPIRIT_SHELL_`). Uses `figment` for multi-source
//! configuration merging.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use figment::providers::{Env, Format, Serialized, Yaml};
use figment::Figment;
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
    /// Only built-in and Unix commands are accepted; no inference routing.
    Cli,
}

impl FromStr for ShellMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "hybrid" => Ok(Self::Hybrid),
            "llm" => Ok(Self::Llm),
            "cli" | "cli-only" => Ok(Self::Cli),
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

/// LLM-specific configuration for natural language processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Model to use for natural language queries.
    #[serde(default = "default_llm_model")]
    pub model: String,

    /// Maximum context tokens for LLM queries.
    #[serde(default = "default_max_context_tokens")]
    pub max_context_tokens: u32,

    /// Whether the LLM should suggest commands.
    #[serde(default = "default_true")]
    pub suggest_commands: bool,

    /// Enable semantic search integration.
    #[serde(default = "default_true")]
    pub semantic_search: bool,

    /// System prompt template for the shell LLM.
    #[serde(default = "default_system_prompt")]
    pub system_prompt: String,
}

/// Prompt display configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptConfig {
    /// Prompt format string. Supports {cwd}, {mode}, {model}, {hostname}.
    #[serde(default = "default_prompt_format")]
    pub format: String,

    /// Show GPU status indicator in prompt.
    #[serde(default)]
    pub show_gpu_status: bool,

    /// Show active inference session count in prompt.
    #[serde(default)]
    pub show_active_sessions: bool,

    /// Show currently loaded model name in prompt.
    #[serde(default = "default_true")]
    pub show_model: bool,
}

/// History configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryConfig {
    /// Enable command history.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Maximum number of history entries.
    #[serde(default = "default_max_history")]
    pub max_entries: usize,

    /// Path to the history file.
    #[serde(default = "default_history_path")]
    pub path: PathBuf,
}

/// Inference endpoint configuration for the shell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceEndpoint {
    /// Ollama API base URL.
    #[serde(default = "default_ollama_url")]
    pub ollama_url: String,

    /// Request timeout in seconds.
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

/// Top-level configuration for the Spirit Shell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellConfig {
    /// The shell operating mode.
    #[serde(default)]
    pub mode: ShellMode,

    /// LLM configuration.
    #[serde(default)]
    pub llm: LlmConfig,

    /// Prompt configuration.
    #[serde(default)]
    pub prompt: PromptConfig,

    /// History configuration.
    #[serde(default)]
    pub history: HistoryConfig,

    /// Inference endpoint.
    #[serde(default)]
    pub inference: InferenceEndpoint,

    /// Command aliases (alias → expansion).
    #[serde(default)]
    pub aliases: std::collections::HashMap<String, String>,
}

// ── Defaults ────────────────────────────────────────────────────────────────

fn default_llm_model() -> String {
    // Priority: SPIRIT_SHELL_MODEL → SPIRIT_DEFAULT_MODEL → hardcoded fallback
    std::env::var("SPIRIT_SHELL_MODEL")
        .or_else(|_| std::env::var("SPIRIT_DEFAULT_MODEL"))
        .unwrap_or_else(|_| "llama3.2:3b".into())
}
fn default_max_context_tokens() -> u32 {
    4096
}
fn default_true() -> bool {
    true
}
fn default_system_prompt() -> String {
    "You are Spirit Shell, the AI assistant built into SpiritROS. \
     You help users manage their system, models, files, and processes. \
     When the user asks a question, answer concisely. \
     When the user asks you to do something, suggest the appropriate `spirit` command \
     or execute it if you have the capability. \
     Available spirit commands: spirit model list/load/unload/pull, \
     spirit mem status/rebalance, spirit sched status, spirit find <query>, \
     spirit similar <file>, spirit summarize <path>, spirit infer <prompt>, \
     spirit agent list/start/stop, spirit config show/get/set."
        .into()
}
fn default_prompt_format() -> String {
    "spirit:{cwd}> ".into()
}
fn default_max_history() -> usize {
    10000
}
fn default_history_path() -> PathBuf {
    resolve_data_path("shell_history")
}
fn default_ollama_url() -> String {
    // Priority: SPIRIT_OLLAMA_URL → OLLAMA_HOST → hardcoded fallback
    // OLLAMA_HOST is a bind address (0.0.0.0:11434); the shell connects to
    // localhost, so we replace 0.0.0.0 with 127.0.0.1.
    std::env::var("SPIRIT_OLLAMA_URL")
        .or_else(|_| std::env::var("OLLAMA_HOST"))
        .map(|h| {
            let url = if h.starts_with("http") {
                h
            } else {
                format!("http://{h}")
            };
            // 0.0.0.0 is a bind address, not a connect address
            url.replace("0.0.0.0", "127.0.0.1")
        })
        .unwrap_or_else(|_| "http://127.0.0.1:11434".into())
}
fn default_timeout() -> u64 {
    300
}

/// Resolve a path under the Spirit data directory.
fn resolve_data_path(name: &str) -> PathBuf {
    // Check SPIRIT_DATA_DIR first (set inside Docker), then fall back to ~/.spirit/
    if let Ok(data_dir) = std::env::var("SPIRIT_DATA_DIR") {
        return PathBuf::from(data_dir).join(name);
    }
    let base = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
    base.join(".spirit").join(name)
}

// ── Implementations ─────────────────────────────────────────────────────────

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            mode: ShellMode::default(),
            llm: LlmConfig::default(),
            prompt: PromptConfig::default(),
            history: HistoryConfig::default(),
            inference: InferenceEndpoint::default(),
            aliases: default_aliases(),
        }
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model: default_llm_model(),
            max_context_tokens: default_max_context_tokens(),
            suggest_commands: true,
            semantic_search: true,
            system_prompt: default_system_prompt(),
        }
    }
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self {
            format: default_prompt_format(),
            show_gpu_status: false,
            show_active_sessions: false,
            show_model: true,
        }
    }
}

impl Default for HistoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries: default_max_history(),
            path: default_history_path(),
        }
    }
}

impl Default for InferenceEndpoint {
    fn default() -> Self {
        Self {
            ollama_url: default_ollama_url(),
            timeout_secs: default_timeout(),
        }
    }
}

fn default_aliases() -> std::collections::HashMap<String, String> {
    let mut m = std::collections::HashMap::new();
    m.insert("ll".into(), "ls -la".into());
    m.insert("models".into(), "spirit model list".into());
    m.insert("mem".into(), "spirit mem status".into());
    m.insert("sched".into(), "spirit sched status".into());
    m
}

impl ShellConfig {
    /// Load configuration from a YAML file with environment variable overrides.
    ///
    /// The YAML file should contain a top-level `shell:` key. Environment variables
    /// are prefixed with `SPIRIT_SHELL_` and use double underscores for nesting.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        Figment::new()
            .merge(Serialized::defaults(Self::default()))
            .merge(Yaml::file(path).nested())
            .merge(Env::prefixed("SPIRIT_SHELL_").split("__"))
            .extract()
            .map_err(|e| e.to_string())
    }

    /// Load from the default `SPIRIT_CONFIG` path or fall back to defaults.
    pub fn load() -> Self {
        if let Ok(config_path) = std::env::var("SPIRIT_CONFIG") {
            let path = Path::new(&config_path);
            if path.exists() {
                match Self::from_file(path) {
                    Ok(cfg) => return cfg,
                    Err(e) => {
                        tracing::warn!("Failed to load shell config from {}: {e}", path.display());
                    }
                }
            }
        }
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shell_mode_default_is_hybrid() {
        assert_eq!(ShellMode::default(), ShellMode::Hybrid);
    }

    #[test]
    fn test_shell_mode_from_str() {
        assert_eq!("hybrid".parse::<ShellMode>().unwrap(), ShellMode::Hybrid);
        assert_eq!("llm".parse::<ShellMode>().unwrap(), ShellMode::Llm);
        assert_eq!("cli".parse::<ShellMode>().unwrap(), ShellMode::Cli);
        assert_eq!("cli-only".parse::<ShellMode>().unwrap(), ShellMode::Cli);
        assert!("invalid".parse::<ShellMode>().is_err());
    }

    #[test]
    fn test_shell_mode_display() {
        assert_eq!(ShellMode::Hybrid.to_string(), "hybrid");
        assert_eq!(ShellMode::Llm.to_string(), "llm");
        assert_eq!(ShellMode::Cli.to_string(), "cli");
    }

    #[test]
    fn test_default_config() {
        let cfg = ShellConfig::default();
        assert_eq!(cfg.mode, ShellMode::Hybrid);
        assert_eq!(cfg.llm.model, "llama3.2:3b");
        assert!(cfg.history.enabled);
        assert_eq!(cfg.history.max_entries, 10000);
        assert!(cfg.aliases.contains_key("ll"));
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let cfg = ShellConfig::default();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: ShellConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg.mode, back.mode);
        assert_eq!(cfg.llm.model, back.llm.model);
    }
}
