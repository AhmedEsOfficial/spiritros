//! Multi-level tab completion for the Spirit Shell.
//!
//! Provides context-aware completion for:
//! - Top-level commands (spirit, help, version, Unix commands)
//! - Spirit subcommands (spirit model, spirit mem, etc.)
//! - Spirit sub-subcommands (spirit model list, spirit model load, etc.)
//! - File paths

use reedline::{Completer, Span, Suggestion};

/// Spirit Shell multi-level completer.
pub struct SpiritCompleter {
    top_level: Vec<CompletionEntry>,
    spirit_subs: Vec<CompletionEntry>,
    model_subs: Vec<CompletionEntry>,
    mem_subs: Vec<CompletionEntry>,
    agent_subs: Vec<CompletionEntry>,
    shell_subs: Vec<CompletionEntry>,
    config_subs: Vec<CompletionEntry>,
    sched_subs: Vec<CompletionEntry>,
}

struct CompletionEntry {
    value: String,
    description: Option<String>,
}

impl CompletionEntry {
    fn new(value: &str, desc: &str) -> Self {
        Self {
            value: value.to_string(),
            description: Some(desc.to_string()),
        }
    }
}

impl SpiritCompleter {
    /// Create a new completer with all known commands.
    #[must_use]
    pub fn new() -> Self {
        Self {
            top_level: vec![
                CompletionEntry::new("spirit", "SpiritROS management commands"),
                CompletionEntry::new("help", "Show help message"),
                CompletionEntry::new("version", "Show version"),
                CompletionEntry::new("status", "System status"),
                CompletionEntry::new("clear", "Clear screen"),
                CompletionEntry::new("exit", "Exit shell"),
                CompletionEntry::new("history", "History info"),
            ],
            spirit_subs: vec![
                CompletionEntry::new("model", "Model management"),
                CompletionEntry::new("mem", "Memory zones"),
                CompletionEntry::new("sched", "Scheduler"),
                CompletionEntry::new("find", "Semantic search"),
                CompletionEntry::new("similar", "Find similar files"),
                CompletionEntry::new("summarize", "Summarize directory"),
                CompletionEntry::new("infer", "Direct inference"),
                CompletionEntry::new("agent", "Agentic tasks"),
                CompletionEntry::new("shell", "Shell settings"),
                CompletionEntry::new("config", "Configuration"),
                CompletionEntry::new("bench", "Benchmarks"),
                CompletionEntry::new("status", "System status"),
                CompletionEntry::new("version", "Version info"),
            ],
            model_subs: vec![
                CompletionEntry::new("list", "List available models"),
                CompletionEntry::new("load", "Load a model"),
                CompletionEntry::new("unload", "Unload a model"),
                CompletionEntry::new("pull", "Pull from registry"),
            ],
            mem_subs: vec![
                CompletionEntry::new("status", "Memory zone utilization"),
                CompletionEntry::new("rebalance", "Resize memory zones"),
            ],
            agent_subs: vec![
                CompletionEntry::new("list", "List agents"),
                CompletionEntry::new("start", "Start an agent"),
                CompletionEntry::new("stop", "Stop an agent"),
            ],
            shell_subs: vec![CompletionEntry::new("mode", "Switch shell mode")],
            config_subs: vec![
                CompletionEntry::new("show", "Show configuration"),
                CompletionEntry::new("get", "Get a config value"),
                CompletionEntry::new("set", "Set a config value"),
            ],
            sched_subs: vec![CompletionEntry::new("status", "Scheduler state")],
        }
    }
}

impl Default for SpiritCompleter {
    fn default() -> Self {
        Self::new()
    }
}

impl Completer for SpiritCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<Suggestion> {
        let input = &line[..pos];
        let words: Vec<&str> = input.split_whitespace().collect();
        let trailing_space = input.ends_with(' ');

        match words.len() {
            // Empty line or cursor at start — show top-level commands
            0 => suggest_from(&self.top_level, "", 0, pos),

            // One word typed (no trailing space) — complete top-level
            1 if !trailing_space => suggest_from(&self.top_level, words[0], 0, pos),

            // One word + space — suggest subcommands for "spirit"
            1 if trailing_space => {
                if words[0] == "spirit" {
                    suggest_from(&self.spirit_subs, "", pos, pos)
                } else {
                    // For other commands, suggest file paths
                    suggest_files("", pos)
                }
            }

            // Two words — depends on context
            2 if !trailing_space => {
                if words[0] == "spirit" {
                    let start = input.rfind(words[1]).unwrap_or(pos);
                    suggest_from(&self.spirit_subs, words[1], start, pos)
                } else {
                    // File path completion for Unix commands
                    let start = input.rfind(words[1]).unwrap_or(pos);
                    suggest_files(words[1], start)
                }
            }

            // Two words + space — suggest sub-subcommands
            2 if trailing_space && words[0] == "spirit" => {
                let entries = match words[1] {
                    "model" => &self.model_subs,
                    "mem" | "memory" => &self.mem_subs,
                    "agent" => &self.agent_subs,
                    "shell" => &self.shell_subs,
                    "config" => &self.config_subs,
                    "sched" | "scheduler" => &self.sched_subs,
                    _ => return Vec::new(),
                };
                suggest_from(entries, "", pos, pos)
            }

            // Three words — completing sub-subcommand
            3 if !trailing_space && words[0] == "spirit" => {
                let entries = match words[1] {
                    "model" => &self.model_subs,
                    "mem" | "memory" => &self.mem_subs,
                    "agent" => &self.agent_subs,
                    "shell" => &self.shell_subs,
                    "config" => &self.config_subs,
                    "sched" | "scheduler" => &self.sched_subs,
                    _ => return Vec::new(),
                };
                let start = input.rfind(words[2]).unwrap_or(pos);
                suggest_from(entries, words[2], start, pos)
            }

            // Three words + space for mode completion
            3 if trailing_space
                && words[0] == "spirit"
                && words[1] == "shell"
                && words[2] == "mode" =>
            {
                let modes = vec![
                    CompletionEntry::new("hybrid", "CLI first, NL fallback"),
                    CompletionEntry::new("cli", "CLI only"),
                    CompletionEntry::new("llm", "LLM only"),
                ];
                suggest_from(&modes, "", pos, pos)
            }

            _ => Vec::new(),
        }
    }
}

/// Generate suggestions from a list of entries matching a prefix.
fn suggest_from(
    entries: &[CompletionEntry],
    prefix: &str,
    start: usize,
    end: usize,
) -> Vec<Suggestion> {
    entries
        .iter()
        .filter(|e| e.value.starts_with(prefix))
        .map(|e| Suggestion {
            value: e.value.clone(),
            description: e.description.clone(),
            style: None,
            extra: None,
            span: Span::new(start, end),
            append_whitespace: true,
        })
        .collect()
}

/// Generate file path suggestions.
fn suggest_files(prefix: &str, start: usize) -> Vec<Suggestion> {
    let (dir, file_prefix) = if prefix.contains('/') {
        let last_slash = prefix.rfind('/').unwrap();
        (&prefix[..=last_slash], &prefix[last_slash + 1..])
    } else {
        ("./", prefix)
    };

    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };

    entries
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with(file_prefix)
        })
        .take(20) // Limit suggestions
        .map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
            let full = if dir == "./" {
                name.clone()
            } else {
                format!("{dir}{name}")
            };
            Suggestion {
                value: full,
                description: if is_dir {
                    Some("directory".into())
                } else {
                    None
                },
                style: None,
                extra: None,
                span: Span::new(start, start + prefix.len()),
                append_whitespace: !is_dir,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completer_top_level() {
        let mut c = SpiritCompleter::new();
        let suggestions = c.complete("sp", 2);
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.value == "spirit"));
    }

    #[test]
    fn test_completer_spirit_subcommands() {
        let mut c = SpiritCompleter::new();
        let suggestions = c.complete("spirit ", 7);
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.value == "model"));
    }

    #[test]
    fn test_completer_model_subcommands() {
        let mut c = SpiritCompleter::new();
        let suggestions = c.complete("spirit model ", 13);
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.value == "list"));
        assert!(suggestions.iter().any(|s| s.value == "load"));
    }

    #[test]
    fn test_completer_partial_match() {
        let mut c = SpiritCompleter::new();
        let suggestions = c.complete("spirit mod", 10);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].value, "model");
    }

    #[test]
    fn test_completer_empty() {
        let mut c = SpiritCompleter::new();
        let suggestions = c.complete("", 0);
        assert!(!suggestions.is_empty());
    }
}
