//! Three-stage command resolution pipeline for the Spirit Shell.
//!
//! Input is resolved through a priority chain:
//! 1. **Aliases** — expand user-defined aliases before further resolution
//! 2. **Spirit built-in commands** — prefixed with `spirit` or matching known builtins
//! 3. **Standard Unix commands** — first word found in `$PATH`
//! 4. **Natural language fallback** — everything else routed to the LLM
//!
//! The resolver also detects pipe-to-LLM syntax: `<unix command> | <natural language>`.

use std::collections::HashMap;

use crate::config::ShellMode;
use crate::types::Resolution;
use crate::unix;

/// Known spirit subcommands (the word after "spirit").
const SPIRIT_SUBCOMMANDS: &[&str] = &[
    "model",
    "mem",
    "memory",
    "sched",
    "scheduler",
    "find",
    "similar",
    "summarize",
    "infer",
    "agent",
    "bench",
    "config",
    "shell",
    "status",
    "version",
];

/// Built-in commands that are always recognized (no "spirit" prefix needed).
const SHELL_BUILTINS: &[&str] = &[
    "help", "version", "status", "clear", "exit", "quit", "history",
];

/// The command resolver — determines how user input should be dispatched.
pub struct CommandResolver {
    aliases: HashMap<String, String>,
    mode: ShellMode,
}

impl CommandResolver {
    /// Create a new resolver with the given aliases and mode.
    pub fn new(aliases: HashMap<String, String>, mode: ShellMode) -> Self {
        Self { aliases, mode }
    }

    /// Update the shell mode.
    pub fn set_mode(&mut self, mode: ShellMode) {
        self.mode = mode;
    }

    /// Get the current mode.
    #[must_use]
    pub fn mode(&self) -> ShellMode {
        self.mode
    }

    /// Resolve a line of user input into a [`Resolution`].
    pub fn resolve(&self, input: &str) -> Resolution {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Resolution::SpiritBuiltin(String::new());
        }

        // ── Mode overrides ──────────────────────────────────────────────
        match self.mode {
            ShellMode::Llm => {
                // In LLM mode, only exit/quit and spirit commands bypass LLM
                let first = unix::first_word(trimmed);
                if first == "exit" || first == "quit" {
                    return Resolution::SpiritBuiltin(trimmed.to_string());
                }
                if first == "spirit" || self.is_shell_builtin(first) {
                    return Resolution::SpiritBuiltin(trimmed.to_string());
                }
                return Resolution::NaturalLanguage(trimmed.to_string());
            }
            ShellMode::Cli => {
                // In CLI mode, no NL fallback — unresolved goes to error
                return self.resolve_cli(trimmed);
            }
            ShellMode::Hybrid => {
                // Fall through to full resolution pipeline
            }
        }

        // ── Hybrid mode resolution pipeline ─────────────────────────────

        // 0. Explicit NL prefix: lines starting with `?` force natural language
        if let Some(query) = trimmed.strip_prefix('?') {
            return Resolution::NaturalLanguage(query.trim().to_string());
        }

        // 0b. Explicit CLI prefix: lines starting with `!` force Unix execution
        if let Some(cmd) = trimmed.strip_prefix('!') {
            return Resolution::UnixCommand(cmd.trim().to_string());
        }

        // 1. Alias expansion
        let first = unix::first_word(trimmed);
        if let Some(expansion) = self.aliases.get(first) {
            let rest = trimmed[first.len()..].trim();
            let expanded = if rest.is_empty() {
                expansion.clone()
            } else {
                format!("{expansion} {rest}")
            };
            return Resolution::Alias(expanded);
        }

        // 2. Pipe-to-LLM detection: `<command> | <natural language>`
        if let Some(resolution) = self.detect_pipe_to_llm(trimmed) {
            return resolution;
        }

        // 3. Spirit built-in commands
        if self.is_spirit_command(trimmed) || self.is_shell_builtin(first) {
            return Resolution::SpiritBuiltin(trimmed.to_string());
        }

        // 4. Unix command (first word found in PATH)
        if unix::command_exists(first) {
            return Resolution::UnixCommand(trimmed.to_string());
        }

        // 5. Natural language fallback
        Resolution::NaturalLanguage(trimmed.to_string())
    }

    /// Resolve in CLI mode: no natural language fallback.
    fn resolve_cli(&self, trimmed: &str) -> Resolution {
        let first = unix::first_word(trimmed);

        // Alias expansion
        if let Some(expansion) = self.aliases.get(first) {
            let rest = trimmed[first.len()..].trim();
            let expanded = if rest.is_empty() {
                expansion.clone()
            } else {
                format!("{expansion} {rest}")
            };
            return Resolution::Alias(expanded);
        }

        // Spirit built-ins
        if self.is_spirit_command(trimmed) || self.is_shell_builtin(first) {
            return Resolution::SpiritBuiltin(trimmed.to_string());
        }

        // Unix commands
        if unix::command_exists(first) {
            return Resolution::UnixCommand(trimmed.to_string());
        }

        // In CLI mode, unknown input is an error (handled by caller)
        Resolution::SpiritBuiltin(trimmed.to_string())
    }

    /// Check if input is a spirit command (starts with "spirit" + known subcommand).
    fn is_spirit_command(&self, input: &str) -> bool {
        let first = unix::first_word(input);
        if first != "spirit" {
            return false;
        }
        // "spirit" alone is a valid command (shows help)
        let rest = input[first.len()..].trim();
        if rest.is_empty() {
            return true;
        }
        let sub = unix::first_word(rest);
        SPIRIT_SUBCOMMANDS.contains(&sub)
    }

    /// Check if the command is a shell built-in (help, version, etc.).
    fn is_shell_builtin(&self, cmd: &str) -> bool {
        SHELL_BUILTINS.contains(&cmd)
    }

    /// Detect pipe-to-LLM pattern: `<unix command> | <natural language query>`.
    ///
    /// Heuristic: if the right side of `|` doesn't start with a known Unix command,
    /// treat it as a natural language query to the LLM.
    fn detect_pipe_to_llm(&self, input: &str) -> Option<Resolution> {
        // Find the last pipe that separates a command from natural language.
        // We iterate from the right to handle chained pipes correctly:
        // `cat file | grep foo | explain this` → command="cat file | grep foo", query="explain this"
        let pipe_pos = input.rfind('|')?;
        let left = input[..pipe_pos].trim();
        let right = input[pipe_pos + 1..].trim();

        if left.is_empty() || right.is_empty() {
            return None;
        }

        let right_first = unix::first_word(right);

        // If the right side starts with a Unix command, it's a normal pipe
        if unix::command_exists(right_first) {
            return None;
        }

        // If the right side starts with a spirit command, it's not pipe-to-LLM
        if right_first == "spirit" {
            return None;
        }

        // It's a pipe-to-LLM
        Some(Resolution::PipeToLlm {
            command: left.to_string(),
            query: right.to_string(),
        })
    }
}

/// Get all completable command names (spirit subcommands + builtins).
pub fn all_command_names() -> Vec<&'static str> {
    let mut names: Vec<&str> = SHELL_BUILTINS.to_vec();
    names.push("spirit");
    names
}

/// Get spirit subcommand names for completion.
pub fn spirit_subcommands() -> &'static [&'static str] {
    SPIRIT_SUBCOMMANDS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_resolver() -> CommandResolver {
        let mut aliases = HashMap::new();
        aliases.insert("ll".into(), "ls -la".into());
        aliases.insert("models".into(), "spirit model list".into());
        CommandResolver::new(aliases, ShellMode::Hybrid)
    }

    #[test]
    fn test_resolve_spirit_command() {
        let r = test_resolver();
        let res = r.resolve("spirit model list");
        assert!(matches!(res, Resolution::SpiritBuiltin(_)));
    }

    #[test]
    fn test_resolve_shell_builtin() {
        let r = test_resolver();
        let res = r.resolve("help");
        assert!(matches!(res, Resolution::SpiritBuiltin(_)));
    }

    #[test]
    fn test_resolve_unix_command() {
        let r = test_resolver();
        let res = r.resolve("ls -la /tmp");
        assert!(matches!(res, Resolution::UnixCommand(_)));
    }

    #[test]
    fn test_resolve_alias() {
        let r = test_resolver();
        let res = r.resolve("ll /tmp");
        match res {
            Resolution::Alias(expanded) => assert_eq!(expanded, "ls -la /tmp"),
            other => panic!("Expected Alias, got {other:?}"),
        }
    }

    #[test]
    fn test_resolve_natural_language() {
        let r = test_resolver();
        let res = r.resolve("What models do I have loaded?");
        assert!(matches!(res, Resolution::NaturalLanguage(_)));
    }

    #[test]
    fn test_resolve_explicit_nl_prefix() {
        let r = test_resolver();
        let res = r.resolve("? What is going on");
        match res {
            Resolution::NaturalLanguage(q) => assert_eq!(q, "What is going on"),
            other => panic!("Expected NL, got {other:?}"),
        }
    }

    #[test]
    fn test_resolve_pipe_to_llm() {
        let r = test_resolver();
        let res = r.resolve("ls -la /var/log | explain what happened today");
        match res {
            Resolution::PipeToLlm { command, query } => {
                assert_eq!(command, "ls -la /var/log");
                assert_eq!(query, "explain what happened today");
            }
            other => panic!("Expected PipeToLlm, got {other:?}"),
        }
    }

    #[test]
    fn test_resolve_normal_pipe() {
        let r = test_resolver();
        // Both sides are Unix commands → should be a plain UnixCommand
        let res = r.resolve("ls | grep foo");
        assert!(matches!(res, Resolution::UnixCommand(_)));
    }

    #[test]
    fn test_resolve_empty() {
        let r = test_resolver();
        let res = r.resolve("");
        assert!(matches!(res, Resolution::SpiritBuiltin(_)));
    }

    #[test]
    fn test_llm_mode_routes_everything() {
        let r = CommandResolver::new(HashMap::new(), ShellMode::Llm);
        let res = r.resolve("ls -la");
        assert!(matches!(res, Resolution::NaturalLanguage(_)));
    }

    #[test]
    fn test_llm_mode_allows_exit() {
        let r = CommandResolver::new(HashMap::new(), ShellMode::Llm);
        let res = r.resolve("exit");
        assert!(matches!(res, Resolution::SpiritBuiltin(_)));
    }
}
