//! Built-in command registry and dispatch for the Spirit Shell.

use crate::types::CommandResult;

/// Trait for a built-in shell command.
pub trait Command {
    /// The primary name of the command (used for dispatch).
    fn name(&self) -> &'static str;

    /// A short description for the help listing.
    fn description(&self) -> &'static str;

    /// Execute the command with the given arguments and return a result.
    fn execute(&self, args: &[&str]) -> CommandResult;
}

// ── Built-in command implementations ────────────────────────────────────────

struct HelpCommand;

impl Command for HelpCommand {
    fn name(&self) -> &'static str {
        "help"
    }

    fn description(&self) -> &'static str {
        "Show available commands"
    }

    fn execute(&self, _args: &[&str]) -> CommandResult {
        let text = "\
Available commands:
  help      Show this help message
  version   Show Spirit Shell version
  status    Show system status
  clear     Clear the screen
  exit      Exit the shell (also: quit)

In Hybrid mode, unrecognized input is forwarded to the inference engine.";
        CommandResult::Output(text.to_string())
    }
}

struct VersionCommand;

impl Command for VersionCommand {
    fn name(&self) -> &'static str {
        "version"
    }

    fn description(&self) -> &'static str {
        "Show Spirit Shell version"
    }

    fn execute(&self, _args: &[&str]) -> CommandResult {
        let version = env!("CARGO_PKG_VERSION");
        CommandResult::Output(format!(
            "Spirit Shell v{version} (Phase 1: Virtualized Deployment)"
        ))
    }
}

struct StatusCommand;

impl Command for StatusCommand {
    fn name(&self) -> &'static str {
        "status"
    }

    fn description(&self) -> &'static str {
        "Show system status"
    }

    // PHASE1-SHIM: Replace with real subsystem queries in later phases.
    fn execute(&self, _args: &[&str]) -> CommandResult {
        let text = "\
SpiritROS System Status (Phase 1 — Stub)
─────────────────────────────────────────
  Inference Engine:  [not connected]
  Memory Zones:      [not configured]
  SpiritFS:          [not mounted]
  GPU:               [not detected]
  Shell Mode:        hybrid";
        CommandResult::Output(text.to_string())
    }
}

struct ClearCommand;

impl Command for ClearCommand {
    fn name(&self) -> &'static str {
        "clear"
    }

    fn description(&self) -> &'static str {
        "Clear the terminal screen"
    }

    fn execute(&self, _args: &[&str]) -> CommandResult {
        CommandResult::Clear
    }
}

struct ExitCommand;

impl Command for ExitCommand {
    fn name(&self) -> &'static str {
        "exit"
    }

    fn description(&self) -> &'static str {
        "Exit the shell"
    }

    fn execute(&self, _args: &[&str]) -> CommandResult {
        CommandResult::Exit
    }
}

// ── Registry ────────────────────────────────────────────────────────────────

/// Registry of all built-in shell commands.
pub struct CommandRegistry {
    commands: Vec<Box<dyn Command>>,
}

impl CommandRegistry {
    /// Create a new registry populated with the default built-in commands.
    #[must_use]
    pub fn new() -> Self {
        Self {
            commands: vec![
                Box::new(HelpCommand),
                Box::new(VersionCommand),
                Box::new(StatusCommand),
                Box::new(ClearCommand),
                Box::new(ExitCommand),
            ],
        }
    }

    /// Return a list of all command names (used by the completer).
    #[must_use]
    pub fn command_names(&self) -> Vec<&str> {
        self.commands.iter().map(|c| c.name()).collect()
    }

    /// Dispatch a line of input to the matching built-in command.
    ///
    /// Returns [`CommandResult::NotBuiltin`] if no command matched.
    #[must_use]
    pub fn dispatch(&self, line: &str) -> CommandResult {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return CommandResult::Output(String::new());
        }

        let mut parts = trimmed.split_whitespace();
        let cmd_name = parts.next().unwrap_or("");
        let args: Vec<&str> = parts.collect();

        // "quit" is an alias for "exit"
        let cmd_name = if cmd_name == "quit" { "exit" } else { cmd_name };

        for cmd in &self.commands {
            if cmd.name() == cmd_name {
                return cmd.execute(&args);
            }
        }

        CommandResult::NotBuiltin
    }
}

impl Default for CommandRegistry {
    fn default() -> Self {
        Self::new()
    }
}
