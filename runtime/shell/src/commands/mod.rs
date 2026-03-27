//! Built-in command registry and dispatch for the Spirit Shell.
//!
//! Commands are organized into modules by subsystem:
//! - `builtin` — core shell commands (help, version, status, clear, exit)
//! - `model` — spirit model {list, load, unload, pull, info}
//! - `mem` — spirit mem {status, rebalance}
//! - `sched` — spirit sched {status}
//! - `find` — spirit find, spirit similar, spirit summarize
//! - `infer` — spirit infer <prompt>
//! - `agent` — spirit agent {list, start, stop}
//! - `shell_cmd` — spirit shell {mode}
//! - `config_cmd` — spirit config {show, get, set}

mod agent;
mod builtin;
mod config_cmd;
mod find;
mod infer;
mod mem;
mod model;
mod sched;
mod shell_cmd;

use std::sync::Arc;

use tokio::sync::RwLock;

use crate::agent_manager::AgentManager;
use crate::config::ShellConfig;
use crate::inference::InferenceClient;
use crate::resolver::CommandResolver;
use crate::types::CommandResult;

/// Shared mutable state accessible to all commands.
pub struct ShellState {
    pub config: ShellConfig,
    pub inference: InferenceClient,
    pub resolver: CommandResolver,
    pub agent_manager: AgentManager,
}

/// Thread-safe handle to shell state.
pub type SharedState = Arc<RwLock<ShellState>>;

/// Create a new shared shell state.
pub fn new_state(config: ShellConfig) -> SharedState {
    let inference = InferenceClient::new(&config);
    let resolver = CommandResolver::new(config.aliases.clone(), config.mode);
    let agent_manager = AgentManager::new();

    Arc::new(RwLock::new(ShellState {
        config,
        inference,
        resolver,
        agent_manager,
    }))
}

/// Dispatch a spirit built-in command.
///
/// Handles routing for all `spirit <subcommand>` commands as well as
/// shell builtins (help, version, status, clear, exit).
pub async fn dispatch_builtin(state: &SharedState, line: &str) -> CommandResult {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return CommandResult::Output(String::new());
    }

    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    let cmd = parts[0];
    let args = &parts[1..];

    // Shell builtins (no "spirit" prefix)
    match cmd {
        "help" => return builtin::help(state).await,
        "version" => return builtin::version(),
        "status" => return builtin::status(state).await,
        "clear" => return CommandResult::Clear,
        "exit" | "quit" => return CommandResult::Exit,
        "history" => return builtin::history(),
        "spirit" => {
            // Route to spirit subcommands
            if args.is_empty() {
                return builtin::spirit_help();
            }
            return dispatch_spirit(state, args).await;
        }
        _ => {}
    }

    // Unknown built-in
    CommandResult::Error(format!(
        "Unknown command: '{cmd}'. Type 'help' for available commands."
    ))
}

/// Dispatch a `spirit <subcommand>` command.
async fn dispatch_spirit(state: &SharedState, args: &[&str]) -> CommandResult {
    let sub = args[0];
    let sub_args = &args[1..];

    match sub {
        "model" => model::dispatch(state, sub_args).await,
        "mem" | "memory" => mem::dispatch(state, sub_args).await,
        "sched" | "scheduler" => sched::dispatch(sub_args),
        "find" => find::find(state, sub_args).await,
        "similar" => find::similar(sub_args),
        "summarize" => find::summarize(sub_args),
        "infer" => infer::dispatch(state, sub_args).await,
        "agent" => agent::dispatch(state, sub_args).await,
        "shell" => shell_cmd::dispatch(state, sub_args).await,
        "config" => config_cmd::dispatch(state, sub_args).await,
        "bench" => CommandResult::Output(
            "Run benchmarks with: spirit-bench\n\
             See benchmarks/ directory for suite configuration."
                .into(),
        ),
        "status" | "version" => {
            // Redirect to top-level builtins
            if sub == "status" {
                builtin::status(state).await
            } else {
                builtin::version()
            }
        }
        _ => CommandResult::Error(format!(
            "Unknown spirit subcommand: '{sub}'. Type 'spirit' for available commands."
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dispatch_help() {
        let state = new_state(ShellConfig::default());
        let result = dispatch_builtin(&state, "help").await;
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[tokio::test]
    async fn test_dispatch_version() {
        let state = new_state(ShellConfig::default());
        let result = dispatch_builtin(&state, "version").await;
        match result {
            CommandResult::Output(s) => assert!(s.contains("Spirit Shell")),
            _ => panic!("Expected Output"),
        }
    }

    #[tokio::test]
    async fn test_dispatch_exit() {
        let state = new_state(ShellConfig::default());
        let result = dispatch_builtin(&state, "exit").await;
        assert!(matches!(result, CommandResult::Exit));
    }

    #[tokio::test]
    async fn test_dispatch_clear() {
        let state = new_state(ShellConfig::default());
        let result = dispatch_builtin(&state, "clear").await;
        assert!(matches!(result, CommandResult::Clear));
    }

    #[tokio::test]
    async fn test_dispatch_spirit_no_args() {
        let state = new_state(ShellConfig::default());
        let result = dispatch_builtin(&state, "spirit").await;
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[tokio::test]
    async fn test_dispatch_unknown() {
        let state = new_state(ShellConfig::default());
        let result = dispatch_builtin(&state, "nonexistent").await;
        assert!(matches!(result, CommandResult::Error(_)));
    }
}
