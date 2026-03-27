//! Spirit shell commands: mode switching.

use crate::commands::SharedState;
use crate::config::ShellMode;
use crate::types::CommandResult;

/// Dispatch `spirit shell <subcommand>`.
pub async fn dispatch(state: &SharedState, args: &[&str]) -> CommandResult {
    if args.is_empty() {
        let mode = state.read().await.config.mode;
        return CommandResult::Output(format!(
            "Usage: spirit shell <mode> [value]\n\n\
             Subcommands:\n  \
             mode <hybrid|cli|llm>   Switch shell operating mode\n\n\
             Current mode: {mode}"
        ));
    }

    match args[0] {
        "mode" => {
            if args.len() < 2 {
                let mode = state.read().await.config.mode;
                return CommandResult::Output(format!(
                    "Current mode: {mode}\n\n\
                     Usage: spirit shell mode <hybrid|cli|llm>\n\
                       hybrid  — CLI commands first, natural language fallback (default)\n\
                       cli     — Only built-in and Unix commands, no LLM\n\
                       llm     — Everything routed to the inference engine"
                ));
            }
            set_mode(state, args[1]).await
        }
        // Allow shorthand: `spirit shell hybrid`
        "hybrid" | "cli" | "llm" | "cli-only" | "llm-only" => set_mode(state, args[0]).await,
        other => CommandResult::Error(format!("Unknown shell subcommand: '{other}'. Use: mode")),
    }
}

/// Set the shell mode.
async fn set_mode(state: &SharedState, mode_str: &str) -> CommandResult {
    match mode_str.parse::<ShellMode>() {
        Ok(mode) => {
            let mut state = state.write().await;
            state.config.mode = mode;
            state.resolver.set_mode(mode);
            CommandResult::Output(format!("Shell mode set to: {mode}"))
        }
        Err(e) => CommandResult::Error(e),
    }
}
