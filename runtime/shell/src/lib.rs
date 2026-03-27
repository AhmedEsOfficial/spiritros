#![warn(clippy::all, clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::default_trait_access,
    clippy::must_use_candidate,
    clippy::redundant_closure_for_method_calls,
    clippy::unused_self,
    clippy::unused_async,
    clippy::map_unwrap_or
)]

//! Spirit Shell — the LLM-native command interface for `SpiritROS`.
//!
//! This crate provides the interactive shell that serves as the primary user-facing
//! entrypoint for `SpiritROS` Phase 1. It supports three modes:
//!
//! - **Hybrid mode** (default): Built-in and Unix commands are dispatched directly;
//!   unrecognized input is forwarded to the inference engine as natural language.
//! - **CLI mode**: Only built-in and Unix commands are accepted; no inference routing.
//! - **LLM mode**: All input is routed to the inference engine for natural language processing.
//!
//! ## Key Features
//!
//! - **Three-stage command resolution**: Spirit builtins → Unix commands → NL fallback
//! - **Pipe-to-LLM**: `ls -la /var/log | explain what happened today`
//! - **Built-in spirit commands**: model, mem, sched, find, infer, agent, config
//! - **Agentic mode**: Long-running autonomous tasks with audit logging
//! - **Rich line editing**: History, multi-level tab completion, syntax highlighting
//! - **Dynamic prompt**: Shows CWD, active model, and shell mode
//! - **Configurable**: YAML + env vars via figment

pub mod agent_manager;
pub mod commands;
pub mod completer;
pub mod config;
pub mod error;
pub mod highlighter;
pub mod inference;
pub mod pipe;
pub mod prompt;
pub mod resolver;
pub mod types;
pub mod unix;

pub use config::{ShellConfig, ShellMode};
pub use error::ShellError;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shell_mode_default_is_hybrid() {
        let mode = ShellMode::default();
        assert_eq!(mode, ShellMode::Hybrid);
    }

    #[test]
    fn test_shell_config_default() {
        let config = ShellConfig::default();
        assert_eq!(config.mode, ShellMode::Hybrid);
        assert_eq!(config.llm.model, "llama3.2:3b");
        assert!(config.history.enabled);
    }

    #[test]
    fn test_shell_error_display_messages() {
        let err = ShellError::CommandNotFound("foo".to_string());
        assert!(err.to_string().contains("foo"));

        let err = ShellError::ParseError("bad input".to_string());
        assert!(err.to_string().contains("bad input"));

        let err = ShellError::InferenceUnavailable("timeout".to_string());
        assert!(err.to_string().contains("unavailable"));
    }

    #[test]
    fn test_shell_mode_from_str() {
        assert_eq!("hybrid".parse::<ShellMode>().unwrap(), ShellMode::Hybrid);
        assert_eq!("llm".parse::<ShellMode>().unwrap(), ShellMode::Llm);
        assert_eq!("cli".parse::<ShellMode>().unwrap(), ShellMode::Cli);
        assert!("invalid".parse::<ShellMode>().is_err());
    }

    #[test]
    fn test_resolution_types() {
        let r = Resolution::SpiritBuiltin("help".into());
        assert!(matches!(r, Resolution::SpiritBuiltin(_)));

        let r = Resolution::UnixCommand("ls -la".into());
        assert!(matches!(r, Resolution::UnixCommand(_)));

        let r = Resolution::NaturalLanguage("what models".into());
        assert!(matches!(r, Resolution::NaturalLanguage(_)));
    }

    #[test]
    fn test_command_result_variants() {
        assert!(matches!(
            CommandResult::Output("hi".into()),
            CommandResult::Output(_)
        ));
        assert!(matches!(CommandResult::Clear, CommandResult::Clear));
        assert!(matches!(CommandResult::Exit, CommandResult::Exit));
        assert!(matches!(
            CommandResult::NotBuiltin,
            CommandResult::NotBuiltin
        ));
        assert!(matches!(
            CommandResult::Error("err".into()),
            CommandResult::Error(_)
        ));
    }
}
