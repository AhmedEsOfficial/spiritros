#![warn(clippy::all, clippy::pedantic)]

//! Spirit Shell — the LLM-native command interface for `SpiritROS`.
//!
//! This crate provides the interactive shell that serves as the primary user-facing
//! entrypoint for `SpiritROS` Phase 1. It supports three modes:
//!
//! - **CLI mode**: Traditional command dispatch for built-in commands.
//! - **LLM mode**: All input is routed to the inference engine for natural language processing.
//! - **Hybrid mode** (default): Built-in commands are dispatched directly; unrecognized input
//!   is forwarded to the inference engine.

pub mod commands;
pub mod completer;
pub mod config;
pub mod error;
pub mod prompt;
pub mod types;

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
    fn test_shell_config_default_prompt_style() {
        let config = ShellConfig::default();
        assert_eq!(config.prompt_style, "spirit");
        assert_eq!(config.shell_mode, ShellMode::Hybrid);
    }

    #[test]
    fn test_shell_error_display_messages() {
        let err = ShellError::CommandNotFound("foo".to_string());
        assert!(err.to_string().contains("foo"));

        let err = ShellError::ParseError("bad input".to_string());
        assert!(err.to_string().contains("bad input"));

        let err = ShellError::InferenceUnavailable;
        assert!(err.to_string().contains("unavailable"));
    }

    #[test]
    fn test_shell_mode_from_str() {
        assert_eq!("hybrid".parse::<ShellMode>().unwrap(), ShellMode::Hybrid);
        assert_eq!("llm".parse::<ShellMode>().unwrap(), ShellMode::Llm);
        assert_eq!("cli".parse::<ShellMode>().unwrap(), ShellMode::Cli);
        assert!("invalid".parse::<ShellMode>().is_err());
    }
}
