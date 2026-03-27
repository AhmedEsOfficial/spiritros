//! Dynamic reedline [`Prompt`] for the Spirit Shell.
//!
//! Displays contextual information including the current working directory,
//! active model, and shell mode.

use std::borrow::Cow;

use reedline::{Prompt, PromptEditMode, PromptHistorySearch, PromptHistorySearchStatus};

use crate::commands::ShellState;
use crate::config::ShellMode;

/// Dynamic Spirit Shell prompt that shows context-sensitive information.
pub struct SpiritPrompt {
    /// Current working directory (cached per-render).
    cwd: String,
    /// Currently configured model name.
    model: String,
    /// Current shell mode.
    mode: ShellMode,
    /// Whether to show the model in the prompt.
    show_model: bool,
}

impl SpiritPrompt {
    /// Create a new prompt with default values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cwd: current_dir_short(),
            model: String::new(),
            mode: ShellMode::Hybrid,
            show_model: true,
        }
    }

    /// Update the prompt state from the shell state.
    pub fn update_from_state(&mut self, state: &ShellState) {
        self.cwd = current_dir_short();
        self.model = state.inference.model().to_string();
        self.mode = state.config.mode;
        self.show_model = state.config.prompt.show_model;
    }
}

impl Default for SpiritPrompt {
    fn default() -> Self {
        Self::new()
    }
}

impl Prompt for SpiritPrompt {
    fn render_prompt_left(&self) -> Cow<'_, str> {
        let mode_indicator = match self.mode {
            ShellMode::Hybrid => "",
            ShellMode::Cli => " \x1b[33m[CLI]\x1b[0m",
            ShellMode::Llm => " \x1b[35m[LLM]\x1b[0m",
        };

        let model_part = if self.show_model && !self.model.is_empty() {
            format!(" \x1b[2m({})\x1b[0m", self.model)
        } else {
            String::new()
        };

        Cow::Owned(format!(
            "\x1b[1;32mspirit\x1b[0m:\x1b[34m{}\x1b[0m{model_part}{mode_indicator}> ",
            self.cwd
        ))
    }

    fn render_prompt_right(&self) -> Cow<'_, str> {
        Cow::Borrowed("")
    }

    fn render_prompt_indicator(&self, _edit_mode: PromptEditMode) -> Cow<'_, str> {
        Cow::Borrowed("")
    }

    fn render_prompt_multiline_indicator(&self) -> Cow<'_, str> {
        Cow::Borrowed("... ")
    }

    fn render_prompt_history_search_indicator(
        &self,
        history_search: PromptHistorySearch,
    ) -> Cow<'_, str> {
        let prefix = match history_search.status {
            PromptHistorySearchStatus::Passing => "",
            PromptHistorySearchStatus::Failing => "[failing] ",
        };
        Cow::Owned(format!("{prefix}(search: '{}') ", history_search.term))
    }
}

/// Get a short representation of the current working directory.
/// Replaces the home directory with `~`.
fn current_dir_short() -> String {
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "?".to_string());

    if let Ok(home) = std::env::var("HOME") {
        if cwd.starts_with(&home) {
            return format!("~{}", &cwd[home.len()..]);
        }
    }

    cwd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_creation() {
        let prompt = SpiritPrompt::new();
        assert!(!prompt.cwd.is_empty());
    }

    #[test]
    fn test_prompt_render() {
        let prompt = SpiritPrompt::new();
        let left = prompt.render_prompt_left();
        assert!(left.contains("spirit"));
    }

    #[test]
    fn test_current_dir_short() {
        let dir = current_dir_short();
        assert!(!dir.is_empty());
    }
}
