//! Custom reedline [`Prompt`] implementation for the Spirit Shell.

use reedline::{Prompt, PromptEditMode, PromptHistorySearch, PromptHistorySearchStatus};
use std::borrow::Cow;

/// The Spirit Shell prompt, displayed before each input line.
pub struct SpiritPrompt;

impl Prompt for SpiritPrompt {
    fn render_prompt_left(&self) -> Cow<'_, str> {
        // ANSI green bold for "spirit> "
        Cow::Borrowed("\x1b[1;32mspirit\x1b[0m> ")
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
