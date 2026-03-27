//! Syntax highlighting for the Spirit Shell.
//!
//! Highlights spirit commands, Unix commands, and natural language prefixes
//! with ANSI colors to provide visual feedback during editing.

use reedline::{Highlighter, StyledText};

use crate::unix;

/// Spirit Shell syntax highlighter.
pub struct SpiritHighlighter;

// ANSI color codes
const GREEN: &str = "\x1b[32m";
const CYAN: &str = "\x1b[36m";
const YELLOW: &str = "\x1b[33m";
const MAGENTA: &str = "\x1b[35m";
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";

impl Highlighter for SpiritHighlighter {
    fn highlight(&self, line: &str, _cursor: usize) -> StyledText {
        let mut styled = StyledText::new();

        let trimmed = line.trim_start();
        let leading_space = &line[..line.len() - trimmed.len()];

        if !leading_space.is_empty() {
            styled.push((Default::default(), leading_space.to_string()));
        }

        if trimmed.is_empty() {
            return styled;
        }

        let first = unix::first_word(trimmed);

        match first {
            // Spirit commands — green bold
            "spirit" => {
                highlight_spirit_command(&mut styled, trimmed);
            }
            // Shell builtins — cyan
            "help" | "version" | "status" | "clear" | "exit" | "quit" | "history" => {
                styled.push((Default::default(), format!("{CYAN}{BOLD}{first}{RESET}")));
                let rest = &trimmed[first.len()..];
                if !rest.is_empty() {
                    styled.push((Default::default(), rest.to_string()));
                }
            }
            // NL prefix — magenta
            _ if trimmed.starts_with('?') => {
                styled.push((Default::default(), format!("{MAGENTA}{trimmed}{RESET}")));
            }
            // CLI force prefix — yellow
            _ if trimmed.starts_with('!') => {
                styled.push((Default::default(), format!("{YELLOW}{trimmed}{RESET}")));
            }
            // Known Unix commands — default with highlighted command name
            _ if unix::command_exists(first) => {
                styled.push((Default::default(), format!("{GREEN}{first}{RESET}")));
                let rest = &trimmed[first.len()..];
                highlight_pipe_to_llm(&mut styled, rest);
            }
            // Unknown — natural language (dim)
            _ => {
                styled.push((Default::default(), format!("{DIM}{trimmed}{RESET}")));
            }
        }

        styled
    }
}

/// Highlight a `spirit <subcommand> <args>` line.
fn highlight_spirit_command(styled: &mut StyledText, line: &str) {
    let parts: Vec<&str> = line.splitn(3, ' ').collect();

    // "spirit" keyword
    styled.push((
        Default::default(),
        format!("{GREEN}{BOLD}{}{RESET}", parts[0]),
    ));

    if parts.len() > 1 {
        // Subcommand
        styled.push((Default::default(), format!(" {CYAN}{}{RESET}", parts[1])));
    }

    if parts.len() > 2 {
        // Arguments
        styled.push((Default::default(), format!(" {}", parts[2])));
    }
}

/// Check for pipe-to-LLM and highlight the NL portion.
fn highlight_pipe_to_llm(styled: &mut StyledText, rest: &str) {
    // Look for `| <natural language>` pattern
    if let Some(pipe_pos) = rest.rfind('|') {
        let before_pipe = &rest[..pipe_pos];
        let after_pipe = rest[pipe_pos + 1..].trim();

        let right_first = unix::first_word(after_pipe);
        if !unix::command_exists(right_first) && !after_pipe.is_empty() {
            // It's a pipe-to-LLM
            styled.push((Default::default(), before_pipe.to_string()));
            styled.push((Default::default(), format!("{YELLOW}|{RESET} ")));
            styled.push((Default::default(), format!("{MAGENTA}{after_pipe}{RESET}")));
            return;
        }
    }

    // Normal rest of command
    if !rest.is_empty() {
        styled.push((Default::default(), rest.to_string()));
    }
}
