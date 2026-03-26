//! Basic reedline [`Completer`] for Spirit Shell built-in commands.

use reedline::{Completer, Span, Suggestion};

/// Completer that suggests built-in command names.
pub struct SpiritCompleter {
    commands: Vec<String>,
}

impl SpiritCompleter {
    /// Create a new completer from a list of command names.
    #[must_use]
    pub fn new(commands: Vec<&str>) -> Self {
        Self {
            commands: commands.into_iter().map(String::from).collect(),
        }
    }
}

impl Completer for SpiritCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<Suggestion> {
        // Only complete the first token (command name).
        let prefix = &line[..pos];
        if prefix.contains(' ') {
            return Vec::new();
        }

        self.commands
            .iter()
            .filter(|cmd| cmd.starts_with(prefix))
            .map(|cmd| Suggestion {
                value: cmd.clone(),
                description: None,
                style: None,
                extra: None,
                span: Span::new(0, pos),
                append_whitespace: true,
            })
            .collect()
    }
}
