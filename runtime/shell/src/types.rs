//! Public types for the Spirit Shell.

/// The result of executing a built-in command.
#[derive(Debug)]
pub enum CommandResult {
    /// The command produced output to display.
    Output(String),
    /// The command requests a screen clear.
    Clear,
    /// The command requests the shell to exit.
    Exit,
    /// The input was not a built-in command and should be routed to inference.
    NotBuiltin,
}
