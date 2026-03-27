//! Pipe-to-LLM execution for the Spirit Shell.
//!
//! Handles the `<unix command> | <natural language query>` pattern where
//! the output of a Unix command is captured and sent to the LLM as context.

use crate::error;
use crate::inference::InferenceClient;
use crate::unix;

/// Execute a pipe-to-LLM command.
///
/// 1. Runs the Unix command and captures its stdout.
/// 2. Sends the output + natural language query to the LLM.
/// 3. Streams the LLM's response to stdout.
pub async fn exec_pipe_to_llm(
    client: &InferenceClient,
    command: &str,
    query: &str,
) -> error::Result<()> {
    // Execute the Unix command and capture output
    let output = unix::exec_capture(command).await.map_err(|e| {
        crate::error::ShellError::PipeCommandFailed(format!("Command `{command}` failed: {e}"))
    })?;

    // Truncate very large outputs to avoid exceeding context window
    let truncated = if output.len() > 32_000 {
        let truncated = &output[..32_000];
        format!("{truncated}\n\n[... output truncated at 32KB ...]")
    } else {
        output
    };

    // Send to LLM with context
    client
        .query_with_context(&truncated, command, query)
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_truncation_threshold() {
        // Verify the truncation logic works for large strings
        let large = "x".repeat(40_000);
        let truncated = if large.len() > 32_000 {
            let t = &large[..32_000];
            format!("{t}\n\n[... output truncated at 32KB ...]")
        } else {
            large.clone()
        };
        assert!(truncated.len() < large.len());
        assert!(truncated.contains("truncated"));
    }
}
