//! Spirit infer command — direct inference invocation.

use crate::commands::SharedState;
use crate::types::CommandResult;

/// Dispatch `spirit infer <prompt>`.
pub async fn dispatch(state: &SharedState, args: &[&str]) -> CommandResult {
    if args.is_empty() {
        return CommandResult::Output(
            "Usage: spirit infer <prompt>\n\n\
             Send a prompt directly to the inference engine.\n\
             The response streams to the terminal in real time.\n\n\
             Examples:\n  \
             spirit infer \"Explain the TensorFirst memory architecture\"\n  \
             spirit infer \"Write a Rust function to sort a Vec<i32>\""
                .into(),
        );
    }

    let prompt = args.join(" ");
    let client = state.read().await.inference.clone();

    match client.generate_streaming(&prompt, None).await {
        Ok(stats) => {
            // Print generation statistics
            let stats_line = format!(
                "\n[{} tokens, {:.1} tok/s, first token {}ms, total {}ms]",
                stats.tokens_generated,
                stats.tokens_per_second,
                stats.first_token_ms,
                stats.total_ms
            );
            CommandResult::Output(stats_line)
        }
        Err(e) => CommandResult::Error(format!("Inference failed: {e}")),
    }
}
