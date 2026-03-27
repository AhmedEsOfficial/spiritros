//! Spirit model commands: list, load, unload, pull.

use crate::commands::SharedState;
use crate::inference::format_bytes;
use crate::types::CommandResult;

/// Dispatch `spirit model <subcommand>`.
pub async fn dispatch(state: &SharedState, args: &[&str]) -> CommandResult {
    if args.is_empty() {
        return CommandResult::Output(
            "Usage: spirit model <list|load|unload|pull> [args...]\n\n\
             Subcommands:\n  \
             list                List available models\n  \
             load <name>         Load a model into memory\n  \
             unload <name>       Unload a model from memory\n  \
             pull <name>         Pull a model from the registry"
                .into(),
        );
    }

    match args[0] {
        "list" | "ls" => list(state).await,
        "load" => {
            if args.len() < 2 {
                return CommandResult::Error("Usage: spirit model load <name>".into());
            }
            load(state, args[1]).await
        }
        "unload" => {
            if args.len() < 2 {
                return CommandResult::Error("Usage: spirit model unload <name>".into());
            }
            unload(state, args[1]).await
        }
        "pull" | "download" => {
            if args.len() < 2 {
                return CommandResult::Error("Usage: spirit model pull <name>".into());
            }
            pull(state, args[1]).await
        }
        other => CommandResult::Error(format!(
            "Unknown model subcommand: '{other}'. Use: list, load, unload, pull"
        )),
    }
}

/// List all models available in Ollama.
async fn list(state: &SharedState) -> CommandResult {
    let client = state.read().await.inference.clone();

    match client.list_models().await {
        Ok(models) => {
            if models.is_empty() {
                return CommandResult::Output(
                    "No models available. Pull one with: spirit model pull <name>".into(),
                );
            }

            let mut lines = Vec::with_capacity(models.len() + 3);
            lines.push("Available Models".to_string());
            lines.push(format!(
                "{:<30} {:<12} {:<8} {:<10}",
                "NAME", "SIZE", "QUANT", "FAMILY"
            ));
            lines.push("─".repeat(64));

            for model in &models {
                let size = format_bytes(model.size);
                let (quant, family) = model.details.as_ref().map_or(("", ""), |d| {
                    (d.quantization_level.as_str(), d.family.as_str())
                });
                lines.push(format!(
                    "{:<30} {:<12} {:<8} {:<10}",
                    model.name, size, quant, family
                ));
            }

            lines.push(format!("\n{} model(s) available", models.len()));
            CommandResult::Output(lines.join("\n"))
        }
        Err(e) => CommandResult::Error(format!("Failed to list models: {e}")),
    }
}

/// Load a model into Ollama (warm it up).
async fn load(state: &SharedState, name: &str) -> CommandResult {
    println!("Loading model '{name}'...");
    let client = state.read().await.inference.clone();

    match client.load_model(name).await {
        Ok(()) => CommandResult::Output(format!("Model '{name}' loaded successfully.")),
        Err(e) => CommandResult::Error(format!("Failed to load model '{name}': {e}")),
    }
}

/// Unload a model from Ollama.
async fn unload(state: &SharedState, name: &str) -> CommandResult {
    let client = state.read().await.inference.clone();

    match client.unload_model(name).await {
        Ok(()) => CommandResult::Output(format!("Model '{name}' unloaded.")),
        Err(e) => CommandResult::Error(format!("Failed to unload model '{name}': {e}")),
    }
}

/// Pull a model from the Ollama registry.
async fn pull(state: &SharedState, name: &str) -> CommandResult {
    println!("Pulling model '{name}' from registry...");
    let client = state.read().await.inference.clone();

    match client.pull_model(name).await {
        Ok(()) => CommandResult::Output(format!("Model '{name}' pulled successfully.")),
        Err(e) => CommandResult::Error(format!("Failed to pull model '{name}': {e}")),
    }
}
