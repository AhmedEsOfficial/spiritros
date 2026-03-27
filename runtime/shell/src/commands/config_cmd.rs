//! Spirit config commands: show, get, set.

use crate::commands::SharedState;
use crate::types::CommandResult;

/// Dispatch `spirit config <subcommand>`.
pub async fn dispatch(state: &SharedState, args: &[&str]) -> CommandResult {
    if args.is_empty() {
        return CommandResult::Output(
            "Usage: spirit config <show|get|set> [args...]\n\n\
             Subcommands:\n  \
             show            Show current shell configuration\n  \
             get <key>       Get a specific configuration value\n  \
             set <key> <val> Set a configuration value (session only)"
                .into(),
        );
    }

    match args[0] {
        "show" => show(state).await,
        "get" => {
            if args.len() < 2 {
                return CommandResult::Error("Usage: spirit config get <key>".into());
            }
            get(state, args[1]).await
        }
        "set" => {
            if args.len() < 3 {
                return CommandResult::Error("Usage: spirit config set <key> <value>".into());
            }
            set(state, args[1], &args[2..].join(" ")).await
        }
        other => CommandResult::Error(format!(
            "Unknown config subcommand: '{other}'. Use: show, get, set"
        )),
    }
}

/// Show the current configuration.
async fn show(state: &SharedState) -> CommandResult {
    let state = state.read().await;
    let config = &state.config;

    let lines = vec![
        "Spirit Shell Configuration".to_string(),
        "─".repeat(40),
        format!("  mode:                {}", config.mode),
        format!("  llm.model:           {}", config.llm.model),
        format!("  llm.max_context:     {}", config.llm.max_context_tokens),
        format!("  llm.suggest_cmds:    {}", config.llm.suggest_commands),
        format!("  llm.semantic_search: {}", config.llm.semantic_search),
        format!("  prompt.format:       {}", config.prompt.format),
        format!("  prompt.show_model:   {}", config.prompt.show_model),
        format!("  history.enabled:     {}", config.history.enabled),
        format!("  history.max_entries: {}", config.history.max_entries),
        format!("  history.path:        {}", config.history.path.display()),
        format!("  inference.ollama_url: {}", config.inference.ollama_url),
        format!("  inference.timeout:   {}s", config.inference.timeout_secs),
        String::new(),
        "Aliases:".to_string(),
    ];

    let mut lines = lines;
    for (alias, expansion) in &config.aliases {
        lines.push(format!("  {alias} = {expansion}"));
    }

    CommandResult::Output(lines.join("\n"))
}

/// Get a specific configuration value.
async fn get(state: &SharedState, key: &str) -> CommandResult {
    let state = state.read().await;
    let config = &state.config;

    let value = match key {
        "mode" => config.mode.to_string(),
        "llm.model" | "model" => config.llm.model.clone(),
        "llm.max_context" | "max_context" => config.llm.max_context_tokens.to_string(),
        "llm.suggest_commands" => config.llm.suggest_commands.to_string(),
        "llm.semantic_search" => config.llm.semantic_search.to_string(),
        "prompt.format" => config.prompt.format.clone(),
        "prompt.show_model" => config.prompt.show_model.to_string(),
        "history.enabled" => config.history.enabled.to_string(),
        "history.max_entries" => config.history.max_entries.to_string(),
        "history.path" => config.history.path.display().to_string(),
        "inference.ollama_url" | "ollama_url" => config.inference.ollama_url.clone(),
        "inference.timeout" | "timeout" => config.inference.timeout_secs.to_string(),
        _ => return CommandResult::Error(format!("Unknown config key: '{key}'")),
    };

    CommandResult::Output(format!("{key} = {value}"))
}

/// Set a configuration value (session-only, not persisted).
async fn set(state: &SharedState, key: &str, value: &str) -> CommandResult {
    let mut state = state.write().await;

    match key {
        "mode" => match value.parse() {
            Ok(mode) => {
                state.config.mode = mode;
                state.resolver.set_mode(mode);
            }
            Err(e) => return CommandResult::Error(e),
        },
        "llm.model" | "model" => {
            state.config.llm.model = value.to_string();
            state.inference.set_model(value.to_string());
        }
        "llm.max_context" | "max_context" => match value.parse::<u32>() {
            Ok(v) => state.config.llm.max_context_tokens = v,
            Err(_) => return CommandResult::Error("max_context must be a number".into()),
        },
        "prompt.format" => state.config.prompt.format = value.to_string(),
        "prompt.show_model" => match value.parse::<bool>() {
            Ok(v) => state.config.prompt.show_model = v,
            Err(_) => return CommandResult::Error("show_model must be true or false".into()),
        },
        _ => return CommandResult::Error(format!("Cannot set '{key}' — unknown or read-only key")),
    }

    CommandResult::Output(format!("{key} = {value} (session only)"))
}
