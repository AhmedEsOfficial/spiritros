//! Core shell built-in commands: help, version, status, history.

use crate::commands::SharedState;
use crate::types::CommandResult;

/// Display help information.
pub async fn help(state: &SharedState) -> CommandResult {
    let mode = state.read().await.config.mode;
    let text = format!(
        "\
Spirit Shell — LLM-Native Interface for SpiritROS
═══════════════════════════════════════════════════

Shell Commands:
  help                     Show this help message
  version                  Show Spirit Shell version
  status                   Show system status overview
  clear                    Clear the terminal screen
  exit / quit              Exit the shell
  history                  Show command history info

Spirit Commands (prefix with 'spirit'):
  spirit model list        List available models
  spirit model load <n>    Load a model into memory
  spirit model unload <n>  Unload a model from memory
  spirit model pull <n>    Pull a model from the registry
  spirit mem status        Show memory zone utilization
  spirit sched status      Show scheduler state
  spirit find <query>      Semantic file search
  spirit infer <prompt>    Direct inference invocation
  spirit agent list        List running agents
  spirit agent start <t>   Start an agentic task
  spirit agent stop <id>   Stop a running agent
  spirit shell mode <m>    Switch shell mode (hybrid/cli/llm)
  spirit config show       Show current configuration

Mode: {mode}
  Hybrid  — CLI commands first, natural language fallback
  CLI     — Only built-in and Unix commands (no LLM)
  LLM     — Everything routed to the inference engine

Prefixes:
  ? <query>    Force natural language interpretation
  ! <cmd>      Force Unix command execution

Pipe-to-LLM:
  <cmd> | <query>   Pipe command output to the LLM
  Example: ls -la /var/log | explain what happened today"
    );
    CommandResult::Output(text)
}

/// Display version information.
pub fn version() -> CommandResult {
    let version = env!("CARGO_PKG_VERSION");
    CommandResult::Output(format!(
        "Spirit Shell v{version} (Phase 1: Virtualized Deployment)"
    ))
}

/// Display system status overview.
pub async fn status(state: &SharedState) -> CommandResult {
    let state = state.read().await;
    let mode = state.config.mode;
    let model = state.inference.model();
    let healthy = state.inference.health_check().await;

    let inference_status = if healthy {
        format!("connected (model: {model})")
    } else {
        "unavailable".to_string()
    };

    let running_agents = state.agent_manager.running_count().await;

    let text = format!(
        "\
SpiritROS System Status (Phase 1)
──────────────────────────────────────────
  Inference Engine:  {inference_status}
  Active Agents:     {running_agents}
  Shell Mode:        {mode}
  Ollama URL:        {}

  Memory Zones:      [cgroups v2 shim]
  Scheduler:         [cpuset/SCHED_FIFO shim]
  SpiritFS:          [FUSE-based]
  Security:          [capability system]",
        state.config.inference.ollama_url
    );
    CommandResult::Output(text)
}

/// Display command history info.
pub fn history() -> CommandResult {
    CommandResult::Output(
        "Command history is stored in ~/.spirit/shell_history\n\
         Use Up/Down arrows to navigate, Ctrl+R to search."
            .into(),
    )
}

/// Display spirit command help (when user types just "spirit").
pub fn spirit_help() -> CommandResult {
    CommandResult::Output(
        "\
Spirit — SpiritROS management commands

Usage: spirit <subcommand> [args...]

Subcommands:
  model       Manage inference models (list, load, unload, pull)
  mem         Memory zone management (status, rebalance)
  sched       Scheduler status and control
  find        Semantic file search
  similar     Find similar files
  summarize   Summarize directory contents
  infer       Direct inference invocation
  agent       Manage agentic tasks (list, start, stop)
  shell       Shell settings (mode switching)
  config      Configuration management (show, get, set)
  bench       Run performance benchmarks
  status      System status overview
  version     Version information"
            .into(),
    )
}
