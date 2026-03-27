//! Spirit agent commands: list, start, stop.

use crate::commands::SharedState;
use crate::types::{AgentStatus, CommandResult};

/// Dispatch `spirit agent <subcommand>`.
pub async fn dispatch(state: &SharedState, args: &[&str]) -> CommandResult {
    if args.is_empty() {
        return CommandResult::Output(
            "Usage: spirit agent <list|start|stop> [args...]\n\n\
             Subcommands:\n  \
             list                List all agents (running and completed)\n  \
             start <task>        Start a new agentic task\n  \
             stop <id>           Stop a running agent\n\n\
             Agentic tasks run autonomously in the background.\n\
             They are governed by the capability system."
                .into(),
        );
    }

    match args[0] {
        "list" | "ls" => list(state).await,
        "start" => {
            if args.len() < 2 {
                return CommandResult::Error("Usage: spirit agent start <task description>".into());
            }
            let task = args[1..].join(" ");
            start(state, &task).await
        }
        "stop" | "kill" => {
            if args.len() < 2 {
                return CommandResult::Error("Usage: spirit agent stop <id>".into());
            }
            stop(state, args[1]).await
        }
        other => CommandResult::Error(format!(
            "Unknown agent subcommand: '{other}'. Use: list, start, stop"
        )),
    }
}

/// List all agents.
async fn list(state: &SharedState) -> CommandResult {
    let agents = state.read().await.agent_manager.list().await;

    if agents.is_empty() {
        return CommandResult::Output("No agents running or completed.".into());
    }

    let mut lines = Vec::with_capacity(agents.len() + 3);
    lines.push(format!(
        "{:<10} {:<14} {:<40} {:<8}",
        "ID", "STATUS", "TASK", "ACTIONS"
    ));
    lines.push("─".repeat(74));

    for agent in &agents {
        let status_indicator = match agent.status {
            AgentStatus::Running => "\x1b[32m●\x1b[0m running",
            AgentStatus::Starting => "\x1b[33m●\x1b[0m starting",
            AgentStatus::AwaitingApproval => "\x1b[33m●\x1b[0m awaiting",
            AgentStatus::Completed => "\x1b[34m●\x1b[0m completed",
            AgentStatus::Failed => "\x1b[31m●\x1b[0m failed",
            AgentStatus::Cancelled => "\x1b[90m●\x1b[0m cancelled",
        };
        let task_display = if agent.task.len() > 38 {
            format!("{}...", &agent.task[..35])
        } else {
            agent.task.clone()
        };
        lines.push(format!(
            "{:<10} {:<14} {:<40} {:<8}",
            agent.id, status_indicator, task_display, agent.actions_taken
        ));
    }

    CommandResult::Output(lines.join("\n"))
}

/// Start a new agent.
async fn start(state: &SharedState, task: &str) -> CommandResult {
    let state_guard = state.read().await;
    let client = state_guard.inference.clone();

    println!("Starting agent for task: {task}");
    println!();

    match state_guard.agent_manager.start(task, client).await {
        Ok(id) => CommandResult::Output(format!(
            "Agent {id} started.\n\
             Monitor with: spirit agent list\n\
             Stop with:    spirit agent stop {id}"
        )),
        Err(e) => CommandResult::Error(format!("Failed to start agent: {e}")),
    }
}

/// Stop a running agent.
async fn stop(state: &SharedState, id_str: &str) -> CommandResult {
    // Parse the agent ID (short form: first 8 chars of UUID)
    let agents = state.read().await.agent_manager.list().await;

    let matching: Vec<_> = agents
        .iter()
        .filter(|a| a.id.short().starts_with(id_str) || a.id.0.to_string().starts_with(id_str))
        .collect();

    match matching.len() {
        0 => CommandResult::Error(format!("No agent found matching '{id_str}'")),
        1 => {
            let id = matching[0].id;
            match state.read().await.agent_manager.stop(id).await {
                Ok(()) => CommandResult::Output(format!("Agent {id} stopped.")),
                Err(e) => CommandResult::Error(format!("Failed to stop agent: {e}")),
            }
        }
        _ => CommandResult::Error(format!(
            "Ambiguous ID '{id_str}' — matches {} agents. Use a longer prefix.",
            matching.len()
        )),
    }
}
