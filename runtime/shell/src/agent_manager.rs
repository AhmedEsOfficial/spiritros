//! Agent manager for the Spirit Shell.
//!
//! Manages agentic workflows — long-running autonomous LLM-driven tasks that
//! run as background processes with their own session, capability set, and
//! audit log.
//!
//! PHASE1-SHIM: Basic agent lifecycle. Phase 2+ integrates with the capability
//! system for fine-grained permission control and the kernel's process model.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::Utc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::error::{self, ShellError};
use crate::inference::InferenceClient;
use crate::types::{AgentId, AgentInfo, AgentStatus};

/// Manages the lifecycle of agentic tasks.
pub struct AgentManager {
    agents: Arc<RwLock<HashMap<AgentId, AgentEntry>>>,
    log_dir: PathBuf,
}

/// Internal agent entry with task handle.
struct AgentEntry {
    info: AgentInfo,
    cancel_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl Default for AgentManager {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentManager {
    /// Create a new agent manager.
    pub fn new() -> Self {
        let log_dir = std::env::var("SPIRIT_LOG_DIR")
            .map_or_else(|_| PathBuf::from("/var/log/spiritros"), PathBuf::from)
            .join("agents");

        // Best-effort directory creation
        let _ = std::fs::create_dir_all(&log_dir);

        Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            log_dir,
        }
    }

    /// Start a new agentic task.
    pub async fn start(&self, task: &str, client: InferenceClient) -> error::Result<AgentId> {
        let id = AgentId::new();
        let log_path = self.log_dir.join(format!("agent-{}.log", id.short()));

        let info = AgentInfo {
            id,
            task: task.to_string(),
            status: AgentStatus::Starting,
            started_at: Utc::now(),
            actions_taken: 0,
            log_path: log_path.to_string_lossy().to_string(),
        };

        let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();

        let entry = AgentEntry {
            info: info.clone(),
            cancel_tx: Some(cancel_tx),
        };

        self.agents.write().await.insert(id, entry);

        // Spawn the agent task
        let agents = self.agents.clone();
        let task_str = task.to_string();
        let log_path_clone = log_path.clone();

        tokio::spawn(async move {
            // Update status to running
            if let Some(entry) = agents.write().await.get_mut(&id) {
                entry.info.status = AgentStatus::Running;
            }

            info!(agent_id = %id, task = %task_str, "Agent started");

            // Run the agent loop
            let result = run_agent_task(&task_str, &client, cancel_rx, &log_path_clone).await;

            // Update final status
            let mut agents = agents.write().await;
            if let Some(entry) = agents.get_mut(&id) {
                match result {
                    Ok(actions) => {
                        entry.info.status = AgentStatus::Completed;
                        entry.info.actions_taken = actions;
                        info!(agent_id = %id, actions, "Agent completed");
                    }
                    Err(e) => {
                        if e.to_string().contains("cancelled") {
                            entry.info.status = AgentStatus::Cancelled;
                            info!(agent_id = %id, "Agent cancelled");
                        } else {
                            entry.info.status = AgentStatus::Failed;
                            warn!(agent_id = %id, error = %e, "Agent failed");
                        }
                    }
                }
                entry.cancel_tx = None;
            }
        });

        Ok(id)
    }

    /// Stop a running agent.
    pub async fn stop(&self, id: AgentId) -> error::Result<()> {
        let mut agents = self.agents.write().await;
        let entry = agents
            .get_mut(&id)
            .ok_or_else(|| ShellError::AgentError(format!("Agent {id} not found")))?;

        if entry.info.status != AgentStatus::Running && entry.info.status != AgentStatus::Starting {
            return Err(ShellError::AgentError(format!(
                "Agent {id} is not running (status: {})",
                entry.info.status
            )));
        }

        if let Some(cancel_tx) = entry.cancel_tx.take() {
            let _ = cancel_tx.send(());
        }
        entry.info.status = AgentStatus::Cancelled;

        Ok(())
    }

    /// List all agents (running and completed).
    pub async fn list(&self) -> Vec<AgentInfo> {
        self.agents
            .read()
            .await
            .values()
            .map(|e| e.info.clone())
            .collect()
    }

    /// Get info about a specific agent.
    pub async fn get(&self, id: AgentId) -> Option<AgentInfo> {
        self.agents.read().await.get(&id).map(|e| e.info.clone())
    }

    /// Get the count of currently running agents.
    pub async fn running_count(&self) -> usize {
        self.agents
            .read()
            .await
            .values()
            .filter(|e| {
                e.info.status == AgentStatus::Running || e.info.status == AgentStatus::Starting
            })
            .count()
    }
}

/// Run the agentic task loop.
///
/// The agent sends the task description to the LLM with a specialized system
/// prompt, then streams the response. In Phase 1, this is a single-shot
/// generation. Phase 2+ will support multi-step planning and execution.
async fn run_agent_task(
    task: &str,
    client: &InferenceClient,
    cancel_rx: tokio::sync::oneshot::Receiver<()>,
    log_path: &std::path::Path,
) -> error::Result<u32> {
    use std::io::Write;

    // Open log file (best effort)
    let mut log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .ok();

    let log = |file: &mut Option<std::fs::File>, msg: &str| {
        let ts = Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ");
        if let Some(f) = file.as_mut() {
            let _ = writeln!(f, "[{ts}] {msg}");
        }
    };

    log(&mut log_file, &format!("Agent task: {task}"));
    log(&mut log_file, "Status: running");

    let system = format!(
        "You are a Spirit Shell agent executing an autonomous task. \
         Your task is: {task}\n\n\
         Analyze the task and provide a step-by-step plan, then describe \
         what actions you would take. For each action, explain what you \
         would do and why.\n\n\
         IMPORTANT: In Phase 1, you can only describe actions — you cannot \
         execute them autonomously. The user will review your plan."
    );

    // Use select to support cancellation
    tokio::select! {
        result = client.generate_streaming(task, Some(&system)) => {
            match result {
                Ok(stats) => {
                    log(&mut log_file, &format!(
                        "Completed: {} tokens in {}ms",
                        stats.tokens_generated, stats.total_ms
                    ));
                    Ok(1) // One action (the generation) in Phase 1
                }
                Err(e) => {
                    log(&mut log_file, &format!("Error: {e}"));
                    Err(e)
                }
            }
        }
        _ = cancel_rx => {
            log(&mut log_file, "Cancelled by user");
            Err(ShellError::AgentError("cancelled".into()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_manager_creation() {
        let _manager = AgentManager::new();
    }

    #[tokio::test]
    async fn test_agent_list_empty() {
        let manager = AgentManager::new();
        let agents = manager.list().await;
        assert!(agents.is_empty());
    }

    #[tokio::test]
    async fn test_agent_running_count_empty() {
        let manager = AgentManager::new();
        assert_eq!(manager.running_count().await, 0);
    }
}
