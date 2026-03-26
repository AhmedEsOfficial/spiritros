# Phase 06: Spirit Shell — LLM-Native Interface

**Roadmap Alignment:** Phase 1 — Virtual Proof of Concept, Milestone M1.1 (prototype), refined through M1.5
**Estimated Duration:** Month 1 (stub), Month 4–5 (full)
**Dependencies:** Phase 01 (Foundation), Phase 03 (Inference Engine), Phase 05 (SpiritFS — for semantic commands)
**Architecture Plan Sections:** §11

---

## Context & Overview

The Spirit Shell is SpiritROS's primary user interface — a command-line environment where natural language and traditional Unix commands coexist. A user can type `ls -la`, a standard Unix command, or "What models do I have loaded?", a natural language query — and the shell handles both seamlessly.

The shell is also the entry point for **agentic workflows** where the LLM can autonomously execute multi-step tasks with capability-governed permissions.

Phase 01 delivers a basic stub (accepts input, echoes back). This phase builds the full Shell with command resolution, LLM integration, and built-in Spirit commands.

---

## Goals

1. **Command resolution pipeline** — Priority chain: Spirit built-ins → Unix commands → natural language fallback
2. **Built-in Spirit commands** — `spirit model`, `spirit mem`, `spirit sched`, `spirit find`, etc.
3. **Natural language mode** — Queries routed to the loaded LLM for interpretation and action
4. **Pipe integration** — `ls -la | explain what happened today` pipes Unix output to LLM
5. **Agentic mode** — Long-running autonomous tasks with capability checks
6. **Mode switching** — Pure CLI mode (no LLM), pure LLM mode, or hybrid (default)
7. **Rich line editing** — History, completion, syntax highlighting via `reedline`
8. **Shell configuration** — Customizable prompt, aliases, startup scripts

---

## Design Decisions

### DD-06.1: Rust with reedline for Line Editing
**Decision:** Use the `reedline` crate (from the Nushell project) for line editing, history, and completion.
**Rationale:** `reedline` is modern, actively maintained, and designed for extensible shells. It supports custom completers, highlighters, and validators — all needed for the dual CLI/NL interface.

### DD-06.2: Three-Stage Command Resolution
**Decision:** Input is resolved through a priority chain:
1. **Spirit built-in commands** (prefixed with `spirit` or matching a known pattern)
2. **Standard Unix commands** (found in `$PATH`, executed via Linux compat layer)
3. **Natural language fallback** (everything else routed to the LLM)
**Rationale:** This ensures existing CLI workflows are never broken while enabling natural language for everything else. Users can force NL mode with quotes or a prefix.

### DD-06.3: Pipe-to-LLM Syntax
**Decision:** The `|` operator followed by a natural language phrase pipes the preceding command's stdout to the LLM as context for the NL query.
**Rationale:** This is the killer UX feature — it lets users combine Unix tools with LLM analysis in a single pipeline. The syntax is natural and consistent with Unix piping conventions.

### DD-06.4: Agentic Tasks as Background Processes
**Decision:** Agentic tasks (multi-step LLM-driven workflows) run as background "agent processes" with their own session, capability set, and audit log.
**Rationale:** Agents may run for minutes or hours. They need to be non-blocking, auditable, and controllable (pause, resume, kill). They also need capability limits to prevent unintended actions.

### DD-06.5: No GUI in Phase 1
**Decision:** The Spirit Shell is CLI-only in Phase 1. A web-based UI (port 8080) is a stretch goal.
**Rationale:** CLI-first development focuses on core functionality. The web UI can be added as a thin layer over the same command resolution engine later.

---

## Components & Tasks

### 1. Shell Core (`runtime/shell/`)
```rust
pub struct SpiritShell {
    editor: Reedline,
    resolver: CommandResolver,
    inference: InferenceClient,  // Connects to Inference Coordinator
    config: ShellConfig,
    mode: ShellMode,
    agent_manager: AgentManager,
}

pub enum ShellMode {
    Hybrid,    // Default: try CLI first, fall back to LLM
    CliOnly,   // No LLM interpretation
    LlmOnly,   // Everything goes to LLM
}

impl SpiritShell {
    pub async fn run(&mut self) -> Result<()> {
        loop {
            let input = self.editor.read_line(&self.prompt())?;
            let result = self.resolver.resolve(&input).await?;
            match result {
                Resolution::SpiritBuiltin(cmd) => self.exec_builtin(cmd).await?,
                Resolution::UnixCommand(cmd) => self.exec_unix(cmd).await?,
                Resolution::NaturalLanguage(query) => self.exec_nl(query).await?,
                Resolution::PipeToLlm(cmd, query) => self.exec_pipe_llm(cmd, query).await?,
                Resolution::Agent(task) => self.start_agent(task).await?,
            }
        }
    }
}
```

### 2. Command Resolver
```rust
pub struct CommandResolver {
    builtins: HashMap<String, BuiltinCommand>,
    path_dirs: Vec<PathBuf>,
}

impl CommandResolver {
    pub async fn resolve(&self, input: &str) -> Result<Resolution> {
        // 1. Check for Spirit built-in commands
        if let Some(builtin) = self.match_builtin(input) {
            return Ok(Resolution::SpiritBuiltin(builtin));
        }

        // 2. Check for pipe-to-LLM pattern (cmd | natural language)
        if let Some((cmd, nl_query)) = self.detect_pipe_to_llm(input) {
            return Ok(Resolution::PipeToLlm(cmd, nl_query));
        }

        // 3. Check for Unix command (first word found in PATH)
        if let Some(unix_cmd) = self.match_unix_command(input) {
            return Ok(Resolution::UnixCommand(unix_cmd));
        }

        // 4. Fall back to natural language
        Ok(Resolution::NaturalLanguage(input.to_string()))
    }
}
```

### 3. Built-in Spirit Commands
| Command | Description | Phase 1 Scope |
|---|---|---|
| `spirit model list` | List loaded/available models | Full |
| `spirit model load <name>` | Load a model into memory | Full |
| `spirit model unload <name>` | Unload a model | Full |
| `spirit model pull <name>` | Pull from registry (wraps Ollama) | Full |
| `spirit mem status` | Show memory zone utilization | Full |
| `spirit mem rebalance` | Dynamically resize zones | Full |
| `spirit sched status` | Show scheduler state and phase | Full |
| `spirit find <query>` | Semantic file search (SpiritFS) | Full |
| `spirit similar <file>` | Find similar files | Full |
| `spirit summarize <path>` | Summarize directory contents | Full |
| `spirit infer <prompt>` | Direct inference invocation | Full |
| `spirit agent list` | List running agents | Basic |
| `spirit agent start <task>` | Start an agentic task | Basic |
| `spirit agent stop <id>` | Stop an agent | Basic |
| `spirit bench` | Run benchmarks | Delegates to spirit-bench |
| `spirit config` | View/edit Spirit configuration | Full |

### 4. Natural Language Execution
```rust
async fn exec_nl(&self, query: String) -> Result<()> {
    // System prompt instructs the LLM about available Spirit commands
    // and system state (loaded models, memory zones, etc.)
    let system_prompt = self.build_system_prompt().await?;

    // Stream LLM response
    let stream = self.inference.prompt_with_system(
        &system_prompt,
        &query,
    ).await?;

    for token in stream {
        print!("{}", token?);
    }
    println!();
    Ok(())
}
```

The system prompt includes:
- Available Spirit commands and their syntax
- Current system state (loaded models, memory stats, scheduler phase)
- User's current directory and recent command history
- Instructions for when to suggest commands vs. execute them

### 5. Pipe-to-LLM
```rust
async fn exec_pipe_llm(&self, cmd: String, query: String) -> Result<()> {
    // Execute the Unix command and capture output
    let output = self.exec_unix_capture(&cmd).await?;

    // Send output + query to LLM
    let prompt = format!(
        "The following is the output of the command `{}`:\n\n```\n{}\n```\n\n{}",
        cmd, output, query
    );

    let stream = self.inference.prompt(&prompt).await?;
    for token in stream {
        print!("{}", token?);
    }
    println!();
    Ok(())
}
```

### 6. Agent Manager
```rust
pub struct AgentManager {
    agents: HashMap<AgentId, AgentHandle>,
}

pub struct AgentHandle {
    id: AgentId,
    task: String,
    status: AgentStatus,
    capabilities: CapabilitySet,
    session: SessionId,
    log_path: PathBuf,
    approval_required_for: Vec<ActionType>,
}

impl AgentManager {
    pub async fn start(&mut self, task: &str, caps: CapabilitySet) -> Result<AgentId>;
    pub async fn stop(&mut self, id: AgentId) -> Result<()>;
    pub async fn status(&self, id: AgentId) -> AgentStatus;
    pub async fn approve(&self, id: AgentId, action: &Action) -> Result<()>;
}
```

### 7. Shell Configuration
```toml
# ~/.spirit/shell.toml
[prompt]
format = "spirit@{hostname} {cwd} $ "
show_model = true        # Show currently loaded model in prompt
show_phase = false       # Show scheduler phase (prefill/decode/idle)

[mode]
default = "hybrid"       # hybrid, cli-only, llm-only

[completion]
enable = true
unix_commands = true
spirit_commands = true
file_paths = true

[history]
max_entries = 10000
file = "~/.spirit/history"

[aliases]
ll = "ls -la"
models = "spirit model list"
mem = "spirit mem status"
```

---

## Acceptance Criteria

- [ ] Shell starts and displays a welcome banner with version and loaded model info
- [ ] Unix commands (ls, cd, grep, cat, git) execute correctly via compat layer
- [ ] Spirit built-in commands work: `spirit model list`, `spirit mem status`, `spirit sched status`
- [ ] Natural language queries are routed to the LLM and responses stream to the terminal
- [ ] Pipe-to-LLM works: `ls -la /var/log/ | explain what happened today`
- [ ] Tab completion works for Spirit commands, Unix commands, and file paths
- [ ] Command history persists across sessions
- [ ] Mode switching works: `spirit shell mode cli-only`, `spirit shell mode llm-only`
- [ ] Basic agentic task can be started, monitored, and stopped
- [ ] Agent actions are logged to `/var/log/spirit/agents/`
- [ ] Shell handles Ctrl+C (interrupt current operation), Ctrl+D (exit) correctly
- [ ] Shell config file is loaded and applied at startup

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| LLM misinterprets commands as natural language (or vice versa) | Wrong execution path | Tune resolver heuristics; allow explicit prefix (`?` for NL, `!` for CLI) |
| Streaming LLM output is slow, feels unresponsive | Poor UX | Show spinner during prefill; ensure token streaming starts within 200ms |
| Agentic tasks execute dangerous operations | Security risk | Capability system limits agent actions; high-impact actions require explicit approval |
| Shell crash loses unsaved context | User frustration | Auto-save history on every command; graceful shutdown handler |
| System prompt for NL mode becomes too large | Exceeds context window | Summarize system state rather than dumping all details; lazy-load details on demand |

---

## Key References

- Architecture Plan §11 (The Spirit Shell — LLM-Native Interface)
- Architecture Plan §11.1 (Overview)
- Architecture Plan §11.2 (Command Resolution)
- Architecture Plan §11.3 (Agentic Capabilities)
