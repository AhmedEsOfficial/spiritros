# Spirit Shell — LLM-Native Interface

The Spirit Shell is SpiritROS's primary user interface — a command-line environment where natural language and traditional Unix commands coexist seamlessly.

## Overview

A user can type any of:

```
spirit@ghost ~ $ ls -la                          # Unix command
spirit@ghost ~ $ spirit model list               # Spirit built-in
spirit@ghost ~ $ What models do I have loaded?   # Natural language
spirit@ghost ~ $ ls -la /var/log | explain what happened today  # Pipe-to-LLM
```

The shell resolves each input through a three-stage priority chain and routes it to the appropriate handler.

## Command Resolution Pipeline

Input is resolved in this order:

1. **Alias expansion** — user-defined aliases are expanded before further resolution
2. **Spirit built-in commands** — commands prefixed with `spirit` or known builtins (`help`, `version`, `status`, `clear`, `exit`)
3. **Standard Unix commands** — if the first word is found in `$PATH`, execute it
4. **Natural language fallback** — everything else is sent to the configured LLM

### Special Prefixes

| Prefix | Behavior |
|--------|----------|
| `?` | Force natural language interpretation |
| `!` | Force Unix command execution |

### Pipe-to-LLM

The `|` operator followed by a natural language phrase pipes the preceding command's stdout to the LLM:

```
ls -la /var/log | explain what happened today
cat /etc/nginx/nginx.conf | find security issues
git log --oneline -20 | summarize recent changes
```

Detection heuristic: if the text after the last `|` doesn't start with a known Unix command, it's treated as a natural language query.

## Shell Modes

| Mode | Behavior |
|------|----------|
| **Hybrid** (default) | CLI commands first, NL fallback for unrecognized input |
| **CLI** | Only built-in and Unix commands; no LLM routing |
| **LLM** | Everything except `exit`/`quit` and `spirit` commands goes to the LLM |

Switch modes with `spirit shell mode <hybrid|cli|llm>`.

## Built-in Spirit Commands

| Command | Description |
|---------|-------------|
| `spirit model list` | List available models |
| `spirit model load <name>` | Load a model into memory |
| `spirit model unload <name>` | Unload a model |
| `spirit model pull <name>` | Pull from the Ollama registry |
| `spirit mem status` | Show memory zone utilization |
| `spirit mem rebalance` | Dynamically resize zones |
| `spirit sched status` | Show scheduler state |
| `spirit find <query>` | Semantic file search |
| `spirit infer <prompt>` | Direct inference invocation |
| `spirit agent list` | List running agents |
| `spirit agent start <task>` | Start an agentic task |
| `spirit agent stop <id>` | Stop a running agent |
| `spirit shell mode <mode>` | Switch shell mode |
| `spirit config show` | Show configuration |
| `spirit config get <key>` | Get a config value |
| `spirit config set <key> <val>` | Set a config value (session) |

## Agentic Mode

The shell supports autonomous LLM-driven tasks:

```
spirit agent start "Monitor inference latency for 1 hour and alert if > 100ms/token"
```

Agents run as background tokio tasks with:
- Unique ID for tracking
- Cancellation support (`spirit agent stop <id>`)
- Audit logging to `/var/log/spiritros/agents/`
- Capability-governed permissions (Phase 2+)

## Configuration

The shell loads config from `spirit.yaml` (under the `shell:` key) with environment variable overrides (`SPIRIT_SHELL_*`):

```yaml
shell:
  mode: "hybrid"
  llm:
    model: "llama3.2:3b"
    max_context_tokens: 4096
  history:
    enabled: true
    max_entries: 10000
  prompt:
    format: "spirit:{cwd}> "
    show_model: true
  inference:
    ollama_url: "http://127.0.0.1:11434"
  aliases:
    ll: "ls -la"
    models: "spirit model list"
```

## Line Editing Features

Built on the `reedline` crate (from the Nushell project):

- **Multi-level tab completion** — completes Spirit commands, subcommands, and file paths
- **Syntax highlighting** — Spirit commands (green), Unix commands (green), NL queries (dim), pipe-to-LLM markers (yellow/magenta)
- **Persistent history** — saved to `~/.spirit/shell_history` (configurable)
- **Dynamic prompt** — shows CWD, active model, and mode indicator
- **Emacs keybindings** — Ctrl+A/E/K/W, Ctrl+R for history search
- **History hinting** — inline suggestions from command history

## Implementation

The shell is implemented in `runtime/shell/` as a standalone Rust crate. Key components:

- **CommandResolver** — three-stage resolution with alias expansion and pipe-to-LLM detection
- **InferenceClient** — lightweight async HTTP client for Ollama (streaming token output)
- **AgentManager** — background task lifecycle with cancellation via tokio oneshot channels
- **ShellState** — shared mutable state (`Arc<RwLock>`) accessible to all command handlers

The REPL loop runs synchronously on the main thread using `rt.block_on()` for async operations, ensuring reedline's terminal handling works correctly.
