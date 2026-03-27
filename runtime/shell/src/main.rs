//! Spirit Shell — the main interactive entrypoint for `SpiritROS` Phase 1.
//!
//! The Spirit Shell is `SpiritROS`'s primary user interface — a command-line
//! environment where natural language and traditional Unix commands coexist.
//! A user can type `ls -la`, a standard Unix command, or "What models do I
//! have loaded?", a natural language query — and the shell handles both.

#![warn(clippy::all, clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::must_use_candidate
)]

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use reedline::{
    default_emacs_keybindings, ColumnarMenu, DefaultHinter, Emacs, FileBackedHistory, KeyCode,
    KeyModifiers, MenuBuilder, Reedline, ReedlineEvent, ReedlineMenu, Signal,
};
use tracing::{info, warn};

use spirit_shell::commands::{self, SharedState};
use spirit_shell::completer::SpiritCompleter;
use spirit_shell::config::ShellConfig;
use spirit_shell::highlighter::SpiritHighlighter;
use spirit_shell::pipe;
use spirit_shell::prompt::SpiritPrompt;
use spirit_shell::resolver::CommandResolver;
use spirit_shell::types::{CommandResult, Resolution};
use spirit_shell::unix;

/// Spirit Shell — LLM-native command interface for `SpiritROS`.
#[derive(Parser, Debug)]
#[command(name = "spirit-shell", version, about)]
struct Cli {
    /// Path to a shell configuration file (YAML).
    #[arg(long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Shell operating mode.
    #[arg(long)]
    mode: Option<String>,

    /// Ollama API URL override.
    #[arg(long, env = "OLLAMA_HOST")]
    ollama_url: Option<String>,

    /// Model to use for inference.
    #[arg(long)]
    model: Option<String>,

    /// Enable verbose (debug-level) logging.
    #[arg(long)]
    verbose: bool,

    /// Execute a single command and exit (non-interactive).
    #[arg(short = 'c')]
    command: Option<String>,
}

const BANNER: &str = r"
═══════════════════════════════════════════════════
  ____        _      _ _   ____   ___  ____
 / ___| _ __ (_)_ __(_) |_|  _ \ / _ \/ ___|
 \___ \| '_ \| | '__| | __| |_) | | | \___ \
  ___) | |_) | | |  | | |_|  _ <| |_| |___) |
 |____/| .__/|_|_|  |_|\__|_| \_\\___/|____/
       |_|

  v0.1.0 — Ghost in the Machine
  Phase 1: Virtualized Deployment
═══════════════════════════════════════════════════";

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing
    let filter = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter)),
        )
        .with_target(false)
        .init();

    // Load configuration
    let mut config = if let Some(ref path) = cli.config {
        ShellConfig::from_file(path).map_err(|e| anyhow::anyhow!("Failed to load config: {e}"))?
    } else {
        ShellConfig::load()
    };

    // Apply CLI overrides
    if let Some(ref mode_str) = cli.mode {
        config.mode = mode_str
            .parse()
            .map_err(|e: String| anyhow::anyhow!(e))
            .context("Invalid --mode value")?;
    }
    if let Some(ref url) = cli.ollama_url {
        let url = if url.starts_with("http") {
            url.clone()
        } else {
            format!("http://{url}")
        };
        config.inference.ollama_url = url;
    }
    if let Some(ref model) = cli.model {
        config.llm.model.clone_from(model);
    }

    info!(mode = %config.mode, model = %config.llm.model, "Starting Spirit Shell");

    // Non-interactive mode: execute a single command
    if let Some(ref cmd) = cli.command {
        return run_command(&config, cmd);
    }

    // Interactive mode
    println!("{BANNER}");
    println!("  Mode: {}  |  Model: {}", config.mode, config.llm.model);
    println!("  Type 'help' for commands, '?' prefix for natural language\n");

    run_repl(config)
}

/// Run a single command non-interactively.
fn run_command(config: &ShellConfig, cmd: &str) -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    let state = commands::new_state(config.clone());

    rt.block_on(async {
        let resolver = CommandResolver::new(config.aliases.clone(), config.mode);
        let resolution = resolver.resolve(cmd);
        handle_resolution(&state, resolution).await;
    });

    Ok(())
}

/// Run the interactive read-eval-print loop.
#[allow(clippy::needless_pass_by_value)]
fn run_repl(config: ShellConfig) -> Result<()> {
    // ── Guard: basic isatty check ───────────────────────────────────────
    // The entrypoint.sh already gates on [ -t 0 ] for docker compose up.
    // Here we catch the case where someone pipes into spirit-shell or
    // redirects stdin. We trust isatty — docker exec -it provides a real
    // TTY that reedline can use.
    if unsafe { libc::isatty(libc::STDIN_FILENO) } == 0 {
        eprintln!(
            "Spirit Shell requires an interactive terminal.\n\
             Use: docker exec -it <container> spirit-shell"
        );
        return Ok(());
    }

    let rt = tokio::runtime::Runtime::new()?;
    let state = commands::new_state(config.clone());

    // ── History ──────────────────────────────────────────────────────────
    let history_path = config.history.path.clone();
    if let Some(parent) = history_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create history dir: {}", parent.display()))?;
    }
    let history = Box::new(
        FileBackedHistory::with_file(config.history.max_entries, history_path.clone())
            .context("Failed to open history file")?,
    );

    // ── Completer ────────────────────────────────────────────────────────
    let completer = Box::new(SpiritCompleter::new());
    let completion_menu = Box::new(ColumnarMenu::default().with_name("completion_menu"));

    // ── Highlighter ──────────────────────────────────────────────────────
    let highlighter = Box::new(SpiritHighlighter);

    // ── Keybindings ──────────────────────────────────────────────────────
    let mut keybindings = default_emacs_keybindings();
    keybindings.add_binding(
        KeyModifiers::NONE,
        KeyCode::Tab,
        ReedlineEvent::UntilFound(vec![
            ReedlineEvent::Menu("completion_menu".to_string()),
            ReedlineEvent::MenuNext,
        ]),
    );
    let edit_mode = Box::new(Emacs::new(keybindings));

    // ── Hinter ───────────────────────────────────────────────────────────
    let hinter = Box::new(DefaultHinter::default());

    // ── Build the line editor ────────────────────────────────────────────
    let mut line_editor = Reedline::create()
        .with_history(history)
        .with_completer(completer)
        .with_menu(ReedlineMenu::EngineCompleter(completion_menu))
        .with_edit_mode(edit_mode)
        .with_hinter(hinter)
        .with_highlighter(highlighter);

    let max_cursor_retries = 3u32;
    let mut prompt = SpiritPrompt::new();
    let mut cursor_retries = 0u32;

    // ── REPL ─────────────────────────────────────────────────────────────
    loop {
        // Update prompt with current state
        {
            let state_guard = rt.block_on(state.read());
            prompt.update_from_state(&state_guard);
        }

        match line_editor.read_line(&prompt) {
            Ok(Signal::Success(line)) => {
                cursor_retries = 0; // reset on success
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                // Resolve the input
                let resolution = {
                    let state_guard = rt.block_on(state.read());
                    state_guard.resolver.resolve(trimmed)
                };

                // Handle the resolution (may involve async operations)
                rt.block_on(handle_resolution(&state, resolution));
            }
            Ok(Signal::CtrlD) => {
                println!("\nGoodbye.");
                break;
            }
            Ok(Signal::CtrlC) => {
                cursor_retries = 0;
                // Ctrl+C cancels the current line, doesn't exit
                println!();
            }
            Err(err) => {
                let msg = err.to_string();
                if msg.contains("cursor position") {
                    cursor_retries += 1;
                    if cursor_retries <= max_cursor_retries {
                        std::thread::sleep(std::time::Duration::from_millis(
                            200 * u64::from(cursor_retries),
                        ));
                        continue;
                    }
                    // Reedline can't work in this terminal — fall back to simple mode
                    warn!(
                        "Reedline cursor query failed after {max_cursor_retries} attempts, \
                         switching to simple input mode"
                    );
                    eprintln!(
                        "\x1b[33mNote:\x1b[0m Terminal does not support cursor queries. \
                         Using simple input mode (no tab completion or syntax highlighting)."
                    );
                    return run_simple_repl(rt, state);
                }
                eprintln!("Error reading input: {err}");
                break;
            }
        }
    }

    Ok(())
}

/// Fallback REPL using basic stdin when reedline cannot work (e.g. Docker pseudo-TTY
/// that doesn't support cursor position queries).
#[allow(clippy::needless_pass_by_value, clippy::unnecessary_wraps)]
fn run_simple_repl(rt: tokio::runtime::Runtime, state: SharedState) -> Result<()> {
    let stdin = std::io::stdin();
    let mut input = String::new();

    loop {
        // Build a simple prompt string
        let prompt_str = {
            let state_guard = rt.block_on(state.read());
            let cwd = std::env::current_dir().map_or_else(
                |_| "?".into(),
                |p| p.file_name().map_or_else(|| "/".into(), |n| n.to_string_lossy().into_owned()),
            );
            format!("spirit:{cwd} ({})> ", state_guard.config.llm.model)
        };

        // Print prompt and flush
        print!("{prompt_str}");
        if std::io::stdout().flush().is_err() {
            break;
        }

        // Read line
        input.clear();
        match stdin.read_line(&mut input) {
            Ok(0) => {
                // EOF (Ctrl+D)
                println!("\nGoodbye.");
                break;
            }
            Ok(_) => {
                let trimmed = input.trim();
                if trimmed.is_empty() {
                    continue;
                }

                let resolution = {
                    let state_guard = rt.block_on(state.read());
                    state_guard.resolver.resolve(trimmed)
                };
                rt.block_on(handle_resolution(&state, resolution));
            }
            Err(e) => {
                eprintln!("Error reading input: {e}");
                break;
            }
        }
    }

    Ok(())
}

/// Handle a resolved command.
async fn handle_resolution(state: &SharedState, resolution: Resolution) {
    match resolution {
        Resolution::SpiritBuiltin(line) => {
            let result = commands::dispatch_builtin(state, &line).await;
            print_result(result);
        }

        Resolution::UnixCommand(cmd) => match unix::exec_interactive(&cmd).await {
            Ok(code) => {
                if code != 0 {
                    eprintln!("\x1b[2m(exit code: {code})\x1b[0m");
                }
            }
            Err(e) => {
                eprintln!("\x1b[31mError:\x1b[0m {e}");
            }
        },

        Resolution::NaturalLanguage(query) => {
            let client = state.read().await.inference.clone();
            match client.query(&query).await {
                Ok(_stats) => {}
                Err(e) => {
                    eprintln!("\x1b[31mInference error:\x1b[0m {e}");
                    eprintln!(
                        "\x1b[2mCheck that Ollama is running at: {}\x1b[0m",
                        state.read().await.config.inference.ollama_url
                    );
                }
            }
        }

        Resolution::PipeToLlm { command, query } => {
            let client = state.read().await.inference.clone();
            match pipe::exec_pipe_to_llm(&client, &command, &query).await {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("\x1b[31mPipe-to-LLM error:\x1b[0m {e}");
                }
            }
        }

        Resolution::Alias(expanded) => {
            // Re-resolve the expanded alias
            let resolution = {
                let state_guard = state.read().await;
                state_guard.resolver.resolve(&expanded)
            };
            // Prevent infinite alias loops by not re-expanding aliases
            match resolution {
                Resolution::Alias(_) => {
                    eprintln!("\x1b[31mError:\x1b[0m Alias loop detected");
                }
                other => {
                    Box::pin(handle_resolution(state, other)).await;
                }
            }
        }
    }
}

/// Print a command result to stdout/stderr.
fn print_result(result: CommandResult) {
    match result {
        CommandResult::Output(text) => {
            if !text.is_empty() {
                println!("{text}");
            }
        }
        CommandResult::Clear => {
            print!("\x1b[2J\x1b[H");
        }
        CommandResult::Exit => {
            println!("Goodbye.");
            std::process::exit(0);
        }
        CommandResult::NotBuiltin => {
            // This shouldn't happen with proper resolution, but handle gracefully
            eprintln!("[internal] Unresolved command — report this as a bug");
        }
        CommandResult::Error(msg) => {
            eprintln!("\x1b[31mError:\x1b[0m {msg}");
        }
    }
}
