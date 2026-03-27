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
    // ── Guard: verify we have a usable interactive terminal ─────────────
    // Docker pseudo-TTYs (tty: true) pass the isatty check but can't handle
    // cursor-position queries that reedline needs. We probe before starting
    // the full editor to give a clear message instead of a cryptic crash.
    if !is_real_interactive_terminal() {
        eprintln!(
            "Spirit Shell requires an interactive terminal.\n\
             Attach with: docker exec -it <container> spirit-shell\n\
             \n\
             Running in daemon mode (background services only)."
        );
        // Block forever so the container stays alive for docker exec
        loop {
            std::thread::sleep(std::time::Duration::from_secs(3600));
        }
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

    let mut prompt = SpiritPrompt::new();

    // ── REPL ─────────────────────────────────────────────────────────────
    loop {
        // Update prompt with current state
        {
            let state_guard = rt.block_on(state.read());
            prompt.update_from_state(&state_guard);
        }

        match line_editor.read_line(&prompt) {
            Ok(Signal::Success(line)) => {
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
                // Ctrl+C cancels the current line, doesn't exit
                println!();
            }
            Err(err) => {
                let msg = err.to_string();
                if msg.contains("cursor position") {
                    eprintln!(
                        "\nTerminal does not support cursor queries (Docker pseudo-TTY?).\n\
                         Attach interactively: docker exec -it <container> spirit-shell"
                    );
                } else {
                    eprintln!("Error reading input: {err}");
                }
                break;
            }
        }
    }

    Ok(())
}

/// Check whether stdin is a real interactive terminal capable of handling
/// cursor-position queries. Docker `tty: true` creates a pseudo-TTY that
/// passes `isatty()` but can't answer `\x1b[6n` (cursor position request).
///
/// We probe by writing the DSR (Device Status Report) escape sequence and
/// checking if we get a response within a short timeout using crossterm's
/// non-blocking event polling.
fn is_real_interactive_terminal() -> bool {
    use crossterm::event;
    use std::io::Write;
    use std::time::Duration;

    // Basic check: is stdin a TTY at all?
    if unsafe { libc::isatty(libc::STDIN_FILENO) } == 0 {
        return false;
    }

    // Probe: send cursor position request and see if we get a reply.
    // This distinguishes a real terminal from a Docker pseudo-TTY that
    // allocates a pts but has no terminal emulator answering queries.
    let Ok(()) = crossterm::terminal::enable_raw_mode() else {
        return false;
    };

    let result = (|| -> bool {
        // Write DSR: \x1b[6n (Device Status Report — cursor position)
        let mut stdout = std::io::stdout();
        if stdout.write_all(b"\x1b[6n").is_err() || stdout.flush().is_err() {
            return false;
        }

        // A real terminal answers with \x1b[<row>;<col>R within milliseconds.
        // A Docker pseudo-TTY with no terminal emulator never answers.
        // Use crossterm::event::poll for non-blocking, timeout-safe waiting.
        match event::poll(Duration::from_millis(500)) {
            Ok(true) => {
                // Something arrived — drain the cursor position response.
                // crossterm may parse it as a key event or we may need to
                // read raw. Either way, arrival means the terminal is real.
                let _ = event::read();
                true
            }
            _ => false,
        }
    })();

    let _ = crossterm::terminal::disable_raw_mode();

    result
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
