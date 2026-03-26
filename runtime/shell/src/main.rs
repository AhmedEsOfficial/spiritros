//! Spirit Shell — the main interactive entrypoint for `SpiritROS` Phase 1.

#![warn(clippy::all, clippy::pedantic)]

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use reedline::{
    default_emacs_keybindings, ColumnarMenu, DefaultHinter, Emacs, FileBackedHistory, KeyCode,
    KeyModifiers, MenuBuilder, Reedline, ReedlineEvent, ReedlineMenu, Signal,
};
use tracing::info;

use spirit_shell::commands::CommandRegistry;
use spirit_shell::completer::SpiritCompleter;
use spirit_shell::config::ShellMode;
use spirit_shell::prompt::SpiritPrompt;
use spirit_shell::types::CommandResult;

/// Spirit Shell — LLM-native command interface for `SpiritROS`.
#[derive(Parser, Debug)]
#[command(name = "spirit-shell", version, about)]
struct Cli {
    /// Path to a shell configuration file (JSON).
    #[arg(long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Shell operating mode.
    #[arg(long, default_value = "hybrid")]
    mode: String,

    /// Enable verbose (debug-level) logging.
    #[arg(long)]
    verbose: bool,
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

    let mode: ShellMode = cli
        .mode
        .parse()
        .map_err(|e: String| anyhow::anyhow!(e))
        .context("Invalid --mode value")?;

    info!(mode = %mode, "Starting Spirit Shell");

    println!("{BANNER}");
    println!("  Mode: {mode}\n");

    run_repl(mode)
}

/// Run the interactive read-eval-print loop.
fn run_repl(mode: ShellMode) -> Result<()> {
    let registry = CommandRegistry::new();

    // ── History ──────────────────────────────────────────────────────────
    let history_path = history_path();
    if let Some(parent) = history_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create history dir: {}", parent.display()))?;
    }
    let history = Box::new(
        FileBackedHistory::with_file(500, history_path.clone())
            .context("Failed to open history file")?,
    );

    // ── Completer ────────────────────────────────────────────────────────
    let mut names = registry.command_names();
    names.push("quit"); // alias
    let completer = Box::new(SpiritCompleter::new(names));

    let completion_menu = Box::new(ColumnarMenu::default().with_name("completion_menu"));

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
        .with_hinter(hinter);

    let prompt = SpiritPrompt;

    // ── REPL ─────────────────────────────────────────────────────────────
    loop {
        match line_editor.read_line(&prompt) {
            Ok(Signal::Success(line)) => {
                let result = dispatch_input(&registry, mode, &line);
                match result {
                    CommandResult::Output(text) => {
                        if !text.is_empty() {
                            println!("{text}");
                        }
                    }
                    CommandResult::Clear => {
                        // ANSI escape: clear screen and move cursor to top-left
                        print!("\x1b[2J\x1b[H");
                    }
                    CommandResult::Exit => {
                        println!("Goodbye.");
                        break;
                    }
                    CommandResult::NotBuiltin => {
                        // PHASE1-SHIM: In later phases this routes to the inference engine.
                        println!("[stub] Command will be routed to inference engine in Phase 06");
                    }
                }
            }
            Ok(Signal::CtrlD | Signal::CtrlC) => {
                println!("\nGoodbye.");
                break;
            }
            Err(err) => {
                eprintln!("Error reading input: {err}");
                break;
            }
        }
    }

    Ok(())
}

/// Route input through the command registry, respecting the current [`ShellMode`].
fn dispatch_input(registry: &CommandRegistry, mode: ShellMode, line: &str) -> CommandResult {
    match mode {
        ShellMode::Hybrid => {
            // In Hybrid mode, if the command is not built-in, we would route to
            // the inference engine. For now the caller handles `NotBuiltin`.
            registry.dispatch(line)
        }
        ShellMode::Cli => {
            let result = registry.dispatch(line);
            if matches!(result, CommandResult::NotBuiltin) {
                let cmd = line.split_whitespace().next().unwrap_or("");
                return CommandResult::Output(format!(
                    "Unknown command: '{cmd}'. Type 'help' for available commands."
                ));
            }
            result
        }
        ShellMode::Llm => {
            // In LLM mode, still honour exit/quit so the user can leave.
            let trimmed = line.trim();
            if trimmed == "exit" || trimmed == "quit" {
                return CommandResult::Exit;
            }
            // Everything else goes to inference.
            CommandResult::NotBuiltin
        }
    }
}

/// Resolve the history file path (`~/.spirit/history.txt`).
fn history_path() -> PathBuf {
    let home = std::env::var("HOME").map_or_else(|_| PathBuf::from("/tmp"), PathBuf::from);
    home.join(".spirit").join("history.txt")
}
