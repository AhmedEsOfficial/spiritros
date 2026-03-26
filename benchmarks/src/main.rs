//! spirit-bench CLI — the SpiritROS benchmarking tool.
//!
//! Run with `--help` for usage information.

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "spirit-bench", version, about = "SpiritROS benchmarking suite")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a specific benchmark suite.
    Run {
        /// The suite to run (inference, memory, filesystem, scheduler).
        suite: String,
    },
    /// List all available benchmark suites.
    List,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    println!("spirit-bench v{}", env!("CARGO_PKG_VERSION"));

    match cli.command {
        Some(Commands::Run { suite }) => match suite.as_str() {
            "inference" => spirit_bench::inference::run(),
            "memory" => spirit_bench::memory::run(),
            "filesystem" => spirit_bench::filesystem::run(),
            "scheduler" => spirit_bench::scheduler::run(),
            other => {
                eprintln!("Unknown suite: {other}");
                eprintln!("Available suites:");
                print_suites();
                std::process::exit(1);
            }
        },
        Some(Commands::List) | None => {
            println!();
            println!("Available benchmark suites:");
            print_suites();
        }
    }

    Ok(())
}

fn print_suites() {
    for suite in spirit_bench::available_suites() {
        println!("  {:<14} {}", suite.name, suite.description);
    }
}
