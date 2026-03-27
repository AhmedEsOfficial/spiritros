//! Spirit find, similar, and summarize commands.
//!
//! These commands integrate with `SpiritFS`'s semantic search capabilities.
//! PHASE1-SHIM: Uses the inference engine for semantic queries when
//! `SpiritFS` is not mounted. Phase 2+ uses `SpiritFS` native APIs.

use crate::commands::SharedState;
use crate::types::CommandResult;

/// Handle `spirit find <query>` — semantic file search.
pub async fn find(state: &SharedState, args: &[&str]) -> CommandResult {
    if args.is_empty() {
        return CommandResult::Output(
            "Usage: spirit find <query>\n\n\
             Search for files by meaning, not just name.\n\
             Examples:\n  \
             spirit find \"database migration scripts\"\n  \
             spirit find \"error handling code\"\n  \
             spirit find \"configuration files for networking\""
                .into(),
        );
    }

    let query = args.join(" ");

    // Check if SpiritFS is mounted
    let mountpoint =
        std::env::var("SPIRITFS_MOUNTPOINT").unwrap_or_else(|_| "/mnt/spiritfs".into());

    if std::path::Path::new(&mountpoint).exists() && std::fs::read_dir(&mountpoint).is_ok() {
        // SpiritFS is available — use its semantic search via xattr or API
        // PHASE1-SHIM: Fall through to LLM-based search
        println!("Searching SpiritFS for: {query}");
    }

    // Fallback: use the LLM to interpret the query and suggest files
    let client = state.read().await.inference.clone();
    let prompt = format!(
        "The user is searching for files matching this description: \"{query}\"\n\n\
         Based on common project structures, suggest what files or directories \
         they might be looking for. Be specific about likely file paths and names."
    );

    match client.generate_streaming(&prompt, None).await {
        Ok(_stats) => CommandResult::Output(String::new()),
        Err(e) => CommandResult::Error(format!("Search failed: {e}")),
    }
}

/// Handle `spirit similar <file>` — find similar files.
pub fn similar(args: &[&str]) -> CommandResult {
    if args.is_empty() {
        return CommandResult::Output(
            "Usage: spirit similar <file>\n\n\
             Find files with similar content using semantic embeddings.\n\
             Requires SpiritFS to be mounted and indexed."
                .into(),
        );
    }

    let file = args[0];

    // PHASE1-SHIM: Requires SpiritFS with Qdrant vector store
    CommandResult::Output(format!(
        "Finding files similar to '{file}'...\n\
         Note: Requires SpiritFS to be mounted with semantic indexing enabled.\n\
         Mount SpiritFS with: spiritfs mount /mnt/spiritfs"
    ))
}

/// Handle `spirit summarize <path>` — summarize directory contents.
pub fn summarize(args: &[&str]) -> CommandResult {
    if args.is_empty() {
        return CommandResult::Output(
            "Usage: spirit summarize <path>\n\n\
             Summarize the contents of a directory using the LLM.\n\
             Requires SpiritFS semantic indexing."
                .into(),
        );
    }

    let path = args[0];

    // PHASE1-SHIM: Requires SpiritFS
    CommandResult::Output(format!(
        "Summarizing '{path}'...\n\
         Note: Requires SpiritFS to be mounted with semantic indexing enabled."
    ))
}
