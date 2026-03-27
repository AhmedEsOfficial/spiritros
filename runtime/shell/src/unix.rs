//! Unix command execution for the Spirit Shell.
//!
//! Handles executing standard Unix commands found in `$PATH` via
//! `tokio::process::Command`. In Phase 1, commands run directly on the
//! Linux host (or inside the Docker container). In Phase 2+, they will
//! route through the Linux compatibility layer.
//!
//! PHASE1-SHIM: Direct process execution; Phase 2+ uses syscall interception.

use std::path::Path;
use std::process::Stdio;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

use crate::error::{self, ShellError};

/// Check if a command exists in `$PATH`.
pub fn command_exists(name: &str) -> bool {
    // Don't allow executing things that look like paths to scripts
    if name.contains('/') {
        return Path::new(name).exists();
    }

    let path_var = std::env::var("PATH").unwrap_or_default();
    path_var
        .split(':')
        .any(|dir| Path::new(dir).join(name).exists())
}

/// Execute a Unix command and stream its output to stdout/stderr.
/// Returns the exit code.
pub async fn exec_interactive(line: &str) -> error::Result<i32> {
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(line)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| ShellError::ExecutionFailed(format!("Failed to spawn: {e}")))?;

    let status = child
        .wait()
        .await
        .map_err(|e| ShellError::ExecutionFailed(format!("Failed to wait: {e}")))?;

    Ok(status.code().unwrap_or(-1))
}

/// Execute a Unix command and capture its output as a string.
/// Used for pipe-to-LLM where we need to feed the output to the LLM.
pub async fn exec_capture(line: &str) -> error::Result<String> {
    let output = Command::new("sh")
        .arg("-c")
        .arg(line)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| ShellError::ExecutionFailed(format!("Failed to spawn: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let code = output.status.code().unwrap_or(-1);
        if !stderr.is_empty() {
            eprintln!("{}", stderr.trim());
        }
        return Err(ShellError::UnixCommandFailed {
            command: line.to_string(),
            code,
        });
    }

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    Ok(stdout)
}

/// Execute a Unix command and stream its output line-by-line, calling
/// a callback for each line. Used for long-running commands in agent mode.
pub async fn exec_streaming<F>(line: &str, mut on_line: F) -> error::Result<i32>
where
    F: FnMut(&str),
{
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(line)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| ShellError::ExecutionFailed(format!("Failed to spawn: {e}")))?;

    if let Some(stdout) = child.stdout.take() {
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            on_line(&line);
        }
    }

    let status = child
        .wait()
        .await
        .map_err(|e| ShellError::ExecutionFailed(format!("Failed to wait: {e}")))?;

    Ok(status.code().unwrap_or(-1))
}

/// Extract the first word (command name) from a shell line.
pub fn first_word(line: &str) -> &str {
    line.split_whitespace().next().unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_exists_true() {
        // `sh` should exist on any Unix system
        assert!(command_exists("sh"));
    }

    #[test]
    fn test_command_exists_false() {
        assert!(!command_exists("definitely_not_a_real_command_xyz"));
    }

    #[test]
    fn test_first_word() {
        assert_eq!(first_word("ls -la /tmp"), "ls");
        assert_eq!(first_word("  git status"), "git");
        assert_eq!(first_word(""), "");
    }

    #[tokio::test]
    async fn test_exec_capture_echo() {
        let output = exec_capture("echo hello").await.unwrap();
        assert_eq!(output.trim(), "hello");
    }

    #[tokio::test]
    async fn test_exec_capture_failure() {
        let result = exec_capture("false").await;
        assert!(result.is_err());
    }
}
