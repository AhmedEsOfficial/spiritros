
How to build and test the container
Prerequisites
You need Docker with nvidia-container-toolkit (for GPU) or it will fall back to CPU-only mode.

Build the image

# From repo root
just docker-build
# or manually:
docker compose -f docker/docker-compose.yml build
Start the stack

just docker-up
# or:
docker compose -f docker/docker-compose.yml up -d
This starts: spiritros (Ollama + spirit-infer + spirit-shell), qdrant, etcd, prometheus, grafana.

Verify Ollama is running

# Shell into the container
docker compose -f docker/docker-compose.yml exec spiritros bash

# Inside the container:
ollama list                          # Should show models (or empty list)
ollama pull qwen3.5:4b              # Pull a small model
ollama run qwen3.5:4b "Hello"       # Test generation
Verify spirit-infer is running

# Inside the container:
cat /var/run/spirit-infer.pid        # Should show a PID
cat /var/log/spiritros/coordinator.log  # Check coordinator logs
Verify end-to-end via IPC
There's no CLI client for the Unix socket yet, but you can test with socat or by watching the logs:


# Check the logs show Ollama model sync
grep "Synced models" /var/log/spiritros/coordinator.log
grep "Model loaded" /var/log/spiritros/coordinator.log
What I fixed just now
The Docker infrastructure had three stale references to a spirit-coordinator binary that didn't exist:

Dockerfile — now builds and copies spirit-infer
entrypoint.sh — now starts spirit-infer (not spirit-coordinator)
health-check.sh — now checks spirit-infer PID
What requires a running container to verify
The checked acceptance criteria are code-complete — the Rust implementation handles all the logic. But the end-to-end flow (Ollama starts -> spirit-infer syncs models -> registry populated) can only be verified with the container running and a GPU (or CPU fallback) available. You can't unit-test "Ollama actually starts" without Docker.

# restart
docker compose -f docker/docker-compose.yml down && just docker-up




# Backing up

What's already persistent (you don't need to do anything)
Build cache — Docker BuildKit stores layer caches automatically. They survive down, up, build, etc. They only get deleted if you explicitly run docker builder prune.
Named volumes (spiritros-models, spiritros-data, etc.) — survive down. Only deleted by docker volume rm or docker compose down -v (the -v flag).
Pulled images (qdrant, etcd, prometheus, grafana, nvidia/cuda, rust) — survive everything. Only deleted by docker image prune or docker rmi.
What WILL delete your cache (avoid these)
Command	What it deletes
docker builder prune	Build layer cache (the big one)
docker system prune	Stopped containers, dangling images, build cache
docker compose down -v	Named volumes (model data, logs)
docker image prune -a	All unused images
To explicitly back up the built image

# Save the compiled image to a tarball (~2-3 GB)
docker save docker-spiritros:latest | zstd -o spiritros-image.tar.zst

# Restore later
zstd -d spiritros-image.tar.zst --stdout | docker load
To pin the build cache permanently
Add this to a ~/.docker/daemon.json (or create it):


{
  "builder": {
    "gc": {
      "enabled": false
    }
  }
}
This disables automatic garbage collection of the build cache. Restart Docker after: sudo systemctl restart docker.

Now go ahead and bring it up:


docker compose -f docker/docker-compose.yml down && just docker-up





# End-to-End Test Commands
Run inside the Docker container (docker compose exec -it spiritros spirit-shell) or locally with Ollama running. The tests are grouped by feature. Expected behavior is noted after each command.

1. Shell Startup & Basics

help
Shows full help with all spirit commands, modes, prefixes, pipe-to-LLM syntax


version
Shows Spirit Shell v0.1.0 (Phase 1: Virtualized Deployment)


status
Shows inference engine status (connected/unavailable), active agents, mode, Ollama URL


history
Shows history info with file path


clear
Clears the screen

2. Spirit Command Routing

spirit
Shows spirit subcommand help listing


spirit model list
Lists all models from Ollama (name, size, quantization, family)


spirit model pull qwen3:1.7b
Streams pull progress to terminal


spirit model load qwen3:1.7b
Loads model into Ollama (warms it up)


spirit model unload qwen3:1.7b
Unloads model from Ollama

3. Memory & Scheduler Status

spirit mem status
Shows memory zone table (or "not configured" outside Docker), plus system memory from /proc/meminfo


spirit sched status
Shows CPU cores, NUMA nodes, GPU info, priority tier table, inference phase

4. Direct Inference

spirit infer Hello, what is SpiritROS?
Streams LLM response token-by-token, then prints stats (tokens, tok/s, latency)

5. Natural Language (Hybrid Mode)

What time is it?
Resolves as NL (not a Unix command), sends to LLM, streams response


Explain the difference between SCHED_FIFO and SCHED_OTHER
Same — NL fallback in hybrid mode

6. Explicit Prefix Routing

? List the files in /tmp
Forces NL interpretation even though ls exists — LLM describes files


! echo hello world
Forces Unix execution — prints hello world

7. Unix Command Execution

ls -la /tmp
Executes as Unix command, shows directory listing


echo "SpiritROS is running"
Executes as Unix command, prints the string


cat /proc/cpuinfo | head -20
Normal Unix pipe (both sides are commands) — shows CPU info

8. Pipe-to-LLM

ls -la /etc | explain what these config files do
Runs ls -la /etc, captures output, sends to LLM with the query, streams response


cat /proc/meminfo | summarize the memory situation
Captures meminfo, LLM summarizes it

9. Alias Expansion

ll /tmp
Expands to ls -la /tmp, executes as Unix command


models
Expands to spirit model list, shows model list


mem
Expands to spirit mem status, shows memory zones

10. Mode Switching

spirit shell mode cli
Switches to CLI mode — confirm with prompt changing to [CLI]


What is 2+2?
In CLI mode: should error with "Unknown command" (no NL fallback)


spirit shell mode llm
Switches to LLM mode — confirm with [LLM] in prompt


ls -la
In LLM mode: goes to inference (not executed as Unix command)


spirit shell mode hybrid
Back to default hybrid mode

11. Configuration

spirit config show
Shows all current config: mode, model, prompt, history, aliases, inference URL


spirit config get model
Shows the current model name


spirit config set model qwen3:1.7b
Changes model for this session — confirm with spirit config get model


spirit config get mode
Shows current mode

12. Agentic Tasks

spirit agent list
Shows "No agents running or completed."


spirit agent start Analyze the system load and suggest optimizations
Starts a background agent, streams its LLM output, shows agent ID


spirit agent list
Shows the agent with status (running/completed), task description, action count


spirit agent stop <id>
Stops the agent (use the short ID from the list output)

13. Semantic Search (SpiritFS Integration)

spirit find database migration scripts
Falls back to LLM-based search (SpiritFS not mounted outside Docker)


spirit similar README.md
Shows note about requiring SpiritFS


spirit summarize /tmp
Shows note about requiring SpiritFS

14. Tab Completion
Press Tab after each partial input:


sp<TAB>          → spirit
spirit <TAB>     → shows: model, mem, sched, find, infer, agent, ...
spirit mod<TAB>  → spirit model
spirit model <TAB> → shows: list, load, unload, pull
spirit shell mode <TAB> → shows: hybrid, cli, llm
he<TAB>          → help
15. Signal Handling & Exit

<Ctrl+C>
Cancels current line, shows new prompt (does NOT exit)


quit
Exits with "Goodbye."

Or:


exit
Same as quit

Or:


<Ctrl+D>
Exits with "Goodbye."

16. Non-Interactive Mode
From outside the shell (in a regular terminal):


spirit-shell -c "spirit model list"
Executes single command and exits


spirit-shell -c "version"
Prints version and exits


spirit-shell --model qwen3:1.7b -c "spirit config get model"
Shows qwen3:1.7b (CLI override applied)