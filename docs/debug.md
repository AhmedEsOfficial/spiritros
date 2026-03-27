
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
ollama pull llama3.2:3b              # Pull a small model
ollama run llama3.2:3b "Hello"       # Test generation
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