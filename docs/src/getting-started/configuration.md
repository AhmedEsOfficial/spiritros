# Configuration

SpiritROS is configured through a combination of configuration files and environment variables.

## Configuration File

The primary configuration file is located at `config/spiritros.toml` within the deployment directory. In Docker, this is mounted at `/etc/spiritros/spiritros.toml`.

```toml
[memory]
# Memory zone sizes in MB
zone_model = 8192
zone_kv = 4096
zone_compute = 2048
zone_general = 4096

# Enable hugepages for model weights
hugepages = true

[scheduler]
# Inference thread scheduling policy
inference_policy = "SCHED_FIFO"
inference_priority = 90

# CPU cores reserved for inference
inference_cpuset = "0-7"

[inference]
# Backend (ollama for Phase 1)
backend = "ollama"

# Maximum concurrent sessions
max_sessions = 4

# Default model
default_model = "llama3.2:8b"

# KV-cache settings
kv_cache_max_mb = 4096
kv_cache_cow = true

[spiritfs]
# Mount point
mount = "/spirit"

# Embedding model for semantic search
embedding_model = "nomic-embed-text"

# Vector database endpoint
vector_db = "http://localhost:6334"

[security]
# Capability enforcement
enforce_capabilities = true

# API bind address (localhost only by default)
api_bind = "127.0.0.1:8080"

# Require authentication for API access
require_auth = true

[logging]
# Log level: trace, debug, info, warn, error
level = "info"

# Log output: stdout, file, both
output = "both"
log_file = "/var/log/spiritros/spiritros.log"
```

## Environment Variables

Environment variables override configuration file values. They use the prefix `SPIRIT_` with double underscores for nesting:

| Variable | Description | Default |
|---|---|---|
| `SPIRIT_MEMORY__ZONE_MODEL` | Model zone size (MB) | `8192` |
| `SPIRIT_MEMORY__ZONE_KV` | KV-cache zone size (MB) | `4096` |
| `SPIRIT_INFERENCE__BACKEND` | Inference backend | `ollama` |
| `SPIRIT_INFERENCE__MAX_SESSIONS` | Max concurrent sessions | `4` |
| `SPIRIT_SECURITY__API_BIND` | API bind address | `127.0.0.1:8080` |
| `SPIRIT_LOGGING__LEVEL` | Log level | `info` |
