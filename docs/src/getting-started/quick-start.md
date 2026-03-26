# Quick Start

This guide walks you through your first interaction with SpiritROS after installation.

## Starting the System

```bash
# Start all services
docker compose -f docker/docker-compose.yml up -d

# Open the Spirit Shell
docker compose -f docker/docker-compose.yml exec spiritros spirit-shell
```

## Loading a Model

From the Spirit Shell, load a model for inference:

```
spirit> model pull llama3.2:8b
Pulling llama3.2:8b... done (4.7 GB)

spirit> model load llama3.2:8b
Loading into ZONE_MODEL... done (3.2s)
VRAM allocated: 4,891 MB
```

## Running Inference

Start an interactive inference session:

```
spirit> infer llama3.2:8b "Explain what an operating system kernel does in two sentences."
The kernel is the core component of an operating system that manages hardware resources
and provides services to application software. It handles memory allocation, process
scheduling, device I/O, and security enforcement as the privileged layer between
applications and physical hardware.

Tokens: 47 | Time: 0.82s | Speed: 57.3 tok/s
```

## Using SpiritFS

SpiritFS provides semantic file operations. Files stored in SpiritFS are automatically indexed for meaning-based search:

```
spirit> cd /spirit/documents
spirit> echo "Q3 2025 revenue exceeded projections by 12%" > quarterly-report.txt
spirit> search "how did we do financially last quarter"
quarterly-report.txt (relevance: 0.94)
```

## Checking System Status

View memory zones and active sessions:

```
spirit> status
Memory Zones:
  ZONE_MODEL:    4,891 MB / 8,192 MB (59.7%)
  ZONE_KV:       1,024 MB / 4,096 MB (25.0%)
  ZONE_COMPUTE:    512 MB / 2,048 MB (25.0%)
  ZONE_GENERAL:    768 MB / 4,096 MB (18.8%)

Active Sessions: 1
Loaded Models: llama3.2:8b
GPU Utilization: 34%
```

## Next Steps

- Read the [Configuration](configuration.md) guide to tune memory zones and scheduling priorities.
- Explore the [Architecture Overview](../architecture/overview.md) to understand how SpiritROS works.
- Check the [SDK documentation](../developer-guide/sdk.md) to build applications on SpiritROS.
