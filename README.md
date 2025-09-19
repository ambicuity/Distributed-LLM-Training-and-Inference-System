# Distributed LLM Training and Inference System

A comprehensive CLI tool (`llmctl`) that orchestrates data preparation, model partitioning, distributed training, checkpointing, evaluation, and low-latency inference for Large Language Models.

## Features

- **Hardware-aware optimization**: Exposes FLOPs ceilings, memory bandwidth, and communication bandwidth knobs
- **Distributed training**: Pipeline, tensor, and sequence parallelism with ZeRO stages
- **Deterministic & reproducible**: Deterministic training runs with full reproducibility
- **Observability**: Built-in telemetry, metrics collection, and tracing
- **Pluggable backends**: Support for PyTorch/XLA, CUDA/HIP, NCCL/Gloo/MPI
- **Production-ready serving**: Low-latency inference with paged attention and dynamic batching

## Architecture

- **CLI (`llmctl`)**: User experience layer for driving workflows
- **Runtime**: Process orchestration, launchers, schedulers, topology detection
- **Partitioner**: Pipeline+tensor+sequence parallelism planner, sharding planner
- **Kernels/Execution**: Fused kernels, activation checkpointing, custom attention
- **Communications**: Collectives, overlap engine, topology planning
- **IO/Storage**: Dataset streaming, memory mapping, checkpointing
- **Serving**: Inference runtime, KV-cache manager, batching scheduler
- **Metrics**: FLOPs estimation, profiling, OpenTelemetry export
- **Configuration**: Validated schemas, hardware profiles, presets
- **Plugins**: Hardware backends, kernels, schedulers, quantizers

## Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```bash
# Initialize a new project
llmctl init --template gpt --size 7b

# Probe hardware and generate profile
llmctl hw probe --emit configs/hw/local.toml

# Compute parallelism plan
llmctl plan --model gpt-7b.json --hardware configs/hw/local.toml --strategy auto

# Launch distributed training
llmctl train --plan plans/local/7b.toml --data configs/data/dataset.toml

# Evaluate checkpoints
llmctl eval --ckpt checkpoints/step-1000 --suite eval_tasks.toml

# Start inference server
llmctl serve --artifact artifacts/7b-model --port 8080
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `llmctl init` | Scaffold project, create configs and directories |
| `llmctl hw` | Probe hardware, generate hardware profiles |
| `llmctl plan` | Compute parallelism plan given model & hardware constraints |
| `llmctl train` | Launch distributed training with computed plan |
| `llmctl eval` | Evaluate checkpoints (perplexity, accuracy, latency) |
| `llmctl export` | Convert checkpoints to deployment formats |
| `llmctl serve` | Start inference server with batching and paged attention |
| `llmctl bench` | Run micro/macro-benchmarks |
| `llmctl trace` | Capture/visualize runtime traces |
| `llmctl replay` | Deterministically replay a run for debugging |
| `llmctl tune` | Auto-tuning for kernels and communication overlap |
| `llmctl health` | Cluster health checks & drift detection |
| `llmctl admin` | Dataset/index operations, checkpoint GC, tensor inspection |

## Configuration

The system uses TOML configuration files with validated schemas. See `configs/` directory for examples.

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black llmctl/
isort llmctl/

# Type checking
mypy llmctl/
```

## License

MIT License