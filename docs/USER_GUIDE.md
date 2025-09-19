# LLMCtl - Distributed LLM Training and Inference System

This document provides a complete guide to using the LLMCtl system for distributed large language model training and inference.

## Overview

LLMCtl is a comprehensive CLI tool that orchestrates:
- Data preparation and preprocessing
- Model partitioning and parallelism planning
- Distributed training across multiple nodes and GPUs
- Checkpointing and model export
- Low-latency inference serving
- Performance monitoring and optimization

## Quick Start Guide

### 1. Installation

```bash
pip install -e .
```

### 2. Initialize a New Project

```bash
llmctl init scaffold --template gpt --size 7b --name my-gpt-project
cd my-gpt-project
```

### 3. Probe Your Hardware

```bash
llmctl hw probe --emit configs/hw/local.toml
```

### 4. Compute Parallelism Plan

```bash
llmctl plan compute \
  --model configs/models/gpt-7b.json \
  --hardware configs/hw/local.toml \
  --out plans/local.toml
```

### 5. Launch Training

```bash
llmctl train launch \
  --plan plans/local.toml \
  --data configs/data/example.toml \
  --launcher local
```

## Command Reference

### `llmctl init`
Initialize new projects with pre-configured templates and directory structures.

**Options:**
- `--template`: Model architecture template (gpt, llama)
- `--size`: Model size (7b, 13b, 30b, 70b)
- `--name`: Project name
- `--output-dir`: Output directory

**Example:**
```bash
llmctl init scaffold --template llama --size 7b --name llama-experiment
```

### `llmctl hw`
Hardware probing and performance profiling.

**Subcommands:**
- `probe`: Detect hardware capabilities
- `benchmark`: Run micro-benchmarks

**Options:**
- `--emit`: Save hardware profile to file
- `--format`: Output format (toml, json)
- `--verbose`: Detailed output

**Example:**
```bash
llmctl hw probe --emit configs/hw/dgx-a100.toml --verbose
llmctl hw benchmark --component memory --duration 30
```

### `llmctl plan`
Compute optimal parallelism strategies based on model and hardware constraints.

**Options:**
- `--model`: Model configuration file
- `--hardware`: Hardware profile file
- `--target-flops`: Target FLOPs ceiling
- `--max-memory-gb`: Maximum memory per GPU
- `--max-comm-bw-gbps`: Maximum communication bandwidth
- `--strategy`: Planning strategy (auto, manual)
- `--out`: Output plan file

**Example:**
```bash
llmctl plan compute \
  --model configs/models/llama-7b.json \
  --hardware configs/hw/a100x8.toml \
  --target-flops 2e15 \
  --max-memory-gb 72 \
  --out plans/llama-7b-a100x8.toml
```

### `llmctl train`
Launch distributed training with automatic parallelism and fault tolerance.

**Options:**
- `--plan`: Parallelism plan file
- `--config`: Training configuration
- `--data`: Data configuration
- `--launcher`: Launcher type (local, slurm, mpi, k8s)
- `--nodes`: Number of nodes
- `--gpus-per-node`: GPUs per node

**Example:**
```bash
llmctl train launch \
  --plan plans/llama-7b-a100x8.toml \
  --data configs/data/the-stack.toml \
  --launcher slurm \
  --nodes 4 \
  --gpus-per-node 8
```

### `llmctl eval`
Evaluate trained models on various benchmarks and tasks.

**Options:**
- `--checkpoint`: Checkpoint directory or file
- `--suite`: Evaluation suite configuration
- `--tasks`: Comma-separated evaluation tasks
- `--output`: Results output file

**Example:**
```bash
llmctl eval run \
  --checkpoint checkpoints/llama-7b/step-50000 \
  --suite configs/eval/lm-eval-harness.toml \
  --output results/eval-50k.json
```

### `llmctl serve`
Start high-performance inference server with dynamic batching.

**Options:**
- `--artifact`: Model artifact path
- `--port`: Server port
- `--host`: Server host
- `--scheduler`: Batching scheduler type
- `--max-batch-tokens`: Maximum tokens per batch

**Example:**
```bash
llmctl serve start \
  --artifact artifacts/llama-7b-awq \
  --port 8080 \
  --scheduler vllm-like \
  --max-batch-tokens 8192
```

## Configuration Files

### Model Configuration
```json
{
  "name": "llama-7b",
  "arch": "decoder-only",
  "layers": 32,
  "hidden": 4096,
  "ffn": 11008,
  "heads": 32,
  "vocab_size": 32000,
  "rope": {
    "base": 10000.0,
    "scaling": "linear"
  }
}
```

### Hardware Profile
```toml
[gpu]
count = 8
[[gpu.devices]]
id = 0
name = "NVIDIA A100-SXM4-80GB"
memory_gb = 80.0

[interconnect]
intra_node = "NVLink"
inter_node = "InfiniBand"

[limits]
estimated_flops = 2.496e15
memory_bw_gbps = 15480
```

### Training Configuration
```toml
[model]
name = "llama-7b"
config_file = "configs/models/llama-7b.json"

[optimizer]
type = "adamw"
lr = 2e-4
betas = [0.9, 0.95]

[parallel]
strategy = "auto"
zero_stage = 2
micro_batch_size = 4
global_batch_size = 512
```

## Best Practices

### Memory Optimization
1. Use appropriate ZeRO stages based on model size
2. Enable gradient checkpointing for large models
3. Tune micro-batch sizes for optimal memory usage

### Communication Optimization
1. Prefer tensor parallelism within nodes
2. Use pipeline parallelism across nodes
3. Enable communication overlap when possible

### Performance Tuning
1. Profile training runs to identify bottlenecks
2. Use mixed precision training (bf16/fp16)
3. Enable Flash Attention for transformer models

## Troubleshooting

### Common Issues

**Memory Out of Error:**
- Reduce micro-batch size
- Increase ZeRO stage
- Use gradient checkpointing

**Slow Training:**
- Check communication bandwidth
- Verify optimal parallelism plan
- Enable kernel fusion

**Convergence Issues:**
- Verify data preprocessing
- Check learning rate schedule
- Monitor gradient norms

## Advanced Usage

### Custom Kernels
Register custom CUDA kernels through the plugin system:

```python
# plugins/custom_kernels.py
def register():
    return {
        'flash_attention_v3': FlashAttentionV3Kernel,
        'fused_rmsnorm': FusedRMSNormKernel,
    }
```

### Custom Schedulers
Implement custom batching schedulers:

```python
# plugins/custom_scheduler.py  
class CustomBatchScheduler:
    def schedule(self, requests):
        # Custom scheduling logic
        return batched_requests
```

### Multi-Node Setup
For multi-node training, ensure:
1. Shared filesystem access (NFS, Lustre)
2. Network connectivity between nodes
3. Proper NCCL configuration for InfiniBand

## Support

For issues and questions:
- GitHub Issues: [Repository Issues](https://github.com/ambicuity/Distributed-LLM-Training-and-Inference-System/issues)
- Documentation: See `docs/` directory
- Examples: See `examples/` directory