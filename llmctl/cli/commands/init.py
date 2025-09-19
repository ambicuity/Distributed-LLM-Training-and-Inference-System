"""
Initialize command - scaffold project structure and configs
"""

import typer
from rich.console import Console
from rich.prompt import Prompt, Confirm
from pathlib import Path
import toml
import json
from typing import Optional

console = Console()
app = typer.Typer(help="Initialize project and create configs")

MODEL_TEMPLATES = {
    "gpt": {
        "7b": {
            "name": "gpt-7b",
            "arch": "decoder-only", 
            "layers": 32,
            "hidden": 4096,
            "ffn": 11008,
            "heads": 32,
            "vocab_size": 32000,
            "rope": {"base": 10000, "scaling": "su"}
        },
        "13b": {
            "name": "gpt-13b",
            "arch": "decoder-only",
            "layers": 40, 
            "hidden": 5120,
            "ffn": 13824,
            "heads": 40,
            "vocab_size": 32000,
            "rope": {"base": 10000, "scaling": "su"}
        }
    },
    "llama": {
        "7b": {
            "name": "llama-7b",
            "arch": "decoder-only",
            "layers": 32,
            "hidden": 4096, 
            "ffn": 11008,
            "heads": 32,
            "vocab_size": 32000,
            "rope": {"base": 10000, "scaling": "linear"}
        }
    }
}

@app.command()
def scaffold(
    template: str = typer.Option("gpt", help="Model template (gpt, llama)"),
    size: str = typer.Option("7b", help="Model size (7b, 13b)"),
    name: Optional[str] = typer.Option(None, help="Project name"),
    output_dir: Path = typer.Option(Path("."), help="Output directory"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing files"),
) -> None:
    """Scaffold a new LLM project with configs and directory structure."""
    
    if name is None:
        name = f"{template}-{size}-project"
    
    project_dir = output_dir / name
    
    # Check if directory exists
    if project_dir.exists() and not force:
        if not Confirm.ask(f"Directory {project_dir} exists. Continue?"):
            raise typer.Abort()
    
    console.print(f"[bold green]Creating project: {name}[/bold green]")
    console.print(f"[blue]Location: {project_dir}[/blue]")
    
    # Create directory structure
    directories = [
        "configs/presets",
        "configs/hw", 
        "configs/data",
        "configs/models",
        "plans",
        "checkpoints",
        "artifacts",
        "logs",
        "data",
        "scripts",
    ]
    
    for dir_name in directories:
        (project_dir / dir_name).mkdir(parents=True, exist_ok=True)
        console.print(f"[dim]Created: {dir_name}[/dim]")
    
    # Generate model config
    if template in MODEL_TEMPLATES and size in MODEL_TEMPLATES[template]:
        model_config = MODEL_TEMPLATES[template][size]
        model_file = project_dir / "configs" / "models" / f"{template}-{size}.json"
        
        with open(model_file, "w") as f:
            json.dump(model_config, f, indent=2)
        console.print(f"[green]Created model config: {model_file}[/green]")
    
    # Generate default config
    default_config = {
        "model": {
            "name": f"{template}-{size}",
            "arch": "decoder-only",
            "config_file": f"configs/models/{template}-{size}.json"
        },
        "optimizer": {
            "type": "adamw",
            "lr": 2e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 0.1,
            "scheduler": {"type": "cosine", "warmup_steps": 2000}
        },
        "data": {
            "train": "data/train",
            "val": "data/val", 
            "tokenizer": "tokenizers/gpt-bpe.json",
            "pack_sequences": True,
            "num_workers": 8
        },
        "hardware": {
            "gpus_per_node": 1,
            "gpu": "auto-detect",
            "memory_gb": "auto-detect",
            "intra_node_interconnect": "auto-detect",
            "inter_node_interconnect": "auto-detect",
            "cpu_pinning": "numa-aware"
        },
        "parallel": {
            "strategy": "auto",
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
            "sequence_parallel": False,
            "zero_stage": 1,
            "activation_checkpoint": "selective",
            "micro_batch_size": 1,
            "global_batch_size": 64
        },
        "limits": {
            "target_flops": 1.0e12,
            "max_memory_gb": 16,
            "max_comm_bw_gbps": 100
        },
        "checkpoint": {
            "path": "checkpoints",
            "interval_steps": 1000,
            "sharded": True,
            "async": True
        },
        "telemetry": {
            "otlp_endpoint": None,
            "metrics": ["flops", "mem_bw", "comm_bw", "latency", "throughput", "loss"],
            "traces": True
        }
    }
    
    config_file = project_dir / "configs" / "default.toml"
    with open(config_file, "w") as f:
        toml.dump(default_config, f)
    console.print(f"[green]Created default config: {config_file}[/green]")
    
    # Create example dataset config
    dataset_config = {
        "name": "example-dataset",
        "format": "json",
        "sources": [
            {"path": "data/train.jsonl", "split": "train"},
            {"path": "data/val.jsonl", "split": "validation"}
        ],
        "preprocessing": {
            "tokenizer": "gpt2",
            "max_length": 2048,
            "padding": "max_length",
            "truncation": True
        }
    }
    
    data_config_file = project_dir / "configs" / "data" / "example.toml"
    with open(data_config_file, "w") as f:
        toml.dump(dataset_config, f)
    console.print(f"[green]Created dataset config: {data_config_file}[/green]")
    
    # Create example training script
    training_script = f'''#!/bin/bash
# Example training script for {name}

# Basic single-node training
llmctl train \\
    --config configs/default.toml \\
    --data configs/data/example.toml \\
    --launcher local \\
    --gpus-per-node 1

# Multi-node training (uncomment for distributed setup)
# llmctl train \\
#     --config configs/default.toml \\
#     --data configs/data/example.toml \\
#     --launcher slurm \\
#     --nodes 4 \\
#     --gpus-per-node 8
'''
    
    script_file = project_dir / "scripts" / "train.sh"
    with open(script_file, "w") as f:
        f.write(training_script)
    script_file.chmod(0o755)
    console.print(f"[green]Created training script: {script_file}[/green]")
    
    # Create README
    readme_content = f"""# {name}

This project was scaffolded using llmctl with template: {template}-{size}

## Quick Start

1. Install dependencies:
   ```bash
   pip install llmctl
   ```

2. Probe hardware:
   ```bash
   llmctl hw probe --emit configs/hw/local.toml
   ```

3. Compute parallelism plan:
   ```bash
   llmctl plan --model configs/models/{template}-{size}.json --hardware configs/hw/local.toml --out plans/local.toml
   ```

4. Launch training:
   ```bash
   ./scripts/train.sh
   ```

## Directory Structure

- `configs/` - Configuration files
- `plans/` - Parallelism plans  
- `checkpoints/` - Model checkpoints
- `artifacts/` - Exported model artifacts
- `logs/` - Training logs
- `data/` - Training data
- `scripts/` - Helper scripts

## Configuration

Main configuration is in `configs/default.toml`. Customize model, training, and hardware settings as needed.
"""
    
    readme_file = project_dir / "README.md"
    with open(readme_file, "w") as f:
        f.write(readme_content)
    console.print(f"[green]Created README: {readme_file}[/green]")
    
    console.print(f"\n[bold green]✅ Project {name} initialized successfully![/bold green]")
    console.print(f"\n[yellow]Next steps:[/yellow]")
    console.print(f"1. cd {project_dir}")
    console.print(f"2. llmctl hw probe --emit configs/hw/local.toml")
    console.print(f"3. llmctl plan --model configs/models/{template}-{size}.json --hardware configs/hw/local.toml")
    console.print(f"4. ./scripts/train.sh")

# Make scaffold the default command
@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Initialize project and create configs."""
    if ctx.invoked_subcommand is None:
        scaffold()