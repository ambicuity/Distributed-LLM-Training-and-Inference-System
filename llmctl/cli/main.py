"""
Main CLI entry point for llmctl
"""

import typer
from rich.console import Console
from rich.traceback import install
from typing import Optional
import sys
from pathlib import Path

# Install rich traceback handler
install(show_locals=True)

# Initialize console for rich output
console = Console()

# Main app
app = typer.Typer(
    name="llmctl",
    help="Distributed LLM Training and Inference System",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Import subcommands
from .commands import (
    init,
    hw,
    plan,
    train,
    eval,
    export,
    serve,
    bench,
    trace,
    replay,
    tune,
    health,
    admin,
)

# Add subcommands
app.add_typer(init.app, name="init", help="Initialize project and create configs")
app.add_typer(hw.app, name="hw", help="Hardware probing and profiling")
app.add_typer(plan.app, name="plan", help="Compute parallelism plans")
app.add_typer(train.app, name="train", help="Launch distributed training")
app.add_typer(eval.app, name="eval", help="Evaluate checkpoints")
app.add_typer(export.app, name="export", help="Export models to deployment formats")
app.add_typer(serve.app, name="serve", help="Start inference server")
app.add_typer(bench.app, name="bench", help="Run benchmarks")
app.add_typer(trace.app, name="trace", help="Capture and visualize traces")
app.add_typer(replay.app, name="replay", help="Replay runs for debugging")
app.add_typer(tune.app, name="tune", help="Auto-tune kernels and communication")
app.add_typer(health.app, name="health", help="Cluster health checks")
app.add_typer(admin.app, name="admin", help="Administrative operations")

# Global options
@app.callback()
def main(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration file path (TOML/YAML)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        "-p",
        help="Hardware/cluster profile name",
    ),
    backend: Optional[str] = typer.Option(
        "torch",
        "--backend",
        "-b",
        help="Backend to use",
    ),
    launcher: Optional[str] = typer.Option(
        "local",
        "--launcher",
        "-l",
        help="Launcher to use",
    ),
    nodes: Optional[int] = typer.Option(
        1,
        "--nodes",
        "-n",
        help="Number of nodes",
        min=1,
    ),
    gpus_per_node: Optional[int] = typer.Option(
        None,
        "--gpus-per-node",
        "-g",
        help="GPUs per node",
        min=1,
    ),
    cpus_per_task: Optional[int] = typer.Option(
        None,
        "--cpus-per-task",
        help="CPUs per task",
        min=1,
    ),
    mixed_precision: Optional[str] = typer.Option(
        "bf16",
        "--mixed-precision",
        help="Mixed precision mode",
    ),
    seed: Optional[int] = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for reproducibility",
    ),
    deterministic: bool = typer.Option(
        False,
        "--deterministic",
        help="Enable deterministic mode",
    ),
    log_level: str = typer.Option(
        "info",
        "--log-level",
        help="Log level",
    ),
    otlp_endpoint: Optional[str] = typer.Option(
        None,
        "--otlp-endpoint",
        help="OpenTelemetry endpoint URL",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """
    Distributed LLM Training and Inference System
    
    A comprehensive CLI tool for orchestrating data preparation, model partitioning,
    distributed training, checkpointing, evaluation, and low-latency inference.
    """
    # Global options are handled by Typer automatically
    # Set up logging and other global configurations here if needed
    if verbose:
        console.print(f"[dim]Global options: backend={backend}, launcher={launcher}, nodes={nodes}[/dim]")

if __name__ == "__main__":
    app()