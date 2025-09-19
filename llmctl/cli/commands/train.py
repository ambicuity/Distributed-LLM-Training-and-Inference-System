"""
Train command - launch distributed training
"""

import typer
from rich.console import Console
from pathlib import Path
from typing import Optional

from llmctl.runtime import LaunchConfig, ProcessOrchestrator

console = Console()
app = typer.Typer(help="Launch distributed training")

@app.command()
def launch(
    plan: Optional[Path] = typer.Option(None, help="Parallelism plan file"),
    config: Optional[Path] = typer.Option(None, help="Training configuration file"),
    data: Optional[Path] = typer.Option(None, help="Data configuration file"),
    model: str = typer.Option("gpt2", help="Model name or path"),
    output_dir: str = typer.Option("./outputs", help="Output directory"),
    checkpoint: Optional[str] = typer.Option(None, help="Checkpoint path for resuming"),
    launcher: str = typer.Option("local", help="Launcher (local, slurm, mpi, k8s)"),
    nodes: int = typer.Option(1, help="Number of nodes"),
    gpus_per_node: int = typer.Option(1, help="GPUs per node"),
    batch_size: int = typer.Option(8, help="Batch size per device"),
    learning_rate: float = typer.Option(5e-5, help="Learning rate"),
    num_epochs: int = typer.Option(3, help="Number of epochs"),
    mixed_precision: str = typer.Option("bf16", help="Mixed precision mode"),
    grad_accum: int = typer.Option(1, help="Gradient accumulation steps"),
    clip_grad: float = typer.Option(1.0, help="Gradient clipping norm"),
    dry_run: bool = typer.Option(False, help="Dry run - show command without executing"),
) -> None:
    """Launch distributed training with the computed plan."""
    
    console.print("[blue]Preparing distributed training...[/blue]")
    
    if plan:
        console.print(f"[green]✓[/green] Using plan: {plan}")
    if config:
        console.print(f"[green]✓[/green] Using config: {config}")
    if data:
        console.print(f"[green]✓[/green] Using data config: {data}")
    
    console.print(f"[yellow]Model: {model}[/yellow]")
    console.print(f"[yellow]Output directory: {output_dir}[/yellow]")
    console.print(f"[yellow]Launcher: {launcher}[/yellow]")
    console.print(f"[yellow]Resources: {nodes} nodes × {gpus_per_node} GPUs[/yellow]")
    console.print(f"[yellow]Batch size: {batch_size}, Learning rate: {learning_rate}[/yellow]")
    
    if dry_run:
        console.print("[yellow]Dry run - would launch training with above configuration[/yellow]")
        return
    
    # Create launch configuration
    launch_config = LaunchConfig(
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        launcher=launcher,
        mixed_precision=mixed_precision,
        config_path=config,
        data_path=data,
        plan_path=plan,
        checkpoint_path=checkpoint,
        gradient_accumulation_steps=grad_accum,
        gradient_clipping=clip_grad
    )
    
    # Create process orchestrator
    orchestrator = ProcessOrchestrator(launch_config)
    
    # Prepare training script arguments
    train_script = str(Path(__file__).parent.parent.parent / "runtime" / "train_script.py")
    
    script_args = [
        "--model-name-or-path", model,
        "--dataset-path", data or "./dummy_dataset",
        "--output-dir", output_dir,
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--num-epochs", str(num_epochs),
        "--mixed-precision", mixed_precision,
        "--gradient-accumulation-steps", str(grad_accum),
        "--gradient-clipping", str(clip_grad),
    ]
    
    if config:
        script_args.extend(["--config", str(config)])
    
    if checkpoint:
        script_args.extend(["--resume-from-checkpoint", checkpoint])
    
    # Launch training
    try:
        success = orchestrator.start_training(train_script, script_args)
        if success:
            console.print("[green]✓ Training completed successfully![/green]")
        else:
            console.print("[red]Training failed![/red]")
            raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        orchestrator.stop_training()
    except Exception as e:
        console.print(f"[red]Training failed with error: {e}[/red]")
        raise typer.Exit(1)

# Make launch the default command
@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Launch distributed training."""
    if ctx.invoked_subcommand is None:
        launch()