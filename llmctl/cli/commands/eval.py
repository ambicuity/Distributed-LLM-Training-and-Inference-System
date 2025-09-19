"""
Evaluation command - evaluate checkpoints
"""

import typer
from rich.console import Console
from pathlib import Path
from typing import Optional

console = Console()
app = typer.Typer(help="Evaluate checkpoints")

@app.command()
def run(
    checkpoint: Path = typer.Option(..., "--ckpt", help="Checkpoint path"),
    suite: Optional[Path] = typer.Option(None, help="Evaluation suite configuration"),
    tasks: Optional[str] = typer.Option("perplexity", help="Comma-separated evaluation tasks"),
    output: Optional[Path] = typer.Option(None, "--out", help="Output results file"),
) -> None:
    """Evaluate checkpoints (perplexity, accuracy tasks, latency)."""
    
    console.print(f"[blue]Evaluating checkpoint: {checkpoint}[/blue]")
    
    if suite:
        console.print(f"[green]✓[/green] Using evaluation suite: {suite}")
    
    task_list = tasks.split(",") if tasks else ["perplexity"]
    console.print(f"[yellow]Tasks: {', '.join(task_list)}[/yellow]")
    
    console.print("[red]Evaluation implementation coming soon![/red]")

# Make run the default command
@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Evaluate checkpoints."""
    if ctx.invoked_subcommand is None:
        run()