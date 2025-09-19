"""
Serve command - start inference server
"""

import typer
from rich.console import Console
from pathlib import Path
from typing import Optional

console = Console()
app = typer.Typer(help="Start inference server")

@app.command()
def start(
    artifact: Path = typer.Option(..., help="Model artifact path"),
    port: int = typer.Option(8080, help="Server port"),
    host: str = typer.Option("0.0.0.0", help="Server host"),
    scheduler: str = typer.Option("vllm-like", help="Batching scheduler"),
    max_batch_tokens: int = typer.Option(8192, help="Maximum batch tokens"),
    max_concurrent: int = typer.Option(128, help="Maximum concurrent requests"),
) -> None:
    """Start inference server with batching and paged attention."""
    
    console.print(f"[blue]Starting inference server...[/blue]")
    console.print(f"[green]Model: {artifact}[/green]")
    console.print(f"[yellow]Server: {host}:{port}[/yellow]")
    console.print(f"[yellow]Scheduler: {scheduler}[/yellow]")
    console.print(f"[yellow]Max batch tokens: {max_batch_tokens}[/yellow]")
    
    console.print("[red]Serving implementation coming soon![/red]")

# Make start the default command
@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Start inference server."""
    if ctx.invoked_subcommand is None:
        start()