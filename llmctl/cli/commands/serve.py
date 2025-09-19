"""
Serve command - start inference server
"""

import asyncio
import typer
from rich.console import Console
from pathlib import Path
from typing import Optional

from llmctl.serve.server import create_inference_server

console = Console()
app = typer.Typer(help="Start inference server")

@app.command()
def start(
    artifact: Path = typer.Option(..., help="Model artifact path"),
    port: int = typer.Option(8080, help="Server port"),
    host: str = typer.Option("0.0.0.0", help="Server host"),
    scheduler: str = typer.Option("dynamic", help="Batching scheduler (dynamic, simple)"),
    max_batch_size: int = typer.Option(8, help="Maximum batch size"),
    max_batch_tokens: int = typer.Option(8192, help="Maximum batch tokens"),
    max_concurrent: int = typer.Option(128, help="Maximum concurrent requests"),
    device: str = typer.Option("auto", help="Device to use (auto, cuda, cpu)"),
) -> None:
    """Start inference server with batching and paged attention."""
    
    console.print(f"[blue]Starting inference server...[/blue]")
    console.print(f"[green]Model: {artifact}[/green]")
    console.print(f"[yellow]Server: {host}:{port}[/yellow]")
    console.print(f"[yellow]Scheduler: {scheduler}[/yellow]")
    console.print(f"[yellow]Max batch size: {max_batch_size}[/yellow]")
    console.print(f"[yellow]Max batch tokens: {max_batch_tokens}[/yellow]")
    console.print(f"[yellow]Max concurrent: {max_concurrent}[/yellow]")
    console.print(f"[yellow]Device: {device}[/yellow]")
    
    # Validate model path
    if not artifact.exists():
        console.print(f"[red]Error: Model path does not exist: {artifact}[/red]")
        raise typer.Exit(1)
    
    try:
        # Create and start inference server
        server = create_inference_server(
            model_path=str(artifact),
            host=host,
            port=port,
            max_batch_size=max_batch_size,
            max_batch_tokens=max_batch_tokens,
            max_concurrent=max_concurrent
        )
        
        # Run the server
        asyncio.run(server.start_server())
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Server shutdown requested[/yellow]")
    except Exception as e:
        console.print(f"[red]Server error: {e}[/red]")
        raise typer.Exit(1)

# Make start the default command
@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Start inference server."""
    if ctx.invoked_subcommand is None:
        # Extract arguments from context and call start with defaults
        start()