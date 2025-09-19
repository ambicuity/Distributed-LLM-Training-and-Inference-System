"""Replay command"""

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Replay runs for debugging")

@app.command()
def run(run_id: str = typer.Option(..., help="Run ID to replay")) -> None:
    """Deterministically replay a run for debugging."""
    console.print(f"[blue]Replaying run: {run_id}[/blue]")
    console.print("[red]Replay implementation coming soon![/red]")