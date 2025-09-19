"""Health monitoring command"""

import typer
import time
from rich.console import Console
from rich.table import Table
from pathlib import Path
from typing import Optional

from llmctl.metrics.health import setup_health_monitoring, HealthStatus

console = Console()
app = typer.Typer(help="Cluster health checks")

@app.command()
def check(
    component: str = typer.Option("all", help="Component to check (system, training, inference, all)"),
    save_report: Optional[Path] = typer.Option(None, help="Save health report to file"),
    monitor_duration: Optional[int] = typer.Option(None, help="Monitor for N seconds"),
    check_interval: float = typer.Option(30.0, help="Check interval in seconds"),
) -> None:
    """Run health checks and show status."""
    
    console.print("[blue]Running health checks...[/blue]")
    
    # Setup health monitoring
    health_manager = setup_health_monitoring(check_interval=check_interval)
    
    if monitor_duration:
        # Start monitoring mode
        console.print(f"[blue]Starting health monitoring for {monitor_duration} seconds...[/blue]")
        health_manager.start_monitoring()
        
        try:
            start_time = time.time()
            while time.time() - start_time < monitor_duration:
                time.sleep(5)  # Check every 5 seconds in monitor mode
                _display_health_status(health_manager, component)
                console.print("\n" + "─" * 80 + "\n")
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring interrupted by user[/yellow]")
        finally:
            health_manager.stop_monitoring()
    else:
        # Single check mode
        _display_health_status(health_manager, component)
    
    # Save report if requested
    if save_report:
        health_manager.save_health_report(str(save_report))

def _display_health_status(health_manager, component_filter: str):
    """Display health status in a formatted table."""
    
    current_health = health_manager.get_current_health()
    overall_status = health_manager.get_overall_status()
    
    # Overall status
    status_color = {
        HealthStatus.HEALTHY: "green",
        HealthStatus.WARNING: "yellow", 
        HealthStatus.CRITICAL: "red",
        HealthStatus.UNKNOWN: "white"
    }
    
    console.print(f"\n[{status_color[overall_status]}]Overall Status: {overall_status.value.upper()}[/{status_color[overall_status]}]")
    console.print(f"Last Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Component details
    table = Table(title="Component Health Details")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Checks", style="blue")
    table.add_column("Key Metrics", style="magenta") 
    table.add_column("Message", style="white")
    
    for name, report in current_health.items():
        if component_filter != "all" and name != component_filter:
            continue
            
        # Status with color
        status_text = f"[{status_color[report.status]}]{report.status.value}[/{status_color[report.status]}]"
        
        # Checks summary
        passed_checks = sum(1 for passed in report.checks.values() if passed)
        total_checks = len(report.checks)
        checks_text = f"{passed_checks}/{total_checks} passed"
        
        # Key metrics
        key_metrics = []
        if "cpu_usage" in report.metrics:
            key_metrics.append(f"CPU: {report.metrics['cpu_usage']:.1f}%")
        if "memory_usage" in report.metrics:
            key_metrics.append(f"Mem: {report.metrics['memory_usage']:.1f}%")
        if "error_rate" in report.metrics:
            key_metrics.append(f"Errors: {report.metrics['error_rate']:.1f}%")
        if "avg_latency" in report.metrics:
            key_metrics.append(f"Latency: {report.metrics['avg_latency']:.3f}s")
        if "active_requests" in report.metrics:
            key_metrics.append(f"Active: {report.metrics['active_requests']}")
        
        metrics_text = "; ".join(key_metrics[:3])  # Show max 3 metrics
        
        table.add_row(
            name.title(),
            status_text,
            checks_text,
            metrics_text,
            report.message[:50] + "..." if len(report.message) > 50 else report.message
        )
    
    console.print(table)

@app.command()
def drift(
    baseline_file: Path = typer.Option(..., help="Baseline health report file"),
    tolerance: float = typer.Option(10.0, help="Drift tolerance percentage"),
) -> None:
    """Check for performance drift."""
    
    console.print("[blue]Checking for health drift...[/blue]")
    
    # Load baseline
    try:
        import json
        with open(baseline_file) as f:
            baseline_data = json.load(f)
        console.print(f"[green]Loaded baseline from {baseline_file}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to load baseline: {e}[/red]")
        raise typer.Exit(1)
    
    # Get current health
    health_manager = setup_health_monitoring()
    current_health = health_manager.get_current_health()
    
    # Compare metrics
    drift_detected = False
    table = Table(title="Health Drift Analysis")
    table.add_column("Component", style="cyan")
    table.add_column("Metric", style="blue")
    table.add_column("Baseline", style="green")
    table.add_column("Current", style="yellow")
    table.add_column("Drift %", style="bold")
    table.add_column("Status", style="white")
    
    for component_name, current_report in current_health.items():
        if component_name not in baseline_data:
            continue
            
        baseline_metrics = baseline_data[component_name].get("metrics", {})
        current_metrics = current_report.metrics
        
        for metric_name in baseline_metrics:
            if metric_name not in current_metrics:
                continue
                
            baseline_value = baseline_metrics[metric_name]
            current_value = current_metrics[metric_name]
            
            if baseline_value == 0:
                drift_percent = 0 if current_value == 0 else float('inf')
            else:
                drift_percent = abs((current_value - baseline_value) / baseline_value) * 100
            
            status = "OK"
            if drift_percent > tolerance:
                status = "DRIFT"
                drift_detected = True
            
            table.add_row(
                component_name,
                metric_name,
                f"{baseline_value:.2f}",
                f"{current_value:.2f}",
                f"{drift_percent:.1f}%",
                f"[red]{status}[/red]" if status == "DRIFT" else f"[green]{status}[/green]"
            )
    
    console.print(table)
    
    if drift_detected:
        console.print(f"[red]⚠️  Health drift detected (>{tolerance}% change)[/red]")
        raise typer.Exit(1)
    else:
        console.print("[green]✓ No significant health drift detected[/green]")