"""Auto-tuning command"""

import typer
from rich.console import Console
from pathlib import Path
from typing import Optional

from llmctl.plugins.autotuning import create_auto_tuner, TuningConfig

console = Console()
app = typer.Typer(help="Auto-tune kernels and communication")

@app.command()
def kernels(
    kernel_type: str = typer.Option("matmul", help="Kernel type to tune (matmul, attention, all)"),
    matrix_size: str = typer.Option("1024x1024x1024", help="Matrix size for matmul (MxKxN)"),
    seq_len: int = typer.Option(512, help="Sequence length for attention"),
    batch_size: int = typer.Option(8, help="Batch size for attention"),
    num_heads: int = typer.Option(8, help="Number of attention heads"),
    head_dim: int = typer.Option(64, help="Head dimension for attention"),
    device: str = typer.Option("auto", help="Device to use (auto, cuda, cpu)"),
    max_iterations: int = typer.Option(50, help="Maximum tuning iterations"),
    timeout: float = typer.Option(300.0, help="Tuning timeout in seconds"),
    save_results: Optional[Path] = typer.Option(None, help="Save results to file"),
    load_cache: Optional[Path] = typer.Option(None, help="Load cached results from file"),
) -> None:
    """Auto-tune kernel performance."""
    console.print("[blue]Auto-tuning kernels...[/blue]")
    
    # Create tuning configuration
    config = TuningConfig(
        max_iterations=max_iterations,
        timeout=timeout
    )
    
    # Create auto-tuner
    tuner = create_auto_tuner(config)
    
    # Load cached results if specified
    if load_cache and load_cache.exists():
        tuner.load_results(str(load_cache))
    
    try:
        if kernel_type == "matmul" or kernel_type == "all":
            console.print("[blue]Tuning matrix multiplication kernels...[/blue]")
            
            # Parse matrix size
            try:
                m, k, n = map(int, matrix_size.split('x'))
                result = tuner.tune_matmul((m, k, n), device)
                console.print(f"[green]MatMul tuning completed: {result.improvement:.1f}% improvement[/green]")
            except ValueError:
                console.print(f"[red]Invalid matrix size format: {matrix_size}. Use MxKxN format.[/red]")
                return
        
        if kernel_type == "attention" or kernel_type == "all":
            console.print("[blue]Tuning attention kernels...[/blue]")
            result = tuner.tune_attention(seq_len, head_dim, batch_size, num_heads, device)
            console.print(f"[green]Attention tuning completed: {result.improvement:.1f}% improvement[/green]")
        
        # Save results if specified
        if save_results:
            tuner.save_results(str(save_results))
            
        console.print("[green]✓ Kernel tuning completed successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]Kernel tuning failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def comms(
    tensor_size: str = typer.Option("1024x1024", help="Tensor size for communication (e.g., 1024x1024)"),
    dtype: str = typer.Option("float32", help="Data type (float32, float16, int32)"),
    max_iterations: int = typer.Option(50, help="Maximum tuning iterations"),
    timeout: float = typer.Option(300.0, help="Tuning timeout in seconds"),
    save_results: Optional[Path] = typer.Option(None, help="Save results to file"),
    load_cache: Optional[Path] = typer.Option(None, help="Load cached results from file"),
) -> None:
    """Auto-tune communication overlap."""
    console.print("[blue]Auto-tuning communication overlap...[/blue]")
    
    # Create tuning configuration
    config = TuningConfig(
        max_iterations=max_iterations,
        timeout=timeout
    )
    
    # Create auto-tuner
    tuner = create_auto_tuner(config)
    
    # Load cached results if specified
    if load_cache and load_cache.exists():
        tuner.load_results(str(load_cache))
    
    try:
        # Parse tensor size
        try:
            size_parts = list(map(int, tensor_size.split('x')))
            tensor_shape = tuple(size_parts)
        except ValueError:
            console.print(f"[red]Invalid tensor size format: {tensor_size}. Use dimensions separated by 'x'.[/red]")
            return
        
        # Parse dtype
        import torch
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "int32": torch.int32,
            "int64": torch.int64,
        }
        
        if dtype not in dtype_map:
            console.print(f"[red]Unsupported dtype: {dtype}. Supported: {list(dtype_map.keys())}[/red]")
            return
        
        # Run communication tuning
        result = tuner.tune_communication(tensor_shape, dtype_map[dtype])
        console.print(f"[green]Communication tuning completed: {result.improvement:.1f}% improvement[/green]")
        console.print(f"[green]Best config: {result.best_config}[/green]")
        
        # Save results if specified
        if save_results:
            tuner.save_results(str(save_results))
            
        console.print("[green]✓ Communication tuning completed successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]Communication tuning failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def full(
    device: str = typer.Option("auto", help="Device to use (auto, cuda, cpu)"),
    max_iterations: int = typer.Option(25, help="Maximum iterations per component"),
    timeout: float = typer.Option(600.0, help="Total tuning timeout in seconds"),
    output_dir: Path = typer.Option("./tuning_results", help="Output directory for results"),
) -> None:
    """Run comprehensive auto-tuning for all components."""
    console.print("[blue]Starting comprehensive auto-tuning...[/blue]")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tuning configuration
    config = TuningConfig(
        max_iterations=max_iterations,
        timeout=timeout / 3  # Divide time among components
    )
    
    # Create auto-tuner
    tuner = create_auto_tuner(config)
    
    results_summary = {}
    
    try:
        # Tune matrix multiplication
        console.print("[blue]1/3 Tuning matrix multiplication...[/blue]")
        matmul_result = tuner.tune_matmul((1024, 1024, 1024), device)
        results_summary["matmul"] = {
            "improvement": matmul_result.improvement,
            "best_config": matmul_result.best_config,
            "time": matmul_result.total_time
        }
        
        # Tune attention
        console.print("[blue]2/3 Tuning attention kernels...[/blue]")
        attention_result = tuner.tune_attention(512, 64, 8, 8, device)
        results_summary["attention"] = {
            "improvement": attention_result.improvement,
            "best_config": attention_result.best_config,
            "time": attention_result.total_time
        }
        
        # Tune communication
        console.print("[blue]3/3 Tuning communication...[/blue]")
        comm_result = tuner.tune_communication((1024, 1024))
        results_summary["communication"] = {
            "improvement": comm_result.improvement,
            "best_config": comm_result.best_config,
            "time": comm_result.total_time
        }
        
        # Save comprehensive results
        import json
        results_file = output_dir / "full_tuning_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        tuner.save_results(str(output_dir / "tuning_cache.json"))
        
        # Display summary
        console.print("\n[green]✓ Comprehensive auto-tuning completed![/green]")
        console.print("[blue]Summary:[/blue]")
        
        total_improvement = 0
        for component, result in results_summary.items():
            improvement = result["improvement"]
            total_improvement += improvement
            console.print(f"  {component}: {improvement:.1f}% improvement in {result['time']:.1f}s")
        
        avg_improvement = total_improvement / len(results_summary)
        console.print(f"[green]Average improvement: {avg_improvement:.1f}%[/green]")
        console.print(f"[green]Results saved to: {output_dir}[/green]")
        
    except Exception as e:
        console.print(f"[red]Comprehensive tuning failed: {e}[/red]")
        raise typer.Exit(1)