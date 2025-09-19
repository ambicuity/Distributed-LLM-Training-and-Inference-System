"""
Plan command - compute parallelism and sharding plans
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
import toml
import json
from typing import Optional, Dict, Any, Tuple
import math

console = Console()
app = typer.Typer(help="Compute parallelism plans")

class ParallelismPlanner:
    """Computes optimal parallelism strategies."""
    
    def __init__(self, model_config: Dict[str, Any], hardware_profile: Dict[str, Any]):
        self.model = model_config
        self.hardware = hardware_profile
        
    def estimate_model_memory(self) -> float:
        """Estimate model memory requirements in GB."""
        params = self.estimate_parameters()
        
        # Model weights (fp16)
        weight_memory = params * 2 / (1024**3)
        
        # Gradients (fp16) 
        grad_memory = params * 2 / (1024**3)
        
        # Optimizer states (AdamW: 2x fp32 + momentum)
        optimizer_memory = params * 8 / (1024**3)
        
        return weight_memory + grad_memory + optimizer_memory
    
    def estimate_parameters(self) -> int:
        """Estimate number of parameters."""
        hidden = self.model.get("hidden", 4096)
        layers = self.model.get("layers", 32)
        vocab_size = self.model.get("vocab_size", 32000)
        ffn = self.model.get("ffn", hidden * 4)
        
        # Embedding layer
        embed_params = vocab_size * hidden
        
        # Each transformer layer
        layer_params = (
            4 * hidden * hidden +  # QKV + output projection
            2 * hidden * ffn +     # FFN up and down
            4 * hidden             # Layer norms and biases
        )
        
        total_params = embed_params + layers * layer_params
        return total_params
    
    def estimate_activation_memory(self, batch_size: int, seq_len: int) -> float:
        """Estimate activation memory in GB."""
        hidden = self.model.get("hidden", 4096)
        layers = self.model.get("layers", 32)
        
        # Activations per layer (rough estimate)
        per_layer = batch_size * seq_len * hidden * 2 / (1024**3)  # fp16
        
        # Attention weights
        attention = batch_size * seq_len * seq_len * 2 / (1024**3)  # fp16
        
        return layers * per_layer + layers * attention
    
    def compute_memory_requirement(self, tp: int, pp: int, zero_stage: int, 
                                 micro_batch: int, seq_len: int = 2048) -> float:
        """Compute total memory requirement per GPU."""
        model_mem = self.estimate_model_memory()
        
        # Tensor parallelism reduces model memory
        if tp > 1:
            model_mem = model_mem / tp
            
        # ZeRO sharding
        if zero_stage >= 2:  # Optimizer sharding
            model_mem = model_mem * 0.6  # Keep weights + grads, shard optimizer
        if zero_stage >= 3:  # Full sharding  
            model_mem = model_mem * 0.3  # Shard everything
            
        # Activation memory
        activation_mem = self.estimate_activation_memory(micro_batch, seq_len)
        
        # Pipeline parallelism reduces activation memory
        if pp > 1:
            activation_mem = activation_mem / pp
            
        return model_mem + activation_mem
    
    def estimate_flops(self, batch_size: int, seq_len: int) -> float:
        """Estimate FLOPs per forward pass."""
        params = self.estimate_parameters()
        
        # Rough estimate: 2 * params * batch_size * seq_len
        return 2 * params * batch_size * seq_len
    
    def estimate_communication_cost(self, tp: int, pp: int, zero_stage: int,
                                   global_batch: int, micro_batch: int) -> float:
        """Estimate communication overhead."""
        hidden = self.model.get("hidden", 4096)
        params = self.estimate_parameters()
        
        # Tensor parallel communication (all-reduce)
        tp_comm = 0
        if tp > 1:
            tp_comm = global_batch * hidden * 2 / (1024**3)  # GB per step
            
        # Pipeline parallel communication
        pp_comm = 0
        if pp > 1:
            pp_comm = micro_batch * hidden * 2 / (1024**3)  # GB per micro-batch
            
        # ZeRO communication
        zero_comm = 0
        if zero_stage >= 1:
            zero_comm = params * 2 / (1024**3)  # Gradient all-reduce
            
        return tp_comm + pp_comm + zero_comm
    
    def search_optimal_plan(self, target_flops: float, max_memory: float, 
                          max_comm_bw: float) -> Dict[str, Any]:
        """Search for optimal parallelism plan."""
        gpu_count = self.hardware.get("gpu", {}).get("count", 1)
        
        best_plan = None
        best_score = float("inf")
        
        # Search space
        for tp in [1, 2, 4, 8]:
            if tp > gpu_count:
                continue
                
            for pp in [1, 2, 4, 8]:
                if tp * pp > gpu_count:
                    continue
                    
                for zero_stage in [0, 1, 2, 3]:
                    for micro_batch in [1, 2, 4, 8]:
                        # Check memory feasibility
                        memory_req = self.compute_memory_requirement(
                            tp, pp, zero_stage, micro_batch
                        )
                        
                        if memory_req > max_memory:
                            continue
                            
                        # Compute global batch size
                        dp = gpu_count // (tp * pp)
                        global_batch = micro_batch * dp * 4  # Some gradient accumulation
                        
                        # Check FLOPs constraint
                        step_flops = self.estimate_flops(global_batch, 2048)
                        if step_flops > target_flops:
                            continue
                            
                        # Check communication constraint
                        comm_cost = self.estimate_communication_cost(
                            tp, pp, zero_stage, global_batch, micro_batch
                        )
                        
                        if comm_cost > max_comm_bw:
                            continue
                            
                        # Score plan (minimize memory usage + communication)
                        score = memory_req + comm_cost * 10
                        
                        if score < best_score:
                            best_score = score
                            best_plan = {
                                "tensor_parallel": tp,
                                "pipeline_parallel": pp,
                                "data_parallel": dp,
                                "zero_stage": zero_stage,
                                "micro_batch_size": micro_batch,
                                "global_batch_size": global_batch,
                                "estimated_memory_gb": memory_req,
                                "estimated_comm_gb": comm_cost,
                                "estimated_flops": step_flops,
                            }
        
        if best_plan is None:
            # Fallback plan
            best_plan = {
                "tensor_parallel": 1,
                "pipeline_parallel": 1,
                "data_parallel": gpu_count,
                "zero_stage": 2,
                "micro_batch_size": 1,
                "global_batch_size": gpu_count,
                "estimated_memory_gb": self.estimate_model_memory(),
                "estimated_comm_gb": 0.1,
                "estimated_flops": self.estimate_flops(gpu_count, 2048),
            }
            
        return best_plan

@app.command()
def compute(
    model: Path = typer.Option(..., help="Model configuration file"),
    hardware: Path = typer.Option(..., help="Hardware profile file"),
    target_flops: Optional[float] = typer.Option(1e14, help="Target FLOPs ceiling"),
    max_memory_gb: Optional[float] = typer.Option(40, help="Max memory per GPU in GB"),
    max_comm_bw_gbps: Optional[float] = typer.Option(100, help="Max communication bandwidth in GB/s"),
    strategy: str = typer.Option("auto", help="Strategy (auto, manual)"),
    tensor_parallel: Optional[int] = typer.Option(None, help="Manual tensor parallel degree"),
    pipeline_parallel: Optional[int] = typer.Option(None, help="Manual pipeline parallel degree"), 
    zero_stage: Optional[int] = typer.Option(None, help="Manual ZeRO stage"),
    output: Optional[Path] = typer.Option(None, "--out", help="Output plan file"),
    dry_run: bool = typer.Option(False, help="Dry run - don't save plan"),
) -> None:
    """Compute parallelism and sharding plan."""
    
    # Load configurations
    console.print("[blue]Loading configurations...[/blue]")
    
    # Load model config
    if model.suffix == ".json":
        with open(model) as f:
            model_config = json.load(f)
    else:
        with open(model) as f:
            model_config = toml.load(f)
    
    # Load hardware profile  
    if hardware.suffix == ".json":
        with open(hardware) as f:
            hw_profile = json.load(f)
    else:
        with open(hardware) as f:
            hw_profile = toml.load(f)
    
    console.print(f"[green]✓[/green] Model: {model_config.get('name', 'Unknown')}")
    console.print(f"[green]✓[/green] Hardware: {hw_profile.get('gpu', {}).get('count', 0)} GPUs")
    
    # Create planner
    planner = ParallelismPlanner(model_config, hw_profile)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        if strategy == "auto":
            task = progress.add_task("Computing optimal plan...", total=1)
            plan = planner.search_optimal_plan(target_flops, max_memory_gb, max_comm_bw_gbps)
            progress.update(task, completed=1)
        else:
            # Manual strategy
            task = progress.add_task("Validating manual plan...", total=1)
            tp = tensor_parallel or 1
            pp = pipeline_parallel or 1
            zs = zero_stage or 1
            
            gpu_count = hw_profile.get("gpu", {}).get("count", 1)
            dp = gpu_count // (tp * pp)
            
            plan = {
                "tensor_parallel": tp,
                "pipeline_parallel": pp,
                "data_parallel": dp,
                "zero_stage": zs,
                "micro_batch_size": 1,
                "global_batch_size": dp * 4,
                "estimated_memory_gb": planner.compute_memory_requirement(tp, pp, zs, 1),
                "estimated_comm_gb": planner.estimate_communication_cost(tp, pp, zs, dp * 4, 1),
                "estimated_flops": planner.estimate_flops(dp * 4, 2048),
            }
            progress.update(task, completed=1)
    
    # Display plan
    console.print("\n[bold blue]Parallelism Plan[/bold blue]")
    
    plan_table = Table()
    plan_table.add_column("Parameter", style="cyan")
    plan_table.add_column("Value", style="green")
    plan_table.add_column("Description", style="dim")
    
    plan_table.add_row("Tensor Parallel", str(plan["tensor_parallel"]), "Degree of tensor parallelism")
    plan_table.add_row("Pipeline Parallel", str(plan["pipeline_parallel"]), "Degree of pipeline parallelism") 
    plan_table.add_row("Data Parallel", str(plan["data_parallel"]), "Degree of data parallelism")
    plan_table.add_row("ZeRO Stage", str(plan["zero_stage"]), "ZeRO optimizer sharding stage")
    plan_table.add_row("Micro Batch Size", str(plan["micro_batch_size"]), "Micro-batch size per GPU")
    plan_table.add_row("Global Batch Size", str(plan["global_batch_size"]), "Global batch size across all GPUs")
    
    console.print(plan_table)
    
    # Resource estimates
    console.print("\n[bold blue]Resource Estimates[/bold blue]")
    
    resource_table = Table()
    resource_table.add_column("Resource", style="cyan")
    resource_table.add_column("Estimate", style="green")
    resource_table.add_column("Limit", style="yellow")
    resource_table.add_column("Status", style="bold")
    
    # Memory
    mem_status = "✓" if plan["estimated_memory_gb"] <= max_memory_gb else "✗"
    resource_table.add_row(
        "Memory per GPU",
        f"{plan['estimated_memory_gb']:.2f} GB",
        f"{max_memory_gb:.2f} GB",
        f"[green]{mem_status}[/green]" if mem_status == "✓" else f"[red]{mem_status}[/red]"
    )
    
    # Communication  
    comm_status = "✓" if plan["estimated_comm_gb"] <= max_comm_bw_gbps else "✗"
    resource_table.add_row(
        "Communication",
        f"{plan['estimated_comm_gb']:.2f} GB/s",
        f"{max_comm_bw_gbps:.2f} GB/s", 
        f"[green]{comm_status}[/green]" if comm_status == "✓" else f"[red]{comm_status}[/red]"
    )
    
    # FLOPs
    flops_status = "✓" if plan["estimated_flops"] <= target_flops else "✗"
    resource_table.add_row(
        "FLOPs per step",
        f"{plan['estimated_flops']:.2e}",
        f"{target_flops:.2e}",
        f"[green]{flops_status}[/green]" if flops_status == "✓" else f"[red]{flops_status}[/red]"
    )
    
    console.print(resource_table)
    
    # Model info
    console.print(f"\n[dim]Model parameters: {planner.estimate_parameters():,}[/dim]")
    console.print(f"[dim]Model memory: {planner.estimate_model_memory():.2f} GB[/dim]")
    
    # Save plan
    if output and not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        
        full_plan = {
            "metadata": {
                "model_file": str(model),
                "hardware_file": str(hardware),
                "strategy": strategy,
                "constraints": {
                    "target_flops": target_flops,
                    "max_memory_gb": max_memory_gb,
                    "max_comm_bw_gbps": max_comm_bw_gbps,
                }
            },
            "parallelism": plan,
            "model": model_config,
            "hardware": hw_profile,
        }
        
        with open(output, "w") as f:
            toml.dump(full_plan, f)
        
        console.print(f"\n[green]✅ Plan saved to: {output}[/green]")
    
    elif dry_run:
        console.print(f"\n[yellow]Dry run - plan not saved[/yellow]")
    
    # Recommendations
    if plan["estimated_memory_gb"] > max_memory_gb:
        console.print(f"\n[red]⚠ Memory requirement exceeds limit![/red]")
        console.print(f"[yellow]Suggestions:[/yellow]")
        console.print(f"  • Increase tensor parallelism")
        console.print(f"  • Use higher ZeRO stage")
        console.print(f"  • Reduce micro-batch size")
    
    if plan["estimated_comm_gb"] > max_comm_bw_gbps:
        console.print(f"\n[red]⚠ Communication requirement exceeds bandwidth![/red]")
        console.print(f"[yellow]Suggestions:[/yellow]")
        console.print(f"  • Reduce tensor parallelism")
        console.print(f"  • Increase gradient accumulation")

# Make compute the default command
@app.callback(invoke_without_command=True)  
def main(ctx: typer.Context) -> None:
    """Compute parallelism plans."""
    if ctx.invoked_subcommand is None:
        # Show help if no model/hardware specified
        console.print("Use 'llmctl plan compute' with --model and --hardware options")
        raise typer.Exit(1)