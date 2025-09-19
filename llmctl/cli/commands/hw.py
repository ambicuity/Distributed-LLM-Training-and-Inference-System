"""
Hardware probing and profiling command
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
import toml
import json
import platform
import subprocess
import psutil
from typing import Optional, Dict, Any
import sys

console = Console()
app = typer.Typer(help="Hardware probing and profiling")

def get_cpu_info() -> Dict[str, Any]:
    """Get CPU information."""
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        return {
            "brand": info.get("brand_raw", "Unknown"),
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "frequency_mhz": info.get("hz_advertised_friendly", "Unknown"),
            "architecture": platform.machine(),
        }
    except ImportError:
        return {
            "brand": "Unknown (install py-cpuinfo for details)",
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True), 
            "frequency_mhz": "Unknown",
            "architecture": platform.machine(),
        }

def get_memory_info() -> Dict[str, Any]:
    """Get memory information."""
    mem = psutil.virtual_memory()
    return {
        "total_gb": round(mem.total / (1024**3), 2),
        "available_gb": round(mem.available / (1024**3), 2),
        "used_gb": round(mem.used / (1024**3), 2),
        "percentage": mem.percent,
    }

def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information."""
    gpu_info = {
        "count": 0,
        "devices": [],
        "total_memory_gb": 0,
        "driver_version": "Unknown",
        "cuda_version": "Unknown",
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["count"] = torch.cuda.device_count()
            gpu_info["cuda_version"] = torch.version.cuda
            
            for i in range(gpu_info["count"]):
                props = torch.cuda.get_device_properties(i)
                device_info = {
                    "id": i,
                    "name": props.name,
                    "memory_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multiprocessors": props.multi_processor_count,
                }
                gpu_info["devices"].append(device_info)
                gpu_info["total_memory_gb"] += device_info["memory_gb"]
        
        # Try to get driver version
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_info["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
        except (ImportError, Exception):
            pass
            
    except ImportError:
        console.print("[yellow]PyTorch not available - GPU detection limited[/yellow]")
    
    return gpu_info

def detect_interconnect() -> Dict[str, Any]:
    """Detect interconnect information."""
    interconnect = {
        "intra_node": "Unknown",
        "inter_node": "Unknown", 
        "topology": "Unknown",
    }
    
    # Try to detect NVLink
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode == 0 and "NV" in result.stdout:
            interconnect["intra_node"] = "NVLink"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Try to detect InfiniBand
    try:
        result = subprocess.run(
            ["ibstat"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            interconnect["inter_node"] = "InfiniBand"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Try alternative detection
        if Path("/sys/class/infiniband").exists():
            interconnect["inter_node"] = "InfiniBand"
        else:
            interconnect["inter_node"] = "Ethernet"
    
    return interconnect

@app.command()
def probe(
    emit: Optional[Path] = typer.Option(None, help="Emit hardware profile to file"),
    format: str = typer.Option("toml", help="Output format (toml, json)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Probe hardware and generate hardware profile."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # CPU Info
        task = progress.add_task("Probing CPU...", total=1)
        cpu_info = get_cpu_info()
        progress.update(task, completed=1)
        
        # Memory Info
        task = progress.add_task("Probing Memory...", total=1)
        memory_info = get_memory_info()
        progress.update(task, completed=1)
        
        # GPU Info
        task = progress.add_task("Probing GPUs...", total=1)
        gpu_info = get_gpu_info()
        progress.update(task, completed=1)
        
        # Interconnect Info
        task = progress.add_task("Detecting Interconnect...", total=1)
        interconnect_info = detect_interconnect()
        progress.update(task, completed=1)
    
    # Create hardware profile
    profile = {
        "system": {
            "hostname": platform.node(),
            "os": f"{platform.system()} {platform.release()}",
            "python_version": platform.python_version(),
            "detected_at": str(Path.cwd()),
        },
        "cpu": cpu_info,
        "memory": memory_info,
        "gpu": gpu_info,
        "interconnect": interconnect_info,
        "limits": {
            "estimated_flops": gpu_info["count"] * 312e12 if gpu_info["count"] > 0 else 1e12,  # Rough estimate for A100
            "memory_bw_gbps": gpu_info["count"] * 1935 if gpu_info["count"] > 0 else 100,  # Rough estimate for A100 HBM
            "intra_node_bw_gbps": 600 if interconnect_info["intra_node"] == "NVLink" else 32,  # NVLink vs PCIe
            "inter_node_bw_gbps": 200 if interconnect_info["inter_node"] == "InfiniBand" else 10,  # IB vs Ethernet
        }
    }
    
    # Display results
    console.print("\n[bold blue]Hardware Profile[/bold blue]")
    
    # System table
    system_table = Table(title="System Information")
    system_table.add_column("Property", style="cyan")
    system_table.add_column("Value", style="green")
    
    system_table.add_row("Hostname", profile["system"]["hostname"])
    system_table.add_row("OS", profile["system"]["os"])
    system_table.add_row("Python", profile["system"]["python_version"])
    console.print(system_table)
    
    # CPU table
    cpu_table = Table(title="CPU Information")
    cpu_table.add_column("Property", style="cyan")
    cpu_table.add_column("Value", style="green")
    
    cpu_table.add_row("Brand", str(cpu_info["brand"]))
    cpu_table.add_row("Cores", str(cpu_info["cores"]))
    cpu_table.add_row("Threads", str(cpu_info["threads"]))
    cpu_table.add_row("Frequency", str(cpu_info["frequency_mhz"]))
    cpu_table.add_row("Architecture", str(cpu_info["architecture"]))
    console.print(cpu_table)
    
    # Memory table
    mem_table = Table(title="Memory Information") 
    mem_table.add_column("Property", style="cyan")
    mem_table.add_column("Value", style="green")
    
    mem_table.add_row("Total", f"{memory_info['total_gb']} GB")
    mem_table.add_row("Available", f"{memory_info['available_gb']} GB")
    mem_table.add_row("Used", f"{memory_info['used_gb']} GB ({memory_info['percentage']:.1f}%)")
    console.print(mem_table)
    
    # GPU table
    if gpu_info["count"] > 0:
        gpu_table = Table(title="GPU Information")
        gpu_table.add_column("ID", style="cyan")
        gpu_table.add_column("Name", style="green")
        gpu_table.add_column("Memory", style="yellow")
        gpu_table.add_column("Compute", style="blue")
        gpu_table.add_column("SMs", style="magenta")
        
        for device in gpu_info["devices"]:
            gpu_table.add_row(
                str(device["id"]),
                device["name"],
                f"{device['memory_gb']:.1f} GB",
                device["compute_capability"],
                str(device["multiprocessors"])
            )
        
        console.print(gpu_table)
        console.print(f"[bold]Total GPU Memory: {gpu_info['total_memory_gb']:.1f} GB[/bold]")
        console.print(f"[bold]CUDA Version: {gpu_info['cuda_version']}[/bold]")
        console.print(f"[bold]Driver Version: {gpu_info['driver_version']}[/bold]")
    else:
        console.print("[red]No GPUs detected[/red]")
    
    # Interconnect table
    interconnect_table = Table(title="Interconnect Information")
    interconnect_table.add_column("Type", style="cyan")
    interconnect_table.add_column("Technology", style="green")
    
    interconnect_table.add_row("Intra-node", interconnect_info["intra_node"])
    interconnect_table.add_row("Inter-node", interconnect_info["inter_node"])
    console.print(interconnect_table)
    
    # Performance estimates table
    perf_table = Table(title="Performance Estimates")
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Estimate", style="green")
    
    perf_table.add_row("Peak FLOPs", f"{profile['limits']['estimated_flops']:.1e}")
    perf_table.add_row("Memory BW", f"{profile['limits']['memory_bw_gbps']} GB/s")
    perf_table.add_row("Intra-node BW", f"{profile['limits']['intra_node_bw_gbps']} GB/s")
    perf_table.add_row("Inter-node BW", f"{profile['limits']['inter_node_bw_gbps']} GB/s")
    console.print(perf_table)
    
    # Save profile if requested
    if emit:
        emit.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            with open(emit, "w") as f:
                json.dump(profile, f, indent=2)
        else:  # toml
            with open(emit, "w") as f:
                toml.dump(profile, f)
        
        console.print(f"\n[green]✅ Hardware profile saved to: {emit}[/green]")
    
    if verbose:
        console.print("\n[dim]Raw profile data:[/dim]")
        console.print(profile)

@app.command()
def benchmark(
    component: str = typer.Option("memory", help="Component to benchmark (memory, compute, network)"),
    duration: int = typer.Option(10, help="Benchmark duration in seconds"),
) -> None:
    """Run hardware micro-benchmarks."""
    console.print(f"[yellow]Running {component} benchmark for {duration}s...[/yellow]")
    
    if component == "memory":
        # Simple memory bandwidth test
        try:
            import numpy as np
            import time
            
            size = 100_000_000  # 100M floats
            data = np.random.rand(size).astype(np.float32)
            
            start = time.time()
            result = np.sum(data)
            end = time.time()
            
            bandwidth = (size * 4) / (end - start) / 1e9  # GB/s
            console.print(f"[green]Memory bandwidth: {bandwidth:.2f} GB/s[/green]")
            
        except ImportError:
            console.print("[red]NumPy required for memory benchmark[/red]")
    
    elif component == "compute":
        # Simple compute test
        try:
            import torch
            import time
            
            if torch.cuda.is_available():
                device = torch.device("cuda")
                size = 4096
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)
                
                # Warmup
                for _ in range(10):
                    c = torch.matmul(a, b)
                
                torch.cuda.synchronize()
                start = time.time()
                
                for _ in range(100):
                    c = torch.matmul(a, b)
                
                torch.cuda.synchronize()
                end = time.time()
                
                flops = 100 * 2 * size**3 / (end - start)  # FLOPs
                console.print(f"[green]Compute performance: {flops:.2e} FLOPs/s[/green]")
            else:
                console.print("[red]CUDA not available for compute benchmark[/red]")
                
        except ImportError:
            console.print("[red]PyTorch required for compute benchmark[/red]")
    
    else:
        console.print(f"[red]Unknown benchmark component: {component}[/red]")

# Make probe the default command
@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Hardware probing and profiling."""
    if ctx.invoked_subcommand is None:
        probe()