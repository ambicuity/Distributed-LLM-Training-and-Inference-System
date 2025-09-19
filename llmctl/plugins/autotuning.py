"""
Auto-tuning for kernels and communication optimization.
"""

import time
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
import itertools

import torch
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

@dataclass
class TuningConfig:
    """Configuration for auto-tuning."""
    max_iterations: int = 50
    min_runtime: float = 0.1  # Minimum runtime for valid measurements
    warmup_iterations: int = 5
    measurement_iterations: int = 10
    tolerance: float = 0.05  # Relative improvement threshold
    timeout: float = 300.0  # Maximum time for tuning session

@dataclass
class TuningResult:
    """Result of auto-tuning operation."""
    best_config: Dict[str, Any]
    best_performance: float
    improvement: float
    total_time: float
    iterations: int
    all_results: List[Dict[str, Any]]

class Tunable(ABC):
    """Base class for tunable components."""
    
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """Get the parameter space for tuning."""
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]):
        """Set parameters for the component."""
        pass
    
    @abstractmethod
    def benchmark(self, input_data: Any) -> float:
        """Run benchmark and return performance metric (lower is better)."""
        pass
    
    @abstractmethod
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate if parameter combination is valid."""
        pass

class MatMulTuner(Tunable):
    """Auto-tuner for matrix multiplication kernels."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.current_params = {}
        
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """Get parameter space for matrix multiplication tuning."""
        return {
            "block_size": [16, 32, 64, 128],
            "num_threads": [32, 64, 128, 256] if self.device == "cuda" else [1, 2, 4, 8],
            "use_tensor_cores": [True, False] if self.device == "cuda" else [False],
            "memory_layout": ["row_major", "col_major"],
        }
    
    def set_parameters(self, params: Dict[str, Any]):
        """Set matrix multiplication parameters."""
        self.current_params = params.copy()
        
        # In a real implementation, this would configure the actual kernel
        if params.get("use_tensor_cores") and self.device == "cuda":
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        
    def benchmark(self, input_data: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Benchmark matrix multiplication with current parameters."""
        a, b = input_data
        
        # Ensure tensors are on the correct device
        a = a.to(self.device)
        b = b.to(self.device)
        
        # Warmup
        for _ in range(5):
            _ = torch.matmul(a, b)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            result = torch.matmul(a, b)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        return (end_time - start_time) / 10  # Average time per operation
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate matrix multiplication parameters."""
        # Check if tensor cores are requested but not available
        if params.get("use_tensor_cores") and self.device != "cuda":
            return False
        
        # Check reasonable block sizes
        if params.get("block_size", 0) <= 0:
            return False
        
        return True

class AttentionTuner(Tunable):
    """Auto-tuner for attention kernels."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.current_params = {}
        
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """Get parameter space for attention tuning."""
        return {
            "block_size_q": [16, 32, 64, 128],
            "block_size_k": [16, 32, 64, 128], 
            "use_flash_attention": [True, False],
            "causal_mask": [True, False],
            "dropout_p": [0.0, 0.1, 0.2] if self.device == "cuda" else [0.0],
        }
    
    def set_parameters(self, params: Dict[str, Any]):
        """Set attention parameters."""
        self.current_params = params.copy()
    
    def benchmark(self, input_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        """Benchmark attention with current parameters."""
        q, k, v = input_data
        
        # Move to device
        q = q.to(self.device)
        k = k.to(self.device) 
        v = v.to(self.device)
        
        # Simple attention implementation for benchmarking
        scale = 1.0 / math.sqrt(q.size(-1))
        
        # Warmup
        for _ in range(5):
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if self.current_params.get("causal_mask"):
                mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
                scores.masked_fill_(mask, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            _ = torch.matmul(attn, v)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if self.current_params.get("causal_mask"):
                mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
                scores.masked_fill_(mask, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            result = torch.matmul(attn, v)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        return (end_time - start_time) / 10
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate attention parameters."""
        # Flash attention only available on CUDA
        if params.get("use_flash_attention") and self.device != "cuda":
            return False
        
        # Check block sizes
        if (params.get("block_size_q", 0) <= 0 or 
            params.get("block_size_k", 0) <= 0):
            return False
        
        return True

class CommunicationTuner(Tunable):
    """Auto-tuner for communication operations."""
    
    def __init__(self):
        self.current_params = {}
        
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """Get parameter space for communication tuning."""
        return {
            "backend": ["nccl", "gloo"] if torch.cuda.is_available() else ["gloo"],
            "bucket_size_mb": [25, 50, 100, 200],
            "overlap_computation": [True, False],
            "compression": ["none", "fp16"] if torch.cuda.is_available() else ["none"],
        }
    
    def set_parameters(self, params: Dict[str, Any]):
        """Set communication parameters."""
        self.current_params = params.copy()
    
    def benchmark(self, input_data: torch.Tensor) -> float:
        """Benchmark communication with current parameters."""
        # For single-node testing, simulate communication delay
        tensor = input_data
        
        # Simulate different backends
        backend = self.current_params.get("backend", "nccl")
        bucket_size = self.current_params.get("bucket_size_mb", 25)
        
        # Simulate processing time based on parameters
        base_time = 0.001  # 1ms base
        
        if backend == "gloo":
            base_time *= 1.5  # Gloo is typically slower
        
        # Larger bucket sizes are generally better for large tensors
        bucket_factor = max(0.5, 100 / bucket_size)
        
        simulated_time = base_time * bucket_factor
        
        # Add some noise
        noise = np.random.normal(0, simulated_time * 0.1)
        
        return max(0.0001, simulated_time + noise)
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate communication parameters."""
        # NCCL only available with CUDA
        if params.get("backend") == "nccl" and not torch.cuda.is_available():
            return False
        
        # Check bucket size
        if params.get("bucket_size_mb", 0) <= 0:
            return False
            
        return True

class AutoTuner:
    """Main auto-tuning orchestrator."""
    
    def __init__(self, config: TuningConfig = None):
        self.config = config or TuningConfig()
        self.results_cache: Dict[str, TuningResult] = {}
        
    def grid_search(self, 
                   tunable: Tunable, 
                   input_data: Any,
                   cache_key: Optional[str] = None) -> TuningResult:
        """Perform grid search over parameter space."""
        
        if cache_key and cache_key in self.results_cache:
            console.print(f"[yellow]Using cached tuning result for {cache_key}[/yellow]")
            return self.results_cache[cache_key]
        
        param_space = tunable.get_parameter_space()
        
        # Generate all parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        param_combinations = list(itertools.product(*param_values))
        
        console.print(f"[blue]Starting auto-tuning with {len(param_combinations)} configurations...[/blue]")
        
        best_config = None
        best_performance = float('inf')
        all_results = []
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task(f"Tuning {len(param_combinations)} configurations...", total=len(param_combinations))
            
            for i, param_combo in enumerate(param_combinations):
                if time.time() - start_time > self.config.timeout:
                    console.print("[yellow]Tuning timeout reached[/yellow]")
                    break
                
                # Create parameter dict
                params = dict(zip(param_names, param_combo))
                
                # Validate parameters
                if not tunable.validate_parameters(params):
                    progress.update(task, advance=1)
                    continue
                
                try:
                    # Set parameters
                    tunable.set_parameters(params)
                    
                    # Benchmark
                    performance = tunable.benchmark(input_data)
                    
                    # Record result
                    result = {
                        "params": params,
                        "performance": performance,
                        "iteration": i
                    }
                    all_results.append(result)
                    
                    # Update best
                    if performance < best_performance:
                        best_performance = performance
                        best_config = params.copy()
                        
                except Exception as e:
                    console.print(f"[red]Error in configuration {params}: {e}[/red]")
                
                progress.update(task, advance=1)
                
                # Early stopping if we've found a good enough solution
                if len(all_results) >= 10:
                    recent_improvements = [
                        r["performance"] for r in all_results[-10:]
                    ]
                    if max(recent_improvements) - min(recent_improvements) < self.config.tolerance * min(recent_improvements):
                        console.print("[green]Early stopping - convergence detected[/green]")
                        break
        
        total_time = time.time() - start_time
        
        # Calculate improvement over baseline (first valid configuration)
        baseline_performance = all_results[0]["performance"] if all_results else best_performance
        improvement = (baseline_performance - best_performance) / baseline_performance * 100
        
        result = TuningResult(
            best_config=best_config or {},
            best_performance=best_performance,
            improvement=improvement,
            total_time=total_time,
            iterations=len(all_results),
            all_results=all_results
        )
        
        # Cache result
        if cache_key:
            self.results_cache[cache_key] = result
        
        console.print(f"[green]✓ Tuning completed in {total_time:.2f}s[/green]")
        console.print(f"[green]Best performance: {best_performance:.6f} ({improvement:.1f}% improvement)[/green]")
        console.print(f"[green]Best config: {best_config}[/green]")
        
        return result
    
    def tune_matmul(self, 
                   matrix_size: Tuple[int, int, int] = (1024, 1024, 1024),
                   device: str = "auto") -> TuningResult:
        """Auto-tune matrix multiplication kernels."""
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create test matrices
        m, k, n = matrix_size
        a = torch.randn(m, k, device=device)
        b = torch.randn(k, n, device=device)
        
        tuner = MatMulTuner(device)
        return self.grid_search(tuner, (a, b), f"matmul_{matrix_size}_{device}")
    
    def tune_attention(self,
                      seq_len: int = 512,
                      head_dim: int = 64, 
                      batch_size: int = 8,
                      num_heads: int = 8,
                      device: str = "auto") -> TuningResult:
        """Auto-tune attention kernels."""
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create test tensors
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        
        tuner = AttentionTuner(device)
        return self.grid_search(tuner, (q, k, v), f"attention_{seq_len}_{head_dim}_{device}")
    
    def tune_communication(self,
                          tensor_size: Tuple[int, ...] = (1024, 1024),
                          dtype: torch.dtype = torch.float32) -> TuningResult:
        """Auto-tune communication operations."""
        
        # Create test tensor
        tensor = torch.randn(*tensor_size, dtype=dtype)
        
        tuner = CommunicationTuner()
        return self.grid_search(tuner, tensor, f"comm_{tensor_size}_{dtype}")
    
    def save_results(self, filepath: str):
        """Save tuning results to file."""
        save_data = {}
        for key, result in self.results_cache.items():
            save_data[key] = {
                "best_config": result.best_config,
                "best_performance": result.best_performance,
                "improvement": result.improvement,
                "total_time": result.total_time,
                "iterations": result.iterations,
            }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        console.print(f"[green]Tuning results saved to {filepath}[/green]")
    
    def load_results(self, filepath: str):
        """Load tuning results from file."""
        try:
            with open(filepath, 'r') as f:
                save_data = json.load(f)
            
            for key, data in save_data.items():
                self.results_cache[key] = TuningResult(
                    best_config=data["best_config"],
                    best_performance=data["best_performance"],
                    improvement=data["improvement"],
                    total_time=data["total_time"],
                    iterations=data["iterations"],
                    all_results=[]  # Not saved in file
                )
            
            console.print(f"[green]Tuning results loaded from {filepath}[/green]")
            
        except FileNotFoundError:
            console.print(f"[yellow]No cached results found at {filepath}[/yellow]")
        except Exception as e:
            console.print(f"[red]Error loading results: {e}[/red]")

def create_auto_tuner(config: TuningConfig = None) -> AutoTuner:
    """Factory function to create an auto-tuner."""
    return AutoTuner(config)