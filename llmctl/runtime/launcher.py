"""
Distributed training launchers for different environments.
"""

import os
import subprocess
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile

import torch
import torch.distributed as dist
from rich.console import Console

console = Console()

@dataclass
class LaunchConfig:
    """Configuration for launching distributed training."""
    nodes: int = 1
    gpus_per_node: int = 1
    cpus_per_task: int = 1
    mixed_precision: str = "bf16"
    launcher: str = "local"
    backend: str = "nccl"
    
    # Training specific
    config_path: Optional[Path] = None
    data_path: Optional[Path] = None
    plan_path: Optional[Path] = None
    checkpoint_path: Optional[str] = None
    
    # Runtime options
    seed: int = 42
    deterministic: bool = False
    log_level: str = "info"
    
    # Advanced options
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0
    
    def total_gpus(self) -> int:
        return self.nodes * self.gpus_per_node

class BaseLauncher(ABC):
    """Base class for distributed training launchers."""
    
    def __init__(self, config: LaunchConfig):
        self.config = config
        
    @abstractmethod
    def launch(self, script_path: str, script_args: List[str]) -> subprocess.Popen:
        """Launch the distributed training job."""
        pass
    
    @abstractmethod
    def get_environment(self) -> Dict[str, str]:
        """Get environment variables for the job."""
        pass

class LocalLauncher(BaseLauncher):
    """Launcher for local single-node training."""
    
    def get_environment(self) -> Dict[str, str]:
        """Get environment variables for local training."""
        env = os.environ.copy()
        
        # Basic distributed training setup
        env.update({
            "WORLD_SIZE": str(self.config.total_gpus()),
            "NPROC_PER_NODE": str(self.config.gpus_per_node),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
            "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(self.config.gpus_per_node)),
        })
        
        # PyTorch specific
        if self.config.backend == "nccl":
            env["NCCL_DEBUG"] = "INFO"
        
        # Deterministic training
        if self.config.deterministic:
            env.update({
                "PYTHONHASHSEED": str(self.config.seed),
                "CUDA_LAUNCH_BLOCKING": "1",
            })
        
        return env
    
    def launch(self, script_path: str, script_args: List[str]) -> subprocess.Popen:
        """Launch local distributed training using torchrun."""
        
        cmd = [
            "python", "-m", "torch.distributed.run",
            f"--nproc_per_node={self.config.gpus_per_node}",
            f"--nnodes={self.config.nodes}",
            "--node_rank=0",
            "--master_addr=localhost",
            "--master_port=29500",
            script_path
        ] + script_args
        
        env = self.get_environment()
        
        console.print(f"[blue]Launching local training with command:[/blue]")
        console.print(f"[yellow]{' '.join(cmd)}[/yellow]")
        
        return subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

class SlurmLauncher(BaseLauncher):
    """Launcher for SLURM-based distributed training."""
    
    def get_environment(self) -> Dict[str, str]:
        """Get environment variables for SLURM training."""
        env = os.environ.copy()
        
        # SLURM will set most distributed variables
        env.update({
            "NCCL_DEBUG": "INFO",
            "NCCL_IB_DISABLE": "1",  # Disable InfiniBand if not available
        })
        
        if self.config.deterministic:
            env.update({
                "PYTHONHASHSEED": str(self.config.seed),
                "CUDA_LAUNCH_BLOCKING": "1",
            })
        
        return env
    
    def create_slurm_script(self, script_path: str, script_args: List[str]) -> str:
        """Create a SLURM batch script."""
        script_content = f"""#!/bin/bash
#SBATCH --job-name=llmctl-train
#SBATCH --nodes={self.config.nodes}
#SBATCH --ntasks-per-node={self.config.gpus_per_node}
#SBATCH --cpus-per-task={self.config.cpus_per_task}
#SBATCH --gres=gpu:{self.config.gpus_per_node}
#SBATCH --time=24:00:00
#SBATCH --output=llmctl-train-%j.out
#SBATCH --error=llmctl-train-%j.err

# Setup environment
source ~/.bashrc
module load python/3.8  # Adjust based on your cluster

# Set distributed training variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Launch training
srun python {script_path} {' '.join(script_args)}
"""
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(script_content)
            return f.name
    
    def launch(self, script_path: str, script_args: List[str]) -> subprocess.Popen:
        """Launch SLURM distributed training."""
        
        # Create SLURM script
        slurm_script = self.create_slurm_script(script_path, script_args)
        
        cmd = ["sbatch", slurm_script]
        
        console.print(f"[blue]Submitting SLURM job with script:[/blue]")
        console.print(f"[yellow]{slurm_script}[/yellow]")
        
        return subprocess.Popen(
            cmd,
            env=self.get_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

class MPILauncher(BaseLauncher):
    """Launcher for MPI-based distributed training."""
    
    def get_environment(self) -> Dict[str, str]:
        """Get environment variables for MPI training."""
        env = os.environ.copy()
        
        env.update({
            "NCCL_DEBUG": "INFO",
            "OMPI_COMM_WORLD_RANK": "0",
        })
        
        if self.config.deterministic:
            env.update({
                "PYTHONHASHSEED": str(self.config.seed),
                "CUDA_LAUNCH_BLOCKING": "1",
            })
        
        return env
    
    def launch(self, script_path: str, script_args: List[str]) -> subprocess.Popen:
        """Launch MPI distributed training."""
        
        cmd = [
            "mpirun",
            "-np", str(self.config.total_gpus()),
            "--bind-to", "none",
            "--map-by", "slot",
            "-x", "NCCL_DEBUG=INFO",
            "-x", "PYTHONPATH",
            "python", script_path
        ] + script_args
        
        console.print(f"[blue]Launching MPI training with command:[/blue]")
        console.print(f"[yellow]{' '.join(cmd)}[/yellow]")
        
        return subprocess.Popen(
            cmd,
            env=self.get_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

def create_launcher(config: LaunchConfig) -> BaseLauncher:
    """Factory function to create the appropriate launcher."""
    if config.launcher == "local":
        return LocalLauncher(config)
    elif config.launcher == "slurm":
        return SlurmLauncher(config)
    elif config.launcher == "mpi":
        return MPILauncher(config)
    else:
        raise ValueError(f"Unknown launcher type: {config.launcher}")

class ProcessOrchestrator:
    """Orchestrates distributed training processes."""
    
    def __init__(self, config: LaunchConfig):
        self.config = config
        self.launcher = create_launcher(config)
        self.process: Optional[subprocess.Popen] = None
        
    def start_training(self, training_script: str, script_args: List[str]) -> bool:
        """Start the distributed training job."""
        try:
            console.print("[blue]Starting distributed training...[/blue]")
            
            self.process = self.launcher.launch(training_script, script_args)
            
            if self.config.launcher == "local":
                # For local launcher, stream output in real-time
                return self._monitor_local_process()
            else:
                # For cluster launchers, just wait for submission
                stdout, stderr = self.process.communicate()
                if self.process.returncode == 0:
                    console.print("[green]✓ Job submitted successfully[/green]")
                    if stdout:
                        console.print(f"Output: {stdout.strip()}")
                    return True
                else:
                    console.print(f"[red]Job submission failed: {stderr}[/red]")
                    return False
                    
        except Exception as e:
            console.print(f"[red]Failed to start training: {e}[/red]")
            return False
    
    def _monitor_local_process(self) -> bool:
        """Monitor local training process and stream output."""
        if not self.process:
            return False
        
        console.print("[blue]Monitoring training progress...[/blue]")
        
        try:
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    console.print(line.rstrip())
                
                # Check if process is still running
                if self.process.poll() is not None:
                    break
            
            # Wait for process to complete and get return code
            return_code = self.process.wait()
            
            if return_code == 0:
                console.print("[green]✓ Training completed successfully[/green]")
                return True
            else:
                console.print(f"[red]Training failed with exit code: {return_code}[/red]")
                return False
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Training interrupted by user[/yellow]")
            self.stop_training()
            return False
    
    def stop_training(self):
        """Stop the training process."""
        if self.process and self.process.poll() is None:
            console.print("[yellow]Stopping training process...[/yellow]")
            self.process.terminate()
            time.sleep(5)
            if self.process.poll() is None:
                self.process.kill()
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the training job."""
        if not self.process:
            return {"status": "not_started"}
        
        poll = self.process.poll()
        if poll is None:
            return {"status": "running", "pid": self.process.pid}
        else:
            return {"status": "completed", "exit_code": poll}