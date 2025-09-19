"""
Training engine implementation with PyTorch integration.
"""

import os
import json
import time
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling
)
from accelerate import Accelerator
import deepspeed
from rich.console import Console

console = Console()

@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Model and data
    model_name_or_path: str
    dataset_path: str
    output_dir: str
    
    # Training hyperparameters
    learning_rate: float = 5e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_epochs: int = 3
    max_steps: int = -1
    warmup_steps: int = 0
    weight_decay: float = 0.01
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "linear"
    gradient_clipping: float = 1.0
    mixed_precision: str = "bf16"
    
    # Distributed training
    distributed_backend: str = "nccl"
    deepspeed_config: Optional[str] = None
    
    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None
    
    # Logging
    logging_steps: int = 10
    log_level: str = "info"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False

class TrainingEngine:
    """Core training engine with PyTorch and distributed support."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.accelerator = None
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.eval_dataloader = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        self._setup_logging()
        self._setup_distributed()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        import logging
        
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            level=getattr(logging, self.config.log_level.upper())
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_distributed(self):
        """Setup distributed training."""
        if torch.cuda.is_available():
            # Initialize accelerator for mixed precision and distributed training
            self.accelerator = Accelerator(
                mixed_precision=self.config.mixed_precision,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps
            )
            
            if self.accelerator.is_main_process:
                console.print(f"[blue]Initializing training engine...[/blue]")
                console.print(f"[yellow]Mixed precision: {self.config.mixed_precision}[/yellow]")
                console.print(f"[yellow]Distributed: {self.accelerator.num_processes} processes[/yellow]")
        else:
            console.print("[yellow]CUDA not available, using CPU training[/yellow]")
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        if self.accelerator and self.accelerator.is_main_process:
            console.print(f"[blue]Loading model: {self.config.model_name_or_path}[/blue]")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch.float16 if self.config.mixed_precision in ["fp16", "bf16"] else torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Resize token embeddings if needed
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        if self.accelerator and self.accelerator.is_main_process:
            console.print(f"[green]✓ Model loaded with {self.model.num_parameters():,} parameters[/green]")
    
    def prepare_datasets(self):
        """Prepare training and evaluation datasets."""
        if self.accelerator and self.accelerator.is_main_process:
            console.print(f"[blue]Loading dataset: {self.config.dataset_path}[/blue]")
        
        # For now, create a simple dummy dataset
        # In practice, this would load from the actual dataset
        from torch.utils.data import TensorDataset
        
        # Create dummy text data
        dummy_texts = [
            "This is a sample training text.",
            "Another example for language modeling.",
            "Training large language models requires careful preparation.",
        ] * 100  # Repeat to have enough data
        
        # Tokenize texts
        tokenized = self.tokenizer(
            dummy_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Create dataset
        dataset = TensorDataset(
            tokenized["input_ids"],
            tokenized["attention_mask"]
        )
        
        # Split into train/eval
        train_size = int(0.9 * len(dataset))
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(
            dataset, [train_size, eval_size]
        )
        
        # Create data loaders
        if self.accelerator:
            train_sampler = DistributedSampler(train_dataset) if self.accelerator.num_processes > 1 else None
            eval_sampler = DistributedSampler(eval_dataset) if self.accelerator.num_processes > 1 else None
        else:
            train_sampler = eval_sampler = None
        
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=self._collate_fn
        )
        
        self.eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            sampler=eval_sampler,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        if self.accelerator and self.accelerator.is_main_process:
            console.print(f"[green]✓ Dataset prepared: {len(train_dataset)} train, {len(eval_dataset)} eval samples[/green]")
    
    def _collate_fn(self, batch):
        """Collate function for data loader."""
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()  # For language modeling
        }
    
    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        
        # Filter parameters that require gradients
        param_groups = [
            {
                "params": [p for p in self.model.parameters() if p.requires_grad],
                "weight_decay": self.config.weight_decay,
            }
        ]
        
        # Create optimizer
        if self.config.optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
        # Calculate total training steps
        if self.config.max_steps > 0:
            max_steps = self.config.max_steps
        else:
            max_steps = len(self.train_dataloader) * self.config.num_epochs
        
        # Create scheduler
        if self.config.scheduler.lower() == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=max_steps
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")
        
        if self.accelerator and self.accelerator.is_main_process:
            console.print(f"[green]✓ Optimizer and scheduler setup complete[/green]")
    
    def train(self):
        """Main training loop."""
        if self.accelerator and self.accelerator.is_main_process:
            console.print("[blue]Starting training...[/blue]")
        
        # Prepare everything with accelerator
        if self.accelerator:
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.eval_dataloader
            )
        
        # Training loop
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            if self.accelerator and self.accelerator.is_main_process:
                console.print(f"[blue]Epoch {epoch + 1}/{self.config.num_epochs}[/blue]")
            
            epoch_loss = 0.0
            num_batches = 0
            
            for step, batch in enumerate(self.train_dataloader):
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                if self.accelerator:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()
                
                # Update weights
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.gradient_clipping > 0:
                        if self.accelerator:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                num_batches += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    if self.accelerator and self.accelerator.is_main_process:
                        console.print(f"Step {self.global_step}: loss = {loss.item():.4f}")
                
                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    eval_loss = self.evaluate()
                    self.model.train()  # Switch back to training mode
                
                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                # Check if max steps reached
                if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                    break
            
            avg_epoch_loss = epoch_loss / num_batches
            if self.accelerator and self.accelerator.is_main_process:
                console.print(f"[green]Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}[/green]")
            
            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                break
        
        # Final save
        self.save_checkpoint(final=True)
        
        if self.accelerator and self.accelerator.is_main_process:
            console.print("[green]✓ Training completed![/green]")
    
    def evaluate(self) -> float:
        """Evaluate the model."""
        if self.accelerator and self.accelerator.is_main_process:
            console.print("[blue]Evaluating...[/blue]")
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if self.accelerator and self.accelerator.is_main_process:
            console.print(f"[yellow]Evaluation loss: {avg_loss:.4f}[/yellow]")
        
        return avg_loss
    
    def save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        if not (self.accelerator and self.accelerator.is_main_process):
            return
        
        output_dir = Path(self.config.output_dir)
        if final:
            checkpoint_dir = output_dir / "final"
        else:
            checkpoint_dir = output_dir / f"checkpoint-{self.global_step}"
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        if self.accelerator:
            self.accelerator.save_model(self.model, checkpoint_dir)
        else:
            self.model.save_pretrained(checkpoint_dir)
        
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": self.config.__dict__
        }
        
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)
        
        console.print(f"[green]✓ Checkpoint saved to {checkpoint_dir}[/green]")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint_dir = Path(checkpoint_path)
        
        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
        
        # Load training state
        state_file = checkpoint_dir / "training_state.json"
        if state_file.exists():
            with open(state_file) as f:
                training_state = json.load(f)
            self.global_step = training_state["global_step"]
            self.epoch = training_state["epoch"]
        
        console.print(f"[green]✓ Checkpoint loaded from {checkpoint_dir}[/green]")

def create_training_config(**kwargs) -> TrainingConfig:
    """Factory function to create training configuration."""
    return TrainingConfig(**kwargs)