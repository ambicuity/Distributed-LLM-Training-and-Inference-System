#!/usr/bin/env python3
"""
Distributed training script for LLM models.
This script is launched by the ProcessOrchestrator.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import llmctl
sys.path.insert(0, str(Path(__file__).parent.parent))

from llmctl.runtime.engine import TrainingEngine, TrainingConfig

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Distributed LLM Training")
    
    # Model and data
    parser.add_argument("--model-name-or-path", type=str, required=True,
                       help="Path to pretrained model or model identifier")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to training dataset")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for checkpoints")
    
    # Training hyperparameters
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--max-steps", type=int, default=-1,
                       help="Maximum training steps")
    parser.add_argument("--warmup-steps", type=int, default=0,
                       help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="Weight decay")
    
    # Optimization
    parser.add_argument("--optimizer", type=str, default="adamw",
                       choices=["adamw"], help="Optimizer")
    parser.add_argument("--scheduler", type=str, default="linear",
                       choices=["linear"], help="Learning rate scheduler")
    parser.add_argument("--gradient-clipping", type=float, default=1.0,
                       help="Gradient clipping norm")
    parser.add_argument("--mixed-precision", type=str, default="bf16",
                       choices=["no", "fp16", "bf16"], help="Mixed precision mode")
    
    # Distributed training
    parser.add_argument("--distributed-backend", type=str, default="nccl",
                       choices=["nccl", "gloo"], help="Distributed backend")
    parser.add_argument("--deepspeed-config", type=str, default=None,
                       help="DeepSpeed configuration file")
    
    # Checkpointing
    parser.add_argument("--save-steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=500,
                       help="Evaluate every N steps")
    parser.add_argument("--save-total-limit", type=int, default=3,
                       help="Maximum number of checkpoints to keep")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                       help="Resume training from checkpoint")
    
    # Logging
    parser.add_argument("--logging-steps", type=int, default=10,
                       help="Log every N steps")
    parser.add_argument("--log-level", type=str, default="info",
                       choices=["debug", "info", "warning", "error"],
                       help="Logging level")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--deterministic", action="store_true",
                       help="Enable deterministic training")
    
    # Configuration file
    parser.add_argument("--config", type=str, default=None,
                       help="Configuration file (JSON)")
    
    return parser.parse_args()

def load_config_from_file(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration from file if provided
    config_dict = {}
    if args.config:
        config_dict = load_config_from_file(args.config)
    
    # Override with command line arguments
    config_dict.update({
        "model_name_or_path": args.model_name_or_path,
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_epochs": args.num_epochs,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "gradient_clipping": args.gradient_clipping,
        "mixed_precision": args.mixed_precision,
        "distributed_backend": args.distributed_backend,
        "deepspeed_config": args.deepspeed_config,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "logging_steps": args.logging_steps,
        "log_level": args.log_level,
        "seed": args.seed,
        "deterministic": args.deterministic,
    })
    
    # Create training configuration
    config = TrainingConfig(**config_dict)
    
    # Create and run training engine
    engine = TrainingEngine(config)
    
    try:
        # Load model and tokenizer
        engine.load_model_and_tokenizer()
        
        # Prepare datasets
        engine.prepare_datasets()
        
        # Setup optimizer and scheduler
        engine.setup_optimizer_and_scheduler()
        
        # Load checkpoint if resuming
        if config.resume_from_checkpoint:
            engine.load_checkpoint(config.resume_from_checkpoint)
        
        # Start training
        engine.train()
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()