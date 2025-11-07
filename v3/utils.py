"""
Utility functions for VideoMAE training and evaluation
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enable cuDNN benchmark for faster training
    # Set to False only if you need exact reproducibility
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    best_val_loss: Optional[float] = None
) -> None:
    """Save model checkpoint with full training state"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    # Save scheduler state
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Save scaler state (for AMP)
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    # Save best validation loss
    if best_val_loss is not None:
        checkpoint['best_val_loss'] = best_val_loss

    # Save random states for reproducibility
    checkpoint['random_states'] = {
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'numpy': np.random.get_state(),
        'random': random.getstate()
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    restore_random_states: bool = True
) -> Dict:
    """Load model checkpoint with full training state"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location='cpu')

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Scheduler state restored")

    # Load scaler state (for AMP)
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"GradScaler state restored")

    # Restore random states for reproducibility
    if restore_random_states and 'random_states' in checkpoint:
        random_states = checkpoint['random_states']
        torch.set_rng_state(random_states['torch'])
        if random_states['torch_cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(random_states['torch_cuda'])
        np.random.set_state(random_states['numpy'])
        random.setstate(random_states['random'])
        print(f"Random states restored")

    print(f"Checkpoint loaded from {filepath} (epoch {checkpoint['epoch']})")

    return checkpoint


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format time in seconds to readable string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def print_model_info(model: nn.Module) -> None:
    """Print model information"""
    num_params = count_parameters(model)
    print(f"Model Parameters: {num_params:,}")

    # Print model architecture summary
    print("\nModel Architecture:")
    for name, module in model.named_children():
        num_params_module = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {num_params_module:,} parameters")


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")

    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average: {meter.avg}")

    # Test time formatting
    print(f"Time format test: {format_time(3665)}")

    print("Utilities test completed!")