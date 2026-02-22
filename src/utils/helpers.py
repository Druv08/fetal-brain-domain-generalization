"""
Utility functions for Fetal Brain MRI Segmentation.

This module provides common utility functions:
- Random seed setting for reproducibility
- Device detection (CPU/GPU)
- Checkpoint saving and loading

Author: Research Team
Date: 2026
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Sets seed for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    
    Args:
        seed: Random seed value (default: 42)
        
    Example:
        >>> set_seed(42)
        >>> # Now all random operations will be reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Device string ('cuda', 'cpu', 'cuda:0', etc.)
                If None, automatically selects CUDA if available
        
    Returns:
        torch.device object
        
    Example:
        >>> device = get_device()
        >>> print(device)
        cuda  # or cpu if CUDA not available
        
        >>> device = get_device("cuda:1")
        >>> model.to(device)
    """
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: Union[str, Path],
    scheduler: Optional[Any] = None,
    metrics: Optional[Dict] = None,
    config: Optional[Dict] = None
) -> None:
    """
    Save a training checkpoint.
    
    Saves model state, optimizer state, and training metadata
    for resuming training later.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        loss: Current loss value
        filepath: Path to save the checkpoint
        scheduler: Optional learning rate scheduler
        metrics: Optional dictionary of evaluation metrics
        config: Optional configuration dictionary
        
    Example:
        >>> save_checkpoint(
        ...     model, optimizer, epoch=10, loss=0.5,
        ...     filepath="outputs/checkpoints/best_model.pth",
        ...     metrics={'dice': 0.85}
        ... )
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    if config is not None:
        checkpoint['config'] = config
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Load a training checkpoint.
    
    Restores model state and optionally optimizer/scheduler states.
    
    Args:
        filepath: Path to the checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to restore
        scheduler: Optional scheduler to restore
        device: Device to load the checkpoint to
        
    Returns:
        Dictionary containing checkpoint metadata (epoch, loss, metrics, etc.)
        
    Example:
        >>> checkpoint_info = load_checkpoint(
        ...     "outputs/checkpoints/best_model.pth",
        ...     model, optimizer
        ... )
        >>> start_epoch = checkpoint_info['epoch']
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded: {filepath}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A')}")
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('config', {})
    }


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Number of parameters
        
    Example:
        >>> model = UNet3D(1, 8)
        >>> params = count_parameters(model)
        >>> print(f"Trainable parameters: {params:,}")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Useful for tracking metrics during training.
    
    Example:
        >>> loss_meter = AverageMeter()
        >>> for batch_loss in batch_losses:
        ...     loss_meter.update(batch_loss)
        >>> print(f"Average loss: {loss_meter.avg:.4f}")
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update with a new value.
        
        Args:
            val: New value to add
            n: Number of samples (for weighted average)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test seed setting
    set_seed(42)
    print("Seed set successfully")
    
    # Test device detection
    device = get_device()
    print(f"Device: {device}")
    
    # Test AverageMeter
    meter = AverageMeter("loss")
    for i in range(10):
        meter.update(i * 0.1)
    print(f"AverageMeter: {meter}")
    
    # Test time formatting
    print(f"Time format: {format_time(3725)}")
    
    print("All tests passed!")
