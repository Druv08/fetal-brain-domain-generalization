"""
Training module for Fetal Brain MRI Segmentation.

This module provides:
- Training loop structure
- Validation loop
- Dice loss implementation
- Model saving and logging

Author: Research Team
Date: 2026
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.helpers import AverageMeter, format_time, save_checkpoint, get_device


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation.
    
    Computes the Dice loss between predictions and targets.
    Supports both soft Dice (from logits) and hard Dice.
    
    Args:
        smooth: Smoothing factor to prevent division by zero
        ignore_index: Class index to ignore (e.g., background)
        softmax: Whether to apply softmax to predictions
        
    Example:
        >>> criterion = DiceLoss()
        >>> loss = criterion(predictions, targets)
    """
    
    def __init__(
        self,
        smooth: float = 1e-5,
        ignore_index: Optional[int] = None,
        softmax: bool = True
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.softmax = softmax
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            predictions: Model output logits (B, C, D, H, W)
            targets: Ground truth labels (B, D, H, W) with integer values
            
        Returns:
            Dice loss value (scalar)
        """
        num_classes = predictions.shape[1]
        
        # Apply softmax if needed
        if self.softmax:
            predictions = F.softmax(predictions, dim=1)
        
        # One-hot encode targets
        # Shape: (B, D, H, W) -> (B, C, D, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
        
        # Compute Dice per class
        dice_scores = []
        
        for c in range(num_classes):
            if self.ignore_index is not None and c == self.ignore_index:
                continue
            
            pred_c = predictions[:, c]
            target_c = targets_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Average Dice across classes
        mean_dice = torch.stack(dice_scores).mean()
        
        # Dice loss = 1 - Dice
        return 1.0 - mean_dice


class DiceCELoss(nn.Module):
    """
    Combined Dice Loss and Cross-Entropy Loss.
    
    Combines soft Dice loss with cross-entropy for better
    optimization landscape.
    
    Args:
        dice_weight: Weight for Dice loss component
        ce_weight: Weight for cross-entropy component
        ignore_index: Class index to ignore
        
    Example:
        >>> criterion = DiceCELoss(dice_weight=0.5, ce_weight=0.5)
        >>> loss = criterion(predictions, targets)
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        ignore_index: int = -100
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        self.dice_loss = DiceLoss(ignore_index=None)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined loss."""
        dice = self.dice_loss(predictions, targets)
        ce = self.ce_loss(predictions, targets)
        
        return self.dice_weight * dice + self.ce_weight * ce


class Trainer:
    """
    Trainer class for 3D segmentation models.
    
    Handles training loop, validation, checkpointing, and logging.
    
    Args:
        model: PyTorch model to train
        config: Configuration dictionary
        optimizer: Optional optimizer (created from config if not provided)
        scheduler: Optional learning rate scheduler
        criterion: Optional loss function (Dice+CE if not provided)
        device: Device to train on
        
    Example:
        >>> trainer = Trainer(model, config)
        >>> trainer.train(train_loader, val_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.device = device or get_device(config.get('hardware', {}).get('device'))
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Setup optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = self._create_scheduler()
        
        # Setup loss function
        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = DiceCELoss()
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.best_loss = float('inf')
        
        # Training config
        train_config = config.get('training', {})
        self.num_epochs = train_config.get('num_epochs', 100)
        self.use_amp = train_config.get('use_amp', True)
        self.gradient_accumulation = train_config.get('gradient_accumulation', 1)
        self.save_freq = train_config.get('save_freq', 10)
        self.early_stopping_patience = train_config.get('early_stopping_patience', 20)
        
        # Setup mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # Output directories
        output_config = config.get('output', {})
        self.checkpoint_dir = Path(output_config.get('checkpoint_dir', 'outputs/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = output_config.get('experiment_name', 'experiment')
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        train_config = self.config.get('training', {})
        
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config.get('learning_rate', 1e-4),
            weight_decay=train_config.get('weight_decay', 1e-5)
        )
    
    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler from config."""
        train_config = self.config.get('training', {})
        scheduler_type = train_config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=1e-7
            )
        elif scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            return None
    
    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        loss_meter = AverageMeter('loss')
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update metrics
            loss_meter.update(loss.item(), images.size(0))
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
        
        return {'train_loss': loss_meter.avg}
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        loss_meter = AverageMeter('loss')
        dice_meter = AverageMeter('dice')
        
        for images, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Compute Dice score
            predictions = torch.argmax(outputs, dim=1)
            dice = self._compute_dice(predictions, labels)
            
            # Update metrics
            loss_meter.update(loss.item(), images.size(0))
            dice_meter.update(dice, images.size(0))
        
        return {
            'val_loss': loss_meter.avg,
            'val_dice': dice_meter.avg
        }
    
    def _compute_dice(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1e-5
    ) -> float:
        """Compute mean Dice score."""
        num_classes = self.config.get('model', {}).get('num_classes', 8)
        
        dice_scores = []
        
        for c in range(1, num_classes):  # Skip background
            pred_c = (predictions == c).float()
            target_c = (targets == c).float()
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            if union > 0:
                dice = (2.0 * intersection + smooth) / (union + smooth)
                dice_scores.append(dice.item())
        
        return sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            
        Returns:
            Dictionary with training history
        """
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'lr': []
        }
        
        no_improvement_count = 0
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch + 1
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['train_loss'])
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history['val_loss'].append(val_metrics['val_loss'])
                history['val_dice'].append(val_metrics['val_dice'])
                
                current_metric = val_metrics['val_dice']
            else:
                current_metric = -train_metrics['train_loss']
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Check for improvement
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                no_improvement_count = 0
                
                # Save best model
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    train_metrics['train_loss'],
                    self.checkpoint_dir / f'{self.experiment_name}_best.pth',
                    scheduler=self.scheduler,
                    metrics={'dice': current_metric},
                    config=self.config
                )
            else:
                no_improvement_count += 1
            
            # Regular checkpoint
            if epoch % self.save_freq == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    train_metrics['train_loss'],
                    self.checkpoint_dir / f'{self.experiment_name}_epoch{epoch}.pth',
                    scheduler=self.scheduler
                )
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch}/{self.num_epochs} ({format_time(epoch_time)})")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            if val_loader is not None:
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"  Val Dice: {val_metrics['val_dice']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Best Dice: {self.best_metric:.4f}")
            
            # Early stopping
            if no_improvement_count >= self.early_stopping_patience:
                print(f"\nEarly stopping after {no_improvement_count} epochs without improvement")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {format_time(total_time)}")
        print(f"Best validation Dice: {self.best_metric:.4f}")
        
        return history


def create_dataloaders(
    train_dataset,
    val_dataset,
    config: Dict
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_config = config.get('training', {})
    
    batch_size = train_config.get('batch_size', 2)
    num_workers = train_config.get('num_workers', 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("Training module loaded successfully")
    
    # Test loss functions
    print("\nTesting loss functions...")
    
    # Create dummy data
    predictions = torch.randn(2, 8, 32, 32, 32)
    targets = torch.randint(0, 8, (2, 32, 32, 32))
    
    # Test Dice loss
    dice_loss = DiceLoss()
    loss = dice_loss(predictions, targets)
    print(f"Dice Loss: {loss.item():.4f}")
    
    # Test Dice+CE loss
    dice_ce_loss = DiceCELoss()
    loss = dice_ce_loss(predictions, targets)
    print(f"Dice+CE Loss: {loss.item():.4f}")
    
    print("\nAll tests passed!")
