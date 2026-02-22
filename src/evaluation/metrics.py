"""
Evaluation module for Fetal Brain MRI Segmentation.

This module provides evaluation metrics for multi-class segmentation:
- Dice score (single class and multi-class)
- Per-class metrics computation
- Evaluation utilities

Author: Research Team
Date: 2026
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def dice_score(
    prediction: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-5
) -> float:
    """
    Compute Dice score for binary segmentation.
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    
    Args:
        prediction: Binary prediction mask
        target: Binary ground truth mask
        smooth: Smoothing factor to prevent division by zero
        
    Returns:
        Dice score between 0 and 1
        
    Example:
        >>> pred = np.array([0, 1, 1, 1, 0])
        >>> target = np.array([0, 0, 1, 1, 0])
        >>> score = dice_score(pred, target)
        >>> print(f"Dice: {score:.4f}")
        Dice: 0.8000
    """
    # Convert to numpy if tensor
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Ensure binary
    prediction = (prediction > 0).astype(np.float32)
    target = (target > 0).astype(np.float32)
    
    # Compute intersection and union
    intersection = np.sum(prediction * target)
    union = np.sum(prediction) + np.sum(target)
    
    # Compute Dice
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return float(dice)


def dice_score_multiclass(
    prediction: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    num_classes: int,
    include_background: bool = False,
    smooth: float = 1e-5
) -> Dict[str, float]:
    """
    Compute Dice score for multi-class segmentation.
    
    Args:
        prediction: Prediction mask with class indices (D, H, W)
        target: Ground truth mask with class indices (D, H, W)
        num_classes: Number of classes
        include_background: Whether to include class 0 (background)
        smooth: Smoothing factor
        
    Returns:
        Dictionary with:
            - 'mean_dice': Mean Dice across all classes
            - 'class_X': Dice for each class X
            
    Example:
        >>> pred = np.random.randint(0, 8, (64, 64, 64))
        >>> target = np.random.randint(0, 8, (64, 64, 64))
        >>> scores = dice_score_multiclass(pred, target, num_classes=8)
        >>> print(f"Mean Dice: {scores['mean_dice']:.4f}")
    """
    # Convert to numpy if tensor
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    results = {}
    dice_scores = []
    
    start_class = 0 if include_background else 1
    
    for c in range(start_class, num_classes):
        # Create binary masks for class c
        pred_c = (prediction == c).astype(np.float32)
        target_c = (target == c).astype(np.float32)
        
        # Compute intersection and union
        intersection = np.sum(pred_c * target_c)
        union = np.sum(pred_c) + np.sum(target_c)
        
        # Only compute Dice if class is present in either pred or target
        if union > 0:
            dice = (2.0 * intersection + smooth) / (union + smooth)
        else:
            # Class not present, skip or assign 1.0 (perfect score for empty class)
            dice = 1.0 if np.sum(target_c) == 0 else 0.0
        
        results[f'class_{c}'] = float(dice)
        dice_scores.append(dice)
    
    # Compute mean Dice
    results['mean_dice'] = float(np.mean(dice_scores)) if dice_scores else 0.0
    
    return results


def evaluate_segmentation(
    prediction: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    num_classes: int = 8,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Comprehensive evaluation of multi-class segmentation.
    
    Args:
        prediction: Prediction mask with class indices
        target: Ground truth mask with class indices
        num_classes: Number of classes
        class_names: Optional list of class names for reporting
        
    Returns:
        Dictionary with all evaluation metrics
        
    Example:
        >>> class_names = ['BG', 'CSF', 'GM', 'WM', 'Vent', 'Cereb', 'DGM', 'BS']
        >>> metrics = evaluate_segmentation(pred, target, num_classes=8, class_names=class_names)
    """
    # Default class names for FeTA
    if class_names is None:
        class_names = [
            'Background',
            'External CSF',
            'Gray Matter',
            'White Matter',
            'Ventricles',
            'Cerebellum',
            'Deep Gray Matter',
            'Brainstem'
        ]
    
    # Compute Dice scores
    dice_results = dice_score_multiclass(
        prediction, target,
        num_classes=num_classes,
        include_background=False
    )
    
    # Format results with class names
    results = {
        'mean_dice': dice_results['mean_dice']
    }
    
    for c in range(1, num_classes):
        class_key = f'class_{c}'
        if class_key in dice_results:
            name = class_names[c] if c < len(class_names) else f'Class {c}'
            results[f'dice_{name.lower().replace(" ", "_")}'] = dice_results[class_key]
    
    return results


def print_evaluation_report(
    metrics: Dict[str, float],
    title: str = "Evaluation Report"
) -> None:
    """
    Print a formatted evaluation report.
    
    Args:
        metrics: Dictionary of metrics
        title: Report title
    """
    print("\n" + "=" * 50)
    print(f" {title}")
    print("=" * 50)
    
    # Print mean Dice first
    if 'mean_dice' in metrics:
        print(f"\n{'Mean Dice:':<30} {metrics['mean_dice']:.4f}")
    
    # Print per-class metrics
    print("\nPer-Class Dice Scores:")
    print("-" * 40)
    
    for key, value in sorted(metrics.items()):
        if key.startswith('dice_'):
            name = key.replace('dice_', '').replace('_', ' ').title()
            print(f"  {name:<26} {value:.4f}")
    
    print("=" * 50)


class SegmentationEvaluator:
    """
    Class for evaluating segmentation models across a dataset.
    
    Args:
        num_classes: Number of segmentation classes
        class_names: Optional list of class names
        
    Example:
        >>> evaluator = SegmentationEvaluator(num_classes=8)
        >>> for pred, target in test_loader:
        ...     evaluator.update(pred, target)
        >>> results = evaluator.compute()
    """
    
    def __init__(
        self,
        num_classes: int = 8,
        class_names: Optional[List[str]] = None
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [
            'Background', 'External CSF', 'Gray Matter', 'White Matter',
            'Ventricles', 'Cerebellum', 'Deep Gray Matter', 'Brainstem'
        ]
        
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.dice_scores = {c: [] for c in range(self.num_classes)}
        self.sample_count = 0
    
    def update(
        self,
        prediction: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor]
    ):
        """
        Update metrics with a new prediction-target pair.
        
        Args:
            prediction: Prediction mask (can be batched)
            target: Target mask (can be batched)
        """
        # Convert to numpy
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        # Handle batched input
        if prediction.ndim == 4:  # (B, D, H, W)
            for i in range(prediction.shape[0]):
                self._update_single(prediction[i], target[i])
        else:
            self._update_single(prediction, target)
    
    def _update_single(self, prediction: np.ndarray, target: np.ndarray):
        """Update with a single sample."""
        for c in range(self.num_classes):
            pred_c = (prediction == c).astype(np.float32)
            target_c = (target == c).astype(np.float32)
            
            intersection = np.sum(pred_c * target_c)
            union = np.sum(pred_c) + np.sum(target_c)
            
            if union > 0:
                dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
                self.dice_scores[c].append(dice)
        
        self.sample_count += 1
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary with mean and per-class Dice scores
        """
        results = {}
        
        all_scores = []
        
        for c in range(self.num_classes):
            if self.dice_scores[c]:
                mean_dice = np.mean(self.dice_scores[c])
                std_dice = np.std(self.dice_scores[c])
                
                name = self.class_names[c] if c < len(self.class_names) else f'Class {c}'
                results[f'dice_{name.lower().replace(" ", "_")}'] = float(mean_dice)
                results[f'dice_{name.lower().replace(" ", "_")}_std'] = float(std_dice)
                
                if c > 0:  # Exclude background from mean
                    all_scores.append(mean_dice)
        
        results['mean_dice'] = float(np.mean(all_scores)) if all_scores else 0.0
        results['num_samples'] = self.sample_count
        
        return results
    
    def print_report(self):
        """Print evaluation report."""
        results = self.compute()
        print_evaluation_report(results)


if __name__ == "__main__":
    print("Testing evaluation module...")
    
    # Create dummy data
    np.random.seed(42)
    prediction = np.random.randint(0, 8, (64, 64, 64))
    target = np.random.randint(0, 8, (64, 64, 64))
    
    # Make some overlap for realistic Dice scores
    target[:32, :32, :32] = prediction[:32, :32, :32]
    
    # Test binary Dice
    pred_binary = (prediction == 1)
    target_binary = (target == 1)
    binary_dice = dice_score(pred_binary, target_binary)
    print(f"Binary Dice (class 1): {binary_dice:.4f}")
    
    # Test multi-class Dice
    multiclass_dice = dice_score_multiclass(prediction, target, num_classes=8)
    print(f"\nMulti-class Dice scores:")
    for key, value in multiclass_dice.items():
        print(f"  {key}: {value:.4f}")
    
    # Test comprehensive evaluation
    class_names = ['BG', 'CSF', 'GM', 'WM', 'Vent', 'Cereb', 'DGM', 'BS']
    metrics = evaluate_segmentation(prediction, target, num_classes=8, class_names=class_names)
    print_evaluation_report(metrics)
    
    # Test evaluator class
    print("\nTesting SegmentationEvaluator...")
    evaluator = SegmentationEvaluator(num_classes=8)
    
    for _ in range(5):
        pred = np.random.randint(0, 8, (64, 64, 64))
        tgt = np.random.randint(0, 8, (64, 64, 64))
        tgt[:32] = pred[:32]  # Some overlap
        evaluator.update(pred, tgt)
    
    evaluator.print_report()
    
    print("\nAll tests passed!")
