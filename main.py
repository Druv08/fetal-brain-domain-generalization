"""
Main entry point for Fetal Brain MRI Segmentation.

This script demonstrates how to use the pipeline end-to-end.

Usage:
    python main.py --config configs/config.yaml

Author: Research Team
Date: 2026
"""

import argparse
from pathlib import Path

import yaml
import torch

from src.data import FetalBrainDataset
from src.preprocessing import PreprocessingPipeline
from src.models import UNet3D
from src.training import Trainer, create_dataloaders
from src.evaluation import SegmentationEvaluator
from src.utils import set_seed, get_device


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main training pipeline."""
    # Load configuration
    print("=" * 60)
    print(" Domain-Generalizable Fetal Brain MRI Segmentation")
    print("=" * 60)
    
    config = load_config(args.config)
    print(f"\nConfiguration loaded from: {args.config}")
    
    # Set random seed for reproducibility
    seed = config.get('hardware', {}).get('seed', 42)
    set_seed(seed)
    print(f"Random seed: {seed}")
    
    # Get device
    device = get_device(config.get('hardware', {}).get('device'))
    print(f"Device: {device}")
    
    # Setup preprocessing pipeline
    preprocess_config = config.get('preprocessing', {})
    transform = PreprocessingPipeline(
        clip_percentiles=(
            preprocess_config.get('clip_percentile_low', 0.5),
            preprocess_config.get('clip_percentile_high', 99.5)
        ),
        normalize=preprocess_config.get('normalize', True),
        crop_size=tuple(preprocess_config.get('crop_size', [128, 128, 128]))
    )
    print(f"Preprocessing: clipping + z-score + crop to {preprocess_config.get('crop_size')}")
    
    # Load datasets
    data_config = config.get('data', {})
    train_dir = data_config.get('train_dir', 'data/feta_2.4/train')
    val_dir = data_config.get('val_dir', 'data/feta_2.4/val')
    
    print(f"\nLoading datasets...")
    print(f"  Train directory: {train_dir}")
    print(f"  Val directory: {val_dir}")
    
    try:
        train_dataset = FetalBrainDataset(
            train_dir,
            image_pattern=data_config.get('image_pattern', '*_T2w.nii.gz'),
            label_pattern=data_config.get('label_pattern', '*_dseg.nii.gz'),
            transform=transform
        )
        print(f"  Training samples: {len(train_dataset)}")
        
        val_dataset = FetalBrainDataset(
            val_dir,
            image_pattern=data_config.get('image_pattern', '*_T2w.nii.gz'),
            label_pattern=data_config.get('label_pattern', '*_dseg.nii.gz'),
            transform=transform
        )
        print(f"  Validation samples: {len(val_dataset)}")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError loading dataset: {e}")
        print("Please ensure the FeTA dataset is downloaded and paths are correct in config.yaml")
        print("\nTo download the FeTA dataset:")
        print("  1. Visit https://feta.grand-challenge.org/")
        print("  2. Register and download FeTA 2.4")
        print("  3. Extract to data/feta_2.4/")
        return
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, config
    )
    
    # Create model
    model_config = config.get('model', {})
    model = UNet3D(
        in_channels=model_config.get('in_channels', 1),
        out_channels=model_config.get('num_classes', 8),
        base_features=model_config.get('base_features', 32),
        num_levels=model_config.get('num_levels', 4),
        dropout=model_config.get('dropout', 0.1)
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {model_config.get('architecture', 'UNet3D')}")
    print(f"  Parameters: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(model, config)
    
    # Train
    print("\n" + "=" * 60)
    print(" Starting Training")
    print("=" * 60)
    
    history = trainer.train(train_loader, val_loader)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print(" Final Evaluation")
    print("=" * 60)
    
    evaluator = SegmentationEvaluator(
        num_classes=model_config.get('num_classes', 8)
    )
    
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            evaluator.update(predictions.cpu(), labels)
    
    evaluator.print_report()
    
    print("\nTraining complete!")
    print(f"Best model saved to: {trainer.checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetal Brain MRI Segmentation Training"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    main(args)
