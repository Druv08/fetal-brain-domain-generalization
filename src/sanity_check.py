"""
Sanity Check Script for Fetal Brain MRI Segmentation Pipeline.

This script verifies that all components of the pipeline work together:
1. Load configuration from YAML
2. Create preprocessing transforms from config
3. Create dataset from config
4. Create dataloader
5. Build model from config
6. Run forward pass through model
7. Print shapes to verify correctness

Usage:
    python src/sanity_check.py --config configs/config.yaml

Expected output:
    Image shape: [B, 1, D, H, W]
    Label shape: [B, D, H, W]
    Model output: [B, 7, D, H, W]  (7 classes for FeTA dataset)

Author: Research Team
Date: 2026
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_config, FetalBrainDatasetFromConfig
from src.preprocessing import PreprocessingPipeline
from src.models.unet3d import build_unet3d
from src.utils import set_seed, get_device_from_config


def run_sanity_check(config_path: str) -> bool:
    """
    Run full pipeline sanity check.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        True if all checks pass, False otherwise
    """
    print("=" * 60)
    print("FETAL BRAIN MRI SEGMENTATION - SANITY CHECK")
    print("=" * 60)
    
    # Step 1: Load configuration
    print("\n[1/6] Loading configuration...")
    try:
        config = load_config(config_path)
        print(f"  ✓ Config loaded from: {config_path}")
        print(f"  ✓ Dataset: {config.get('data', {}).get('images_dir', 'N/A')}")
        print(f"  ✓ Model out_channels: {config.get('model', {}).get('out_channels', 'N/A')}")
    except Exception as e:
        print(f"  ✗ Failed to load config: {e}")
        return False
    
    # Step 2: Create preprocessing transforms
    print("\n[2/6] Creating preprocessing transforms...")
    try:
        transforms = PreprocessingPipeline.from_config(config)
        print(f"  ✓ PreprocessingPipeline created")
        print(f"  ✓ Clip range: {config.get('preprocessing', {}).get('clip_range', 'N/A')}")
    except Exception as e:
        print(f"  ✗ Failed to create transforms: {e}")
        return False
    
    # Step 3: Create dataset
    print("\n[3/6] Creating dataset...")
    try:
        dataset = FetalBrainDatasetFromConfig(config, transform=transforms)
        print(f"  ✓ Dataset created with {len(dataset)} samples")
    except Exception as e:
        print(f"  ✗ Failed to create dataset: {e}")
        return False
    
    # Step 4: Create dataloader
    print("\n[4/6] Creating dataloader...")
    try:
        batch_size = config.get('training', {}).get('batch_size', 1)
        num_workers = config.get('system', {}).get('num_workers', 0)
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        print(f"  ✓ DataLoader created")
        print(f"  ✓ Batch size: {batch_size}, Workers: {num_workers}")
    except Exception as e:
        print(f"  ✗ Failed to create dataloader: {e}")
        return False
    
    # Step 5: Build model
    print("\n[5/6] Building model...")
    try:
        model = build_unet3d(config)
        device = get_device_from_config(config)
        model = model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  ✓ 3D U-Net model built")
        print(f"  ✓ Device: {device}")
        print(f"  ✓ Total parameters: {total_params:,}")
        print(f"  ✓ Trainable parameters: {trainable_params:,}")
    except Exception as e:
        print(f"  ✗ Failed to build model: {e}")
        return False
    
    # Step 6: Forward pass
    print("\n[6/6] Running forward pass...")
    try:
        # Get one batch
        batch = next(iter(dataloader))
        images, labels = batch  # Dataset returns (image, label) tuple
        
        # Move to device
        images = images.to(device)
        labels = labels.to(device)
        
        print(f"  → Image tensor shape: {list(images.shape)}")
        print(f"  → Label tensor shape: {list(labels.shape)}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(images)
        
        print(f"  → Model output shape: {list(outputs.shape)}")
        
        # Validate shapes
        expected_out_channels = config.get('model', {}).get('out_channels', 7)
        assert outputs.shape[1] == expected_out_channels, \
            f"Expected {expected_out_channels} output channels, got {outputs.shape[1]}"
        assert outputs.shape[0] == images.shape[0], "Batch size mismatch"
        assert outputs.shape[2:] == images.shape[2:], "Spatial dimensions mismatch"
        
        print(f"  ✓ All shape checks passed!")
        
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("SANITY CHECK PASSED ✓")
    print("=" * 60)
    print("\nPipeline components verified:")
    print(f"  • Config loading:     OK")
    print(f"  • Preprocessing:      OK")
    print(f"  • Dataset:            OK ({len(dataset)} samples)")
    print(f"  • DataLoader:         OK (batch_size={batch_size})")
    print(f"  • Model:              OK ({trainable_params:,} params)")
    print(f"  • Forward pass:       OK")
    print("\nExpected shapes:")
    print(f"  Image shape:  {list(images.shape)}")
    print(f"  Label shape:  {list(labels.shape)}")
    print(f"  Output shape: {list(outputs.shape)}")
    
    return True, model, dataloader, device, config


def run_one_step_training(model, dataloader, device, config):
    """
    Run one training iteration to verify backward pass works.
    
    Args:
        model: The UNet3D model
        dataloader: DataLoader with training data
        device: Device to train on
        config: Configuration dictionary
    
    Returns:
        True if training step succeeds
    """
    from src.training.trainer import DiceCELoss
    
    print("\n" + "=" * 60)
    print("ONE-STEP TRAINING TEST")
    print("=" * 60)
    
    try:
        # Setup
        model.train()
        criterion = DiceCELoss()
        
        lr = config.get('training', {}).get('learning_rate', 0.0001)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        print(f"\n  Learning rate: {lr}")
        print(f"  Loss function: Dice + CrossEntropy")
        
        # Get one batch
        print("\n[1/4] Loading batch...")
        images, labels = next(iter(dataloader))
        images = images.to(device)
        labels = labels.to(device)
        print(f"  ✓ Batch loaded: images={list(images.shape)}, labels={list(labels.shape)}")
        
        # Forward pass
        print("\n[2/4] Forward pass...")
        outputs = model(images)
        print(f"  ✓ Output shape: {list(outputs.shape)}")
        
        # Compute loss
        print("\n[3/4] Computing loss...")
        loss = criterion(outputs, labels)
        print(f"  ✓ Loss value: {loss.item():.4f}")
        
        # Backward pass
        print("\n[4/4] Backward pass + optimizer step...")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"  ✓ Gradients computed and weights updated!")
        
        # Check gradients exist
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        print(f"  ✓ Gradient norm: {grad_norm:.4f}")
        
        print("\n" + "=" * 60)
        print("ONE-STEP TRAINING PASSED ✓")
        print("=" * 60)
        print(f"\n  Loss: {loss.item():.4f}")
        print(f"  Training pipeline is fully functional!")
        
        return True
        
    except Exception as e:
        print(f"\n  ✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run sanity check on the fetal brain segmentation pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Also run one-step training test"
    )
    
    args = parser.parse_args()
    
    # Run sanity check
    result = run_sanity_check(args.config)
    
    if isinstance(result, tuple):
        success, model, dataloader, device, config = result
    else:
        success = result
        model = dataloader = device = config = None
    
    # Optionally run one-step training
    if success and args.train and model is not None:
        train_success = run_one_step_training(model, dataloader, device, config)
        success = success and train_success
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
