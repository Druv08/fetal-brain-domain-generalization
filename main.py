"""
Main entry point for Fetal Brain MRI Segmentation.

Loads config, builds dataset / model, runs multi-epoch training with
per-class Dice evaluation, and checkpoints the best model.

Usage:
    python main.py --config configs/config.yaml
    python main.py --config configs/config.yaml --epochs 5

Author: Research Team
Date: 2026
"""

import argparse
import sys
from pathlib import Path

import torch

from src.data import load_config, FetalBrainDatasetFromConfig
from src.preprocessing import PreprocessingPipeline
from src.models.unet3d import build_unet3d
from src.training.trainer import Trainer
from src.evaluation import SegmentationEvaluator, FETA_CLASS_NAMES
from src.utils import set_seed, get_device_from_config


def main(args):
    """Main training pipeline."""
    print("=" * 60)
    print(" Domain-Generalizable Fetal Brain MRI Segmentation")
    print("=" * 60)

    # ── 1. Configuration ───────────────────────────────────────
    print("\n[1/5] Loading configuration...")
    config = load_config(args.config)
    set_seed(config.get("system", {}).get("seed", 42))
    device = get_device_from_config(config)

    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs

    num_epochs = config["training"]["num_epochs"]
    print(f"  Device : {device}")
    print(f"  Epochs : {num_epochs}")

    # ── 2. Dataset ─────────────────────────────────────────────
    print("\n[2/5] Building dataset...")
    transforms = PreprocessingPipeline.from_config(config)
    dataset = FetalBrainDatasetFromConfig(config, transform=transforms)

    if dataset._use_dummy:
        print("  ERROR: No real FeTA data found.")
        print("  Place NIfTI subjects under data/sub-*/anat/")
        sys.exit(1)

    n_total = len(dataset)
    print(f"  Subjects found: {n_total}")

    # Train / Val split (80/20), or single-subject sanity mode
    if n_total > 1:
        n_train = max(1, int(0.8 * n_total))
        n_val = n_total - n_train
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val]
        )
        print(f"  Train: {n_train}  |  Val: {n_val}")
    else:
        train_dataset = dataset
        val_dataset = dataset
        print("  Single-subject mode (train = val = 1)")

    # ── 3. DataLoaders ─────────────────────────────────────────
    print("\n[3/5] Creating DataLoaders...")
    batch_size = min(
        config.get("training", {}).get("batch_size", 1), len(train_dataset)
    )
    num_workers = config.get("system", {}).get("num_workers", 0)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    print(f"  batch_size={batch_size}, workers={num_workers}")

    # ── 4. Model ───────────────────────────────────────────────
    print("\n[4/5] Building 3D U-Net...")
    model = build_unet3d(config)
    total_params = sum(p.numel() for p in model.parameters())
    out_ch = config.get("model", {}).get("out_channels", 8)
    print(f"  Parameters    : {total_params:,}")
    print(f"  Output classes: {out_ch}")

    # ── 5. Train ───────────────────────────────────────────────
    print("\n[5/5] Starting training...\n")
    trainer = Trainer(model, config, device=device)
    history = trainer.train(train_loader, val_loader)

    # ── Final evaluation on val set ────────────────────────────
    print("\n" + "=" * 60)
    print(" Final Evaluation")
    print("=" * 60)

    evaluator = SegmentationEvaluator(num_classes=out_ch)
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            evaluator.update(predictions.cpu(), labels)
    evaluator.print_report()

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Final train loss : {history['train_loss'][-1]:.4f}")
    if history["val_dice"]:
        print(f"  Best val Dice    : {max(history['val_dice']):.4f}")
    print(f"  Checkpoints in   : outputs/checkpoints/")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetal Brain MRI Segmentation Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override num_epochs from config",
    )
    args = parser.parse_args()
    main(args)
