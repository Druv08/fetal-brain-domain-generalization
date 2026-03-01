"""
Training Script — Train 3D U-Net on FeTA dataset.

Loads config, builds dataset/model, and runs multi-epoch training
with batch-wise loss, validation Dice score, and best-model checkpointing.

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --epochs 5

Author: Druv
Date: 2026
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_config, FetalBrainDatasetFromConfig
from src.preprocessing import PreprocessingPipeline
from src.models.unet3d import build_unet3d
from src.training.trainer import Trainer
from src.utils import set_seed, get_device_from_config


def main():
    parser = argparse.ArgumentParser(description="Train 3D U-Net on FeTA data")
    parser.add_argument("--config", default="configs/config.yaml", help="Config path")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    args = parser.parse_args()

    print("=" * 60)
    print("FETAL BRAIN MRI SEGMENTATION — TRAINING")
    print("=" * 60)

    # ── 1. Config ──────────────────────────────────────────────
    print("\n[1/5] Loading configuration...")
    config = load_config(args.config)
    set_seed(config.get("system", {}).get("seed", 42))
    device = get_device_from_config(config)

    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs

    num_epochs = config["training"]["num_epochs"]
    print(f"  ✓ Device: {device}")
    print(f"  ✓ Epochs: {num_epochs}")

    # ── 2. Dataset ─────────────────────────────────────────────
    print("\n[2/5] Building dataset...")
    transforms = PreprocessingPipeline.from_config(config)
    dataset = FetalBrainDatasetFromConfig(config, transform=transforms)

    if dataset._use_dummy:
        print("  ✗ No real FeTA data found. Place data under data/sub-*/anat/")
        sys.exit(1)

    n_total = len(dataset)
    print(f"  ✓ {n_total} subject(s) found")

    # Split into train/val (80/20) if more than 1 subject, else use same for both
    if n_total > 1:
        n_train = max(1, int(0.8 * n_total))
        n_val = n_total - n_train
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
        print(f"  ✓ Train: {n_train}, Val: {n_val}")
    else:
        # Single subject: use it for both train and val (sanity mode)
        train_dataset = dataset
        val_dataset = dataset
        print(f"  ✓ Single subject mode (train=val=1)")

    # ── 3. DataLoaders ─────────────────────────────────────────
    print("\n[3/5] Creating DataLoaders...")
    batch_size = min(config.get("training", {}).get("batch_size", 1), len(train_dataset))
    num_workers = config.get("system", {}).get("num_workers", 0)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    print(f"  ✓ batch_size={batch_size}, workers={num_workers}")

    # ── 4. Model ───────────────────────────────────────────────
    print("\n[4/5] Building 3D U-Net...")
    model = build_unet3d(config)
    total_params = sum(p.numel() for p in model.parameters())
    out_ch = config.get("model", {}).get("out_channels", 8)
    print(f"  ✓ Parameters: {total_params:,}")
    print(f"  ✓ Output classes: {out_ch}")

    # ── 5. Train ───────────────────────────────────────────────
    print("\n[5/5] Starting training...\n")
    trainer = Trainer(model, config, device=device)
    history = trainer.train(train_loader, val_loader)

    # ── Done ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Final train loss : {history['train_loss'][-1]:.4f}")
    if history['val_dice']:
        print(f"  Best val Dice    : {max(history['val_dice']):.4f}")
    print(f"  Checkpoints in   : outputs/checkpoints/")
    print()


if __name__ == "__main__":
    main()
