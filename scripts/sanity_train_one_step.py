"""
Sanity Training Check — One iteration on real FeTA data.

Loads config, creates dataset/dataloader/model, runs ONE training step
(forward + loss + backward + optimizer), and prints the loss.

NO full epochs. Just proof that training works end-to-end.

Usage:
    python scripts/sanity_train_one_step.py

Author: Druv
Date: 2026
"""

import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_config, FetalBrainDatasetFromConfig
from src.preprocessing import PreprocessingPipeline
from src.models.unet3d import build_unet3d
from src.training.trainer import DiceCELoss
from src.utils import set_seed, get_device_from_config


def main():
    config_path = "configs/config.yaml"

    print("=" * 60)
    print("SANITY TRAINING CHECK — ONE STEP ON REAL DATA")
    print("=" * 60)

    # ── 1. Config ──────────────────────────────────────────────
    print("\n[1/7] Loading configuration...")
    config = load_config(config_path)
    set_seed(config.get("system", {}).get("seed", 42))
    device = get_device_from_config(config)
    print(f"  ✓ Device: {device}")

    # ── 2. Dataset ─────────────────────────────────────────────
    print("\n[2/7] Building dataset...")
    transforms = PreprocessingPipeline.from_config(config)
    dataset = FetalBrainDatasetFromConfig(config, transform=transforms)

    if dataset._use_dummy:
        print("  ✗ No real FeTA subjects found. Place data under data/sub-*/anat/")
        sys.exit(1)

    print(f"  ✓ {len(dataset)} subject(s): {[s['subject_id'] for s in dataset.samples]}")

    # ── 3. DataLoader ──────────────────────────────────────────
    print("\n[3/7] Creating DataLoader...")
    batch_size = min(config.get("training", {}).get("batch_size", 1), len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    print(f"  ✓ batch_size={batch_size}")

    # ── 4. Load one batch ──────────────────────────────────────
    print("\n[4/7] Loading one batch...")
    images, labels = next(iter(dataloader))
    images = images.to(device)
    labels = labels.to(device)
    print(f"  ✓ images: {list(images.shape)}")
    print(f"  ✓ labels: {list(labels.shape)}")

    # ── 5. Model ───────────────────────────────────────────────
    print("\n[5/7] Building 3D U-Net...")
    model = build_unet3d(config).to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Parameters: {total_params:,}")

    # ── 6. Forward pass + loss ─────────────────────────────────
    print("\n[6/7] Forward pass + Dice+CE loss...")
    criterion = DiceCELoss()
    outputs = model(images)
    loss = criterion(outputs, labels)
    print(f"  ✓ Output shape: {list(outputs.shape)}")
    print(f"  ✓ Loss: {loss.item():.4f}")

    # ── 7. Backward + optimizer step ───────────────────────────
    print("\n[7/7] Backward pass + optimizer step...")
    lr = config.get("training", {}).get("learning_rate", 0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    print(f"  ✓ Gradients computed  (norm: {grad_norm:.4f})")
    print(f"  ✓ Weights updated     (lr: {lr})")

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ONE-STEP TRAINING PASSED ✓")
    print("=" * 60)
    print(f"\n  Loss: {loss.item():.4f}")
    print(f"  The model is now learning on real fetal brain MRI data.")
    print()


if __name__ == "__main__":
    main()
