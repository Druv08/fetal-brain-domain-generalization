"""
Sanity Forward Pass — Real FeTA Data through 3D U-Net.

Loads config, builds dataset + dataloader + model, runs one forward pass
on real fetal brain MRI data, and prints input/output shapes.

NO training is performed.

Usage:
    python scripts/sanity_forward_pass.py

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
from src.utils import set_seed, get_device_from_config


def main():
    config_path = "configs/config.yaml"

    print("=" * 60)
    print("SANITY FORWARD PASS — REAL FeTA DATA")
    print("=" * 60)

    # ── 1. Load config ─────────────────────────────────────────
    print("\n[1/5] Loading configuration...")
    config = load_config(config_path)
    set_seed(config.get("system", {}).get("seed", 42))
    device = get_device_from_config(config)
    print(f"  ✓ Config loaded")
    print(f"  ✓ Device: {device}")

    # ── 2. Build dataset ───────────────────────────────────────
    print("\n[2/5] Building dataset...")
    transforms = PreprocessingPipeline.from_config(config)
    dataset = FetalBrainDatasetFromConfig(config, transform=transforms)

    if dataset._use_dummy:
        print("\n  ✗ No real FeTA subjects found in data/")
        print("    Place sub-*/anat/*T2w.nii.gz + *dseg.nii.gz under data/")
        sys.exit(1)

    print(f"  ✓ Found {len(dataset)} real subject(s):")
    for s in dataset.samples:
        print(f"      • {s['subject_id']}: {s['image'].name}")

    # ── 3. Create dataloader ───────────────────────────────────
    print("\n[3/5] Creating DataLoader...")
    batch_size = min(config.get("training", {}).get("batch_size", 1), len(dataset))
    num_workers = config.get("system", {}).get("num_workers", 0)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    print(f"  ✓ DataLoader ready  (batch_size={batch_size})")

    # ── 4. Build model ─────────────────────────────────────────
    print("\n[4/5] Building 3D U-Net...")
    model = build_unet3d(config).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    out_channels = config.get("model", {}).get("out_channels", 7)
    print(f"  ✓ Parameters : {total_params:,}")
    print(f"  ✓ Classes    : {out_channels}")

    # ── 5. Forward pass ────────────────────────────────────────
    print("\n[5/5] Running forward pass on real data...")
    images, labels = next(iter(dataloader))
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)

    print(f"\n  Input shape  : {list(images.shape)}")
    print(f"  Label shape  : {list(labels.shape)}")
    print(f"  Output shape : {list(outputs.shape)}")
    print(f"  Num classes  : {outputs.shape[1]}")

    # Quick prediction stats
    preds = torch.argmax(outputs, dim=1)
    unique_preds = torch.unique(preds).cpu().tolist()
    print(f"  Predicted classes in batch: {unique_preds}")

    # ── Done ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FORWARD PASS PASSED ✓")
    print("=" * 60)
    print(f"\n  Your 3D U-Net just processed real fetal brain MRI data.")
    print(f"  Input:  [B={images.shape[0]}, C=1, D={images.shape[2]}, H={images.shape[3]}, W={images.shape[4]}]")
    print(f"  Output: [B={outputs.shape[0]}, Classes={outputs.shape[1]}, D={outputs.shape[2]}, H={outputs.shape[3]}, W={outputs.shape[4]}]")
    print()


if __name__ == "__main__":
    main()
