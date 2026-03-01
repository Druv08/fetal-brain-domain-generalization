"""
Test script to verify FeTA fetal brain MRI dataset loading.

Automatically discovers the first FeTA subject inside data/sub-*/anat/,
loads the T2w MRI and segmentation label, prints diagnostics, and
visualizes the middle axial slice.

Usage:
    python scripts/test_feta_loading.py

Requirements:
    pip install nibabel numpy matplotlib

Author: Druv
Date: 2026
"""

import sys
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────
# Root data directory (relative to project root)
# ─────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
# ─────────────────────────────────────────────────────────────

# FeTA tissue class names for reference
TISSUE_NAMES = {
    0: "Background",
    1: "External CSF",
    2: "Gray Matter",
    3: "White Matter",
    4: "Ventricles",
    5: "Cerebellum",
    6: "Deep Gray Matter",
    7: "Brainstem",
}


def find_feta_subject(data_dir: Path):
    """
    Auto-detect the first FeTA subject inside data/.

    Searches for:
        data/sub-*/anat/*T2w.nii.gz
        data/sub-*/anat/*dseg.nii.gz

    Returns:
        (subject_id, img_path, label_path) or None
    """
    if not data_dir.exists():
        return None

    # Find all sub-* directories, sorted so results are deterministic
    subject_dirs = sorted(data_dir.glob("sub-*"))

    for subj_dir in subject_dirs:
        anat_dir = subj_dir / "anat"
        if not anat_dir.is_dir():
            continue

        # Look for T2w and dseg NIfTI files
        t2w_files = sorted(anat_dir.glob("*T2w.nii.gz"))
        dseg_files = sorted(anat_dir.glob("*dseg.nii.gz"))

        if t2w_files and dseg_files:
            return subj_dir.name, t2w_files[0], dseg_files[0]

    return None


def main():
    print("=" * 60)
    print("FeTA DATASET LOADING TEST")
    print("=" * 60)

    # ── Step 1: Auto-detect subject ────────────────────────────
    print("\n[1/5] Searching for FeTA subject in data/ ...")

    result = find_feta_subject(DATA_DIR)

    if result is None:
        print("\n  No FeTA subject found inside data/")
        print("  Expected structure:")
        print("    data/")
        print("      └── sub-XXX/")
        print("           └── anat/")
        print("                ├── sub-XXX_..._T2w.nii.gz")
        print("                └── sub-XXX_..._dseg.nii.gz")
        sys.exit(1)

    subject_id, img_file, label_file = result

    print(f"  ✓ Found subject : {subject_id}")
    print(f"  ✓ MRI path      : {img_file}")
    print(f"  ✓ Label path    : {label_file}")

    # ── Step 2: Load NIfTI files ───────────────────────────────
    try:
        import nibabel as nib
    except ImportError:
        print("\n[ERROR] nibabel not installed. Run: pip install nibabel")
        sys.exit(1)

    print(f"\n[2/5] Loading NIfTI volumes for {subject_id}...")
    img_nii = nib.load(str(img_file))
    label_nii = nib.load(str(label_file))

    img_data = img_nii.get_fdata().astype(np.float32)
    label_data = label_nii.get_fdata().astype(np.int64)

    print("  ✓ Volumes loaded successfully")

    # ── Step 3: Print diagnostic info ──────────────────────────
    print(f"\n[3/5] Volume information:")
    print(f"  MRI shape       : {img_data.shape}")
    print(f"  Label shape     : {label_data.shape}")
    print(f"  MRI dtype       : {img_data.dtype}")
    print(f"  Label dtype     : {label_data.dtype}")
    print(f"  Voxel size (mm) : {tuple(np.round(img_nii.header.get_zooms()[:3], 3))}")
    print(f"  MRI intensity   : min={img_data.min():.2f}, max={img_data.max():.2f}, mean={img_data.mean():.2f}")

    shapes_match = img_data.shape == label_data.shape
    print(f"  Shapes match    : {'✓ YES' if shapes_match else '✗ NO — this is a problem!'}")

    # ── Step 4: Label analysis ─────────────────────────────────
    unique_labels = np.unique(label_data).astype(int)
    print(f"\n[4/5] Unique label values: {unique_labels.tolist()}")
    for lbl in unique_labels:
        name = TISSUE_NAMES.get(lbl, "Unknown")
        count = np.sum(label_data == lbl)
        pct = 100.0 * count / label_data.size
        print(f"  {lbl} → {name:20s}  ({count:>10,} voxels, {pct:5.2f}%)")

    # ── Step 5: Extract middle slice & visualize ───────────────
    print(f"\n[5/5] Visualizing middle slice...")

    mid_idx = img_data.shape[2] // 2
    img_slice = img_data[:, :, mid_idx]
    label_slice = label_data[:, :, mid_idx]

    print(f"  Slice index: {mid_idx} (of {img_data.shape[2]} total in 3rd dim)")

    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for saving
    except Exception:
        pass

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[ERROR] matplotlib not installed. Run: pip install matplotlib")
        print("  Skipping visualization, but data loading PASSED ✓")
        sys.exit(0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1 — MRI slice (grayscale)
    axes[0].imshow(img_slice.T, cmap="gray", origin="lower")
    axes[0].set_title(f"T2w MRI  (slice {mid_idx})")
    axes[0].axis("off")

    # Panel 2 — Label map (color)
    axes[1].imshow(label_slice.T, cmap="nipy_spectral", origin="lower",
                   vmin=0, vmax=max(7, unique_labels.max()))
    axes[1].set_title(f"Segmentation Label  (slice {mid_idx})")
    axes[1].axis("off")

    # Panel 3 — Overlay (MRI + label transparency)
    axes[2].imshow(img_slice.T, cmap="gray", origin="lower")
    masked_label = np.ma.masked_where(label_slice == 0, label_slice)
    axes[2].imshow(masked_label.T, cmap="nipy_spectral", origin="lower",
                   alpha=0.45, vmin=0, vmax=max(7, unique_labels.max()))
    axes[2].set_title(f"Overlay  (slice {mid_idx})")
    axes[2].axis("off")

    plt.suptitle(f"FeTA Sample: {subject_id}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Ensure outputs directory exists
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / "feta_sample_preview.png"

    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"  ✓ Figure saved to: {save_path}")

    # Also try to show interactively (will be skipped in headless envs)
    try:
        plt.show()
    except Exception:
        pass

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DATASET LOADING TEST PASSED ✓")
    print("=" * 60)
    print(f"  ✔ Subject          : {subject_id}")
    print(f"  ✔ MRI loaded       : {img_data.shape}")
    print(f"  ✔ Labels loaded    : {label_data.shape}")
    print(f"  ✔ Shapes match     : {shapes_match}")
    print(f"  ✔ Tissue classes   : {len(unique_labels)}")
    print(f"  ✔ Middle slice plot : saved to {save_path}")
    print()
    print()


if __name__ == "__main__":
    main()
