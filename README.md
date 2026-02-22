# Domain-Generalizable Multi-Tissue Fetal Brain MRI Segmentation

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Pipeline](https://img.shields.io/badge/Pipeline-Verified-success.svg)](#sanity-check)

## Research Problem

Fetal brain MRI segmentation is a critical task in prenatal diagnostics, enabling quantitative analysis of brain development and early detection of neurological abnormalities. However, existing segmentation models often struggle with **domain shift** — performance degradation when applied to data from different scanners, acquisition protocols, or patient populations.

This project addresses the challenge of building **domain-generalizable** segmentation models that maintain robust performance across diverse clinical settings without requiring retraining.

## Objective

Develop a deep learning pipeline for multi-tissue fetal brain MRI segmentation that:

1. **Accurately segments** 7 brain tissue classes
2. **Generalizes across domains** (different scanners, sites, gestational ages)
3. **Provides reproducible** and clinically useful results
4. **Enables research** into domain adaptation and generalization techniques

## Dataset

### FeTA 2.4 Challenge Dataset

The [Fetal Tissue Annotation (FeTA)](https://feta.grand-challenge.org/) dataset provides:

- **3D T2-weighted MRI scans** of the fetal brain
- **Multi-class segmentation labels** with 7 tissue classes
- **Multi-site data** from different hospitals and scanners
- **Range of gestational ages** (typically 20-35 weeks)

**Tissue Classes:**
| Label | Tissue |
|-------|--------|
| 0 | Background |
| 1 | External CSF |
| 2 | Gray Matter |
| 3 | White Matter |
| 4 | Ventricles |
| 5 | Cerebellum |
| 6 | Deep Gray Matter |

## Project Structure

```
fetal-brain-domain-generalization/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataloader.py       # NIfTI loading, Dataset class
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── transforms.py       # Z-score, clipping, cropping
│   ├── models/
│   │   ├── __init__.py
│   │   └── unet3d.py          # 3D U-Net (26M params)
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py         # Training loop, Dice+CE loss
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py         # Multi-class Dice score
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py         # Seed, device utilities
│   └── sanity_check.py        # Pipeline verification script
├── notebooks/
│   └── 01_eda.ipynb           # Exploratory data analysis
├── configs/
│   └── config.yaml            # All hyperparameters
├── outputs/                    # Checkpoints (gitignored)
├── main.py                     # Training entry point
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Druv08/fetal-brain-domain-generalization.git
cd fetal-brain-domain-generalization

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install torch nibabel pyyaml numpy tqdm
```

## Configuration

Edit `configs/config.yaml`:

```yaml
data:
  images_dir: "data/feta_2.4/images"
  labels_dir: "data/feta_2.4/labels"
  num_classes: 7

model:
  in_channels: 1
  out_channels: 7
  features: [32, 64, 128, 256]

training:
  batch_size: 2
  num_epochs: 100
  learning_rate: 0.0001

system:
  num_workers: 0
  device: "cpu"  # or "cuda"
  seed: 42
```

## Sanity Check

Verify the entire pipeline works end-to-end:

```bash
python src/sanity_check.py --config configs/config.yaml --train
```

**Expected Output:**
```
============================================================
FETAL BRAIN MRI SEGMENTATION - SANITY CHECK
============================================================

[1/6] Loading configuration...
  ✓ Config loaded from: configs/config.yaml

[2/6] Creating preprocessing transforms...
  ✓ PreprocessingPipeline created

[3/6] Creating dataset...
  ✓ Dataset created with 1 samples

[4/6] Creating dataloader...
  ✓ DataLoader created

[5/6] Building model...
  ✓ 3D U-Net model built
  ✓ Total parameters: 26,321,671

[6/6] Running forward pass...
  → Image tensor shape: [1, 1, 128, 128, 128]
  → Label tensor shape: [1, 128, 128, 128]
  → Model output shape: [1, 7, 128, 128, 128]
  ✓ All shape checks passed!

============================================================
SANITY CHECK PASSED ✓
============================================================

============================================================
ONE-STEP TRAINING TEST
============================================================

  ✓ Loss value: 2.0496
  ✓ Gradients computed and weights updated!

============================================================
ONE-STEP TRAINING PASSED ✓
============================================================
```

## Usage

### 1. Exploratory Data Analysis

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Training

```bash
python main.py --config configs/config.yaml
```

### 3. Programmatic Usage

```python
from src.data import load_config, FetalBrainDatasetFromConfig
from src.preprocessing import PreprocessingPipeline
from src.models.unet3d import build_unet3d
from src.training import Trainer

# Load config
config = load_config("configs/config.yaml")

# Setup pipeline
transform = PreprocessingPipeline(crop_size=(128, 128, 128))
dataset = FetalBrainDatasetFromConfig(config, transform=transform)
model = build_unet3d(config)

# Train
trainer = Trainer(model, config)
trainer.train(train_loader, val_loader)
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CONFIG (config.yaml)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   DATA LOADING (NIfTI)                       │
│  • Load 3D volumes with nibabel                              │
│  • Discover image/label pairs                                │
│  • Return [1, D, H, W] tensors                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PREPROCESSING                             │
│  • Intensity clipping (0.5-99.5 percentile)                  │
│  • Z-score normalization                                     │
│  • Center crop to [128, 128, 128]                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    3D U-NET MODEL                            │
│  • Input:  [B, 1, 128, 128, 128]                             │
│  • Output: [B, 7, 128, 128, 128]                             │
│  • Parameters: 26,321,671                                    │
│  • Features: [32, 64, 128, 256]                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING                                  │
│  • Loss: Dice + CrossEntropy                                 │
│  • Optimizer: Adam (lr=0.0001)                               │
│  • Validation: Multi-class Dice score                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION                                │
│  • Per-class Dice scores                                     │
│  • Mean Dice across tissues                                  │
└─────────────────────────────────────────────────────────────┘
```

## Model Architecture

**3D U-Net** with encoder-decoder structure:

| Component | Details |
|-----------|---------|
| Input | `[B, 1, D, H, W]` - Single channel MRI |
| Encoder | 4 levels: 32→64→128→256 features |
| Bottleneck | 512 features |
| Decoder | 4 levels with skip connections |
| Output | `[B, 7, D, H, W]` - 7 class logits |
| Parameters | 26,321,671 |

## Results

| Metric | Value |
|--------|-------|
| Image Shape | `[1, 1, 128, 128, 128]` |
| Label Shape | `[1, 128, 128, 128]` |
| Output Shape | `[1, 7, 128, 128, 128]` |
| Initial Loss | ~2.0 |
| Parameters | 26.3M |

## Roadmap

- [x] Project structure setup
- [x] Config-driven pipeline
- [x] Data loading module
- [x] Preprocessing transforms
- [x] 3D U-Net model
- [x] Training loop skeleton
- [x] Evaluation metrics
- [x] Sanity check script
- [x] One-step training verification
- [ ] Full training on FeTA dataset
- [ ] Domain generalization techniques
- [ ] Evaluation on held-out domains

## Citation

```bibtex
@misc{fetal_brain_dg,
  title={Domain-Generalizable Multi-Tissue Fetal Brain MRI Segmentation},
  author={Druv},
  year={2026},
  url={https://github.com/Druv08/fetal-brain-domain-generalization}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [FeTA Challenge](https://feta.grand-challenge.org/) for the dataset
- [PyTorch](https://pytorch.org/) for the deep learning framework
