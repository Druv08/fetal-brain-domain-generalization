# Domain-Generalizable Multi-Tissue Fetal Brain MRI Segmentation

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.3+-green.svg)](https://monai.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Research Problem

Fetal brain MRI segmentation is a critical task in prenatal diagnostics, enabling quantitative analysis of brain development and early detection of neurological abnormalities. However, existing segmentation models often struggle with **domain shift** — performance degradation when applied to data from different scanners, acquisition protocols, or patient populations.

This project addresses the challenge of building **domain-generalizable** segmentation models that maintain robust performance across diverse clinical settings without requiring retraining.

## Objective

Develop a deep learning pipeline for multi-tissue fetal brain MRI segmentation that:

1. **Accurately segments** multiple brain tissue classes (white matter, gray matter, CSF, etc.)
2. **Generalizes across domains** (different scanners, sites, gestational ages)
3. **Provides reproducible** and clinically useful results
4. **Enables research** into domain adaptation and generalization techniques

## Dataset

### FeTA 2.4 Challenge Dataset

The [Fetal Tissue Annotation (FeTA)](https://feta.grand-challenge.org/) dataset provides:

- **3D T2-weighted MRI scans** of the fetal brain
- **Multi-class segmentation labels** with 7+ tissue classes
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
| 7 | Brainstem |

## Methodology

### Pipeline Overview

```
Data Loading → Preprocessing → Augmentation → 3D U-Net → Training → Evaluation
```

### Key Components

1. **Data Pipeline**: NIfTI loading, normalization, and PyTorch Dataset integration
2. **Preprocessing**: Z-score normalization, intensity clipping, resampling
3. **Model**: 3D U-Net architecture with configurable depth and channels
4. **Training**: Dice loss optimization, learning rate scheduling
5. **Evaluation**: Multi-class Dice score, per-tissue analysis

### Domain Generalization Strategies (Planned)

- Data augmentation for robustness
- Domain-invariant feature learning
- Test-time adaptation techniques

## Project Structure

```
FeTa/
├── src/
│   ├── data/              # Data loading and dataset classes
│   ├── preprocessing/     # Image preprocessing transforms
│   ├── models/            # Neural network architectures
│   ├── training/          # Training loops and utilities
│   ├── evaluation/        # Metrics and evaluation functions
│   └── utils/             # Helper functions
├── notebooks/             # Jupyter notebooks for EDA
├── outputs/               # Model checkpoints and results
├── configs/               # Configuration files
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/FeTa.git
cd FeTa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Configure Paths

Edit `configs/config.yaml` with your dataset paths:

```yaml
data:
  data_dir: "/path/to/feta_dataset"
  train_dir: "/path/to/feta_dataset/train"
  val_dir: "/path/to/feta_dataset/val"
```

### 2. Exploratory Data Analysis

```bash
# Run the EDA notebook
jupyter notebook notebooks/01_eda.ipynb
```

### 3. Train the Model

```python
from src.data import FetalBrainDataset
from src.models import UNet3D
from src.training import Trainer

# Load configuration
import yaml
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize components
dataset = FetalBrainDataset(config["data"]["train_dir"])
model = UNet3D(in_channels=1, out_channels=config["model"]["num_classes"])
trainer = Trainer(model, config)

# Train
trainer.train(dataset)
```

### 4. Evaluate

```python
from src.evaluation import dice_score_multiclass

# Load model and predict
predictions = model(test_images)
dice_scores = dice_score_multiclass(predictions, labels, num_classes=8)
print(f"Mean Dice: {dice_scores.mean():.4f}")
```

## Running Order

1. **Setup**: Install requirements and configure paths
2. **EDA**: Run `notebooks/01_eda.ipynb` to understand the data
3. **Preprocessing**: Verify preprocessing functions work correctly
4. **Training**: Train the 3D U-Net model
5. **Evaluation**: Evaluate on validation/test sets

## Results

*Results will be added as experiments progress.*

| Metric | Score |
|--------|-------|
| Mean Dice | TBD |
| WM Dice | TBD |
| GM Dice | TBD |

## Citation

If you use this code, please cite:

```bibtex
@misc{feta_segmentation,
  title={Domain-Generalizable Multi-Tissue Fetal Brain MRI Segmentation},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/FeTa}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FeTA Challenge](https://feta.grand-challenge.org/) for the dataset
- [MONAI](https://monai.io/) for medical imaging tools
- [PyTorch](https://pytorch.org/) for the deep learning framework
