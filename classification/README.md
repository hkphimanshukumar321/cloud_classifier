# CloudDenseNet-Lite Classification Module

## Quick Start (Run on Server)

**1. Full Research Pipeline (Recommended)**
Runs ablation study, cross-validation, and baseline comparisons in one go.

```bash
cd classification
python run.py
```

**2. Individual Experiments**

*   **Ablation Study** (Hyperparameter search: 35 configs × 3 seeds):
    ```bash
    python experiments/run_ablation.py
    ```

*   **Cross-Validation** (Robustness check):
    ```bash
    python experiments/run_cross_validation.py --folds 5
    ```

*   **Baseline Comparison** (vs MobileNetV2, EfficientNet):
    ```bash
    python experiments/run_baselines.py
    ```

## Configuration

Edit `classification/config.py` to change parameters:
- `data_dir`: Path to dataset (default: `../data/classification/raw`)
- `img_size`: `(64, 64)` (optimized for speed)
- `batch_size`: `32`
- `epochs`: `80`

## Key Features

- **Model**: `CloudDenseNet-Lite` (Novel architecture)
- **Input**: 64x64 RGB images
- **Output**: 5 density levels (Very Low → Very High)
- **Technique**: DS-Dense Blocks + Coordinate Attention + Ordinal Regression
