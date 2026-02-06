# Classification Module

## Quick Start

```bash
cd classification
python run.py --quick  # Quick test
python run.py          # Full pipeline
```

## Structure

```
classification/
├── config.py              # Configuration (data paths, hyperparams)
├── run.py                 # Master runner
├── src/
│   ├── models.py          # Custom CNN + Baseline models
│   ├── data_loader.py     # Image loading & preprocessing
│   ├── training.py        # Training loop
│   └── visualization.py   # Plots & metrics
├── experiments/
│   ├── run_ablation.py    # Hyperparameter search
│   ├── run_baselines.py   # Compare with pretrained models
│   ├── run_cross_validation.py
│   └── run_snr_robustness.py  # RF-specific SNR testing
└── tests/
```

## Configuration

Edit `config.py` to set:
- `data_dir`: Path to your image dataset
- `img_size`: Image resolution (default: 64x64)
- `epochs`, `batch_size`, `learning_rate`

## Dataset Format

```
data/
├── class_0/
│   ├── image1.jpg
│   └── image2.png
├── class_1/
│   └── ...

### Importing Flat Data (Excel/CSV)

If you have a flat folder of images and an Excel/CSV file with labels:

```bash
python tools/organize_data.py \
  --csv "path/to/labels.xlsx" \
  --images "path/to/flat_images_folder" \
  --file_col "filename" \
  --label_col "label"
```

This will automatically sort your images into the correct folders in `data/classification/raw`.

### Option 2: Label from Filename (No Excel)

If your files are named like `cat_01.jpg`, `dog_002.jpg` (label is the first part):

```bash
python tools/organize_data.py \
  --images "path/to/images" \
  --delimiter "_" \
  --index 0
```

```

## Baselines

Available pretrained models (transfer learning):
- MobileNetV2
- EfficientNetV2B0
- ResNet50V2
- DenseNet121
- VGG16

Run: `python experiments/run_baselines.py`

## Experiments

| Experiment | Command | Output |
|------------|---------|--------|
| Ablation | `python experiments/run_ablation.py` | `results/ablation_*.csv` |
| Cross-Val | `python experiments/run_cross_validation.py` | `results/cv_*.csv` |
| Baselines | `python experiments/run_baselines.py` | `results/baselines.csv` |
| SNR Test | `python experiments/run_snr_robustness.py` | `results/snr_*.csv` |

## Metrics

- Accuracy, Precision, Recall, F1
- Confusion Matrix
- Per-class performance
- Training curves
