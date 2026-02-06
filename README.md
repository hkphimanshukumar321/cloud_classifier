# Omni Workbench


Multi-task machine learning template supporting **Classification**, **Segmentation**, and **Detection**.

## Choose Your Task

| Task | Use Case | Go To |
|------|----------|-------|
| **Classification** | Categorize images into classes | [classification/](classification/) |
| **Segmentation** | Pixel-level labeling (masks) | [segmentation/](segmentation/) |
| **Detection** | Locate objects with bounding boxes | [detection/](detection/) |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Choose your task and run
cd classification && python run.py --quick
# OR
cd segmentation && python run.py --quick
# OR
cd detection && python run.py --quick
```

## Directory Structure

```
omni_workbench_MLsuite/
├── common/                  # Shared utilities
│   ├── experiments/         # Base experiment runners
│   ├── data_loader.py       # Image loading & preprocessing
│   ├── visualization.py     # Journal-quality plots
│   ├── hardware.py          # GPU setup
│   └── logger.py            # Experiment logging
│
├── data/                    # [SHARED] Dataset Root
│   ├── classification/      # -> Put class folders in 'raw/'
│   ├── segmentation/        # -> Put 'images/' and 'masks/'
│   └── detection/           # -> Put 'images/' and 'labels/'
│
├── classification/          # Image Classification
│   ├── README.md            # <- Start here
│   ├── config.py
│   ├── run.py
│   ├── src/
│   └── experiments/
│
├── segmentation/            # Semantic Segmentation
│   ├── README.md            # <- Start here
│   ├── config.py
│   ├── run.py
│   ├── src/
│   └── experiments/
│
├── detection/               # Object Detection
│   ├── README.md            # <- Start here
│   ├── config.py
│   ├── run.py
│   ├── src/
│   └── experiments/
│
└── requirements.txt
```

## Workflow

1. **Choose task** → Read the task-specific README
2. **Prepare data** → Follow dataset format in README
3. **Configure** → Edit `config.py` (paths, hyperparams)
4. **Train** → Run `python run.py`
5. **Evaluate** → Run experiments, compare baselines
6. **Publish** → Generate journal-quality figures

## Common Features (All Tasks)

| Feature | Location | Description |
|---------|----------|-------------|
| Ablation Study | `common/experiments/` | Hyperparameter search |
| Cross-Validation | `common/experiments/` | K-fold statistical validation |
| Baseline Comparison | Task-specific | Compare with pretrained models |
| Journal Plots | `common/visualization.py` | Radar, Pareto, confusion matrix |
| GPU Setup | `common/hardware.py` | Auto-detect, multi-GPU |

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- NumPy, Pandas, Matplotlib, Seaborn
- OpenCV, scikit-learn, tqdm
