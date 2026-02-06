# Segmentation Module

## Quick Start

```bash
cd segmentation
python run.py --quick  # Quick test
python run.py          # Full pipeline
```

## Structure

```
segmentation/
├── config.py              # Segmentation config
├── run.py                 # Master runner
├── src/
│   ├── models.py          # U-Net + SOTA references (SAM, DeepLabV3+)
│   └── __init__.py
├── experiments/
│   └── run_iou_analysis.py  # IoU/Dice per class analysis
└── tests/
```

## Configuration

Edit `config.py` to set:
- `data_dir`: Path to images
- `mask_dir`: Path to segmentation masks
- `img_size`: Resolution (default: 128x128)
- `num_classes`: Number of segmentation classes

## Dataset Format

```
data/segmentation/
├── images/
│   ├── img_001.png
│   └── img_002.png
└── masks/
    ├── img_001.png  # Same name, different folder
    └── img_002.png
```

Masks should be single-channel with class indices (0, 1, 2, ...).

## Models

### Custom U-Net
```python
from src.models import create_unet_model
model = create_unet_model(input_shape=(128, 128, 3), num_classes=3)
```

### SOTA References
| Model | Paper | Install |
|-------|-------|---------|
| SAM | Segment Anything (Meta, 2023) | `pip install segment-anything` |
| DeepLabV3+ | Atrous Convolution (Google, 2018) | TensorFlow built-in |

## Baselines

- U-Net with ResNet encoder
- U-Net with MobileNet encoder
- DeepLabV3+ with MobileNetV2

## Experiments

| Experiment | Command | Output |
|------------|---------|--------|
| IoU Analysis | `python experiments/run_iou_analysis.py` | `results/iou_analysis.csv` |

## Metrics

- IoU (Intersection over Union) per class
- Dice coefficient per class
- Mean IoU (mIoU)
- Pixel accuracy
