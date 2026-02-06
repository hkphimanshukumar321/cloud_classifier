# Detection Module

## Quick Start

```bash
cd detection
python run.py --quick  # Quick test
python run.py          # Full pipeline
```

## Structure

```
detection/
├── config.py              # Detection config
├── run.py                 # Master runner
├── src/
│   ├── models.py          # SSD-lite + SOTA references (YOLOv8, RT-DETR)
│   └── __init__.py
├── experiments/
│   └── run_map_analysis.py  # mAP analysis at different IoU thresholds
└── tests/
```

## Configuration

Edit `config.py` to set:
- `data_dir`: Path to images
- `annotations_dir`: Path to bounding box annotations
- `img_size`: Resolution (default: 416x416)
- `num_classes`: Number of object classes
- `anchors`: Anchor box sizes
- `confidence_threshold`, `nms_threshold`

## Dataset Formats

### YOLO Format
```
data/detection/
├── images/
│   ├── img_001.jpg
│   └── img_002.jpg
└── labels/
    ├── img_001.txt  # class x_center y_center width height
    └── img_002.txt
```

### COCO Format (JSON)
```
annotations/
└── instances.json
```

## Models

### Custom SSD-Lite
```python
from src.models import create_detection_model
model = create_detection_model(input_shape=(416, 416, 3), num_classes=10)
```

### SOTA References
| Model | Paper | Install |
|-------|-------|---------|
| YOLOv8 | Ultralytics (2023) | `pip install ultralytics` |
| RT-DETR | DETRs Beat YOLOs (2023) | `pip install ultralytics` |
| EfficientDet | Compound Scaling (2020) | TensorFlow Model Garden |

## Baselines

- YOLOv8n (nano) - fastest
- YOLOv8s (small) - balanced
- RT-DETR-L - highest accuracy

## Experiments

| Experiment | Command | Output |
|------------|---------|--------|
| mAP Analysis | `python experiments/run_map_analysis.py` | `results/map_analysis.csv` |

## Metrics

- mAP@0.5 (PASCAL VOC style)
- mAP@0.75 (strict IoU)
- mAP@0.5:0.95 (COCO style)
- Precision, Recall per class
- Inference latency (FPS)
