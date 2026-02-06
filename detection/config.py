# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Detection Configuration
=======================
"""

from dataclasses import dataclass, field
from typing import Tuple, List
from pathlib import Path
import sys

# Add root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.config_base import BaseTrainingConfig, BaseDataConfig, BaseModelConfig


@dataclass
class DetectionDataConfig(BaseDataConfig):
    data_dir: Path = Path(__file__).parent.parent / "data" / "detection"
    annotations_dir: str = None  # Path to annotations
    format: str = "yolo"  # yolo or coco
    num_classes: int = 10
    img_size: Tuple[int, int] = (416, 416)


@dataclass
class DetectionModelConfig(BaseModelConfig):
    name: str = "SSD_Lite"
    anchors: List[Tuple[int, int]] = field(
        default_factory=lambda: [(10,13), (16,30), (33,23)]
    )
    num_classes: int = 10
    nms_threshold: float = 0.5
    confidence_threshold: float = 0.5


@dataclass
class DetectionConfig:
    data: DetectionDataConfig = field(default_factory=DetectionDataConfig)
    model: DetectionModelConfig = field(default_factory=DetectionModelConfig)
    training: BaseTrainingConfig = field(default_factory=BaseTrainingConfig)