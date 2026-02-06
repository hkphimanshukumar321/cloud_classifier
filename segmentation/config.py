# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Segmentation Configuration
==========================
"""

from dataclasses import dataclass, field
from typing import Tuple
from pathlib import Path
import sys

# Add root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.config_base import BaseTrainingConfig, BaseDataConfig, BaseModelConfig


@dataclass
class SegmentationDataConfig(BaseDataConfig):
    data_dir: Path = Path(__file__).parent.parent / "data" / "segmentation"
    mask_dir: str = None  # Path to masks
    num_classes: int = 3
    img_size: Tuple[int, int] = (128, 128)


@dataclass
class SegmentationModelConfig(BaseModelConfig):
    name: str = "UNet"
    encoder_filters: Tuple[int, ...] = (16, 32, 64, 128)
    num_classes: int = 3


@dataclass
class SegmentationConfig:
    data: SegmentationDataConfig = field(default_factory=SegmentationDataConfig)
    model: SegmentationModelConfig = field(default_factory=SegmentationModelConfig)
    training: BaseTrainingConfig = field(default_factory=BaseTrainingConfig)