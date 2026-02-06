# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Base Configuration Module
=========================

Shared configuration dataclasses for all tasks.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class BaseDataConfig:
    """Base dataset configuration."""
    data_dir: str = None
    img_size: Tuple[int, int] = (64, 64)
    batch_size: int = 32
    val_split: float = 0.2
    test_split: float = 0.1
    num_workers: int = 4


@dataclass
class BaseModelConfig:
    """Base model configuration."""
    name: str = "BaseModel"
    num_classes: int = 10
    pretrained: bool = True
    freeze_backbone: bool = False


@dataclass
class BaseTrainingConfig:
    """Universal training parameters."""
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    
    # Reproducibility
    use_multiple_seeds: bool = True
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    
    # Common flags
    enable_cross_validation: bool = False
    cv_folds: int = 5


@dataclass
class BaseOutputConfig:
    """Universal output settings."""
    results_dir: Path = Path("results")
    experiment_name: str = "default_exp"
    save_models: bool = True