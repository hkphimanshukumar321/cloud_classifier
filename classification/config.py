# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Experiment Configuration
========================

Configuration for CloudDenseNet-Lite experiments.

Model: CloudDenseNet-Lite (DS-Dense Blocks + Coordinate Attention)
Output modes: 'softmax' (standard) or 'ordinal' (density levels)
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import sys

# Add parent to path for common imports
sys.path.append(str(Path(__file__).parent.parent))

from common.config_base import BaseTrainingConfig, BaseOutputConfig


# =============================================================================
# DATA CONFIG
# =============================================================================

@dataclass
class DataConfig:
    """Dataset parameters."""
    data_dir: Path = Path(__file__).parent.parent / "data" / "classification" / "raw"
    img_size: Tuple[int, int] = (64, 64)  # Low-res for density estimation
    batch_size: int = 32
    validation_split: float = 0.15
    test_split: float = 0.15
    max_images_per_class: Optional[int] = None

    # Augmentation
    use_augmentation: bool = True
    use_balanced_sampling: bool = True  # Class-aware sampling for imbalance

    # MixUp (works with both softmax and ordinal)
    use_mixup: bool = True
    mixup_alpha: float = 0.2


# =============================================================================
# MODEL CONFIG
# =============================================================================

@dataclass
class ModelConfig:
    """CloudDenseNet-Lite hyperparameters."""
    # Architecture
    growth_rate: int = 12
    compression: float = 0.5
    depth: Tuple[int, ...] = (3, 4, 3)
    initial_filters: int = 24
    dropout_rate: float = 0.30
    weight_decay: float = 1e-4
    use_coord_att: bool = True
    use_in_model_aug: bool = True

    # Output
    output_mode: str = "softmax"  # "softmax" or "ordinal"


# =============================================================================
# TRAINING CONFIG
# =============================================================================

@dataclass
class TrainingConfig(BaseTrainingConfig):
    """Training hyperparameters."""

    # Experiment flags
    use_multiple_seeds: bool = True
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])

    enable_cross_validation: bool = True
    cv_folds: int = 3

    enable_snr_testing: bool = False
    snr_levels_db: List[int] = field(default_factory=lambda: [0, 5, 10, 15, 20, 25, 30])

    # Training parameters
    epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 1e-3

    # LR schedule
    use_warmup: bool = True
    warmup_epochs: int = 3
    lr_schedule: str = 'cosine_warmup'
    min_lr: float = 1e-6

    # Optimizer
    optimizer: str = 'adam'
    weight_decay: float = 1e-4
    gradient_clip_value: float = 1.0

    # Loss
    loss_type: str = 'sparse_categorical_crossentropy'

    # Class imbalance
    use_class_weights: bool = True

    # Callbacks
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 10
    reduce_lr_factor: float = 0.5


# =============================================================================
# ABLATION CONFIG
# =============================================================================

@dataclass
class AblationConfig:
    """Ablation parameters for CloudDenseNet-Lite."""

    # Batch B: Architecture ablation
    growth_rates: List[int] = field(default_factory=lambda: [8, 12, 16])
    compressions: List[float] = field(default_factory=lambda: [0.4, 0.5, 0.6])
    depths: List[Tuple[int, ...]] = field(default_factory=lambda: [
        (2, 3, 2),
        (3, 4, 3),
        (4, 6, 4),
    ])

    # Batch C: Batch size
    batch_sizes: List[int] = field(default_factory=lambda: [16, 32, 64])

    # Batch D: Resolution
    resolutions: List[int] = field(default_factory=lambda: [48, 64, 96])

    # Batch E: Coordinate Attention ablation
    coord_att_options: List[bool] = field(default_factory=lambda: [False, True])

    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])


# =============================================================================
# BASELINE CONFIG
# =============================================================================

@dataclass
class BaselineConfig:
    """Baseline models for comparison (MobileNetV2 transfer learning)."""
    baseline_models: List[str] = field(default_factory=lambda: [
        'MobileNetV2',
        'EfficientNetV2B0',
    ])
    use_pretrained: bool = True
    freeze_base: bool = True


# =============================================================================
# OUTPUT CONFIG
# =============================================================================

@dataclass
class OutputConfig(BaseOutputConfig):
    figure_dpi: int = 300
    save_history: bool = True
    save_best_only: bool = True
    save_confusion_matrix: bool = True
    save_classification_report: bool = True


# =============================================================================
# MASTER CONFIG
# =============================================================================

@dataclass
class ResearchConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    experiment_name: str = "CloudDenseNet_Lite"
    description: str = "Cloud density estimation using lightweight DS-Dense architecture"
    version: str = "4.0"


# =============================================================================
# PRESETS
# =============================================================================

def get_quick_test_config() -> ResearchConfig:
    """Quick smoke test config."""
    config = ResearchConfig()
    config.data.max_images_per_class = 50
    config.training.epochs = 3
    config.training.enable_cross_validation = False
    config.training.use_multiple_seeds = False
    config.experiment_name = "Quick_Test"
    return config


def get_full_run_config() -> ResearchConfig:
    """Full experiment config."""
    config = ResearchConfig()
    config.experiment_name = "CloudDenseNet_Lite_Full"
    return config


# =============================================================================
# SUMMARY
# =============================================================================

def print_experiment_summary(config: ResearchConfig):
    print("=" * 60)
    print(f"EXPERIMENT: {config.experiment_name} (v{config.version})")
    print("=" * 60)

    print(f"\n[Data]")
    print(f"  Directory : {config.data.data_dir}")
    print(f"  Image Size: {config.data.img_size}")
    print(f"  Batch Size: {config.data.batch_size}")
    print(f"  Balanced  : {config.data.use_balanced_sampling}")

    print(f"\n[Model] CloudDenseNet-Lite")
    print(f"  Growth Rate    : {config.model.growth_rate}")
    print(f"  Compression    : {config.model.compression}")
    print(f"  Depth          : {config.model.depth}")
    print(f"  Initial Filters: {config.model.initial_filters}")
    print(f"  Coord Attention: {config.model.use_coord_att}")
    print(f"  Dropout        : {config.model.dropout_rate}")
    print(f"  Output Mode    : {config.model.output_mode}")

    print(f"\n[Training]")
    print(f"  Epochs     : {config.training.epochs}")
    print(f"  LR         : {config.training.learning_rate}")
    print(f"  Optimizer  : {config.training.optimizer}")
    print(f"  Loss       : {config.training.loss_type}")
    print(f"  Class Wts  : {config.training.use_class_weights}")
    print(f"  CV Folds   : {config.training.cv_folds} (enabled={config.training.enable_cross_validation})")
    print(f"  Seeds      : {config.training.seeds}")
    print("=" * 60)


if __name__ == "__main__":
    cfg = ResearchConfig()
    print_experiment_summary(cfg)
