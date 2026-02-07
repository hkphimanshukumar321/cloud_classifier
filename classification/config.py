# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Experiment Configuration
========================

UPDATED FOR CloudLiteNet-TL + optional Ordinal Learning

Key updates:
- ModelConfig now supports:
    - model_type: 'cloudlitenet_tl' (recommended), 'legacy_rf_cloudnet'
    - backbone: 'MobileNetV2' / 'EfficientNetV2B0'
    - output_mode: 'softmax' / 'ordinal'
    - freeze_backbone, head_dim, weight_decay, use_gem, use_in_model_aug
- TrainingConfig loss supports:
    - 'categorical_crossentropy' (softmax)
    - 'binary_crossentropy' (ordinal)
- Augmentation defaults adjusted for ground-based cloud images:
    - rotation_range reduced
    - vertical_flip default OFF
(If your images are satellite/overhead, you can turn vertical_flip ON and increase rotation_range.)
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
    img_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    validation_split: float = 0.2
    test_split: float = 0.1
    max_images_per_class: Optional[int] = None  # For debugging

    # Augmentation
    use_augmentation: bool = True
    augmentation_strength: str = 'medium'  # 'light', 'medium', 'strong'

    # NOTE:
    # For ground-based sky images, full 180° rotation + vertical flip can be harmful.
    # For satellite images, they can be OK.
    rotation_range: int = 25              # was 180
    zoom_range: float = 0.15              # was 0.2
    horizontal_flip: bool = True
    vertical_flip: bool = False           # was True (set True only if satellite-like)
    brightness_range: Tuple[float, float] = (0.85, 1.15)
    contrast_range: float = 0.2

    # Advanced augmentation
    # MixUp works with BOTH softmax and ordinal (labels become fractional; BCE can handle).
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0


# =============================================================================
# MODEL CONFIG
# =============================================================================

@dataclass
class ModelConfig:
    """
    Model architecture hyperparameters.

    model_type:
      - 'cloudlitenet_tl'      : (RECOMMENDED) Transfer Learning, lightweight + stable on small datasets
      - 'legacy_rf_cloudnet'   : Your older dense/CBAM architecture (kept for ablation/compare)
    """
    # === SELECTION ===
    model_type: str = 'cloudlitenet_tl'

    # === CloudLiteNet-TL params (NEW) ===
    backbone: str = 'MobileNetV2'      # 'MobileNetV2' or 'EfficientNetV2B0'
    freeze_backbone: bool = True      # Stage-1 freeze; then fine-tune (your training code should unfreeze later)
    input_scale: str = "0_1"          # "0_1" if pipeline outputs [0,1]; "0_255" if [0,255]
    output_mode: str = "softmax"      # "softmax" or "ordinal"
    head_dim: int = 128               # 128 good for MobileNetV2
    weight_decay: float = 1e-4
    dropout_rate: float = 0.40
    use_gem: bool = True
    use_in_model_aug: bool = True     # if you already do aug in tf.data, set False

    # Optional label smoothing (ONLY for softmax)
    use_label_smoothing: bool = True
    label_smoothing: float = 0.05

    # === LEGACY RF-CloudNet params (kept for comparison) ===
    growth_rate: int = 12
    compression: float = 0.5
    depth: Tuple[int, int, int, int] = (4, 6, 8, 6)
    initial_filters: int = 64
    use_multi_scale: bool = True
    use_cbam: bool = True
    use_bottleneck: bool = True
    l2_decay: float = 1e-4


# =============================================================================
# TRAINING CONFIG
# =============================================================================

@dataclass
class TrainingConfig(BaseTrainingConfig):
    """Training hyperparameters and experiment flags."""

    # === EXPERIMENT FLAGS ===
    use_multiple_seeds: bool = True
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])

    enable_cross_validation: bool = True
    cv_folds: int = 3

    # SNR testing is not meaningful for vision classification unless you synthetically corrupt images
    enable_snr_testing: bool = False
    snr_levels_db: List[int] = field(default_factory=lambda: [0, 5, 10, 15, 20, 25, 30])

    # === TRAINING PARAMETERS ===
    # For TL on small dataset, you usually do NOT need 150 epochs.
    epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 1e-3

    # LR schedule
    use_warmup: bool = True
    warmup_epochs: int = 3
    lr_schedule: str = 'cosine_warmup'   # 'constant', 'cosine_warmup', 'exponential', 'reduce_on_plateau'
    min_lr: float = 1e-6

    # Optimizer
    optimizer: str = 'adamw'             # 'adam', 'adamw', 'sgd'
    weight_decay: float = 1e-4
    gradient_clip_value: float = 1.0

    # === LOSS FUNCTION ===
    # For softmax: categorical_crossentropy (optionally label smoothing handled in loss)
    # For ordinal: binary_crossentropy
    loss_type: str = 'categorical_crossentropy'  # 'categorical_crossentropy' or 'binary_crossentropy' or 'focal'
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # === CLASS IMBALANCE HANDLING ===
    use_class_weights: bool = True
    class_weight_method: str = 'effective_num'  # 'balanced', 'inverse_freq', 'effective_num'

    # === CALLBACKS ===
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 10
    reduce_lr_factor: float = 0.5

    # === TEST-TIME AUG ===
    use_tta: bool = False
    tta_rounds: int = 5

    # === OPTIONAL: two-stage fine-tune switches (NEW) ===
    # (Your training script must implement these if you want automatic unfreezing)
    two_stage_finetune: bool = True
    head_train_epochs: int = 20
    finetune_epochs: int = 30
    finetune_lr: float = 1e-4
    unfreeze_last_ratio: float = 0.3   # unfreeze last 30% layers of backbone in stage-2


# =============================================================================
# ABLATION / BASELINES / OUTPUT (UNCHANGED MOSTLY)
# =============================================================================

@dataclass
class AblationConfig:
    """Ablation parameters."""
    architecture_variants: List[Dict[str, bool]] = field(default_factory=lambda: [
        {'use_multi_scale': False, 'use_cbam': False, 'use_bottleneck': False},
        {'use_multi_scale': True,  'use_cbam': False, 'use_bottleneck': False},
        {'use_multi_scale': True,  'use_cbam': True,  'use_bottleneck': False},
        {'use_multi_scale': True,  'use_cbam': True,  'use_bottleneck': True},
    ])

    growth_rates: List[int] = field(default_factory=lambda: [8, 12, 16])
    compressions: List[float] = field(default_factory=lambda: [0.4, 0.5, 0.6])
    depths: List[Tuple[int, int, int, int]] = field(default_factory=lambda: [
        (3, 4, 6, 4),
        (4, 6, 8, 6),
        (6, 8, 10, 8),
    ])

    initial_filters_list: List[int] = field(default_factory=lambda: [32, 64, 96])

    loss_functions: List[str] = field(default_factory=lambda: [
        'categorical_crossentropy',
        'focal',
        'binary_crossentropy',
    ])

    augmentation_strengths: List[str] = field(default_factory=lambda: ['light', 'medium', 'strong'])

    batch_sizes: List[int] = field(default_factory=lambda: [16, 32, 64])
    learning_rates: List[float] = field(default_factory=lambda: [1e-3, 5e-4, 1e-4])
    resolutions: List[int] = field(default_factory=lambda: [128, 224, 256])
    dropout_rates: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.4])
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])


@dataclass
class BaselineConfig:
    baseline_models: List[str] = field(default_factory=lambda: [
        'ResNet50V2',
        'DenseNet121',
        'EfficientNetV2B0',
        'MobileNetV2',
    ])
    use_pretrained: bool = True
    freeze_base: bool = True
    unfreeze_after_epochs: int = 20


@dataclass
class OutputConfig(BaseOutputConfig):
    figure_dpi: int = 300
    save_history: bool = True
    save_best_only: bool = True

    save_confusion_matrix: bool = True
    save_classification_report: bool = True
    save_attention_maps: bool = False  # CBAM-only legacy
    save_feature_maps: bool = False


@dataclass
class ResearchConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    experiment_name: str = "CloudLiteNet_TL"
    description: str = "Lightweight Transfer Learning for cloudiness classification"
    version: str = "3.0"


# =============================================================================
# PRESETS
# =============================================================================

def get_quick_test_config() -> ResearchConfig:
    config = ResearchConfig()
    config.data.img_size = (160, 160)
    config.data.max_images_per_class = 50
    config.training.epochs = 5
    config.training.enable_cross_validation = False
    config.training.use_multiple_seeds = False
    config.experiment_name = "Quick_Test"
    return config


def get_recommended_tl_softmax_config() -> ResearchConfig:
    """
    Recommended first run:
    - TL + softmax
    - stable baseline to judge confusion matrix
    """
    config = ResearchConfig()
    config.model.model_type = 'cloudlitenet_tl'
    config.model.backbone = 'MobileNetV2'
    config.model.freeze_backbone = True
    config.model.output_mode = 'softmax'
    config.training.loss_type = 'categorical_crossentropy'
    config.experiment_name = "CloudLiteNet_TL_Softmax"
    return config


def get_recommended_tl_ordinal_config() -> ResearchConfig:
    """
    Ordinal run:
    - TL + ordinal head
    NOTE: your training pipeline must convert labels to ordinal targets.
    """
    config = ResearchConfig()
    config.model.model_type = 'cloudlitenet_tl'
    config.model.backbone = 'MobileNetV2'
    config.model.freeze_backbone = True
    config.model.output_mode = 'ordinal'
    config.training.loss_type = 'binary_crossentropy'
    # label smoothing is not applicable the same way for ordinal; keep it off
    config.model.use_label_smoothing = False
    config.experiment_name = "CloudLiteNet_TL_Ordinal"
    return config


def get_legacy_rf_cloudnet_config() -> ResearchConfig:
    """
    Legacy compare (not recommended for your small dataset).
    """
    config = ResearchConfig()
    config.model.model_type = 'legacy_rf_cloudnet'
    config.experiment_name = "Legacy_RF_CloudNet"
    return config


# =============================================================================
# SUMMARY UTILS
# =============================================================================

def print_experiment_summary(config: ResearchConfig):
    print("=" * 70)
    print(f"EXPERIMENT CONFIGURATION: {config.experiment_name}")
    print("=" * 70)

    print(f"\n[*] Data:")
    print(f"  - Directory: {config.data.data_dir}")
    print(f"  - Image Size: {config.data.img_size}")
    print(f"  - Batch Size (Data): {config.data.batch_size}")
    print(f"  - Augmentation: {config.data.augmentation_strength}")
    print(f"  - Rot/Flip: rot={config.data.rotation_range}, hflip={config.data.horizontal_flip}, vflip={config.data.vertical_flip}")
    print(f"  - Mixup: {config.data.use_mixup} (alpha={config.data.mixup_alpha})")

    print(f"\n[*] Model:")
    print(f"  - Type: {config.model.model_type}")
    print(f"  - Backbone: {config.model.backbone}")
    print(f"  - Freeze backbone: {config.model.freeze_backbone}")
    print(f"  - Output mode: {config.model.output_mode}")
    print(f"  - Head dim: {config.model.head_dim}")
    print(f"  - Dropout: {config.model.dropout_rate}")
    print(f"  - Weight decay: {config.model.weight_decay}")
    print(f"  - GeM pooling: {config.model.use_gem}")

    print(f"\n[*] Training:")
    print(f"  - Epochs: {config.training.epochs}")
    print(f"  - Batch Size (Train): {config.training.batch_size}")
    print(f"  - LR: {config.training.learning_rate} | schedule={config.training.lr_schedule}")
    print(f"  - Optimizer: {config.training.optimizer} (wd={config.training.weight_decay})")
    print(f"  - Loss: {config.training.loss_type}")
    print(f"  - Focal loss: {config.training.use_focal_loss}")
    print(f"  - Class weights: {config.training.use_class_weights} ({config.training.class_weight_method})")
    print(f"  - Two-stage finetune: {config.training.two_stage_finetune} "
          f"(head={config.training.head_train_epochs}, ft={config.training.finetune_epochs}, ft_lr={config.training.finetune_lr})")

    print(f"\n[*] Experiments:")
    print(f"  - Cross-Validation: {config.training.enable_cross_validation} ({config.training.cv_folds} folds)")
    print(f"  - Multiple Seeds: {config.training.use_multiple_seeds} ({len(config.training.seeds)} seeds)")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    cfg = get_recommended_tl_softmax_config()
    print_experiment_summary(cfg)

    cfg_ord = get_recommended_tl_ordinal_config()
    print_experiment_summary(cfg_ord)
