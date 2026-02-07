# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Experiment Configuration
========================

Centralized configuration management for reproducibility.
Updated with ADVANCED RF-CloudNet configurations for >95% accuracy target.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import sys

# Add parent to path for common imports
sys.path.append(str(Path(__file__).parent.parent))

from common.config_base import BaseTrainingConfig, BaseOutputConfig


@dataclass
class DataConfig:
    """Dataset parameters."""
    data_dir: Path = Path(__file__).parent.parent / "data" / "classification" / "raw"
    img_size: Tuple[int, int] = (224, 224)  # UPDATED: Larger for better accuracy (was 64x64)
    batch_size: int = 32
    validation_split: float = 0.2
    test_split: float = 0.1
    max_images_per_class: Optional[int] = None  # For debugging
    
    # Augmentation (ENHANCED)
    use_augmentation: bool = True
    augmentation_strength: str = 'medium'  # NEW: 'light', 'medium', 'strong'
    rotation_range: int = 180  # UPDATED: Full rotation (clouds are rotation-invariant)
    zoom_range: float = 0.2
    horizontal_flip: bool = True
    vertical_flip: bool = True  # NEW: Satellite images benefit from both flips
    brightness_range: Tuple[float, float] = (0.8, 1.2)  # NEW: Photometric augmentation
    contrast_range: float = 0.3  # NEW
    
    # Advanced augmentation (NEW)
    use_mixup: bool = True  # NEW: Mixup augmentation
    mixup_alpha: float = 0.2  # NEW
    use_cutmix: bool = False  # NEW: Alternative to mixup (use one or the other)
    cutmix_alpha: float = 1.0  # NEW


@dataclass
class ModelConfig:
    """Model architecture hyperparameters (UPDATED FOR ADVANCED RF-CloudNet)."""
    
    # === ARCHITECTURE SELECTION ===
    model_type: str = 'rf_cloudnet'  # NEW: 'original', 'rf_cloudnet', 'lightweight', 'heavy'
    
    # === CUSTOM RF-CloudNet PARAMETERS ===
    # Dense block configuration
    growth_rate: int = 12      # UPDATED: More filters (was 8)
    compression: float = 0.5   # Keep same
    depth: Tuple[int, int, int, int] = (4, 6, 8, 6)  # UPDATED: 4 blocks, deeper (was 3,3,3)
    
    # Initial feature extraction
    initial_filters: int = 64  # UPDATED: Richer features (was 16)
    
    # Advanced features (NEW)
    use_multi_scale: bool = True      # NEW: Multi-scale Inception stem
    use_cbam: bool = True              # NEW: CBAM attention modules
    use_bottleneck: bool = True        # NEW: Bottleneck layers for efficiency
    
    # Regularization (ENHANCED)
    dropout_rate: float = 0.3          # UPDATED: Stronger (was 0.2)
    l2_decay: float = 1e-4             # Keep same
    use_label_smoothing: bool = True   # NEW: Label smoothing
    label_smoothing: float = 0.1       # NEW


@dataclass
class TrainingConfig(BaseTrainingConfig):
    """Training hyperparameters and experiment flags (ENHANCED)."""
    
    # === EXPERIMENT ENABLE FLAGS ===
    # Multiple seeds for statistical significance
    use_multiple_seeds: bool = True  # True = 3 seeds
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    
    # Cross-validation (3-fold)
    enable_cross_validation: bool = True
    cv_folds: int = 3
    
    # SNR robustness testing
    enable_snr_testing: bool = True
    snr_levels_db: List[int] = field(default_factory=lambda: [0, 5, 10, 15, 20, 25, 30])
    
    # === TRAINING PARAMETERS (UPDATED) ===
    epochs: int = 150          # UPDATED: More epochs for convergence (was 40)
    batch_size: int = 32       # UPDATED: Larger batches (was 5)
    learning_rate: float = 1e-3  # Keep same
    
    # Learning rate schedule (NEW)
    use_warmup: bool = True              # NEW: LR warmup
    warmup_epochs: int = 5               # NEW
    lr_schedule: str = 'cosine_warmup'   # NEW: 'constant', 'cosine_warmup', 'exponential', 'reduce_on_plateau'
    min_lr: float = 1e-6                 # NEW: Minimum LR for cosine decay
    
    # Optimizer settings (NEW)
    optimizer: str = 'adamw'             # NEW: 'adam', 'adamw', 'sgd'
    weight_decay: float = 1e-4           # NEW: For AdamW
    gradient_clip_value: float = 1.0     # Keep same
    
    # === LOSS FUNCTION (NEW) ===
    loss_type: str = 'combined'          # NEW: 'categorical_crossentropy', 'focal', 'label_smoothing', 'combined'
    use_focal_loss: bool = True          # NEW: Handle class imbalance
    focal_alpha: float = 0.25            # NEW
    focal_gamma: float = 2.0             # NEW
    
    # === CLASS IMBALANCE HANDLING (NEW) ===
    use_class_weights: bool = True       # NEW: Compute class weights
    class_weight_method: str = 'effective_num'  # NEW: 'balanced', 'inverse_freq', 'effective_num'
    
    # === CALLBACKS (UPDATED) ===
    early_stopping_patience: int = 20    # UPDATED: More patience (was 15)
    reduce_lr_patience: int = 5          # NEW: ReduceLROnPlateau patience
    reduce_lr_factor: float = 0.5        # NEW
    
    # === TEST-TIME AUGMENTATION (NEW) ===
    use_tta: bool = True                 # NEW: Test-time augmentation
    tta_rounds: int = 10                 # NEW: Number of TTA iterations


@dataclass
class AblationConfig:
    """
    Ablation parameters for grouped study (UPDATED FOR RF-CloudNet).
    
    Study groups:
    -------------
    A. Architecture Components (Ablation)
       - Multi-scale stem: ON/OFF
       - CBAM attention: ON/OFF  
       - Bottleneck: ON/OFF
       - Depth variations
    
    B. Dense Block Parameters
       - Growth rates: [8, 12, 16]
       - Compressions: [0.4, 0.5, 0.6]
       - Depths: [(3,4,6,4), (4,6,8,6), (6,8,10,8)]
    
    C. Training Strategies
       - Loss functions: [CE, Focal, Combined]
       - Augmentation: [Light, Medium, Strong]
       - Batch sizes: [16, 32, 64]
    
    D. Input Resolution
       - Resolutions: [128, 224, 256]
    """
    
    # === ARCHITECTURE COMPONENTS (NEW - for ablation) ===
    architecture_variants: List[Dict[str, bool]] = field(default_factory=lambda: [
        # Baseline (original RF-DenseNet)
        {'use_multi_scale': False, 'use_cbam': False, 'use_bottleneck': False},
        # Progressive additions
        {'use_multi_scale': True,  'use_cbam': False, 'use_bottleneck': False},
        {'use_multi_scale': True,  'use_cbam': True,  'use_bottleneck': False},
        {'use_multi_scale': True,  'use_cbam': True,  'use_bottleneck': True},  # Full
    ])
    
    # === DENSE BLOCK PARAMETERS (UPDATED) ===
    growth_rates: List[int] = field(default_factory=lambda: [8, 12, 16])  # UPDATED: Added 16
    compressions: List[float] = field(default_factory=lambda: [0.4, 0.5, 0.6])  # UPDATED
    depths: List[Tuple[int, int, int, int]] = field(default_factory=lambda: [
        (3, 4, 6, 4),   # Lightweight
        (4, 6, 8, 6),   # Standard (recommended)
        (6, 8, 10, 8),  # Heavy
    ])
    
    # === INITIAL FILTERS (NEW) ===
    initial_filters_list: List[int] = field(default_factory=lambda: [32, 64, 96])
    
    # === TRAINING STRATEGIES (NEW) ===
    loss_functions: List[str] = field(default_factory=lambda: [
        'categorical_crossentropy',
        'focal',
        'label_smoothing', 
        'combined'
    ])
    
    augmentation_strengths: List[str] = field(default_factory=lambda: [
        'light', 'medium', 'strong'
    ])
    
    # === TRAINING PARAMETERS ===
    batch_sizes: List[int] = field(default_factory=lambda: [16, 32, 64])  # UPDATED
    learning_rates: List[float] = field(default_factory=lambda: [1e-3, 5e-4, 1e-4])  # UPDATED
    
    # === INPUT RESOLUTION (UPDATED) ===
    resolutions: List[int] = field(default_factory=lambda: [128, 224, 256])  # UPDATED: Larger
    
    # === REGULARIZATION (NEW) ===
    dropout_rates: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.4])
    
    # Multiple seeds
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])


@dataclass
class BaselineConfig:
    """
    Configuration for baseline model comparisons (NEW).
    
    Baselines to compare against:
    - ResNet50V2
    - DenseNet121
    - EfficientNetV2B0
    - MobileNetV2
    - Original RF-DenseNet
    """
    baseline_models: List[str] = field(default_factory=lambda: [
        'ResNet50V2',
        'DenseNet121', 
        'EfficientNetV2B0',
        'MobileNetV2',
    ])
    
    use_pretrained: bool = True  # Use ImageNet weights
    freeze_base: bool = True     # Freeze base, train only head initially
    unfreeze_after_epochs: int = 20  # Fine-tune after N epochs


@dataclass
class OutputConfig(BaseOutputConfig):
    """Output directories and settings (UPDATED)."""
    figure_dpi: int = 300  # UPDATED: Higher quality (was 30)
    save_history: bool = True
    save_best_only: bool = True  # NEW: Only save best model
    
    # Additional outputs (NEW)
    save_confusion_matrix: bool = True
    save_classification_report: bool = True
    save_attention_maps: bool = True  # NEW: Visualize CBAM attention
    save_feature_maps: bool = True    # NEW: Visualize learned features


@dataclass
class ResearchConfig:
    """Master configuration (UPDATED)."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)  # NEW
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Experiment metadata (NEW)
    experiment_name: str = "RF_CloudNet_Advanced"
    description: str = "Advanced RF-CloudNet for cloud classification (>95% accuracy target)"
    version: str = "2.0"


# =============================================================================
# PRESET CONFIGURATIONS (NEW)
# =============================================================================

def get_quick_test_config() -> ResearchConfig:
    """Quick configuration for testing/debugging."""
    config = ResearchConfig()
    config.data.img_size = (64, 64)
    config.data.max_images_per_class = 100
    config.training.epochs = 5
    config.training.enable_cross_validation = False
    config.training.enable_snr_testing = False
    config.training.use_multiple_seeds = False
    config.experiment_name = "Quick_Test"
    return config


def get_original_config() -> ResearchConfig:
    """Configuration matching original RF-DenseNet."""
    config = ResearchConfig()
    config.data.img_size = (64, 64)
    config.model.model_type = 'original'
    config.model.growth_rate = 8
    config.model.depth = (3, 3, 3)
    config.model.initial_filters = 16
    config.model.use_multi_scale = False
    config.model.use_cbam = False
    config.model.use_bottleneck = False
    config.model.dropout_rate = 0.2
    config.training.epochs = 40
    config.training.use_focal_loss = False
    config.training.use_class_weights = False
    config.experiment_name = "Original_RF_DenseNet"
    return config


def get_advanced_config() -> ResearchConfig:
    """Configuration for advanced RF-CloudNet (>95% accuracy target)."""
    config = ResearchConfig()
    config.data.img_size = (224, 224)
    config.data.use_mixup = True
    config.model.model_type = 'rf_cloudnet'
    config.model.growth_rate = 12
    config.model.depth = (4, 6, 8, 6)
    config.model.initial_filters = 64
    config.model.use_multi_scale = True
    config.model.use_cbam = True
    config.model.use_bottleneck = True
    config.model.dropout_rate = 0.3
    config.training.epochs = 150
    config.training.use_focal_loss = True
    config.training.use_class_weights = True
    config.training.use_tta = True
    config.experiment_name = "Advanced_RF_CloudNet"
    return config


def get_lightweight_config() -> ResearchConfig:
    """Lightweight configuration for faster training."""
    config = ResearchConfig()
    config.data.img_size = (128, 128)
    config.model.model_type = 'lightweight'
    config.model.growth_rate = 8
    config.model.depth = (3, 4, 6, 4)
    config.model.initial_filters = 32
    config.model.use_multi_scale = True
    config.model.use_cbam = False
    config.model.use_bottleneck = True
    config.model.dropout_rate = 0.3
    config.training.epochs = 100
    config.training.batch_size = 64
    config.experiment_name = "Lightweight_RF_CloudNet"
    return config


def get_heavy_config() -> ResearchConfig:
    """Heavy configuration for maximum accuracy."""
    config = ResearchConfig()
    config.data.img_size = (256, 256)
    config.model.model_type = 'heavy'
    config.model.growth_rate = 16
    config.model.depth = (6, 8, 10, 8)
    config.model.initial_filters = 96
    config.model.use_multi_scale = True
    config.model.use_cbam = True
    config.model.use_bottleneck = True
    config.model.dropout_rate = 0.4
    config.training.epochs = 200
    config.training.batch_size = 16
    config.training.use_tta = True
    config.experiment_name = "Heavy_RF_CloudNet"
    return config


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_experiment_summary(config: ResearchConfig):
    """Print configuration summary (UPDATED)."""
    print("=" * 70)
    print(f"EXPERIMENT CONFIGURATION: {config.experiment_name}")
    print("=" * 70)
    
    print(f"\n[*] Data:")
    print(f"  - Directory: {config.data.data_dir}")
    print(f"  - Image Size: {config.data.img_size}")
    print(f"  - Batch Size: {config.data.batch_size}")
    print(f"  - Augmentation: {config.data.augmentation_strength}")
    print(f"  - Mixup: {config.data.use_mixup}")
    
    print(f"\n[*] Model Architecture:")
    print(f"  - Type: {config.model.model_type}")
    print(f"  - Growth Rate: {config.model.growth_rate}")
    print(f"  - Depth: {config.model.depth}")
    print(f"  - Initial Filters: {config.model.initial_filters}")
    print(f"  - Multi-scale Stem: {config.model.use_multi_scale}")
    print(f"  - CBAM Attention: {config.model.use_cbam}")
    print(f"  - Bottleneck Layers: {config.model.use_bottleneck}")
    print(f"  - Dropout: {config.model.dropout_rate}")
    
    print(f"\n[*] Training:")
    print(f"  - Epochs: {config.training.epochs}")
    print(f"  - Batch Size: {config.training.batch_size}")
    print(f"  - Learning Rate: {config.training.learning_rate}")
    print(f"  - LR Schedule: {config.training.lr_schedule}")
    print(f"  - Optimizer: {config.training.optimizer}")
    print(f"  - Loss Function: {config.training.loss_type}")
    print(f"  - Focal Loss: {config.training.use_focal_loss}")
    print(f"  - Class Weights: {config.training.use_class_weights}")
    print(f"  - Test-Time Aug: {config.training.use_tta}")
    
    print(f"\n[*] Experiments:")
    print(f"  - Cross-Validation: {config.training.enable_cross_validation} ({config.training.cv_folds} folds)")
    print(f"  - SNR Robustness: {config.training.enable_snr_testing}")
    print(f"  - Multiple Seeds: {config.training.use_multiple_seeds} ({len(config.training.seeds)} seeds)")
    
    print(f"\n[*] Ablation Study:")
    print(f"  - Architecture Variants: {len(config.ablation.architecture_variants)}")
    print(f"  - Growth Rates: {config.ablation.growth_rates}")
    print(f"  - Depths: {len(config.ablation.depths)}")
    print(f"  - Loss Functions: {config.ablation.loss_functions}")
    print(f"  - Batch Sizes: {config.ablation.batch_sizes}")
    print(f"  - Resolutions: {config.ablation.resolutions}")
    
    # Calculate total experiments
    total_arch = len(config.ablation.architecture_variants)
    total_dense = (len(config.ablation.growth_rates) * 
                   len(config.ablation.compressions) * 
                   len(config.ablation.depths))
    total_training = (len(config.ablation.loss_functions) * 
                      len(config.ablation.augmentation_strengths) *
                      len(config.ablation.batch_sizes))
    total_resolution = len(config.ablation.resolutions)
    total_seeds = len(config.ablation.seeds)
    
    total_experiments = (total_arch + total_dense + total_training + total_resolution) * total_seeds
    
    print(f"\n[*] Total Ablation Experiments:")
    print(f"  - Architecture: {total_arch} configs")
    print(f"  - Dense Params: {total_dense} configs")
    print(f"  - Training: {total_training} configs")
    print(f"  - Resolution: {total_resolution} configs")
    print(f"  - Seeds: {total_seeds}")
    print(f"  - TOTAL: {total_experiments} experiments")
    
    print("=" * 70)


def get_ablation_config_by_name(name: str) -> ResearchConfig:
    """Get a specific ablation configuration by name."""
    configs = {
        'quick_test': get_quick_test_config,
        'original': get_original_config,
        'advanced': get_advanced_config,
        'lightweight': get_lightweight_config,
        'heavy': get_heavy_config,
    }
    
    if name not in configs:
        raise ValueError(f"Unknown config: {name}. Choose from: {list(configs.keys())}")
    
    return configs[name]()


def compare_configs(config1: ResearchConfig, config2: ResearchConfig):
    """Compare two configurations side-by-side."""
    print("=" * 70)
    print("CONFIGURATION COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Parameter':<30} {'Config 1':<20} {'Config 2':<20}")
    print("-" * 70)
    
    # Model comparison
    print(f"{'Model Type':<30} {config1.model.model_type:<20} {config2.model.model_type:<20}")
    print(f"{'Growth Rate':<30} {config1.model.growth_rate:<20} {config2.model.growth_rate:<20}")
    print(f"{'Depth':<30} {str(config1.model.depth):<20} {str(config2.model.depth):<20}")
    print(f"{'Initial Filters':<30} {config1.model.initial_filters:<20} {config2.model.initial_filters:<20}")
    print(f"{'Multi-scale':<30} {config1.model.use_multi_scale:<20} {config2.model.use_multi_scale:<20}")
    print(f"{'CBAM':<30} {config1.model.use_cbam:<20} {config2.model.use_cbam:<20}")
    
    # Training comparison
    print(f"{'Epochs':<30} {config1.training.epochs:<20} {config2.training.epochs:<20}")
    print(f"{'Batch Size':<30} {config1.training.batch_size:<20} {config2.training.batch_size:<20}")
    print(f"{'Focal Loss':<30} {config1.training.use_focal_loss:<20} {config2.training.use_focal_loss:<20}")
    print(f"{'TTA':<30} {config1.training.use_tta:<20} {config2.training.use_tta:<20}")
    
    # Data comparison
    print(f"{'Image Size':<30} {str(config1.data.img_size):<20} {str(config2.data.img_size):<20}")
    print(f"{'Mixup':<30} {config1.data.use_mixup:<20} {config2.data.use_mixup:<20}")
    
    print("=" * 70)


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RF-CloudNet Configuration System")
    print("=" * 70)
    
    # Example 1: Default advanced config
    print("\n1. Advanced RF-CloudNet Configuration:")
    config = get_advanced_config()
    print_experiment_summary(config)
    
    # Example 2: Original for comparison
    print("\n2. Original RF-DenseNet Configuration:")
    config_original = get_original_config()
    print_experiment_summary(config_original)
    
    # Example 3: Compare
    print("\n3. Comparison:")
    compare_configs(config_original, config)
    
    print("\n" + "=" * 70)
    print("Use get_advanced_config() for >95% accuracy target!")
    print("=" * 70)