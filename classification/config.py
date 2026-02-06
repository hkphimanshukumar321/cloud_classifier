# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Experiment Configuration
========================

Centralized configuration management for reproducibility.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import sys

# Add parent to path for common imports
sys.path.append(str(Path(__file__).parent.parent))

from common.config_base import BaseTrainingConfig, BaseOutputConfig


@dataclass
class DataConfig:
    """Dataset parameters."""
    data_dir: Path = Path(__file__).parent.parent / "data" / "classification" / "raw"
    img_size: Tuple[int, int] = (64, 64)
    batch_size: int = 32
    validation_split: float = 0.2
    test_split: float = 0.1
    max_images_per_class: Optional[int] = None  # For debugging
    
    # Augmentation
    use_augmentation: bool = True
    rotation_range: int = 20
    zoom_range: float = 0.2
    horizontal_flip: bool = True


@dataclass
class ModelConfig:
    """Model architecture hyperparameters."""
    # Custom CNN parameters
    growth_rate: int = 8      # Filters added per layer
    compression: float = 0.5  # Compression in transition blocks
    depth: Tuple[int, int, int] = (3, 3, 3) # Layers per dense block
    
    # Regularization
    dropout_rate: float = 0.2
    l2_decay: float = 1e-4
    
    # Initialization
    initial_filters: int = 24


@dataclass
class TrainingConfig(BaseTrainingConfig):
    """Training hyperparameters and experiment flags."""
    
    # SNR robustness testing (Classification specific)
    enable_snr_testing: bool = True
    snr_levels_db: List[int] = field(default_factory=lambda: [0, 5, 10, 15, 20, 25, 30])
    
    # Training callbacks
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5


@dataclass
class AblationConfig:
    """Ablation study search spaces."""
    growth_rates: List[int] = field(default_factory=lambda: [8, 12,16])
    compressions: List[float] = field(default_factory=lambda: [0.2,0.5, 0.75])
    depths: List[Tuple[int, ...]] = field(default_factory=lambda: [(2, 2, 2),(3,3,3) , (4, 4, 4)])
    batch_sizes: List[int] = field(default_factory=lambda: [16, 32, 64])
    resolutions: List[int] = field(default_factory=lambda: [32,64, 128])
    learning_rates: List[float] = field(default_factory=lambda: [1e-3, 1e-4])


@dataclass
class OutputConfig(BaseOutputConfig):
    """Output directories and settings."""
    figure_dpi: int = 30
    save_history: bool = True


@dataclass
class ResearchConfig:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def print_experiment_summary(config: ResearchConfig):
    """Print configuration summary."""
    print("=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)
    
    print(f"\n[*] Data:")
    print(f"  - Directory: {config.data.data_dir}")
    print(f"  - Image Size: {config.data.img_size}")
    
    print(f"\n[*] Model:")
    print(f"  - Growth Rate: {config.model.growth_rate}")
    print(f"  - Depth: {config.model.depth}")
    
    print(f"\n[*] Training:")
    print(f"  - Epochs: {config.training.epochs}")
    print(f"  - Batch Size: {config.training.batch_size}")
    print(f"  - Setup: {'Multi-GPU' if config.training.use_multiple_seeds else 'Standard'}")
    
    print(f"\n[*] Experiments:")
    print(f"  - Cross-Validation: {config.training.enable_cross_validation}")
    print(f"  - SNR Robustness: {config.training.enable_snr_testing}")
    
    print("=" * 60)