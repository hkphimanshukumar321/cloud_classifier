# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Common Ablation Framework
=========================

Base class for ablation studies that each task extends.
Classification, Segmentation, and Detection inherit from this.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from pathlib import Path
import itertools
import time
import json
import pandas as pd
import numpy as np


@dataclass
class AblationParameter:
    """Single ablation parameter."""
    name: str
    values: List[Any]
    description: str = ""


class BaseAblationStudy(ABC):
    """
    Base class for ablation studies.
    
    Each task (classification, segmentation, detection) extends this
    with task-specific parameters and evaluation metrics.
    """
    
    def __init__(self, results_dir: Path, seeds: List[int] = None):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seeds = seeds or [42, 123, 456]
        self.results = []
        self.start_time = None
    
    @abstractmethod
    def get_parameters(self) -> List[AblationParameter]:
        """Return list of parameters to ablate. Override in subclass."""
        pass
    
    @abstractmethod
    def run_single_experiment(self, config: Dict[str, Any], seed: int) -> Dict[str, Any]:
        """Run single experiment with given config. Override in subclass."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> List[str]:
        """Return list of metrics to track. Override in subclass."""
        pass
    
    def generate_configurations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        params = self.get_parameters()
        names = [p.name for p in params]
        values = [p.values for p in params]
        
        configs = []
        for combo in itertools.product(*values):
            configs.append(dict(zip(names, combo)))
        
        return configs
    
    def run_full_ablation(self, quick_test: bool = False) -> pd.DataFrame:
        """Run complete ablation study."""
        print("=" * 60)
        print("ABLATION STUDY")
        print("=" * 60)
        
        configs = self.generate_configurations()
        seeds = [self.seeds[0]] if quick_test else self.seeds
        total = len(configs) * len(seeds)
        
        print(f"\nConfigurations: {len(configs)}")
        print(f"Seeds: {len(seeds)}")
        print(f"Total experiments: {total}")
        
        self.start_time = time.time()
        self.results = []
        
        for i, config in enumerate(configs):
            for seed in seeds:
                print(f"\n[{i*len(seeds)+seeds.index(seed)+1}/{total}] Config: {config}, Seed: {seed}")
                
                try:
                    result = self.run_single_experiment(config, seed)
                    result['config'] = str(config)
                    result['seed'] = seed
                    self.results.append(result)
                except Exception as e:
                    print(f"  [ERROR] {e}")
                    self.results.append({
                        'config': str(config),
                        'seed': seed,
                        'error': str(e)
                    })
        
        # Save results
        df = pd.DataFrame(self.results)
        df.to_csv(self.results_dir / "ablation_results.csv", index=False)
        
        elapsed = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"ABLATION COMPLETE ({elapsed/60:.1f} min)")
        print(f"Results: {self.results_dir}")
        print(f"{'='*60}")
        
        return df
    
    def compute_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute summary statistics grouped by config."""
        metrics = self.get_metrics()
        
        agg_dict = {m: ['mean', 'std'] for m in metrics if m in df.columns}
        
        if agg_dict:
            summary = df.groupby('config').agg(agg_dict).reset_index()
            summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
            summary.to_csv(self.results_dir / "ablation_summary.csv", index=False)
            return summary
        
        return df


# Task-specific ablation configs (examples)
@dataclass
class ClassificationAblationConfig:
    """Classification-specific ablation parameters."""
    growth_rates: List[int] = field(default_factory=lambda: [8, 16, 32])
    compressions: List[float] = field(default_factory=lambda: [0.5, 1.0])
    depths: List[Tuple[int, ...]] = field(default_factory=lambda: [(2, 2, 2), (4, 4, 4)])
    batch_sizes: List[int] = field(default_factory=lambda: [16, 32])
    resolutions: List[int] = field(default_factory=lambda: [64, 128])


@dataclass
class SegmentationAblationConfig:
    """Segmentation-specific ablation parameters."""
    encoder_depths: List[int] = field(default_factory=lambda: [3, 4, 5])
    decoder_filters: List[List[int]] = field(default_factory=lambda: [[64, 32], [128, 64, 32]])
    skip_connections: List[bool] = field(default_factory=lambda: [True, False])
    loss_functions: List[str] = field(default_factory=lambda: ["dice", "focal", "ce"])


@dataclass
class DetectionAblationConfig:
    """Detection-specific ablation parameters."""
    backbones: List[str] = field(default_factory=lambda: ["darknet", "resnet", "efficientnet"])
    anchor_scales: List[List[int]] = field(default_factory=lambda: [[8, 16, 32], [16, 32, 64]])
    nms_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    confidence_thresholds: List[float] = field(default_factory=lambda: [0.25, 0.5])