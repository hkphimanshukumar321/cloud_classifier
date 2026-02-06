# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Base Ablation Runner
====================

Common ablation study framework that each task extends.
Override `get_search_space()` and `run_experiment()` in subclasses.
"""

import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import itertools
import time
import pandas as pd
import numpy as np

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class AblationResult:
    """Single ablation experiment result."""
    config: Dict[str, Any]
    seed: int
    metrics: Dict[str, float]
    training_time: float
    error: str = None


class BaseAblationRunner(ABC):
    """
    Base class for ablation studies.
    
    Each task (classification, segmentation, detection) extends this
    with task-specific search spaces and evaluation.
    
    Usage:
        class ClassificationAblation(BaseAblationRunner):
            def get_search_space(self):
                return {'growth_rate': [8, 16], 'depth': [(2,2,2), (4,4,4)]}
            
            def run_experiment(self, config, seed):
                # Train and evaluate model
                return {'accuracy': 0.95, 'f1': 0.94}
    """
    
    def __init__(self, results_dir: Path, seeds: List[int] = None):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seeds = seeds or [42, 123, 456]
        self.results: List[AblationResult] = []
    
    @abstractmethod
    def get_search_space(self) -> Dict[str, List[Any]]:
        """
        Return hyperparameter search space.
        
        Returns:
            Dict mapping parameter names to list of values
            
        Example:
            {'growth_rate': [8, 16, 32], 'batch_size': [16, 32]}
        """
        pass
    
    @abstractmethod
    def run_experiment(self, config: Dict[str, Any], seed: int) -> Dict[str, float]:
        """
        Run single experiment with given config.
        
        Args:
            config: Hyperparameter configuration
            seed: Random seed
            
        Returns:
            Dict of metric_name -> value
        """
        pass
    
    def get_metrics(self) -> List[str]:
        """Return list of metrics to track. Override if needed."""
        return ['accuracy', 'loss', 'training_time']
    
    def generate_configs(self) -> List[Dict[str, Any]]:
        """Generate all hyperparameter combinations."""
        space = self.get_search_space()
        names = list(space.keys())
        values = list(space.values())
        
        configs = []
        for combo in itertools.product(*values):
            configs.append(dict(zip(names, combo)))
        
        return configs
    
    def run(self, quick_test: bool = False) -> pd.DataFrame:
        """Run complete ablation study."""
        print("=" * 60)
        print("ABLATION STUDY")
        print("=" * 60)
        
        configs = self.generate_configs()
        seeds = [self.seeds[0]] if quick_test else self.seeds
        total = len(configs) * len(seeds)
        
        print(f"\nSearch Space: {self.get_search_space()}")
        print(f"Configurations: {len(configs)}")
        print(f"Seeds: {len(seeds)}")
        print(f"Total experiments: {total}")
        print("=" * 60)
        
        start_time = time.time()
        self.results = []
        
        for i, config in enumerate(configs):
            for j, seed in enumerate(seeds):
                exp_num = i * len(seeds) + j + 1
                print(f"\n[{exp_num}/{total}] {config} | seed={seed}")
                
                try:
                    exp_start = time.time()
                    metrics = self.run_experiment(config, seed)
                    exp_time = time.time() - exp_start
                    
                    self.results.append(AblationResult(
                        config=config,
                        seed=seed,
                        metrics=metrics,
                        training_time=exp_time
                    ))
                    
                    # Print key metrics
                    metric_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                    print(f"  -> {metric_str} ({exp_time:.1f}s)")
                    
                except Exception as e:
                    print(f"  [ERROR] {e}")
                    self.results.append(AblationResult(
                        config=config,
                        seed=seed,
                        metrics={},
                        training_time=0,
                        error=str(e)
                    ))
        
        # Convert to DataFrame
        df = self._results_to_dataframe()
        df.to_csv(self.results_dir / "ablation_results.csv", index=False)
        
        # Summary
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"ABLATION COMPLETE ({elapsed/60:.1f} min)")
        print(f"Results saved: {self.results_dir / 'ablation_results.csv'}")
        print(f"{'='*60}")
        
        return df
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        rows = []
        for r in self.results:
            row = {**r.config, 'seed': r.seed, 'training_time': r.training_time}
            row.update(r.metrics)
            if r.error:
                row['error'] = r.error
            rows.append(row)
        return pd.DataFrame(rows)
    
    def get_best_config(self, metric: str = 'accuracy') -> Dict[str, Any]:
        """Get best configuration based on metric."""
        df = self._results_to_dataframe()
        if metric not in df.columns:
            raise ValueError(f"Metric {metric} not found")
        
        # Group by config, average across seeds
        config_cols = list(self.get_search_space().keys())
        summary = df.groupby(config_cols)[metric].mean().reset_index()
        best_idx = summary[metric].idxmax()
        
        return summary.iloc[best_idx].to_dict()