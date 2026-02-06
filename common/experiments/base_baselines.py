# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Base Baseline Runner
====================

Compare custom model against pretrained baselines.
Each task extends with task-specific model loading and evaluation.
"""

import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import time
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class BaseBaselineRunner(ABC):
    """
    Base class for baseline comparison.
    
    Each task extends with task-specific baselines.
    
    Usage:
        class ClassificationBaselines(BaseBaselineRunner):
            def get_baselines(self):
                return ['MobileNetV2', 'EfficientNetV2B0', 'ResNet50V2']
            
            def evaluate_baseline(self, name, X_test, y_test):
                return {'accuracy': 0.92, 'params': 3_000_000}
    """
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    @abstractmethod
    def get_baselines(self) -> List[str]:
        """Return list of baseline model names."""
        pass
    
    @abstractmethod
    def get_custom_model_metrics(self) -> Dict[str, float]:
        """Return metrics for custom model (already trained)."""
        pass
    
    @abstractmethod
    def evaluate_baseline(self, model_name: str) -> Dict[str, float]:
        """
        Evaluate a baseline model.
        
        Args:
            model_name: Name of baseline
            
        Returns:
            Dict with metrics (accuracy, params, latency, etc.)
        """
        pass
    
    def run(self) -> pd.DataFrame:
        """Run baseline comparison."""
        print("=" * 60)
        print("BASELINE COMPARISON")
        print("=" * 60)
        
        baselines = self.get_baselines()
        print(f"Baselines: {baselines}")
        
        start_time = time.time()
        results = []
        
        # Custom model
        print("\n[1] Custom Model")
        custom_metrics = self.get_custom_model_metrics()
        custom_metrics['model'] = 'Custom'
        results.append(custom_metrics)
        print(f"  -> Accuracy: {custom_metrics.get('accuracy', 'N/A')}")
        
        # Baselines
        for i, name in enumerate(baselines):
            print(f"\n[{i+2}] {name}")
            try:
                metrics = self.evaluate_baseline(name)
                metrics['model'] = name
                results.append(metrics)
                print(f"  -> Accuracy: {metrics.get('accuracy', 'N/A')}")
            except Exception as e:
                print(f"  [ERROR] {e}")
                results.append({'model': name, 'error': str(e)})
        
        df = pd.DataFrame(results)
        df.to_csv(self.results_dir / "baseline_comparison.csv", index=False)
        
        # Summary
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print("BASELINE COMPARISON RESULTS")
        print("=" * 60)
        
        if 'accuracy' in df.columns:
            df_sorted = df.sort_values('accuracy', ascending=False)
            for _, row in df_sorted.iterrows():
                acc = row.get('accuracy', 'N/A')
                params = row.get('params', 'N/A')
                print(f"  {row['model']:20} | Acc: {acc:.4f} | Params: {params:,}" 
                      if isinstance(acc, float) else f"  {row['model']:20} | Error")
        
        print(f"\nTotal time: {elapsed/60:.1f} min")
        print(f"Results: {self.results_dir / 'baseline_comparison.csv'}")
        print("=" * 60)
        
        return df