# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Base Cross-Validation Runner
============================

K-Fold cross-validation framework that each task extends.
"""

import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import time
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class BaseCrossValidation(ABC):
    """
    Base class for K-Fold cross-validation.
    
    Each task extends with task-specific data loading and evaluation.
    
    Usage:
        class ClassificationCV(BaseCrossValidation):
            def load_data(self):
                return X, y
            
            def train_fold(self, X_train, y_train, X_val, y_val, fold):
                # Train and return metrics
                return {'val_accuracy': 0.95}
    """
    
    def __init__(self, results_dir: Path, n_folds: int = 5, seeds: List[int] = None):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.n_folds = n_folds
        self.seeds = seeds or [42]
        self.results = []
    
    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and return (X, y) data."""
        pass
    
    @abstractmethod
    def train_fold(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        fold: int
    ) -> Dict[str, float]:
        """Train on fold and return validation metrics."""
        pass
    
    def get_fold_indices(self, n_samples: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate stratified fold indices."""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=seed)
        return list(kf.split(np.arange(n_samples)))
    
    def run(self) -> pd.DataFrame:
        """Run cross-validation."""
        print("=" * 60)
        print(f"{self.n_folds}-FOLD CROSS-VALIDATION")
        print("=" * 60)
        
        X, y = self.load_data()
        print(f"Data: {X.shape[0]} samples")
        
        start_time = time.time()
        all_results = []
        
        for seed in self.seeds:
            print(f"\n--- Seed: {seed} ---")
            folds = self.get_fold_indices(len(X), seed)
            
            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                print(f"\nFold {fold_idx + 1}/{self.n_folds}")
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                try:
                    metrics = self.train_fold(X_train, y_train, X_val, y_val, fold_idx)
                    metrics['fold'] = fold_idx
                    metrics['seed'] = seed
                    all_results.append(metrics)
                    
                    metric_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() 
                                          if k not in ['fold', 'seed'])
                    print(f"  -> {metric_str}")
                    
                except Exception as e:
                    print(f"  [ERROR] {e}")
                    all_results.append({'fold': fold_idx, 'seed': seed, 'error': str(e)})
        
        df = pd.DataFrame(all_results)
        df.to_csv(self.results_dir / "cross_validation_results.csv", index=False)
        
        # Summary statistics
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 60)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['fold', 'seed']:
                mean = df[col].mean()
                std = df[col].std()
                print(f"  {col}: {mean:.4f} +/- {std:.4f}")
        
        print(f"\nTotal time: {elapsed/60:.1f} min")
        print(f"Results: {self.results_dir / 'cross_validation_results.csv'}")
        print("=" * 60)
        
        return df