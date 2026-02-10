#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
K-Fold Cross-Validation
=======================

Run K-fold cross-validation for robust model evaluation.

Features:
- Stratified k-fold splitting
- Per-fold metrics collection
- Aggregate statistics with confidence intervals

Usage:
    python run_cross_validation.py
    python run_cross_validation.py --folds 10
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

from config import ResearchConfig
from src.models import create_model, get_model_metrics
from src.data_loader import validate_dataset_directory, load_dataset
from src.training import train_model, compile_model, setup_gpu
from src.visualization import plot_confusion_matrix
from common.logger import ExperimentLogger


def run_cross_validation(n_folds: int = None, quick_test: bool = False):
    """
    Run k-fold cross-validation.
    
    Args:
        n_folds: Number of folds (default from config)
        quick_test: Use 2 epochs
    """
    print("=" * 70)
    print("K-FOLD CROSS-VALIDATION")
    print("=" * 70)
    
    config = ResearchConfig()
    
    if not config.training.enable_cross_validation:
        print("⚠ Cross-validation disabled in config")
        return None
    
    n_folds = n_folds or config.training.cv_folds
    epochs = 2 if quick_test else config.training.epochs
    
    results_dir = config.output.results_dir / "cross_validation"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/3] Loading Dataset...")
    try:
        categories, _ = validate_dataset_directory(config.data.data_dir)
    except Exception as e:
        print(f"❌ Dataset not found. Configure 'data_dir' in config.py")
        return None
    
    X, Y = load_dataset(
        config.data.data_dir, categories, config.data.img_size,
        config.data.max_images_per_class, show_progress=True
    )
    num_classes = len(categories)
    input_shape = (*config.data.img_size, 3)
    
    # Cross-validation
    print(f"\n[2/3] Running {n_folds}-Fold CV...")
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, Y)):
        print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = Y[train_idx], Y[val_idx]
        
        # Create and train model
        tf.keras.backend.clear_session()
        
        model = create_model(
            input_shape=input_shape,
            num_classes=num_classes,
            **{k: getattr(config.model, k) for k in ['growth_rate', 'compression', 'depth', 'dropout_rate', 'initial_filters']}
        )

        steps_per_epoch = max(1, len(X_train) // config.training.batch_size)
        total_steps = steps_per_epoch * epochs
        warmup_steps = steps_per_epoch * config.training.warmup_epochs if config.training.lr_schedule == 'cosine_warmup' else 0
        use_cosine = config.training.lr_schedule in ('cosine', 'cosine_warmup')

        mixup_alpha = config.data.mixup_alpha if config.data.use_mixup else 0.0

        model = compile_model(
            model, config.training.learning_rate,
            loss=config.training.loss_type,
            label_smoothing=config.training.label_smoothing,
            total_steps=total_steps if use_cosine else 0,
            warmup_steps=warmup_steps,
            use_mixup=(mixup_alpha > 0)
        )
        
        fold_dir = results_dir / f"fold_{fold_idx + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        history = train_model(
            model, X_train, y_train, X_val, y_val,
            run_dir=fold_dir, epochs=epochs,
            batch_size=config.training.batch_size,
            use_cosine_lr=use_cosine,
            mixup_alpha=mixup_alpha,
            num_classes=num_classes,
            verbose=2
        )
        
        # Evaluate
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        f1 = f1_score(y_val, y_pred, average='macro')
        
        fold_results.append({
            'fold': fold_idx + 1,
            'val_accuracy': val_acc * 100,
            'val_loss': val_loss,
            'macro_f1': f1 * 100,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx)
        })
        
        print(f"   Accuracy: {val_acc*100:.2f}%, F1: {f1*100:.2f}%")
    
    # Save results
    print("\n[3/3] Saving Results...")
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(results_dir / "cv_results.csv", index=False)
    
    # Summary
    summary = {
        'accuracy_mean': results_df['val_accuracy'].mean(),
        'accuracy_std': results_df['val_accuracy'].std(),
        'f1_mean': results_df['macro_f1'].mean(),
        'f1_std': results_df['macro_f1'].std(),
        'n_folds': n_folds
    }
    
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)
    print(f"   Accuracy: {summary['accuracy_mean']:.2f}% ± {summary['accuracy_std']:.2f}%")
    print(f"   F1 Score: {summary['f1_mean']:.2f}% ± {summary['f1_std']:.2f}%")
    print(f"   Results: {results_dir}")
    print("=" * 70)
    
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='K-fold cross-validation')
    parser.add_argument('--folds', type=int, help='Number of folds')
    parser.add_argument('--quick', action='store_true', help='Quick test')
    args = parser.parse_args()
    
    run_cross_validation(n_folds=args.folds, quick_test=args.quick)