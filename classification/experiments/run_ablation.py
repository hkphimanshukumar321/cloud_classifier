#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Full Factorial Ablation Study
==============================

Run systematic ablation study varying:
- Architecture parameters (growth rate, compression, depth)
- Training parameters (batch size)
- Input resolution

Features:
- Multiple seeds for statistical significance
- Multi-GPU support
- Progress tracking with ETA
- Journal-quality plots

Usage:
    python run_ablation.py           # Full run
    python run_ablation.py --quick   # Quick test (2 epochs)
    python run_ablation.py --single-seed  # Single seed only
"""

import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple, Any

# Add paths for imports
# 1. classification/ directory (for config, src)
sys.path.append(str(Path(__file__).parent.parent))
# 2. template/ directory (for common)
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import ResearchConfig
from src.models import create_model, get_model_metrics
from src.data_loader import (
    validate_dataset_directory, load_dataset, split_dataset
)
from src.training import (
    train_model, compile_model, setup_multi_gpu,
    benchmark_inference, get_device_info, compute_metrics
)
from src.visualization import (
    plot_training_history, plot_confusion_matrix, close_all_figures
)
from src.visualization import (
    plot_training_history, plot_confusion_matrix, close_all_figures
)
from common.logger import ExperimentLogger


class AblationProgress:
    """Track ablation progress with ETA."""
    
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.start = time.time()
        self.category = ""
    
    def set_category(self, cat: str):
        self.category = cat
    
    def update(self):
        self.completed += 1
        pct = self.completed / self.total
        elapsed = time.time() - self.start
        eta = (elapsed / self.completed) * (self.total - self.completed) if self.completed > 0 else 0
        eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.1f}m"
        bar = "█" * int(40 * pct) + "░" * (40 - int(40 * pct))
        print(f"\r[{bar}] {pct*100:.1f}% | {self.completed}/{self.total} | ETA: {eta_str} | {self.category}", 
              end="", flush=True)
        if self.completed == self.total:
            print()


def run_ablation(quick_test: bool = False, single_seed: bool = False):
    """
    Run full factorial ablation study.
    
    Args:
        quick_test: Use 2 epochs per experiment
        single_seed: Use only first seed
    """
    import tensorflow as tf
    from sklearn.metrics import f1_score
    
    print("=" * 70)
    print("FULL FACTORIAL ABLATION STUDY")
    print("=" * 70)
    
    config = ResearchConfig()
    results_dir = config.output.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup
    epochs = 2 if quick_test else config.training.epochs
    use_single = single_seed or not config.training.use_multiple_seeds
    seeds = [config.training.seeds[0]] if use_single else config.training.seeds
    
    # Logger
    exp_logger = ExperimentLogger(results_dir / "logs")
    exp_logger.log_machine_info()
    exp_logger.log_config(config)
    
    # Multi-GPU
    strategy = setup_multi_gpu()
    
    # Load data
    print("\n[1/4] Loading Dataset...")
    data_dir = config.data.data_dir
    
    try:
        categories, _ = validate_dataset_directory(data_dir, min_classes=2)
    except Exception as e:
        print(f"❌ Dataset not found: {data_dir}")
        print(f"   Error: {e}")
        print("\n   Please configure 'data_dir' in config.py")
        return None
    
    X, Y = load_dataset(
        data_dir=data_dir,
        categories=categories,
        img_size=config.data.img_size,
        max_images_per_class=config.data.max_images_per_class,
        show_progress=True
    )
    num_classes = len(categories)
    
    # Experiment design
    print("\n[2/4] Experiment Design...")
    arch_combos = list(product(
        config.ablation.growth_rates,
        config.ablation.compressions,
        config.ablation.depths
    ))
    n_arch = len(arch_combos)
    n_batch = len(config.ablation.batch_sizes)
    n_resolution = len(config.ablation.resolutions)
    n_seeds = len(seeds)
    
    n_configs = n_arch + n_batch + n_resolution
    total_experiments = n_configs * n_seeds
    
    print(f"\n   Architecture combos: {n_arch}")
    print(f"   Batch sizes: {config.ablation.batch_sizes}")
    print(f"   Resolutions: {config.ablation.resolutions}")
    print(f"   Seeds: {seeds}")
    print(f"   TOTAL: {n_configs} × {n_seeds} = {total_experiments} experiments")
    
    progress = AblationProgress(total_experiments)
    all_results = []
    
    # Helper function
    def run_single(exp_id, gr, comp, depth, batch, res, seed):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Load at correct resolution
        if res != config.data.img_size[0]:
            X_exp, Y_exp = load_dataset(
                data_dir, categories, img_size=(res, res),
                max_images_per_class=config.data.max_images_per_class,
                show_progress=False
            )
        else:
            X_exp, Y_exp = X, Y
        
        splits = split_dataset(X_exp, Y_exp, seed=seed)
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']
        X_test, y_test = splits['test']
        
        inp_shape = (res, res, 3)
        
        with strategy.scope():
            model = create_model(
                input_shape=inp_shape,
                num_classes=num_classes,
                growth_rate=gr,
                compression=comp,
                depth=depth,
                dropout_rate=config.model.dropout_rate,
                initial_filters=config.model.initial_filters
            )
            model = compile_model(model, learning_rate=config.training.learning_rate)
        
        metrics_info = get_model_metrics(model)
        
        run_dir = results_dir / "runs" / f"{exp_id}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        history = train_model(
            model=model,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            run_dir=run_dir,
            epochs=epochs,
            batch_size=batch,
            early_stopping_patience=config.training.early_stopping_patience,
            verbose=0
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        macro_f1 = f1_score(y_test, y_pred, average='macro') * 100
        
        # Benchmark
        latency = benchmark_inference(model, inp_shape, warmup_runs=3, benchmark_runs=10)
        
        tf.keras.backend.clear_session()
        close_all_figures()
        
        return {
            'experiment_id': exp_id,
            'seed': seed,
            'growth_rate': gr,
            'compression': comp,
            'depth': str(depth),
            'batch_size': batch,
            'resolution': res,
            'test_accuracy': test_acc * 100,
            'test_loss': test_loss,
            'macro_f1': macro_f1,
            'val_accuracy': max(history.get('val_accuracy', [0])) * 100,
            'total_params': metrics_info.total_params,
            'inference_ms': latency['batch_1']['mean_ms'],
        }
    
    # Run experiments
    print("\n[3/4] Running Experiments...")
    print("-" * 70)
    
    # Architecture ablation
    print("\nGROUP 1: ARCHITECTURE")
    for gr, comp, depth in arch_combos:
        depth_str = '_'.join(map(str, depth))
        exp_id = f"arch_gr{gr}_c{comp}_d{depth_str}"
        
        for seed in seeds:
            progress.set_category(f"ARCH: GR={gr},C={comp},D={depth}")
            result = run_single(exp_id, gr, comp, depth, config.training.batch_size, 
                               config.data.img_size[0], seed)
            result['ablation_group'] = 'architecture'
            all_results.append(result)
            exp_logger.log_experiment(exp_id, result)
            progress.update()
    
    # Batch size ablation
    print("\n\nGROUP 2: BATCH SIZE")
    for bs in config.ablation.batch_sizes:
        exp_id = f"batch_{bs}"
        for seed in seeds:
            progress.set_category(f"BATCH: {bs}")
            result = run_single(exp_id, config.model.growth_rate, config.model.compression,
                               config.model.depth, bs, config.data.img_size[0], seed)
            result['ablation_group'] = 'batch_size'
            all_results.append(result)
            exp_logger.log_experiment(exp_id, result)
            progress.update()
    
    # Resolution ablation
    print("\n\nGROUP 3: RESOLUTION")
    for res in config.ablation.resolutions:
        exp_id = f"res_{res}"
        for seed in seeds:
            progress.set_category(f"RESOLUTION: {res}×{res}")
            result = run_single(exp_id, config.model.growth_rate, config.model.compression,
                               config.model.depth, config.training.batch_size, res, seed)
            result['ablation_group'] = 'resolution'
            all_results.append(result)
            exp_logger.log_experiment(exp_id, result)
            progress.update()
    
    # Save results
    print("\n\n[4/4] Saving Results...")
    results_df = pd.DataFrame(all_results)
    results_csv = results_dir / "ablation_full_factorial.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"✓ Saved: {results_csv}")
    
    # Summary
    summary_df = results_df.groupby('experiment_id').agg({
        'test_accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'inference_ms': ['mean', 'std'],
        'total_params': 'first',
        'ablation_group': 'first'
    }).reset_index()
    summary_df.columns = ['experiment_id', 'accuracy_mean', 'accuracy_std',
                          'f1_mean', 'f1_std', 'latency_mean', 'latency_std',
                          'params', 'group']
    summary_csv = results_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Saved: {summary_csv}")
    
    exp_logger.save_all()
    
    print("\n" + "=" * 70)
    print("🎉 ABLATION STUDY COMPLETE!")
    print(f"   Best Accuracy: {results_df['test_accuracy'].max():.2f}%")
    print(f"   Results: {results_dir}")
    print("=" * 70)
    
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Full factorial ablation study')
    parser.add_argument('--quick', action='store_true', help='Quick test (2 epochs)')
    parser.add_argument('--single-seed', action='store_true', help='Single seed only')
    args = parser.parse_args()
    
    run_ablation(quick_test=args.quick, single_seed=args.single_seed)