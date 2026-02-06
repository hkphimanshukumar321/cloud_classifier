#!/usr/bin/env python3
"""
Grouped Ablation Study (Batches A, B, C, D)
===========================================

Experimental Design:
- Batch A: Baseline Performance (SNR & Cross-Validation)
- Batch B: Architecture Ablation (Growth Rate × Compression × Depth) - 27 configs
- Batch C: Batch Size Ablation - 3 configs
- Batch D: Input Resolution Ablation - 3 configs

Total: 33 unique configurations × 3 seeds = 99 experiments
"""

import sys
import time
import socket
import platform
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd
from itertools import product
from typing import List, Dict, Tuple, Optional

# Add research root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import ResearchConfig, print_experiment_summary
from src.training import (
    train_model, setup_gpu, compile_model, compute_metrics, 
    generate_run_id, benchmark_inference, get_device_info, setup_multi_gpu
)
from src.models import create_rf_densenet, get_model_metrics
from src.data_loader import validate_dataset_directory, load_dataset_numpy, split_dataset
from src.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_ablation_study,
    plot_model_comparison_bar,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_radar_chart,
    plot_accuracy_vs_latency,
    close_all_figures,
    set_publication_style
)


def get_machine_info() -> Dict:
    """Get machine-specific info for reproducibility."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    return {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'tensorflow_version': tf.__version__,
        'num_gpus': len(gpus),
        'gpu_names': [gpu.name for gpu in gpus],
        'timestamp': datetime.now().isoformat()
    }


class AblationProgress:
    """Track ablation study progress with ETA."""
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
        eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.1f}m" if eta > 60 else f"{eta:.0f}s"
        bar = "█" * int(40 * pct) + "░" * (40 - int(40 * pct))
        print(f"\r[{bar}] {pct*100:5.1f}% | {self.completed}/{self.total} | ETA: {eta_str} | {self.category}", 
              end="", flush=True)
        if self.completed == self.total:
            print()


def run_experiment(exp_id, batch_type, gr, comp, depth, batch_size, res, lr, seed, config, strategy, machine_info, categories, results_dir, epochs):
    """Run a single experiment configuration."""
    from sklearn.metrics import f1_score
    import tensorflow as tf
    from sklearn.utils import class_weight
    
    # Set seed
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    data_dir = config.data.data_dir
    num_classes = len(categories)
    
    # Reload data if resolution differs from default
    X_exp, Y_exp = load_dataset_numpy(
        data_dir=data_dir, categories=categories,
        img_size=(res, res), max_images_per_class=config.data.max_images_per_class,
        show_progress=False
    )
    
    splits = split_dataset(X_exp, Y_exp, test_size=0.15, val_size=0.15, seed=seed)
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    class_weights_vals = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(enumerate(class_weights_vals))
    
    inp_shape = (res, res, 3)
    
    with strategy.scope():
        model = create_rf_densenet(
            input_shape=inp_shape, num_classes=num_classes,
            growth_rate=gr, compression=comp, depth=depth,
            dropout_rate=config.model.dropout_rate,
            initial_filters=config.model.initial_filters
        )
        model = compile_model(model, learning_rate=lr)
    
    metrics_info = get_model_metrics(model)
    run_dir = results_dir / "runs" / f"Batch{batch_type}_{exp_id}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    history = train_model(
        model=model, X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val, run_dir=run_dir,
        epochs=epochs, batch_size=batch_size, class_weights=class_weights,
        early_stopping_patience=config.training.early_stopping_patience,
        verbose=0
    )
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    macro_f1 = f1_score(y_test, y_pred, average='macro') * 100
    
    latency = benchmark_inference(model, inp_shape, warmup_runs=5, benchmark_runs=20)
    close_all_figures()
    
    # Cleanup
    tf.keras.backend.clear_session()
    
    return {
        'experiment_id': exp_id,
        'batch': batch_type,
        'seed': seed,
        'growth_rate': gr,
        'compression': comp,
        'depth': str(depth),
        'batch_size': batch_size,
        'resolution': res,
        'learning_rate': lr,
        'test_accuracy': test_acc * 100,
        'test_loss': test_loss,
        'macro_f1': macro_f1,
        'total_params': metrics_info.total_params,
        'inference_ms': latency['batch_1']['mean_ms'],
        'machine': machine_info['hostname']
    }


def run_ablation(quick_test: bool = False, single_seed: bool = False):
    """Run grouped ablation study (Batches A, B, C, D)."""
    config = ResearchConfig()
    results_dir = config.output.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = 2 if quick_test else config.training.epochs
    seeds = [config.ablation.seeds[0]] if single_seed else config.ablation.seeds
    
    machine_info = get_machine_info()
    strategy = setup_multi_gpu()
    
    data_dir = config.data.data_dir
    categories, _ = validate_dataset_directory(data_dir, min_classes=2)
    
    # 1. BATCH B: Architecture Ablation (27 configs)
    arch_combos = list(product(
        config.ablation.growth_rates,
        config.ablation.compressions,
        config.ablation.depths
    ))
    
    # 2. BATCH C: Batch Size Ablation (3 configs)
    # Keeping arch and res at defaults from config
    batch_combos = config.ablation.batch_sizes
    
    # 3. BATCH D: Resolution Ablation (3 configs)
    res_combos = config.ablation.resolutions
    
    # Total experiments: (27 + 3 + 3) * seeds
    total_configs = len(arch_combos) + len(batch_combos) + len(res_combos)
    total_experiments = total_configs * len(seeds)
    
    print(f"\n🚀 GROUPED ABLATION STUDY (99 Experiments)")
    print(f"   Batch B (Arch): {len(arch_combos)} configs")
    print(f"   Batch C (BatchSize): {len(batch_combos)} configs")
    print(f"   Batch D (Resolution): {len(res_combos)} configs")
    print(f"   Seeds per config: {len(seeds)}")
    print(f"   ─────────────────────────────────")
    print(f"   Total: {total_configs} configs × {len(seeds)} seeds = {total_experiments}\n")
    
    progress = AblationProgress(total_experiments + 1) # +1 for Batch A
    all_results = []
    
    # --- Execute Batch B (Architecture) ---
    for gr, comp, depth in arch_combos:
        depth_str = '_'.join(map(str, depth))
        exp_id = f"gr{gr}_c{comp}_d{depth_str}"
        for seed in seeds:
            progress.set_category(f"Batch B: {exp_id} Seed {seed}")
            res = result = run_experiment(
                exp_id, "B", gr, comp, depth, 
                config.training.batch_size, config.data.img_size[0], 
                config.training.learning_rate, seed, config, strategy, machine_info, categories, results_dir, epochs
            )
            all_results.append(res)
            progress.update()

    # --- Execute Batch C (Batch Size) ---
    # Use default arch (config.model)
    for bs in batch_combos:
        exp_id = f"bs{bs}"
        for seed in seeds:
            progress.set_category(f"Batch C: {exp_id} Seed {seed}")
            res = result = run_experiment(
                exp_id, "C", config.model.growth_rate, config.model.compression, config.model.depth, 
                bs, config.data.img_size[0], 
                config.training.learning_rate, seed, config, strategy, machine_info, categories, results_dir, epochs
            )
            all_results.append(res)
            progress.update()

    # --- Execute Batch D (Resolution) ---
    # Use default arch and default batch size
    for res_val in res_combos:
        exp_id = f"res{res_val}"
        for seed in seeds:
            progress.set_category(f"Batch D: {exp_id} Seed {seed}")
            res = result = run_experiment(
                exp_id, "D", config.model.growth_rate, config.model.compression, config.model.depth, 
                config.training.batch_size, res_val, 
                config.training.learning_rate, seed, config, strategy, machine_info, categories, results_dir, epochs
            )
            all_results.append(res)
            progress.update()

    # Save Results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_dir / "ablation_grouped_results.csv", index=False)
    
    # Generate Summary Plots
    _generate_grouped_plots(results_df, results_dir / "figures")
    
    print("\n" + "=" * 70)
    print("🎉 GROUPED ABLATION COMPLETE!")
    print(f"   Results saved to: {results_dir / 'ablation_grouped_results.csv'}")
    print("=" * 70)


def _generate_grouped_plots(df: pd.DataFrame, figures_dir: Path):
    """Generate plots specifically for grouped ablation batches."""
    import matplotlib.pyplot as plt
    figures_dir.mkdir(parents=True, exist_ok=True)
    set_publication_style()
    
    # 1. Batch B: Arch Analysis (Heatmap or Bar)
    # For simplicity, we'll plot mean accuracy per GR/Comp/Depth
    # ... (Plotting code here) ...
    
    # 2. Batch C: Batch Size Trend
    c_df = df[df['batch'] == 'C']
    if not c_df.empty:
        grouped = c_df.groupby('batch_size')['test_accuracy'].agg(['mean', 'std']).reset_index()
        plt.figure(figsize=(8, 6))
        plt.errorbar(grouped['batch_size'], grouped['mean'], yerr=grouped['std'], fmt='o-', capsize=5)
        plt.xlabel('Batch Size')
        plt.ylabel('Accuracy (%)')
        plt.title('Batch Size Sensitivity (Batch C)')
        plt.savefig(figures_dir / 'batch_size_ablation.png')
        plt.close()

    # 3. Batch D: Resolution Trend
    d_df = df[df['batch'] == 'D']
    if not d_df.empty:
        grouped = d_df.groupby('resolution')['test_accuracy'].agg(['mean', 'std']).reset_index()
        plt.figure(figsize=(8, 6))
        plt.errorbar(grouped['resolution'], grouped['mean'], yerr=grouped['std'], fmt='s-', color='orange', capsize=5)
        plt.xlabel('Resolution')
        plt.ylabel('Accuracy (%)')
        plt.title('Resolution Sensitivity (Batch D)')
        plt.savefig(figures_dir / 'resolution_ablation.png')
        plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Grouped ablation study')
    parser.add_argument('--quick', action='store_true', help='Quick test (2 epochs)')
    parser.add_argument('--single-seed', action='store_true', help='Use single seed only')
    args = parser.parse_args()
    
    run_ablation(quick_test=args.quick, single_seed=args.single_seed)