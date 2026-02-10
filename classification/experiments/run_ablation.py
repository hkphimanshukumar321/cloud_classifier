#!/usr/bin/env python3
"""
Grouped Ablation Study for CloudDenseNet-Lite
==============================================

Experimental Design:
- Batch B: Architecture Ablation (Growth Rate × Compression × Depth) - 27 configs
- Batch C: Batch Size Ablation - 3 configs
- Batch D: Resolution Ablation - 3 configs
- Batch E: Coordinate Attention Ablation - 2 configs

Total: 35 unique configurations × 3 seeds = 105 experiments
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
from src.models import create_cloud_densenet_lite, get_model_metrics
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
        'python_version': platform.python_version(),
        'tensorflow_version': tf.__version__,
        'num_gpus': len(gpus),
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


def run_experiment(
    exp_id, batch_type, growth_rate, compression, depth, initial_filters,
    use_coord_att, batch_size, resolution, lr, dropout_rate, seed,
    config, strategy, machine_info, categories, results_dir, epochs
):
    """Run a single CloudDenseNet-Lite experiment."""
    from sklearn.metrics import f1_score
    import tensorflow as tf
    from sklearn.utils import class_weight

    # Set seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    data_dir = config.data.data_dir
    num_classes = len(categories)

    # Load data at specified resolution
    X_exp, Y_exp = load_dataset_numpy(
        data_dir=data_dir, categories=categories,
        img_size=(resolution, resolution),
        max_images_per_class=config.data.max_images_per_class,
        show_progress=True
    )

    splits = split_dataset(X_exp, Y_exp, test_size=0.15, val_size=0.15, seed=seed)
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    class_weights_vals = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights_vals))

    inp_shape = (resolution, resolution, 3)

    with strategy.scope():
        model = create_cloud_densenet_lite(
            input_shape=inp_shape,
            num_classes=num_classes,
            growth_rate=growth_rate,
            compression=compression,
            depth=depth,
            initial_filters=initial_filters,
            dropout_rate=dropout_rate,
            use_coord_att=use_coord_att,
            use_in_model_aug=True,
            use_pretrained_stem=config.model.use_pretrained_stem,
            output_mode=config.model.output_mode,
            name=f"CDN_{exp_id}"
        )

        # Cosine LR schedule
        steps_per_epoch = max(1, len(X_train) // batch_size)
        total_steps = steps_per_epoch * epochs
        warmup_steps = steps_per_epoch * config.training.warmup_epochs if config.training.lr_schedule == 'cosine_warmup' else 0
        use_cosine = config.training.lr_schedule in ('cosine', 'cosine_warmup')

        # Mixup config
        mixup_alpha = config.data.mixup_alpha if config.data.use_mixup else 0.0

        model = compile_model(
            model, learning_rate=lr,
            loss=config.training.loss_type,
            label_smoothing=config.training.label_smoothing,
            total_steps=total_steps if use_cosine else 0,
            warmup_steps=warmup_steps,
            use_mixup=(mixup_alpha > 0)
        )

    metrics_info = get_model_metrics(model)
    run_dir = results_dir / "runs" / f"Batch{batch_type}_{exp_id}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    history = train_model(
        model=model, X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val, run_dir=run_dir,
        epochs=epochs, batch_size=batch_size,
        class_weights=class_weights_dict,
        early_stopping_patience=config.training.early_stopping_patience,
        use_cosine_lr=use_cosine,
        mixup_alpha=mixup_alpha,
        num_classes=num_classes,
        verbose=2
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    macro_f1 = f1_score(y_test, y_pred, average='macro') * 100

    latency = benchmark_inference(model, inp_shape, warmup_runs=3, benchmark_runs=10)
    close_all_figures()

    print(f"  >> {exp_id} seed={seed}: acc={test_acc*100:.1f}%  F1={macro_f1:.1f}%  params={metrics_info.total_params:,}")

    tf.keras.backend.clear_session()

    return {
        'experiment_id': exp_id,
        'batch': batch_type,
        'seed': seed,
        'growth_rate': growth_rate,
        'compression': compression,
        'depth': str(depth),
        'initial_filters': initial_filters,
        'use_coord_att': use_coord_att,
        'batch_size': batch_size,
        'resolution': resolution,
        'learning_rate': lr,
        'dropout_rate': dropout_rate,
        'test_accuracy': test_acc * 100,
        'test_loss': test_loss,
        'macro_f1': macro_f1,
        'total_params': metrics_info.total_params,
        'memory_mb': metrics_info.memory_mb,
        'inference_ms': latency['batch_1']['mean_ms'],
        'machine': machine_info['hostname']
    }


def run_ablation(quick_test: bool = False, single_seed: bool = False):
    """Run grouped ablation study for CloudDenseNet-Lite."""
    config = ResearchConfig()
    results_dir = config.output.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    epochs = 2 if quick_test else config.training.epochs
    seeds = [config.ablation.seeds[0]] if single_seed else config.ablation.seeds

    machine_info = get_machine_info()
    strategy = setup_multi_gpu()

    data_dir = config.data.data_dir
    categories, class_counts = validate_dataset_directory(data_dir, min_classes=2)
    num_classes = len(categories)

    # Default architecture params
    default_gr = config.model.growth_rate
    default_comp = config.model.compression
    default_depth = config.model.depth
    default_filters = config.model.initial_filters
    default_bs = config.training.batch_size
    default_res = config.data.img_size[0]
    default_lr = config.training.learning_rate
    default_dropout = config.model.dropout_rate

    # --- Count experiments ---
    arch_combos = list(product(
        config.ablation.growth_rates,
        config.ablation.compressions,
        config.ablation.depths
    ))
    batch_combos = config.ablation.batch_sizes
    res_combos = config.ablation.resolutions
    ca_combos = config.ablation.coord_att_options

    total_configs = len(arch_combos) + len(batch_combos) + len(res_combos) + len(ca_combos)
    total_experiments = total_configs * len(seeds)

    print(f"\n🚀 CloudDenseNet-Lite ABLATION STUDY")
    print(f"   Dataset: {num_classes} classes, {sum(class_counts.values())} images")
    print(f"   Batch B (Arch GR×Comp×Depth): {len(arch_combos)} configs")
    print(f"   Batch C (Batch Size)         : {len(batch_combos)} configs")
    print(f"   Batch D (Resolution)         : {len(res_combos)} configs")
    print(f"   Batch E (Coord Attention)    : {len(ca_combos)} configs")
    print(f"   Seeds per config: {len(seeds)}")
    print(f"   ─────────────────────────────────")
    print(f"   Total: {total_configs} configs × {len(seeds)} seeds = {total_experiments}\n")

    progress = AblationProgress(total_experiments)
    all_results = []

    # --- Batch B: Architecture ---
    for gr, comp, depth in arch_combos:
        depth_str = '_'.join(map(str, depth))
        exp_id = f"gr{gr}_c{comp}_d{depth_str}"
        for seed in seeds:
            progress.set_category(f"B: {exp_id} s{seed}")
            result = run_experiment(
                exp_id, "B", gr, comp, depth, default_filters,
                True, default_bs, default_res, default_lr, default_dropout,
                seed, config, strategy, machine_info, categories, results_dir, epochs
            )
            all_results.append(result)
            progress.update()

    # --- Batch C: Batch Size ---
    for bs in batch_combos:
        exp_id = f"bs{bs}"
        for seed in seeds:
            progress.set_category(f"C: {exp_id} s{seed}")
            result = run_experiment(
                exp_id, "C", default_gr, default_comp, default_depth, default_filters,
                True, bs, default_res, default_lr, default_dropout,
                seed, config, strategy, machine_info, categories, results_dir, epochs
            )
            all_results.append(result)
            progress.update()

    # --- Batch D: Resolution ---
    for res_val in res_combos:
        exp_id = f"res{res_val}"
        for seed in seeds:
            progress.set_category(f"D: {exp_id} s{seed}")
            result = run_experiment(
                exp_id, "D", default_gr, default_comp, default_depth, default_filters,
                True, default_bs, res_val, default_lr, default_dropout,
                seed, config, strategy, machine_info, categories, results_dir, epochs
            )
            all_results.append(result)
            progress.update()

    # --- Batch E: Coordinate Attention ---
    for use_ca in ca_combos:
        exp_id = f"ca{'ON' if use_ca else 'OFF'}"
        for seed in seeds:
            progress.set_category(f"E: {exp_id} s{seed}")
            result = run_experiment(
                exp_id, "E", default_gr, default_comp, default_depth, default_filters,
                use_ca, default_bs, default_res, default_lr, default_dropout,
                seed, config, strategy, machine_info, categories, results_dir, epochs
            )
            all_results.append(result)
            progress.update()

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_dir / "ablation_results.csv", index=False)

    # Generate plots
    _generate_plots(results_df, results_dir / "figures")

    print(f"\n{'='*60}")
    print(f"🎉 ABLATION COMPLETE!")
    print(f"   Results: {results_dir / 'ablation_results.csv'}")
    print(f"{'='*60}")

    return results_df


def _generate_plots(df: pd.DataFrame, figures_dir: Path):
    """Generate ablation plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures_dir.mkdir(parents=True, exist_ok=True)
    set_publication_style()

    # Batch C: Batch Size
    c_df = df[df['batch'] == 'C']
    if not c_df.empty:
        grouped = c_df.groupby('batch_size')['test_accuracy'].agg(['mean', 'std']).reset_index()
        plt.figure(figsize=(8, 6))
        plt.errorbar(grouped['batch_size'], grouped['mean'], yerr=grouped['std'], fmt='o-', capsize=5)
        plt.xlabel('Batch Size')
        plt.ylabel('Accuracy (%)')
        plt.title('Batch Size Sensitivity')
        plt.savefig(figures_dir / 'batch_size_ablation.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Batch D: Resolution
    d_df = df[df['batch'] == 'D']
    if not d_df.empty:
        grouped = d_df.groupby('resolution')['test_accuracy'].agg(['mean', 'std']).reset_index()
        plt.figure(figsize=(8, 6))
        plt.errorbar(grouped['resolution'], grouped['mean'], yerr=grouped['std'], fmt='s-', color='orange', capsize=5)
        plt.xlabel('Resolution')
        plt.ylabel('Accuracy (%)')
        plt.title('Resolution Sensitivity')
        plt.savefig(figures_dir / 'resolution_ablation.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Batch E: Coord Attention
    e_df = df[df['batch'] == 'E']
    if not e_df.empty:
        grouped = e_df.groupby('use_coord_att')['test_accuracy'].agg(['mean', 'std']).reset_index()
        plt.figure(figsize=(6, 5))
        labels = ['Without CA', 'With CA']
        plt.bar(labels, grouped['mean'], yerr=grouped['std'], capsize=5, color=['#e57373', '#81c784'])
        plt.ylabel('Accuracy (%)')
        plt.title('Coordinate Attention Impact')
        plt.savefig(figures_dir / 'coord_attention_ablation.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Params vs Accuracy scatter
    plt.figure(figsize=(8, 6))
    for batch in df['batch'].unique():
        sub = df[df['batch'] == batch]
        plt.scatter(sub['total_params'], sub['test_accuracy'], label=f'Batch {batch}', alpha=0.7)
    plt.xlabel('Parameters')
    plt.ylabel('Accuracy (%)')
    plt.title('Parameters vs Accuracy')
    plt.legend()
    plt.savefig(figures_dir / 'params_vs_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='CloudDenseNet-Lite ablation study')
    parser.add_argument('--quick', action='store_true', help='Quick test (2 epochs)')
    parser.add_argument('--single-seed', action='store_true', help='Single seed only')
    args = parser.parse_args()

    run_ablation(quick_test=args.quick, single_seed=args.single_seed)