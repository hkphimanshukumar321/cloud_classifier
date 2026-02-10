#!/usr/bin/env python3
"""
Baseline Model Comparisons
==========================

Compare CloudDenseNet-Lite against standard transfer-learning baselines
(MobileNetV2, EfficientNetV2B0, DenseNet121, ResNet50V2).

Usage:
    python run_baselines.py
    python run_baselines.py --models MobileNetV2 DenseNet121
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import tensorflow as tf
from sklearn.metrics import f1_score

from config import ResearchConfig
from src.models import create_cloud_densenet_lite, get_model_metrics
from src.data_loader import validate_dataset_directory, load_dataset, split_dataset
from src.training import train_model, compile_model, benchmark_inference
from src.visualization import plot_model_comparison_bar


# Supported baseline models (TF.Keras Applications)
BASELINE_MODELS = [
    'MobileNetV2', 'EfficientNetV2B0', 'DenseNet121', 'ResNet50V2'
]


def _create_baseline_model(
    model_name: str,
    input_shape,
    num_classes: int,
    freeze_base: bool = True
) -> tf.keras.Model:
    """Create a transfer-learning baseline model."""
    base_cls = {
        'MobileNetV2': tf.keras.applications.MobileNetV2,
        'EfficientNetV2B0': tf.keras.applications.EfficientNetV2B0,
        'DenseNet121': tf.keras.applications.DenseNet121,
        'ResNet50V2': tf.keras.applications.ResNet50V2,
    }

    if model_name not in base_cls:
        raise ValueError(f"Unknown baseline: {model_name}. Choose from {list(base_cls.keys())}")

    base = base_cls[model_name](
        include_top=False, weights='imagenet', input_shape=input_shape
    )
    base.trainable = not freeze_base

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=not freeze_base)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs, name=f"Baseline_{model_name}")


def run_baselines(models: List[str] = None, quick_test: bool = False):
    """Compare CloudDenseNet-Lite against baseline models."""
    print("=" * 60)
    print("BASELINE MODEL COMPARISONS")
    print("=" * 60)

    config = ResearchConfig()
    epochs = 2 if quick_test else config.training.epochs

    if models is None:
        models = config.baseline.baseline_models

    valid_models = [m for m in models if m in BASELINE_MODELS]
    if len(valid_models) < len(models):
        invalid = set(models) - set(valid_models)
        print(f"⚠ Skipping unknown models: {invalid}")

    results_dir = config.output.results_dir / "baselines"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/3] Loading Dataset...")
    try:
        categories, _ = validate_dataset_directory(config.data.data_dir)
    except Exception as e:
        print(f"❌ Dataset not found: {e}")
        return None

    X, Y = load_dataset(
        config.data.data_dir, categories, config.data.img_size,
        config.data.max_images_per_class, show_progress=True
    )
    splits = split_dataset(X, Y, seed=42)
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    num_classes = len(categories)
    input_shape = (*config.data.img_size, 3)

    print("\n[2/3] Training Models...")
    results = []

    # --- Our model: CloudDenseNet-Lite ---
    print("\n--- CloudDenseNet-Lite (ours) ---")
    tf.keras.backend.clear_session()

    model = create_cloud_densenet_lite(
        input_shape=input_shape,
        num_classes=num_classes,
        growth_rate=config.model.growth_rate,
        compression=config.model.compression,
        depth=config.model.depth,
        initial_filters=config.model.initial_filters,
        dropout_rate=config.model.dropout_rate,
        use_coord_att=config.model.use_coord_att,
        use_pretrained_stem=config.model.use_pretrained_stem,
        output_mode=config.model.output_mode
    )

    steps_per_epoch = max(1, len(X_train) // config.training.batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * config.training.warmup_epochs if config.training.lr_schedule == 'cosine_warmup' else 0
    use_cosine = config.training.lr_schedule in ('cosine', 'cosine_warmup')

    model = compile_model(
        model, config.training.learning_rate,
        loss=config.training.loss_type,
        label_smoothing=config.training.label_smoothing,
        total_steps=total_steps if use_cosine else 0,
        warmup_steps=warmup_steps
    )
    metrics = get_model_metrics(model)

    mixup_alpha = config.data.mixup_alpha if config.data.use_mixup else 0.0

    train_model(
        model, X_train, y_train, X_val, y_val,
        run_dir=results_dir / "CloudDenseNet_Lite", epochs=epochs,
        batch_size=config.training.batch_size,
        use_cosine_lr=use_cosine,
        mixup_alpha=mixup_alpha,
        num_classes=num_classes,
        verbose=2
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    latency = benchmark_inference(model, input_shape)

    results.append({
        'model': 'CloudDenseNet-Lite (ours)',
        'accuracy': acc * 100,
        'loss': loss,
        'params': metrics.total_params,
        'memory_mb': metrics.memory_mb,
        'inference_ms': latency['batch_1']['mean_ms']
    })
    print(f"   Accuracy: {acc*100:.2f}%, Params: {metrics.total_params:,}")

    # --- Baseline models ---
    for model_name in valid_models:
        print(f"\n--- {model_name} ---")
        tf.keras.backend.clear_session()

        try:
            model = _create_baseline_model(
                model_name, input_shape, num_classes,
                freeze_base=config.baseline.freeze_base
            )
            model = compile_model(model, config.training.learning_rate, loss='sparse_categorical_crossentropy')
            metrics = get_model_metrics(model)

            train_model(
                model, X_train, y_train, X_val, y_val,
                run_dir=results_dir / model_name, epochs=epochs,
                batch_size=config.training.batch_size, verbose=2
            )

            loss, acc = model.evaluate(X_test, y_test, verbose=0)
            latency = benchmark_inference(model, input_shape)

            results.append({
                'model': model_name,
                'accuracy': acc * 100,
                'loss': loss,
                'params': metrics.total_params,
                'memory_mb': metrics.memory_mb,
                'inference_ms': latency['batch_1']['mean_ms']
            })
            print(f"   Accuracy: {acc*100:.2f}%, Params: {metrics.total_params:,}")

        except Exception as e:
            print(f"   ❌ Failed: {e}")

    # Save results
    print("\n[3/3] Saving Results...")
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('accuracy', ascending=False)
    results_df.to_csv(results_dir / "baseline_comparison.csv", index=False)

    try:
        plot_model_comparison_bar(
            results_df['model'].tolist(),
            results_df['accuracy'].tolist(),
            save_path=results_dir / 'model_comparison.png'
        )
    except Exception as e:
        print(f"   Plot warning: {e}")

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\n   Results: {results_dir}")
    print("=" * 60)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline model comparisons')
    parser.add_argument('--models', nargs='+', help='Baseline models to test')
    parser.add_argument('--quick', action='store_true', help='Quick test')
    parser.add_argument('--list', action='store_true', help='List available models')
    args = parser.parse_args()

    if args.list:
        print("Available baseline models:")
        for name in BASELINE_MODELS:
            print(f"  - {name}")
    else:
        run_baselines(models=args.models, quick_test=args.quick)