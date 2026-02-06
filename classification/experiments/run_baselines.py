#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Baseline Model Comparisons
==========================

Compare your custom model against standard baseline architectures.

Features:
- Transfer learning from ImageNet
- Parameter count comparison
- Latency benchmarking

Usage:
    python run_baselines.py
    python run_baselines.py --models   ResNet50V2 MobileNetV2
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

from config import ResearchConfig
from src.models import create_model, create_baseline_model, BASELINE_MODELS, get_model_metrics
from src.data_loader import validate_dataset_directory, load_dataset, split_dataset
from src.training import train_model, compile_model, benchmark_inference
from src.visualization import plot_model_comparison_bar
from common.logger import ExperimentLogger


def run_baselines(models: List[str] = None, quick_test: bool = False):
    """
    Compare against baseline models.
    
    Args:
        models: List of baseline model names
        quick_test: Use 2 epochs
    """
    print("=" * 70)
    print("BASELINE MODEL COMPARISONS")
    print("=" * 70)
    
    config = ResearchConfig()
    epochs = 2 if quick_test else config.training.epochs
    
    # Default to a subset of models for reasonable runtime
    if models is None:
        models = ['MobileNetV2', 'EfficientNetV2B0', 'DenseNet121']
    
    # Filter valid models
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
        print(f"❌ Dataset not found. Configure 'data_dir' in config.py")
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
    
    # Run custom model first
    print("\n[2/3] Training Models...")
    results = []
    
    # Custom model
    print("\n--- CustomModel (Your Model) ---")
    tf.keras.backend.clear_session()
    
    model = create_model(input_shape, num_classes)
    model = compile_model(model, config.training.learning_rate)
    metrics = get_model_metrics(model)
    
    train_model(
        model, X_train, y_train, X_val, y_val,
        run_dir=results_dir / "CustomModel", epochs=epochs,
        batch_size=config.training.batch_size, verbose=0
    )
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    latency = benchmark_inference(model, input_shape)
    
    results.append({
        'model': 'CustomModel',
        'accuracy': acc * 100,
        'loss': loss,
        'params': metrics.total_params,
        'inference_ms': latency['batch_1']['mean_ms']
    })
    print(f"   Accuracy: {acc*100:.2f}%, Params: {metrics.total_params:,}")
    
    # Baseline models
    for model_name in valid_models:
        print(f"\n--- {model_name} ---")
        tf.keras.backend.clear_session()
        
        try:
            model = create_baseline_model(
                model_name, input_shape, num_classes,
                use_pretrained=True, freeze_base=True
            )
            model = compile_model(model, config.training.learning_rate)
            metrics = get_model_metrics(model)
            
            train_model(
                model, X_train, y_train, X_val, y_val,
                run_dir=results_dir / model_name, epochs=epochs,
                batch_size=config.training.batch_size, verbose=0
            )
            
            loss, acc = model.evaluate(X_test, y_test, verbose=0)
            latency = benchmark_inference(model, input_shape)
            
            results.append({
                'model': model_name,
                'accuracy': acc * 100,
                'loss': loss,
                'params': metrics.total_params,
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
    
    # Plot
    plot_model_comparison_bar(
        results_df['model'].tolist(),
        results_df['accuracy'].tolist(),
        save_path=results_dir / 'model_comparison.png'
    )
    
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\n   Results: {results_dir}")
    print("=" * 70)
    
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