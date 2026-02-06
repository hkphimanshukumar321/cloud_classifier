#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
SNR Robustness Testing
======================

Test model robustness to noise at various SNR levels.

Features:
- Gaussian noise injection at multiple SNR levels
- Accuracy degradation curves
- Publication-quality plots

Usage:
    python run_snr_robustness.py
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import tensorflow as tf

from config import ResearchConfig
from src.models import create_model
from src.data_loader import validate_dataset_directory, load_dataset, split_dataset, add_noise
from src.training import train_model, compile_model
from src.visualization import set_publication_style, close_all_figures, COLORS
from common.logger import ExperimentLogger


def run_snr_robustness(quick_test: bool = False):
    """
    Test model robustness to noise.
    
    Args:
        quick_test: Use 2 epochs
    """
    import matplotlib.pyplot as plt
    
    print("=" * 70)
    print("SNR ROBUSTNESS TESTING")
    print("=" * 70)
    
    config = ResearchConfig()
    
    if not config.training.enable_snr_testing:
        print("⚠ SNR testing disabled in config")
        return None
    
    epochs = 2 if quick_test else config.training.epochs
    snr_levels = config.training.snr_levels_db
    
    results_dir = config.output.results_dir / "snr_robustness"
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
    
    # Train baseline model on clean data
    print("\n[2/3] Training Baseline Model...")
    tf.keras.backend.clear_session()
    
    model = create_model(
        input_shape=input_shape,
        num_classes=num_classes,
        growth_rate=config.model.growth_rate,
        compression=config.model.compression,
        depth=config.model.depth
    )
    model = compile_model(model, config.training.learning_rate)
    
    train_model(
        model, X_train, y_train, X_val, y_val,
        run_dir=results_dir / "baseline",
        epochs=epochs,
        batch_size=config.training.batch_size,
        verbose=1
    )
    
    # Test at different SNR levels
    print("\n[3/3] Testing at SNR Levels...")
    results = []
    
    for snr in snr_levels:
        print(f"   Testing SNR = {snr} dB...", end=" ")
        
        # Add noise to test set
        X_test_noisy = add_noise(X_test, snr)
        
        # Evaluate
        loss, acc = model.evaluate(X_test_noisy, y_test, verbose=0)
        
        results.append({
            'snr_db': snr,
            'accuracy': acc * 100,
            'loss': loss
        })
        
        print(f"Accuracy: {acc*100:.2f}%")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_dir / "snr_results.csv", index=False)
    
    # Plot
    set_publication_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(results_df['snr_db'], results_df['accuracy'], 'o-', 
            linewidth=2, markersize=8, color=COLORS['primary'])
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Robustness to Noise')
    ax.grid(True, alpha=0.3)
    
    fig.savefig(results_dir / 'snr_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("\n" + "=" * 70)
    print("SNR ROBUSTNESS RESULTS")
    print("=" * 70)
    print(f"   Clean (∞ dB): {results_df['accuracy'].max():.2f}%")
    print(f"   Noisy ({min(snr_levels)} dB): {results_df[results_df['snr_db'] == min(snr_levels)]['accuracy'].values[0]:.2f}%")
    print(f"   Results: {results_dir}")
    print("=" * 70)
    
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SNR robustness testing')
    parser.add_argument('--quick', action='store_true', help='Quick test')
    args = parser.parse_args()
    
    run_snr_robustness(quick_test=args.quick)