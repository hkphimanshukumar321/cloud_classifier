# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Training Utilities Module
=========================

Training infrastructure including:
- GPU setup and multi-GPU strategy
- Model compilation
- Training loop with callbacks
- Inference benchmarking
"""

import os
import time
import json
import socket
import platform
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from datetime import datetime
import sys

# Add parent to path for common imports
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
)

from common.hardware import setup_gpu, setup_multi_gpu, get_device_info

logger = logging.getLogger(__name__)


# =============================================================================
# GPU SETUP (Delegated to common.hardware)
# =============================================================================
# Functions imported from common.hardware:
# - setup_gpu
# - setup_multi_gpu
# - get_device_info


# =============================================================================
# MODEL COMPILATION
# =============================================================================

def ordinal_loss(y_true, y_pred):
    """Correlation-aware ordinal loss (BCE on derived targets)."""
    # Infer num_classes from y_pred shape: (B, K-1)
    K_minus_1 = tf.shape(y_pred)[-1]
    y_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    # Ordinal encoding: for class c, bits 0..c-1 = 1, rest = 0
    targets = tf.cast(tf.sequence_mask(y_int, maxlen=K_minus_1), tf.float32)
    return tf.keras.losses.binary_crossentropy(targets, y_pred)


def compile_model(
    model: Model,
    learning_rate: float = 1e-3,
    optimizer: str = 'adam',
    loss: str = 'sparse_categorical_crossentropy',
    label_smoothing: float = 0.0,
    total_steps: int = 0,
    warmup_steps: int = 0,
    gradient_clip_value: float = 1.0,
    metrics: list = None
) -> Model:
    """
    Compile model with optimizer and loss.
    
    Args:
        model: Keras model
        learning_rate: Peak learning rate
        optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')
        loss: Loss function name or object
        label_smoothing: Label smoothing factor (0=off, 0.1=recommended)
        total_steps: Total training steps (for cosine decay; 0=constant LR)
        warmup_steps: Linear warmup steps before cosine decay
        gradient_clip_value: Max gradient value (stabilizes training)
        metrics: List of metrics (default: ['accuracy'])
        
    Returns:
        Compiled model
    """
    if metrics is None:
        metrics = ['accuracy']
    
    # Learning rate schedule: cosine decay with optional warmup
    if total_steps > 0:
        cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=total_steps,
            alpha=1e-6  # minimum LR
        )
        if warmup_steps > 0:
            # Linear warmup: ramp from 0 to peak LR over warmup_steps
            lr_schedule = WarmupCosineSchedule(
                learning_rate, total_steps, warmup_steps
            )
        else:
            lr_schedule = cosine_decay
    else:
        lr_schedule = learning_rate
    
    # Gradient clipping prevents large gradient spikes that cause loss fluctuation
    clip_kw = {'clipvalue': gradient_clip_value} if gradient_clip_value > 0 else {}
    
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, **clip_kw)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, **clip_kw)
    elif optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, **clip_kw)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, **clip_kw)
    
    # Handle loss selection
    if loss == 'ordinal_loss':
        loss_fn = ordinal_loss
        metrics = [ordinal_accuracy]
    elif label_smoothing > 0:
        # Sparse labels + label smoothing = custom wrapper
        loss_fn = _smooth_sparse_crossentropy(label_smoothing)
    else:
        loss_fn = loss

    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=metrics
    )
    
    return model


class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay."""
    
    def __init__(self, peak_lr, total_steps, warmup_steps):
        super().__init__()
        self.peak_lr = peak_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)
        
        # Linear warmup
        warmup_lr = self.peak_lr * (step / tf.maximum(warmup, 1.0))
        
        # Cosine decay after warmup
        progress = (step - warmup) / tf.maximum(total - warmup, 1.0)
        cosine_lr = self.peak_lr * 0.5 * (1.0 + tf.cos(np.pi * progress))
        
        return tf.where(step < warmup, warmup_lr, cosine_lr)
    
    def get_config(self):
        return {
            'peak_lr': self.peak_lr,
            'total_steps': self.total_steps,
            'warmup_steps': self.warmup_steps
        }


def _smooth_sparse_crossentropy(label_smoothing=0.1):
    """
    Crossentropy with label smoothing that handles both:
    - Integer (sparse) labels from validation data
    - One-hot / soft labels from mixup training data
    """
    def loss_fn(y_true, y_pred):
        num_classes = tf.shape(y_pred)[-1]
        
        # Detect if labels are already one-hot (rank 2) or sparse (rank 1)
        if len(y_true.shape) > 1 and y_true.shape[-1] is not None and y_true.shape[-1] > 1:
            # Already one-hot / soft labels (from mixup)
            y_one_hot = y_true
        else:
            # Integer labels → convert to one-hot
            y_int = tf.cast(tf.squeeze(y_true), tf.int32)
            y_one_hot = tf.one_hot(y_int, num_classes)
        
        return tf.keras.losses.categorical_crossentropy(
            y_one_hot, y_pred, label_smoothing=label_smoothing
        )
    loss_fn.__name__ = 'smooth_sparse_crossentropy'
    return loss_fn


def ordinal_accuracy(y_true, y_pred):
    """Accuracy for ordinal regression (sum of sigmoids > 0.5)."""
    pred_labels = tf.reduce_sum(tf.cast(y_pred > 0.5, tf.int32), axis=-1)
    true_labels = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    return tf.cast(tf.equal(true_labels, pred_labels), tf.float32)


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    run_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    class_weights: Optional[Dict] = None,
    early_stopping_patience: int = 15,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    use_balanced_sampling: bool = True,
    use_cosine_lr: bool = False,
    mixup_alpha: float = 0.0,
    num_classes: int = 5,
    verbose: int = 1
) -> Dict[str, list]:
    """
    Train model with callbacks and optional class-balanced sampling.
    
    Args:
        model: Compiled Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        run_dir: Directory for saving checkpoints and logs
        epochs: Maximum epochs
        batch_size: Batch size
        class_weights: Optional class weights
        early_stopping_patience: Early stopping patience
        reduce_lr_patience: LR reduction patience
        reduce_lr_factor: LR reduction factor
        use_balanced_sampling: Use class-balanced sampling
        use_cosine_lr: If True, skip ReduceLROnPlateau (LR handled by schedule)
        mixup_alpha: MixUp alpha (0 = disabled)
        num_classes: Number of classes
        verbose: Verbosity level
        
    Returns:
        Training history dict
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            str(run_dir / 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        ),
        CSVLogger(str(run_dir / 'training_log.csv'))
    ]
    
    # Only add ReduceLROnPlateau if NOT using cosine schedule
    if not use_cosine_lr:
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                patience=reduce_lr_patience,
                factor=reduce_lr_factor,
                min_lr=1e-7,
                verbose=1
            )
        )
    
    # Compute steps_per_epoch for balanced sampling
    steps_per_epoch = max(1, len(X_train) // batch_size)
    
    if use_balanced_sampling:
        # Use class-aware balanced sampling + optional mixup
        try:
            from .data_loader import create_balanced_tf_dataset
        except ImportError:
            from data_loader import create_balanced_tf_dataset
        train_ds = create_balanced_tf_dataset(
            X_train, y_train,
            batch_size=batch_size,
            augment=True,
            mixup_alpha=mixup_alpha,
            num_classes=num_classes
        )
        
        history = model.fit(
            train_ds,
            validation_data=(X_val, y_val),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=verbose
        )
    else:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=verbose
        )
    
    # Save history
    history_path = run_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
    
    return history.history


# =============================================================================
# INFERENCE BENCHMARKING
# =============================================================================

def benchmark_inference(
    model: Model,
    input_shape: Tuple[int, int, int],
    warmup_runs: int = 5,
    benchmark_runs: int = 20,
    batch_sizes: list = None
) -> Dict[str, Any]:
    """
    Benchmark model inference time.
    
    Args:
        model: Keras model
        input_shape: Input shape (H, W, C)
        warmup_runs: Warmup iterations
        benchmark_runs: Benchmark iterations
        batch_sizes: Batch sizes to test
        
    Returns:
        Benchmark results dict
    """
    if batch_sizes is None:
        batch_sizes = [1, 8, 32]
    
    results = {}
    
    for bs in batch_sizes:
        dummy_input = np.random.randn(bs, *input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(warmup_runs):
            _ = model.predict(dummy_input, verbose=0)
        
        # Benchmark
        times = []
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            _ = model.predict(dummy_input, verbose=0)
            times.append((time.perf_counter() - start) * 1000)  # ms
        
        results[f'batch_{bs}'] = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'throughput_fps': (bs * 1000) / np.mean(times)
        }
    
    return results


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_prob: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_prob: Prediction probabilities (optional)
        
    Returns:
        Dict of metrics
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
    }
    
    return metrics


# =============================================================================
# UTILITIES
# =============================================================================

def generate_run_id() -> str:
    """Generate unique run ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_model_summary(model: Model, path: Path) -> None:
    """Save model summary to text file."""
    with open(path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))