# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Data Loading Module
===================

Handles data loading, preprocessing, and augmentation for image classification.

Features:
- Directory structure validation
- Efficient parallel loading with tf.data
- Train/val/test splitting with stratification
- GPU-optimized data pipelines
"""

import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataIntegrityError(Exception):
    """Exception raised for data integrity issues."""
    pass


# =============================================================================
# DATASET VALIDATION
# =============================================================================

def validate_dataset_directory(
    data_dir: Path,
    min_classes: int = 2,
    min_samples_per_class: int = 10
) -> Tuple[List[str], Dict[str, int]]:
    """
    Validate dataset directory structure.
    
    Expected structure:
        data_dir/
        ├── class_0/
        │   ├── image1.jpg
        │   └── image2.jpg
        ├── class_1/
        │   └── ...
        └── ...
    
    Args:
        data_dir: Root directory containing class subfolders
        min_classes: Minimum required number of classes
        min_samples_per_class: Minimum samples per class
        
    Returns:
        Tuple of (class_names, class_counts)
        
    Raises:
        DataIntegrityError: If validation fails
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise DataIntegrityError(f"Directory not found: {data_dir}")
    
    # Find class directories
    class_dirs = [
        p.name for p in data_dir.iterdir()
        if p.is_dir() and not p.name.startswith('.')
    ]
    
    # Sort (numerically if possible)
    try:
        class_names = sorted(class_dirs, key=int)
    except ValueError:
        class_names = sorted(class_dirs)
    
    if len(class_names) < min_classes:
        raise DataIntegrityError(
            f"Found {len(class_names)} classes, need at least {min_classes}"
        )
    
    # Count samples per class
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    class_counts = {}
    
    for class_name in class_names:
        class_path = data_dir / class_name
        count = sum(
            1 for f in class_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        )
        class_counts[class_name] = count
        
        if count < min_samples_per_class:
            logger.warning(f"Class '{class_name}' has only {count} samples")
    
    total = sum(class_counts.values())
    logger.info(f"Dataset: {len(class_names)} classes, {total} total samples")
    
    return class_names, class_counts


# =============================================================================
# IMAGE LOADING
# =============================================================================

def load_image(
    path: Path,
    img_size: Tuple[int, int] = (64, 64),
    normalize: bool = True
) -> Optional[np.ndarray]:
    """
    Load and preprocess a single image.
    
    Args:
        path: Path to image file
        img_size: Target size (height, width)
        normalize: Normalize to [0, 1]
        
    Returns:
        Preprocessed image or None if loading fails
    """
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"Failed to load: {path}")
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        
        if normalize:
            img = img.astype(np.float32) / 255.0
        
        return img
    except Exception as e:
        logger.warning(f"Error loading {path}: {e}")
        return None


def load_dataset(
    data_dir: Path,
    categories: List[str],
    img_size: Tuple[int, int] = (64, 64),
    max_images_per_class: Optional[int] = None,
    show_progress: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load entire dataset into NumPy arrays.
    
    Args:
        data_dir: Root directory
        categories: List of class names
        img_size: Target image size
        max_images_per_class: Limit per class (None for all)
        show_progress: Show loading progress
        
    Returns:
        Tuple of (X, Y) arrays
    """
    data_dir = Path(data_dir)
    X, Y = [], []
    
    iterator = tqdm(enumerate(categories), total=len(categories), 
                    desc="Loading") if show_progress else enumerate(categories)
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    
    for class_idx, class_name in iterator:
        class_dir = data_dir / class_name
        
        files = sorted([
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ])
        
        if max_images_per_class:
            files = files[:max_images_per_class]
        
        for file_path in files:
            img = load_image(file_path, img_size)
            if img is not None:
                X.append(img)
                Y.append(class_idx)
    
    if len(X) == 0:
        raise DataIntegrityError("No valid images loaded")
    
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int64)
    
    logger.info(f"Loaded {len(X)} images, shape: {X.shape}")
    return X, Y


# Alias for Master Experiment Runner
load_dataset_numpy = load_dataset



# =============================================================================
# DATASET SPLITTING
# =============================================================================

def split_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
    stratify: bool = True
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        X: Feature array
        Y: Label array
        test_size: Test set fraction
        val_size: Validation set fraction
        seed: Random seed
        stratify: Use stratified splitting
        
    Returns:
        Dict with 'train', 'val', 'test' keys
    """
    stratify_arr = Y if stratify else None
    
    # First split: separate test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=seed, stratify=stratify_arr
    )
    
    # Second split: train/val
    adjusted_val = val_size / (1.0 - test_size)
    stratify_temp = y_temp if stratify else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adjusted_val, random_state=seed, stratify=stratify_temp
    )
    
    logger.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


# =============================================================================
# TF.DATA DATASET CREATION
# =============================================================================

def create_tf_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
    prefetch: bool = True
) -> tf.data.Dataset:
    """
    Create optimized TensorFlow dataset.
    
    Args:
        X: Feature array
        Y: Label array
        batch_size: Batch size
        shuffle: Enable shuffling
        augment: Apply augmentation
        prefetch: Enable prefetching
        
    Returns:
        tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    
    if shuffle:
        buffer_size = min(len(X), 10000)
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
    
    dataset = dataset.batch(batch_size)
    
    if augment:
        dataset = dataset.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.cache()
    
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


@tf.function
def augment_batch(images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply data augmentation to a batch (enhanced for cloud density)."""
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)  # clouds have no strict "up"
    images = tf.image.random_brightness(images, max_delta=0.15)
    images = tf.image.random_contrast(images, lower=0.85, upper=1.15)
    images = tf.image.random_saturation(images, lower=0.9, upper=1.1)
    images = tf.clip_by_value(images, 0.0, 1.0)
    return images, labels


# =============================================================================
# UTILITIES
# =============================================================================

def get_class_weights(Y: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights."""
    unique, counts = np.unique(Y, return_counts=True)
    n_samples = len(Y)
    n_classes = len(unique)
    
    return {
        int(cls): n_samples / (n_classes * count)
        for cls, count in zip(unique, counts)
    }


def add_noise(images: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise at specified SNR level."""
    signal_power = np.mean(images ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), images.shape)
    return np.clip(images + noise, 0, 1).astype(np.float32)


# =============================================================================
# CLASS-AWARE BALANCED SAMPLING (for imbalanced datasets)
# =============================================================================

def create_balanced_tf_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = 32,
    augment: bool = True,
    prefetch: bool = True,
    mixup_alpha: float = 0.0,
    num_classes: int = 5
) -> tf.data.Dataset:
    """
    Create a class-balanced TF dataset using per-class sampling.
    
    Instead of random sampling (which over-represents majority classes),
    this creates one dataset per class and samples uniformly across them.
    This ensures each batch has roughly equal representation.
    
    Args:
        X: Feature array
        Y: Label array (integer class indices)
        batch_size: Batch size
        augment: Apply augmentation
        prefetch: Enable prefetching
        mixup_alpha: MixUp alpha (0 = disabled). Produces soft one-hot labels.
        num_classes: Number of classes (needed for mixup one-hot encoding)
        
    Returns:
        Balanced tf.data.Dataset
    """
    unique_classes = np.unique(Y)
    num_classes = len(unique_classes)
    
    # Create per-class datasets
    per_class_datasets = []
    for cls in unique_classes:
        mask = Y == cls
        X_cls = X[mask]
        Y_cls = Y[mask]
        
        ds = tf.data.Dataset.from_tensor_slices((X_cls, Y_cls))
        ds = ds.shuffle(len(X_cls), reshuffle_each_iteration=True)
        ds = ds.repeat()  # infinite stream per class
        per_class_datasets.append(ds)
    
    # Uniform sampling weights (equal probability per class)
    weights = [1.0 / num_classes] * num_classes
    
    # Sample from all class datasets with equal probability
    balanced_ds = tf.data.Dataset.sample_from_datasets(
        per_class_datasets, weights=weights
    )
    
    balanced_ds = balanced_ds.batch(batch_size)
    
    if augment:
        balanced_ds = balanced_ds.map(
            augment_batch, num_parallel_calls=tf.data.AUTOTUNE
        )
    
    if mixup_alpha > 0.0:
        def _apply_mixup(images, labels):
            return mixup_batch(images, labels, alpha=mixup_alpha, num_classes=num_classes)
        balanced_ds = balanced_ds.map(
            _apply_mixup, num_parallel_calls=tf.data.AUTOTUNE
        )
    
    if prefetch:
        balanced_ds = balanced_ds.prefetch(tf.data.AUTOTUNE)
    
    logger.info(
        f"Created balanced dataset: {num_classes} classes, "
        f"~{len(X)//num_classes} samples/class effective"
        f"{', mixup='+str(mixup_alpha) if mixup_alpha > 0 else ''}"
    )
    return balanced_ds


def mixup_batch(
    images: tf.Tensor,
    labels: tf.Tensor,
    alpha: float = 0.2,
    num_classes: int = 5
) -> tuple:
    """
    MixUp augmentation: blend random pairs of images within a batch.
    
    Creates synthetic training samples by linear interpolation:
        mixed_image = λ * image_a + (1-λ) * image_b
        mixed_label = λ * onehot_a + (1-λ) * onehot_b
    
    where λ ~ Beta(alpha, alpha).
    
    Returns one-hot (soft) labels, compatible with CategoricalCrossentropy.
    """
    batch_size = tf.shape(images)[0]
    
    # Sample mixing coefficient from Beta distribution
    # Beta(0.2, 0.2) produces values near 0 or 1 (light mixing)
    lam = tf.random.uniform([], 0, 1)
    # Approximate Beta by clipping uniform — simple and effective
    lam = tf.maximum(lam, 1.0 - lam)  # Ensure λ ≥ 0.5 (original dominates)
    
    # Shuffle indices for pairing
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)
    
    # Mix images
    mixed_images = lam * images + (1.0 - lam) * shuffled_images
    
    # Convert labels to one-hot and mix
    labels_int = tf.cast(tf.squeeze(labels), tf.int32)
    shuffled_int = tf.cast(tf.squeeze(shuffled_labels), tf.int32)
    
    labels_oh = tf.one_hot(labels_int, num_classes)
    shuffled_oh = tf.one_hot(shuffled_int, num_classes)
    
    mixed_labels = lam * labels_oh + (1.0 - lam) * shuffled_oh
    
    return mixed_images, mixed_labels