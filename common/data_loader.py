# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Common Data Loading Module
==========================

Shared utilities for image loading, preprocessing, and augmentation.
Each task imports these and adds task-specific extensions.

Features:
- Image loading with OpenCV
- Resolution resizing
- Normalization
- Augmentation pipelines
- Logging of data operations
"""

import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any, Callable
from abc import ABC, abstractmethod

import numpy as np
import cv2

logger = logging.getLogger(__name__)


# =============================================================================
# IMAGE I/O
# =============================================================================

def load_image(
    path: Path,
    img_size: Tuple[int, int] = (64, 64),
    normalize: bool = True,
    color_mode: str = 'rgb'
) -> Optional[np.ndarray]:
    """
    Load and preprocess a single image.
    
    Args:
        path: Path to image file
        img_size: Target size (height, width)
        normalize: Normalize to [0, 1]
        color_mode: 'rgb', 'bgr', or 'grayscale'
        
    Returns:
        Preprocessed image or None if loading fails
    """
    try:
        if color_mode == 'grayscale':
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = img[..., np.newaxis]  # Add channel dim
        else:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        
        if img is None:
            logger.warning(f"Failed to load: {path}")
            return None
        
        if color_mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        
        if normalize:
            img = img.astype(np.float32) / 255.0
        
        return img
        
    except Exception as e:
        logger.warning(f"Error loading {path}: {e}")
        return None


def save_image(
    img: np.ndarray,
    path: Path,
    denormalize: bool = True
) -> bool:
    """
    Save image to disk.
    
    Args:
        img: Image array (RGB or BGR)
        path: Output path
        denormalize: Convert from [0,1] to [0,255]
        
    Returns:
        True if successful
    """
    try:
        if denormalize and img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        if len(img.shape) == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(str(path), img)
        return True
        
    except Exception as e:
        logger.error(f"Error saving {path}: {e}")
        return False


# =============================================================================
# RESOLUTION & PREPROCESSING
# =============================================================================

class ImagePreprocessor:
    """
    Configurable image preprocessor.
    
    Usage:
        preprocessor = ImagePreprocessor(
            img_size=(128, 128),
            normalize=True,
            augment=True
        )
        
        img = preprocessor(load_image(path))
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (64, 64),
        normalize: bool = True,
        normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        use_imagenet_norm: bool = False
    ):
        self.img_size = img_size
        self.normalize = normalize
        self.mean = np.array(normalize_mean)
        self.std = np.array(normalize_std)
        self.use_imagenet_norm = use_imagenet_norm
        
        logger.info(f"Preprocessor: size={img_size}, normalize={normalize}")
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline."""
        # Resize if needed
        if img.shape[:2] != self.img_size:
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        
        # Normalize
        if self.normalize:
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
            
            if self.use_imagenet_norm:
                img = (img - self.mean) / self.std
        
        return img
    
    def inverse(self, img: np.ndarray) -> np.ndarray:
        """Inverse preprocessing for visualization."""
        if self.use_imagenet_norm:
            img = img * self.std + self.mean
        
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        return img


# =============================================================================
# AUGMENTATION
# =============================================================================

class BaseAugmentation(ABC):
    """Base class for augmentation transforms."""
    
    @abstractmethod
    def __call__(self, img: np.ndarray) -> np.ndarray:
        pass


class RandomFlip(BaseAugmentation):
    """Random horizontal/vertical flip."""
    
    def __init__(self, horizontal: bool = True, vertical: bool = False, p: float = 0.5):
        self.horizontal = horizontal
        self.vertical = vertical
        self.p = p
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.horizontal and np.random.random() < self.p:
            img = np.fliplr(img)
        if self.vertical and np.random.random() < self.p:
            img = np.flipud(img)
        return np.ascontiguousarray(img)


class RandomBrightness(BaseAugmentation):
    """Random brightness adjustment."""
    
    def __init__(self, delta: float = 0.2, p: float = 0.5):
        self.delta = delta
        self.p = p
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            factor = 1.0 + np.random.uniform(-self.delta, self.delta)
            img = np.clip(img * factor, 0, 1)
        return img


class RandomRotation(BaseAugmentation):
    """Random rotation."""
    
    def __init__(self, max_angle: float = 15, p: float = 0.5):
        self.max_angle = max_angle
        self.p = p
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            angle = np.random.uniform(-self.max_angle, self.max_angle)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return img


class AugmentationPipeline:
    """
    Compose multiple augmentations.
    
    Usage:
        augmentations = AugmentationPipeline([
            RandomFlip(horizontal=True),
            RandomBrightness(delta=0.2),
            RandomRotation(max_angle=15)
        ])
        
        augmented_img = augmentations(img)
    """
    
    def __init__(self, transforms: List[BaseAugmentation]):
        self.transforms = transforms
        logger.info(f"Augmentation pipeline: {len(transforms)} transforms")
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            img = t(img)
        return img


# =============================================================================
# LOGGING & STATS
# =============================================================================

def log_dataset_stats(
    X: np.ndarray,
    Y: np.ndarray,
    class_names: Optional[List[str]] = None,
    name: str = "Dataset"
):
    """Log dataset statistics."""
    logger.info(f"\n{'='*50}")
    logger.info(f"{name} Statistics")
    logger.info(f"{'='*50}")
    logger.info(f"  Shape: {X.shape}")
    logger.info(f"  Dtype: {X.dtype}")
    logger.info(f"  Range: [{X.min():.3f}, {X.max():.3f}]")
    
    unique, counts = np.unique(Y, return_counts=True)
    logger.info(f"  Classes: {len(unique)}")
    
    for i, (cls, count) in enumerate(zip(unique, counts)):
        name = class_names[cls] if class_names else str(cls)
        logger.info(f"    {name}: {count} ({count/len(Y)*100:.1f}%)")
    
    logger.info(f"{'='*50}\n")


def validate_images(
    data_dir: Path,
    img_size: Tuple[int, int],
    sample_count: int = 5
) -> bool:
    """
    Validate sample images from directory.
    
    Args:
        data_dir: Directory with images
        img_size: Expected image size
        sample_count: Number of images to validate
        
    Returns:
        True if validation passes
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    
    images = list(data_dir.rglob('*'))
    images = [p for p in images if p.suffix.lower() in image_extensions]
    
    if not images:
        logger.error(f"No images found in {data_dir}")
        return False
    
    sample = np.random.choice(images, min(sample_count, len(images)), replace=False)
    
    for path in sample:
        img = load_image(path, img_size)
        if img is None:
            return False
        if img.shape[:2] != img_size:
            logger.error(f"Size mismatch: {img.shape[:2]} != {img_size}")
            return False
    
    logger.info(f"Validated {len(sample)} images from {data_dir}")
    return True