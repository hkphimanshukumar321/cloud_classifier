# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Error Handler Module
====================

Provides helpful error messages and quick-fix suggestions for common issues.
"""

import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


class DatasetError(Exception):
    """Raised when dataset is missing or invalid."""
    pass


class EnvironmentError(Exception):
    """Raised when environment is misconfigured."""
    pass


def validate_path(path: Path, name: str, create_if_missing: bool = False) -> bool:
    """
    Validate a path exists.
    
    Args:
        path: Path to validate
        name: Human-readable name for error messages
        create_if_missing: Create directory if missing
        
    Returns:
        True if valid
        
    Raises:
        DatasetError: If path doesn't exist and not created
    """
    if path.exists():
        return True
    
    if create_if_missing:
        path.mkdir(parents=True, exist_ok=True)
        print(f"[*] Created directory: {path}")
        return True
    
    raise DatasetError(
        f"\n{'='*60}\n"
        f"[ERROR] {name} NOT FOUND\n"
        f"{'='*60}\n"
        f"  Path: {path.absolute()}\n\n"
        f"  Quick Fixes:\n"
        f"  1. Create the directory manually\n"
        f"  2. Update 'data_dir' in config.py\n"
        f"  3. Use --create-dirs flag to auto-create\n"
        f"{'='*60}"
    )


def validate_config(config: Any, required_fields: Dict[str, type]) -> bool:
    """
    Validate configuration has required fields.
    
    Args:
        config: Configuration object
        required_fields: Dict of field_name -> expected_type
        
    Returns:
        True if valid
        
    Raises:
        ConfigurationError: If validation fails
    """
    errors = []
    
    for field, expected_type in required_fields.items():
        parts = field.split('.')
        value = config
        try:
            for part in parts:
                value = getattr(value, part)
            
            if not isinstance(value, expected_type):
                errors.append(f"  - {field}: Expected {expected_type.__name__}, got {type(value).__name__}")
        except AttributeError:
            errors.append(f"  - {field}: Missing")
    
    if errors:
        raise ConfigurationError(
            f"\n{'='*60}\n"
            f"[ERROR] CONFIGURATION INVALID\n"
            f"{'='*60}\n"
            f"{'chr(10)'.join(errors)}\n\n"
            f"  Quick Fix: Check config.py for typos\n"
            f"{'='*60}"
        )
    
    return True


def check_gpu_available() -> Dict[str, Any]:
    """
    Check GPU availability and configuration.
    
    Returns:
        Dict with GPU info
    """
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        
        return {
            'available': len(gpus) > 0,
            'count': len(gpus),
            'names': [gpu.name for gpu in gpus],
            'memory_growth_enabled': True,  # Default
            'strategy': 'MirroredStrategy' if len(gpus) > 1 else 'Default'
        }
    except Exception as e:
        return {
            'available': False,
            'count': 0,
            'error': str(e)
        }


def check_multiprocessing() -> Dict[str, Any]:
    """
    Check multiprocessing capabilities.
    
    Returns:
        Dict with multiprocessing info
    """
    import multiprocessing
    import os
    
    cpu_count = multiprocessing.cpu_count()
    
    # Check TensorFlow parallel settings
    try:
        import tensorflow as tf
        tf_threads = tf.config.threading.get_inter_op_parallelism_threads()
        tf_intra = tf.config.threading.get_intra_op_parallelism_threads()
    except:
        tf_threads = 0
        tf_intra = 0
    
    return {
        'cpu_count': cpu_count,
        'recommended_workers': min(cpu_count - 1, 8),
        'tf_inter_threads': tf_threads,
        'tf_intra_threads': tf_intra,
        'parallel_enabled': True
    }


def print_environment_summary():
    """Print summary of environment configuration."""
    print("=" * 60)
    print("ENVIRONMENT SUMMARY")
    print("=" * 60)
    
    # GPU
    gpu_info = check_gpu_available()
    print(f"\n[*] GPU:")
    print(f"    Available: {gpu_info['available']}")
    print(f"    Count: {gpu_info['count']}")
    if gpu_info['available']:
        print(f"    Strategy: {gpu_info['strategy']}")
    
    # Multiprocessing
    mp_info = check_multiprocessing()
    print(f"\n[*] Multiprocessing:")
    print(f"    CPU Cores: {mp_info['cpu_count']}")
    print(f"    Recommended Workers: {mp_info['recommended_workers']}")
    
    print("=" * 60)


def handle_exception(exc: Exception, context: str = ""):
    """
    Handle exception with helpful message.
    
    Args:
        exc: Exception to handle
        context: Additional context
    """
    print(f"\n{'='*60}")
    print(f"[ERROR] {type(exc).__name__}")
    print(f"{'='*60}")
    
    if context:
        print(f"Context: {context}")
    
    print(f"\nMessage: {exc}")
    
    # Provide suggestions based on error type
    if "No module named" in str(exc):
        module = str(exc).split("'")[1] if "'" in str(exc) else "unknown"
        print(f"\nQuick Fix: pip install {module}")
    
    elif "CUDA" in str(exc) or "GPU" in str(exc):
        print("\nQuick Fix: Set environment variable TF_ENABLE_ONEDNN_OPTS=0")
        print("           or run with CPU: CUDA_VISIBLE_DEVICES=-1")
    
    elif "path" in str(exc).lower() or "directory" in str(exc).lower():
        print("\nQuick Fix: Check data_dir in config.py")
        print("           Run: python run.py --create-dirs")
    
    print(f"\n{'='*60}")
    
    # Log full traceback for debugging
    logger.error(f"Exception in {context}: {exc}", exc_info=True)