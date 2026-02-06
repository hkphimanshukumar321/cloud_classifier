# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Hardware Utilities Module
=========================

Functions for GPU detection, memory management, and hardware acceleration setup.
"""

import os
import platform
import logging
import socket
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def get_gpu_info() -> Dict[str, Any]:
    """
    Get information about available GPUs.
    """
    info = {
        "num_gpus": 0,
        "gpu_names": [],
        "has_gpu": False
    }
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        info["num_gpus"] = len(gpus)
        info["has_gpu"] = len(gpus) > 0
        info["gpu_names"] = [gpu.name for gpu in gpus]
        
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Error detecting GPUs: {e}")
        
    return info


def setup_gpu_memory_growth() -> None:
    """Enable GPU memory growth."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Enabled memory growth for {len(gpus)} GPUs")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to set memory growth: {e}")


def setup_gpu(memory_growth: bool = True) -> None:
    """
    Legacy compatibility wrapper for GPU setup.
    """
    if memory_growth:
        setup_gpu_memory_growth()


def setup_multi_gpu():
    """
    Setup multi-GPU training strategy.
    
    Returns:
        tf.distribute.Strategy
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            logger.info(f"Multi-GPU enabled: {len(gpus)} GPUs")
            return strategy
        else:
            return tf.distribute.get_strategy()
    except ImportError:
        return None


def get_system_info() -> Dict[str, str]:
    """Get basic system information."""
    return {
        "system": platform.system(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "hostname": socket.gethostname()
    }


def get_device_info() -> Dict[str, Any]:
    """Legacy alias for device info."""
    info = get_gpu_info()
    info.update(get_system_info())
    try:
        import tensorflow as tf
        info['tensorflow_version'] = tf.__version__
    except ImportError:
        info['tensorflow_version'] = "unknown"
    return info