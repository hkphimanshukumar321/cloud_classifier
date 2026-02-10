# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Research Template - Source Package
===================================

Core modules for cloud density classification using CloudDenseNet-Lite.
"""

from .models import create_model, create_cloud_densenet_lite, get_model_metrics
from .data_loader import load_dataset, split_dataset, create_tf_dataset
from .training import train_model, compile_model, setup_gpu
from .visualization import plot_training_history, plot_confusion_matrix

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from common.logger import ExperimentLogger

__all__ = [
    'create_model',
    'create_cloud_densenet_lite',
    'get_model_metrics',
    'load_dataset',
    'split_dataset',
    'create_tf_dataset',
    'train_model',
    'compile_model',
    'setup_gpu',
    'plot_training_history',
    'plot_confusion_matrix',
    'ExperimentLogger',
]