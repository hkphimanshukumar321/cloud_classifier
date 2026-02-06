# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Experiment Logger Module
========================

Centralized logging for experiments.
"""

import os
import json
import csv
import socket
import platform
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path = None, level: int = logging.INFO) -> None:
    """
    Setup global logging configuration.
    
    Args:
        log_dir: Directory to save log files (optional)
        level: Logging level
    """
    handlers = [logging.StreamHandler()]
    
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"run_{timestamp}.log"
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=handlers
    )


class ExperimentLogger:
    """
    Logger for tracking experiments and results.
    """
    
    def __init__(self, log_dir: str = "results/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments: List[Dict[str, Any]] = []
        self.machine_info: Optional[Dict[str, Any]] = None
        self.start_time = datetime.now()
        
    def log_machine_info(self) -> Dict[str, Any]:
        """Capture and log machine information."""
        try:
            import tensorflow as tf
            tf_version = tf.__version__
            gpus = tf.config.list_physical_devices('GPU')
            gpu_names = [gpu.name for gpu in gpus]
        except ImportError:
            tf_version = "not installed"
            gpu_names = []
        
        self.machine_info = {
            'hostname': socket.gethostname(),
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'tensorflow_version': tf_version,
            'num_gpus': len(gpu_names),
            'gpu_names': gpu_names,
            'timestamp': self.start_time.isoformat(),
        }
        
        machine_path = self.log_dir / 'machine_info.json'
        with open(machine_path, 'w') as f:
            json.dump(self.machine_info, f, indent=2)
        
        return self.machine_info
    
    def log_experiment(self, experiment_id: str, results: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> None:
        entry = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            **results
        }
        if config:
            entry['config'] = config
        self.experiments.append(entry)
    
    def save_all(self) -> None:
        if self.machine_info is None:
            self.log_machine_info()
            
        # Save JSON
        with open(self.log_dir / 'experiments.json', 'w') as f:
            json.dump(self.experiments, f, indent=2, default=str)
            
        # Save CSV
        if self.experiments:
            keys = set().union(*(d.keys() for d in self.experiments))
            with open(self.log_dir / 'experiments.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(keys))
                writer.writeheader()
                writer.writerows(self.experiments)