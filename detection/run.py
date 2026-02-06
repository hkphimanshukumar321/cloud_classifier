# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Detection Master Runner
=======================

Main entry point for detection experiments.
"""

import sys
import argparse
import logging
from pathlib import Path

# Fix paths to allow imports from common and local src
current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))  # Template root
sys.path.insert(0, str(current_dir))  # Detection root

from common.logger import setup_logging
from common.hardware import get_gpu_info
from config import DetectionConfig

# Setup logger
logger = logging.getLogger("detection_runner")


def run_all(quick_test: bool = False) -> bool:
    """
    Run full detection pipeline.
    
    Args:
        quick_test: Run with minimal data/epochs for verification
        
    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print("DETECTION MASTER RUNNER")
    print("=" * 60)
    
    # 1. Setup
    setup_logging(log_dir=current_dir / "logs")
    gpu_info = get_gpu_info()
    logger.info(f"Hardware: {gpu_info}")
    
    config = DetectionConfig()
    if quick_test:
        config.training.epochs = 1
        config.data.img_size = (128, 128)
    
    print(f"\n[*] Config: Image Size={config.data.img_size}, Classes={config.model.num_classes}")
    
    # 2. Check Data
    data_dir = Path(config.data.data_dir or (current_dir / "data"))
    if not data_dir.exists():
        print(f"\n[!] Data directory not found: {data_dir}")
        print("    Please add 'images' and 'labels' folders.")
        print("    Skipping training/evaluation loop.")
        return True
        
    # 3. Validation success
    logger.info("Environment ready. Add training logic here.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detection Runner")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke test")
    args = parser.parse_args()
    
    try:
        success = run_all(quick_test=args.quick)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Runner failed: {e}", exc_info=True)
        sys.exit(1)