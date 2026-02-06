# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Segmentation Baseline Comparison
================================

Compare custom model against pretrained segmentation baselines.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.experiments.base_baselines import BaseBaselineRunner


class SegmentationBaselines(BaseBaselineRunner):
    """
    Baseline comparison for segmentation.
    
    Uses pretrained encoder backbones from ImageNet.
    """
    
    def get_baselines(self):
        return [
            'UNet_ResNet50',
            'UNet_MobileNetV2', 
            'DeepLabV3_MobileNetV2',
        ]
    
    def get_custom_model_metrics(self):
        """Return metrics for your custom model (already trained)."""
        # TODO: Load your trained model and compute metrics
        return {
            'accuracy': 0.0,
            'mean_iou': 0.0,
            'mean_dice': 0.0,
            'params': 0
        }
    
    def evaluate_baseline(self, model_name: str):
        """
        Evaluate a baseline segmentation model.
        
        Uses pretrained encoders from TensorFlow.
        """
        from tensorflow.keras.applications import ResNet50V2, MobileNetV2
        
        print(f"  Loading {model_name}...")
        
        if 'ResNet50' in model_name:
            encoder = ResNet50V2(include_top=False, weights='imagenet')
            params = encoder.count_params()
        elif 'MobileNet' in model_name:
            encoder = MobileNetV2(include_top=False, weights='imagenet')
            params = encoder.count_params()
        else:
            params = 0
        
        # TODO: Build full model, train on your data, evaluate
        
        return {
            'accuracy': 0.0,
            'mean_iou': 0.0,
            'mean_dice': 0.0,
            'params': params
        }


def run_segmentation_baselines(results_dir: Path = None):
    """Run segmentation baseline comparison."""
    if results_dir is None:
        results_dir = Path(__file__).parent.parent / 'results'
    
    runner = SegmentationBaselines(results_dir)
    return runner.run()


if __name__ == "__main__":
    run_segmentation_baselines()