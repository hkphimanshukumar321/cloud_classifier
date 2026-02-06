# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Segmentation IoU Analysis
=========================

Task-specific experiment for analyzing IoU per class.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def compute_iou(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    """
    Compute IoU (Intersection over Union) per class.
    
    Args:
        y_true: Ground truth masks (N, H, W) or (N, H, W, C)
        y_pred: Predicted masks
        num_classes: Number of classes
        
    Returns:
        Dict with per-class IoU and mean IoU
    """
    if len(y_true.shape) == 4:
        y_true = np.argmax(y_true, axis=-1)
    if len(y_pred.shape) == 4:
        y_pred = np.argmax(y_pred, axis=-1)
    
    ious = {}
    for c in range(num_classes):
        true_mask = (y_true == c)
        pred_mask = (y_pred == c)
        
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        
        if union > 0:
            ious[f'iou_class_{c}'] = intersection / union
        else:
            ious[f'iou_class_{c}'] = 1.0  # Both empty = perfect
    
    ious['mean_iou'] = np.mean(list(ious.values()))
    return ious


def compute_dice(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    """
    Compute Dice coefficient per class.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """
    if len(y_true.shape) == 4:
        y_true = np.argmax(y_true, axis=-1)
    if len(y_pred.shape) == 4:
        y_pred = np.argmax(y_pred, axis=-1)
    
    dice_scores = {}
    for c in range(num_classes):
        true_mask = (y_true == c)
        pred_mask = (y_pred == c)
        
        intersection = 2 * np.logical_and(true_mask, pred_mask).sum()
        total = true_mask.sum() + pred_mask.sum()
        
        if total > 0:
            dice_scores[f'dice_class_{c}'] = intersection / total
        else:
            dice_scores[f'dice_class_{c}'] = 1.0
    
    dice_scores['mean_dice'] = np.mean(list(dice_scores.values()))
    return dice_scores


def run_iou_analysis(
    model,
    test_data,
    class_names: list,
    results_dir: Path
) -> pd.DataFrame:
    """
    Run comprehensive IoU analysis.
    
    Args:
        model: Trained segmentation model
        test_data: (X_test, y_test) or generator
        class_names: List of class names
        results_dir: Directory to save results
    """
    print("=" * 60)
    print("SEGMENTATION IoU ANALYSIS")
    print("=" * 60)
    
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if isinstance(test_data, tuple):
        X_test, y_test = test_data
        y_pred = model.predict(X_test)
    else:
        # Generator
        y_true_list, y_pred_list = [], []
        for X_batch, y_batch in test_data:
            y_true_list.append(y_batch)
            y_pred_list.append(model.predict(X_batch))
        y_test = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)
    
    num_classes = len(class_names)
    
    # Compute metrics
    iou_scores = compute_iou(y_test, y_pred, num_classes)
    dice_scores = compute_dice(y_test, y_pred, num_classes)
    
    # Per-class results
    results = []
    for i, name in enumerate(class_names):
        results.append({
            'class': name,
            'iou': iou_scores[f'iou_class_{i}'],
            'dice': dice_scores[f'dice_class_{i}']
        })
    
    df = pd.DataFrame(results)
    df.to_csv(results_dir / "iou_analysis.csv", index=False)
    
    # Summary
    print(f"\nPer-Class Results:")
    print("-" * 40)
    for _, row in df.iterrows():
        print(f"  {row['class']:15} | IoU: {row['iou']:.4f} | Dice: {row['dice']:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Mean IoU:  {iou_scores['mean_iou']:.4f}")
    print(f"Mean Dice: {dice_scores['mean_dice']:.4f}")
    print(f"{'='*60}")
    
    return df