# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Detection mAP Analysis
======================

Task-specific experiment for analyzing mAP at different IoU thresholds.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def compute_iou_boxes(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two bounding boxes.
    
    Boxes format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def compute_ap(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> float:
    """
    Compute Average Precision for a single class.
    
    Args:
        predictions: List of {'box': [x1,y1,x2,y2], 'score': conf, 'image_id': id}
        ground_truths: List of {'box': [x1,y1,x2,y2], 'image_id': id}
        iou_threshold: IoU threshold for match
        
    Returns:
        AP value
    """
    if not predictions or not ground_truths:
        return 0.0
    
    # Sort predictions by confidence
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    # Track matched ground truths
    gt_matched = {i: False for i in range(len(ground_truths))}
    
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    
    for pred_idx, pred in enumerate(predictions):
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if gt['image_id'] != pred['image_id']:
                continue
            if gt_matched[gt_idx]:
                continue
            
            iou = compute_iou_boxes(pred['box'], gt['box'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[pred_idx] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[pred_idx] = 1
    
    # Cumulative sums
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Precision and Recall
    recall = tp_cumsum / len(ground_truths)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # AP (area under PR curve - 11-point interpolation)
    ap = 0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t]
        ap += max(p) if len(p) > 0 else 0
    ap /= 11
    
    return ap


def compute_map(
    all_predictions: Dict[str, List[Dict]],
    all_ground_truths: Dict[str, List[Dict]],
    iou_thresholds: List[float] = [0.5, 0.75]
) -> Dict[str, float]:
    """
    Compute mAP across all classes at various IoU thresholds.
    
    Args:
        all_predictions: {class_name: [predictions]}
        all_ground_truths: {class_name: [ground_truths]}
        iou_thresholds: List of IoU thresholds
        
    Returns:
        Dict with mAP values
    """
    results = {}
    
    for iou_threshold in iou_thresholds:
        aps = []
        for class_name in all_ground_truths.keys():
            preds = all_predictions.get(class_name, [])
            gts = all_ground_truths.get(class_name, [])
            ap = compute_ap(preds, gts, iou_threshold)
            aps.append(ap)
            results[f'AP_{class_name}@{iou_threshold}'] = ap
        
        results[f'mAP@{iou_threshold}'] = np.mean(aps) if aps else 0
    
    # COCO-style mAP (average across 0.5:0.95:0.05)
    coco_ious = np.arange(0.5, 1.0, 0.05)
    coco_aps = []
    for iou in coco_ious:
        for class_name in all_ground_truths.keys():
            preds = all_predictions.get(class_name, [])
            gts = all_ground_truths.get(class_name, [])
            coco_aps.append(compute_ap(preds, gts, iou))
    
    results['mAP@0.5:0.95'] = np.mean(coco_aps) if coco_aps else 0
    
    return results


def run_map_analysis(
    model,
    test_data,
    class_names: list,
    results_dir: Path,
    confidence_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Run comprehensive mAP analysis.
    
    Args:
        model: Trained detection model
        test_data: Test dataset with images and annotations
        class_names: List of class names
        results_dir: Directory to save results
        confidence_threshold: Confidence threshold for predictions
    """
    print("=" * 60)
    print("DETECTION mAP ANALYSIS")
    print("=" * 60)
    
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # TODO: Implement model inference and ground truth extraction
    # This is a placeholder - actual implementation depends on model format
    
    print("\n[INFO] mAP analysis requires:")
    print("  1. Model predictions in format: {'class': [{'box', 'score', 'image_id'}]}")
    print("  2. Ground truths in format: {'class': [{'box', 'image_id'}]}")
    print("\nFor production use, consider using:")
    print("  - COCO API: pycocotools")
    print("  - Ultralytics mAP: model.val()")
    
    # Example output structure
    results = {
        'metric': ['mAP@0.5', 'mAP@0.75', 'mAP@0.5:0.95'],
        'value': [0.0, 0.0, 0.0]  # Placeholder
    }
    
    df = pd.DataFrame(results)
    df.to_csv(results_dir / "map_analysis.csv", index=False)
    
    print(f"\n{'='*60}")
    print("Analysis template created.")
    print(f"Results: {results_dir / 'map_analysis.csv'}")
    print("=" * 60)
    
    return df