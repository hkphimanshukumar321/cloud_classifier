# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Detection Models
================

Includes:
1. Custom SSD-lite detector
2. SOTA Reference: YOLOv8
3. RT-DETR reference
"""

from tensorflow.keras import layers, Model, Input
from typing import Tuple, List


def create_detection_model(
    input_shape: Tuple[int, int, int] = (416, 416, 3),
    num_classes: int = 10,
    num_anchors: int = 3
) -> Model:
    """
    Custom SSD-lite style Object Detector.
    
    Args:
        input_shape: (H, W, C)
        num_classes: Number of object classes
        num_anchors: Anchors per grid cell
        
    Returns:
        Keras Model
    """
    inputs = Input(shape=input_shape)
    
    # Backbone
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Detection heads at multiple scales
    # Output: (batch, grid_h, grid_w, anchors * (5 + classes))
    # 5 = [x, y, w, h, objectness]
    
    output_channels = num_anchors * (5 + num_classes)
    
    # Large objects (26x26 grid for 416 input)
    det_large = layers.Conv2D(output_channels, 1, name='detection_large')(x)
    
    # Medium objects
    x = layers.Conv2D(512, 3, strides=2, padding='same', activation='relu')(x)
    det_medium = layers.Conv2D(output_channels, 1, name='detection_medium')(x)
    
    # Small objects
    x = layers.Conv2D(1024, 3, strides=2, padding='same', activation='relu')(x)
    det_small = layers.Conv2D(output_channels, 1, name='detection_small')(x)
    
    return Model(inputs, [det_large, det_medium, det_small], name="SSD_Lite")


# =============================================================================
# SOTA REFERENCES
# =============================================================================

def get_yolov8_model_info() -> dict:
    """
    YOLOv8 - Ultralytics
    
    Current SOTA for real-time object detection. Requires:
    - pip install ultralytics
    
    Paper: YOLOv8 Docs (Ultralytics, 2023)
    GitHub: https://github.com/ultralytics/ultralytics
    """
    return {
        'name': 'YOLOv8',
        'variants': ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
        'url': 'https://github.com/ultralytics/ultralytics',
        'install': 'pip install ultralytics',
        'usage': '''
from ultralytics import YOLO

# Load pretrained
model = YOLO('yolov8n.pt')

# Train on custom data
model.train(data='dataset.yaml', epochs=100)

# Inference
results = model('image.jpg')
        ''',
        'strengths': [
            'Best speed-accuracy trade-off',
            'Easy to use API',
            'Supports detection, segmentation, classification'
        ],
        'use_case': 'Production real-time detection'
    }


def get_rtdetr_model_info() -> dict:
    """
    RT-DETR - Baidu
    
    Real-time DETR (Detection Transformer). Requires:
    - pip install ultralytics
    
    Paper: "RT-DETR: DETRs Beat YOLOs on Real-time Object Detection" (2023)
    """
    return {
        'name': 'RT-DETR',
        'paper': 'DETRs Beat YOLOs on Real-time Object Detection (2023)',
        'url': 'https://github.com/ultralytics/ultralytics',
        'install': 'pip install ultralytics',
        'usage': '''
from ultralytics import RTDETR

model = RTDETR('rtdetr-l.pt')
results = model('image.jpg')
        ''',
        'strengths': [
            'Transformer-based (no NMS needed)',
            'Better accuracy than YOLO at same speed',
            'End-to-end detection'
        ],
        'use_case': 'When accuracy matters more than speed'
    }


def get_efficientdet_model_info() -> dict:
    """
    EfficientDet - Google
    
    Scalable and efficient object detection.
    
    Paper: "EfficientDet: Scalable and Efficient Object Detection" (2020)
    """
    return {
        'name': 'EfficientDet',
        'paper': 'EfficientDet: Scalable and Efficient Object Detection (2020)',
        'variants': ['D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7'],
        'tensorflow': 'tf.keras.applications.EfficientNetB0 + BiFPN',
        'strengths': [
            'Compound scaling',
            'BiFPN feature fusion',
            'High efficiency'
        ],
        'use_case': 'When you need scalable detection with TensorFlow'
    }


# =============================================================================
# MODEL REGISTRY
# =============================================================================

DETECTION_MODELS = {
    'ssd_lite': create_detection_model,
}

SOTA_REFERENCES = {
    'YOLOv8': get_yolov8_model_info,
    'RT-DETR': get_rtdetr_model_info,
    'EfficientDet': get_efficientdet_model_info,
}