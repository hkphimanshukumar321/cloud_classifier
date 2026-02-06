# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Segmentation Models
===================

Includes:
1. Custom U-Net (lightweight)
2. SOTA Reference: SAM (Segment Anything)
3. DeepLabV3+ reference
"""

from tensorflow.keras import layers, Model, Input
from typing import Tuple, List


def create_unet_model(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 3,
    encoder_filters: List[int] = [16, 32, 64, 128],
    dropout_rate: float = 0.2
) -> Model:
    """
    Custom U-Net for Semantic Segmentation.
    
    Args:
        input_shape: (H, W, C)
        num_classes: Number of mask classes
        encoder_filters: Filters per encoder block
        dropout_rate: Dropout rate
        
    Returns:
        Keras Model
    """
    inputs = Input(shape=input_shape)
    
    # Encoder path
    skip_connections = []
    x = inputs
    
    for filters in encoder_filters:
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        skip_connections.append(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Bottleneck
    x = layers.Conv2D(encoder_filters[-1] * 2, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Decoder path
    for i, filters in enumerate(reversed(encoder_filters)):
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
        x = layers.Concatenate()([x, skip_connections[-(i+1)]])
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
    
    # Output
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(x)
    
    return Model(inputs, outputs, name="UNet")


# =============================================================================
# SOTA REFERENCES
# =============================================================================

def get_sam_model_info() -> dict:
    """
    SAM (Segment Anything Model) - Meta AI
    
    SOTA for zero-shot segmentation. Requires:
    - pip install segment-anything
    - Pretrained weights from Meta
    
    Paper: "Segment Anything" (Kirillov et al., 2023)
    GitHub: https://github.com/facebookresearch/segment-anything
    """
    return {
        'name': 'SAM (Segment Anything)',
        'paper': 'Segment Anything (2023)',
        'url': 'https://github.com/facebookresearch/segment-anything',
        'install': 'pip install segment-anything',
        'weights': 'sam_vit_h_4b8939.pth (2.4 GB)',
        'strengths': [
            'Zero-shot segmentation',
            'Interactive prompting',
            'State-of-the-art quality'
        ],
        'use_case': 'When you need best quality without task-specific training'
    }


def get_deeplabv3_model_info() -> dict:
    """
    DeepLabV3+ - Google
    
    SOTA for semantic segmentation. Available in TensorFlow.
    
    Paper: "Encoder-Decoder with Atrous Separable Convolution" (Chen et al., 2018)
    """
    return {
        'name': 'DeepLabV3+',
        'paper': 'Encoder-Decoder with Atrous Separable Convolution (2018)',
        'tensorflow': 'tf.keras.applications.MobileNetV2 + ASPP',
        'strengths': [
            'Atrous Spatial Pyramid Pooling',
            'Multi-scale context',
            'Encoder-Decoder architecture'
        ],
        'use_case': 'When you need production-ready semantic segmentation'
    }


def create_deeplabv3_backbone(input_shape: Tuple[int, int, int]) -> Model:
    """Create DeepLabV3 backbone using MobileNetV2."""
    from tensorflow.keras.applications import MobileNetV2
    
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    
    # Extract features at multiple scales
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
    ]
    
    outputs = [base.get_layer(name).output for name in layer_names]
    
    return Model(inputs=base.input, outputs=outputs, name="DeepLabV3_Backbone")


# =============================================================================
# MODEL REGISTRY
# =============================================================================

SEGMENTATION_MODELS = {
    'unet': create_unet_model,
    'deeplabv3_backbone': create_deeplabv3_backbone,
}

SOTA_REFERENCES = {
    'SAM': get_sam_model_info,
    'DeepLabV3+': get_deeplabv3_model_info,
}