# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Model Architecture Module
=========================

This module contains model definitions for your research.
Now updated with your Custom RF-DenseNet Architecture.
"""

import logging
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import (
    VGG16, VGG19,
    ResNet50V2, ResNet101V2, ResNet152V2,
    DenseNet121, DenseNet169, DenseNet201,
    MobileNetV2,
    EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3,
    InceptionV3, InceptionResNetV2,
    Xception,
    NASNetMobile,
    ConvNeXtTiny, ConvNeXtSmall
)

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL METRICS
# =============================================================================

@dataclass
class ModelMetrics:
    """Container for model architecture metrics."""
    total_params: int
    trainable_params: int
    non_trainable_params: int
    memory_mb: float
    
    def __str__(self) -> str:
        return (
            f"Parameters: {self.total_params:,} "
            f"(Trainable: {self.trainable_params:,})\n"
            f"Memory: {self.memory_mb:.2f} MB"
        )


def get_model_metrics(model: Model) -> ModelMetrics:
    """Compute model metrics for analysis."""
    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) 
        for w in model.trainable_weights
    )
    non_trainable_params = total_params - trainable_params
    memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
    
    return ModelMetrics(
        total_params=total_params,
        trainable_params=trainable_params,
        non_trainable_params=non_trainable_params,
        memory_mb=memory_mb
    )


# =============================================================================
# YOUR CUSTOM MODEL: RF-DenseNet
# =============================================================================

def _dense_block(
    x: tf.Tensor,
    num_layers: int,
    growth_rate: int,
    name: str
) -> tf.Tensor:
    """Dense Block implementation for RF-DenseNet."""
    for i in range(num_layers):
        # Composite function: BN → ReLU → Conv
        out = layers.BatchNormalization(name=f"{name}_bn_{i}")(x)
        out = layers.Activation('relu', name=f"{name}_relu_{i}")(out)
        out = layers.Conv2D(
            filters=growth_rate,
            kernel_size=3,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            name=f"{name}_conv_{i}"
        )(out)
        
        # Dense connection: concatenate input with output
        x = layers.Concatenate(name=f"{name}_concat_{i}")([x, out])
    
    return x


def _transition_block(
    x: tf.Tensor,
    compression: float,
    name: str
) -> tf.Tensor:
    """Transition Block for feature map compression."""
    # Compute reduced filter count
    num_filters = int(x.shape[-1])
    reduced_filters = max(1, int(num_filters * compression))
    
    # Compression pathway
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation('relu', name=f"{name}_relu")(x)
    x = layers.Conv2D(
        filters=reduced_filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        name=f"{name}_conv"
    )(x)
    
    # Spatial downsampling
    x = layers.AveragePooling2D(pool_size=2, strides=2, name=f"{name}_pool")(x)
    
    return x


def create_custom_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    growth_rate: int = 8,
    compression: float = 0.5,
    depth: Tuple[int, int, int] = (3, 3, 3),
    dropout_rate: float = 0.2,
    initial_filters: int = 16,
    name: str = "RF_DenseNet"
) -> Model:
    """
    Creates the RF-DenseNet architecture.
    """
    inputs = Input(shape=input_shape, name="input")
    
    # Initial Convolution
    x = layers.BatchNormalization(name="initial_bn")(inputs)
    x = layers.Conv2D(
        filters=initial_filters,
        kernel_size=3,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        name="initial_conv"
    )(x)
    x = layers.Activation('relu', name="initial_relu")(x)
    
    # Dense Blocks + Transition Layers
    for block_idx, num_layers in enumerate(depth):
        x = _dense_block(
            x,
            num_layers=num_layers,
            growth_rate=growth_rate,
            name=f"dense_block_{block_idx}"
        )
        
        # Add transition after each block except the last
        if block_idx < len(depth) - 1:
            x = _transition_block(
                x,
                compression=compression,
                name=f"transition_{block_idx}"
            )
    
    # Classification Head
    x = layers.BatchNormalization(name="final_bn")(x)
    x = layers.Activation('relu', name="final_relu")(x)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="dropout")(x)
    
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        kernel_initializer='he_normal',
        name="predictions"
    )(x)
    
    model = Model(inputs, outputs, name=name)
    logger.info(f"Created {name} with {model.count_params():,} parameters")
    
    return model


def create_simple_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    name: str = "SimpleCNN"
) -> Model:
    """Simple 3-layer CNN as minimal baseline."""
    inputs = Input(shape=input_shape, name="input")
    
    x = layers.Conv2D(32, 3, padding='same', activation='relu', name="conv1")(inputs)
    x = layers.MaxPooling2D(2, name="pool1")(x)
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu', name="conv2")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)
    
    x = layers.Conv2D(128, 3, padding='same', activation='relu', name="conv3")(x)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="predictions")(x)
    
    return Model(inputs, outputs, name=name)


# =============================================================================
# MAIN INTERFACE
# =============================================================================

def create_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    **kwargs
) -> Model:
    """Delegates to custom model."""
    return create_custom_model(input_shape, num_classes, **kwargs)


# =============================================================================
# BASELINE MODELS
# =============================================================================

BASELINE_MODELS = {
    "VGG16": VGG16,
    "VGG19": VGG19,
    "ResNet50V2": ResNet50V2,
    "ResNet101V2": ResNet101V2,
    "ResNet152V2": ResNet152V2,
    "DenseNet121": DenseNet121,
    "DenseNet169": DenseNet169,
    "DenseNet201": DenseNet201,
    "MobileNetV2": MobileNetV2,
    "EfficientNetV2B0": EfficientNetV2B0,
    "EfficientNetV2B1": EfficientNetV2B1,
    "EfficientNetV2B2": EfficientNetV2B2,
    "EfficientNetV2B3": EfficientNetV2B3,
    "InceptionV3": InceptionV3,
    "InceptionResNetV2": InceptionResNetV2,
    "Xception": Xception,
    "NASNetMobile": NASNetMobile,
    "ConvNeXtTiny": ConvNeXtTiny,
    "ConvNeXtSmall": ConvNeXtSmall,
}

def create_baseline_model(
    model_name: str,
    input_shape: Tuple[int, int, int],
    num_classes: int,
    use_pretrained: bool = True,
    freeze_base: bool = True,
    dropout_rate: float = 0.2
) -> Model:
    """Create a baseline model using transfer learning."""
    if model_name not in BASELINE_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    weights = 'imagenet' if use_pretrained else None
    
    base_model = BASELINE_MODELS[model_name](
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        pooling='avg'
    )
    
    if freeze_base:
        base_model.trainable = False
    
    inputs = Input(shape=input_shape, name="input")
    x = base_model(inputs, training=not freeze_base)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="dropout")(x)
    
    outputs = layers.Dense(
        num_classes, activation='softmax',
        kernel_initializer='he_normal', name="predictions"
    )(x)
    
    model = Model(inputs, outputs, name=f"{model_name}_transfer")
    logger.info(f"Created {model_name} baseline with {model.count_params():,} parameters")
    
    return model

def get_all_model_variants() -> Dict[str, callable]:
    models = {
        "CustomModel": create_custom_model,
        "SimpleCNN": create_simple_cnn,
    }
    for name in BASELINE_MODELS:
        models[name] = lambda inp, nc, n=name: create_baseline_model(n, inp, nc)
    return models
