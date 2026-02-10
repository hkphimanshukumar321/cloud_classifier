# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Model Architecture Module
=========================

CloudDenseNet-Lite: A novel lightweight architecture for cloud density
estimation from UAV imagery.

Key innovations:
1. DS-Dense Blocks: DenseNet connectivity + MobileNet separable convolutions
2. Coordinate Attention: Encodes spatial cloud patterns efficiently
3. Low-Resolution Design: Operates on 64x64 inputs for UAV efficiency
4. Ordinal Head: Predicts density levels (not classes) for visibility

Supports two output modes:
- output_mode="ordinal"  -> (K-1) sigmoid thresholds + BCE loss (recommended)
- output_mode="softmax"  -> K-way softmax (standard classification)
"""

import logging
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

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
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable_params = total_params - trainable_params
    memory_mb = (total_params * 4) / (1024 * 1024)

    return ModelMetrics(
        total_params=total_params,
        trainable_params=trainable_params,
        non_trainable_params=non_trainable_params,
        memory_mb=memory_mb
    )


# =============================================================================
# ORDINAL LEARNING HELPERS
# =============================================================================

def ordinal_targets_from_int(y: tf.Tensor, num_classes: int) -> tf.Tensor:
    """
    Convert integer label y in [0..K-1] -> ordinal targets z in {0,1}^{K-1}:
      z_t = 1[y > t], for t=0..K-2

    Example (K=5):
      y=0 -> [0,0,0,0]
      y=2 -> [1,1,0,0]
      y=4 -> [1,1,1,1]
    """
    y = tf.cast(y, tf.int32)
    t = tf.range(num_classes - 1, dtype=tf.int32)
    return tf.cast(y > t, tf.float32)


def ordinal_probs_to_label(p: tf.Tensor, threshold: float = 0.5) -> tf.Tensor:
    """Convert ordinal probabilities p (B, K-1) -> integer label (B,)."""
    return tf.reduce_sum(tf.cast(p > threshold, tf.int32), axis=-1)


class OrdinalAccuracy(tf.keras.metrics.Metric):
    """Accuracy for ordinal targets (B, K-1)."""
    def __init__(self, name="ordinal_accuracy", threshold: float = 0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_lbl = tf.reduce_sum(tf.cast(y_true > 0.5, tf.int32), axis=-1)
        y_pred_lbl = tf.reduce_sum(tf.cast(y_pred > self.threshold, tf.int32), axis=-1)
        matches = tf.cast(tf.equal(y_true_lbl, y_pred_lbl), tf.float32)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            matches = matches * sample_weight
            denom = tf.reduce_sum(sample_weight)
        else:
            denom = tf.cast(tf.size(matches), tf.float32)
        self.total.assign_add(tf.reduce_sum(matches))
        self.count.assign_add(denom)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


class OrdinalMAE(tf.keras.metrics.Metric):
    """MAE on class index for ordinal outputs (severity error)."""
    def __init__(self, name="ordinal_mae", threshold: float = 0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_lbl = tf.cast(
            tf.reduce_sum(tf.cast(y_true > 0.5, tf.int32), axis=-1), tf.float32
        )
        y_pred_lbl = tf.cast(
            tf.reduce_sum(tf.cast(y_pred > self.threshold, tf.int32), axis=-1), tf.float32
        )
        err = tf.abs(y_true_lbl - y_pred_lbl)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            err = err * sample_weight
            denom = tf.reduce_sum(sample_weight)
        else:
            denom = tf.cast(tf.size(err), tf.float32)
        self.total.assign_add(tf.reduce_sum(err))
        self.count.assign_add(denom)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


# =============================================================================
# COORDINATE ATTENTION
# =============================================================================

class CoordinateAttention(layers.Layer):
    """
    Coordinate Attention (Hou et al., CVPR 2021).

    Encodes long-range dependencies along one spatial direction while
    preserving positional information along the other. Ideal for cloud
    density where horizontal/vertical streaks carry density information.
    """
    def __init__(self, reduction: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        C = input_shape[-1]
        if C is None:
            # Shape unknown at build time (e.g. from pretrained sub-model)
            # Use a reasonable default; will be rebuilt on first call
            C = 24
        mid = max(8, C // self.reduction)
        self.conv_reduce = layers.Conv2D(
            mid, 1, padding='same', use_bias=False,
            kernel_initializer='he_normal', name=f"{self.name}_reduce"
        )
        self.bn = layers.BatchNormalization(name=f"{self.name}_bn")
        self.conv_h = layers.Conv2D(
            C, 1, padding='same', use_bias=False,
            kernel_initializer='he_normal', name=f"{self.name}_conv_h"
        )
        self.conv_w = layers.Conv2D(
            C, 1, padding='same', use_bias=False,
            kernel_initializer='he_normal', name=f"{self.name}_conv_w"
        )
        super().build(input_shape)

    def call(self, x):
        _, H, W, C = tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
        pool_h = tf.reduce_mean(x, axis=2, keepdims=True)
        pool_w = tf.reduce_mean(x, axis=1, keepdims=True)
        pool_w_t = tf.transpose(pool_w, perm=[0, 2, 1, 3])

        y = tf.concat([pool_h, pool_w_t], axis=1)
        y = self.conv_reduce(y)
        y = self.bn(y)
        y = tf.nn.swish(y)

        h_att, w_att = tf.split(y, [H, W], axis=1)
        w_att = tf.transpose(w_att, perm=[0, 2, 1, 3])

        h_att = tf.sigmoid(self.conv_h(h_att))
        w_att = tf.sigmoid(self.conv_w(w_att))

        return x * h_att * w_att


# =============================================================================
# DS-DENSE BLOCK (Novel: DenseNet + MobileNet Separable Convs)
# =============================================================================

def _ds_dense_block(
    x: tf.Tensor,
    num_layers: int,
    growth_rate: int,
    name: str,
    use_coord_att: bool = True
) -> tf.Tensor:
    """
    Depthwise-Separable Dense Block (DS-Dense Block).

    Novel: Replaces 3x3 Conv in DenseNet with DepthwiseConv3x3 +
    PointwiseConv1x1 (MobileNet-style), reducing parameters ~8x
    while preserving dense connectivity and feature reuse.
    """
    for i in range(num_layers):
        layer_input = x

        out = layers.BatchNormalization(name=f"{name}_bn_{i}")(x)
        out = layers.Activation('swish', name=f"{name}_swish_{i}")(out)

        # Depthwise Separable Conv (MobileNet-style)
        out = layers.DepthwiseConv2D(
            kernel_size=3, padding='same', use_bias=False,
            depthwise_initializer='he_normal',
            name=f"{name}_dw_{i}"
        )(out)
        out = layers.BatchNormalization(name=f"{name}_dw_bn_{i}")(out)
        out = layers.Activation('swish', name=f"{name}_dw_swish_{i}")(out)

        out = layers.Conv2D(
            filters=growth_rate, kernel_size=1, padding='same',
            use_bias=False, kernel_initializer='he_normal',
            name=f"{name}_pw_{i}"
        )(out)

        # Dense connection
        x = layers.Concatenate(name=f"{name}_concat_{i}")([layer_input, out])

        # Coordinate Attention every 2 layers
        if use_coord_att and (i % 2 == 1):
            x = CoordinateAttention(
                reduction=4, name=f"{name}_ca_{i}"
            )(x)

    return x


# =============================================================================
# TRANSITION BLOCK
# =============================================================================

def _lite_transition(
    x: tf.Tensor,
    compression: float,
    name: str
) -> tf.Tensor:
    """Lightweight Transition: BN -> Swish -> Conv1x1(compress) -> AvgPool(2x2)."""
    # Handle potentially None static shape (from pretrained sub-model)
    num_filters = x.shape[-1]
    if num_filters is None:
        num_filters = tf.keras.backend.int_shape(x)[-1] or 24
    num_filters = int(num_filters)
    reduced = max(8, int(num_filters * compression))

    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation('swish', name=f"{name}_swish")(x)
    x = layers.Conv2D(
        reduced, 1, padding='same', use_bias=False,
        kernel_initializer='he_normal', name=f"{name}_conv"
    )(x)
    x = layers.AveragePooling2D(2, strides=2, name=f"{name}_pool")(x)
    return x


# =============================================================================
# CloudDenseNet-Lite MODEL BUILDER
# =============================================================================

def create_cloud_densenet_lite(
    input_shape: Tuple[int, int, int] = (64, 64, 3),
    num_classes: int = 5,
    growth_rate: int = 12,
    compression: float = 0.5,
    depth: Tuple[int, ...] = (3, 4, 3),
    initial_filters: int = 24,
    dropout_rate: float = 0.30,
    weight_decay: float = 1e-4,
    use_coord_att: bool = True,
    use_in_model_aug: bool = True,
    use_pretrained_stem: bool = False,
    output_mode: str = "softmax",
    name: str = "CloudDenseNet_Lite"
) -> Model:
    """
    CloudDenseNet-Lite: Novel lightweight architecture for cloud density.

    Novelty:
    1. DS-Dense Blocks: DenseNet + MobileNet separable convolutions
    2. Coordinate Attention: Spatial cloud pattern encoding
    3. Optional Pretrained Stem: MobileNetV2 first block for transfer learning
    4. Ordinal Head: Density levels (not classes) for visibility

    Expected: ~40-150K parameters, <2MB memory, <5ms inference.
    """
    if output_mode not in ("softmax", "ordinal"):
        raise ValueError("output_mode must be 'softmax' or 'ordinal'")

    inputs = Input(shape=input_shape, name="input")
    x = layers.Lambda(lambda t: tf.cast(t, tf.float32), name="to_float32")(inputs)

    # In-model augmentation (only active during training)
    if use_in_model_aug:
        aug = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.10),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.20),
        ], name="augment")
        x = aug(x)

    # Stem
    if use_pretrained_stem:
        # Use MobileNetV2 first inverted residual as pretrained stem
        # Rescale [0,1] -> [-1,1] for MobileNetV2 compatibility
        x_scaled = layers.Rescaling(scale=2.0, offset=-1.0, name="rescale_for_mobilenet")(x)

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        # Extract first inverted residual block output (lightweight: ~1.7K params)
        stem_layer = base_model.get_layer('expanded_conv_project_BN')
        stem_extractor = tf.keras.Model(
            inputs=base_model.input,
            outputs=stem_layer.output,
            name='pretrained_stem'
        )
        # Freeze pretrained weights
        for layer in stem_extractor.layers:
            layer.trainable = False

        x = stem_extractor(x_scaled)

        # Project pretrained features to initial_filters
        x = layers.Conv2D(
            initial_filters, 1, padding='same', use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            name="stem_project"
        )(x)
        x = layers.BatchNormalization(name="stem_project_bn")(x)
        x = layers.Activation('swish', name="stem_project_swish")(x)
    else:
        # Original lightweight stem
        x = layers.Conv2D(
            initial_filters, 3, padding='same', use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            name="stem_conv"
        )(x)
        x = layers.BatchNormalization(name="stem_bn")(x)
        x = layers.Activation('swish', name="stem_swish")(x)

    # DS-Dense Blocks + Transitions
    for block_idx, num_layers in enumerate(depth):
        x = _ds_dense_block(
            x,
            num_layers=num_layers,
            growth_rate=growth_rate,
            name=f"ds_dense_{block_idx}",
            use_coord_att=use_coord_att
        )
        if block_idx < len(depth) - 1:
            x = _lite_transition(
                x, compression=compression,
                name=f"transition_{block_idx}"
            )

    # Final BN + Multi-Pool
    x = layers.BatchNormalization(name="final_bn")(x)
    x = layers.Activation('swish', name="final_swish")(x)

    gap = layers.GlobalAveragePooling2D(name="gap")(x)
    gmp = layers.GlobalMaxPooling2D(name="gmp")(x)
    x = layers.Concatenate(name="pool_concat")([gap, gmp])

    # Classification head
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    x = layers.Dense(
        64, activation='swish',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        name="fc1"
    )(x)
    x = layers.Dropout(dropout_rate / 2, name="dropout2")(x)

    # Output
    if output_mode == "softmax":
        outputs = layers.Dense(
            num_classes, activation='softmax', name='predictions'
        )(x)
    else:
        outputs = layers.Dense(
            num_classes - 1, activation='sigmoid', name='predictions_ordinal'
        )(x)

    model = Model(inputs, outputs, name=name)
    logger.info(f"Created {name} ({output_mode}) with {model.count_params():,} params")
    return model


# Aliases for backward compatibility with experiment runner
create_model = create_cloud_densenet_lite
create_custom_model = create_cloud_densenet_lite
create_rf_densenet = create_cloud_densenet_lite


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CloudDenseNet-Lite Architecture Test")
    print("=" * 60)

    print("\n1) Softmax mode:")
    m1 = create_cloud_densenet_lite((64, 64, 3), 5, output_mode="softmax")
    print(get_model_metrics(m1))
    print(f"   Output shape: {m1.output_shape}")

    print("\n2) Ordinal mode:")
    m2 = create_cloud_densenet_lite((64, 64, 3), 5, output_mode="ordinal")
    print(get_model_metrics(m2))
    print(f"   Output shape: {m2.output_shape}")
