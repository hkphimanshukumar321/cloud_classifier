# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Visualization Module
====================

Generate publication-quality figures for research papers.

Features:
- Training history plots
- Confusion matrices
- ROC and PR curves
- Ablation study visualizations
- Model comparison charts
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

logger = logging.getLogger(__name__)


# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

def set_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def close_all_figures():
    """Close all matplotlib figures to free memory."""
    plt.close('all')


# Color palette (colorblind-friendly)
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'success': '#28A745',
    'danger': '#DC3545',
}


# =============================================================================
# TRAINING HISTORY
# =============================================================================

def plot_training_history(
    history: Dict[str, list],
    save_path: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Training history dict
        save_path: Path to save figure
        show: Display figure
        
    Returns:
        Figure object
    """
    set_publication_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['loss'], 'o-', label='Training', 
                 color=COLORS['primary'], linewidth=2, markersize=4)
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 's-', label='Validation',
                     color=COLORS['secondary'], linewidth=2, markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    
    # Accuracy
    axes[1].plot(epochs, history['accuracy'], 'o-', label='Training',
                 color=COLORS['primary'], linewidth=2, markersize=4)
    if 'val_accuracy' in history:
        axes[1].plot(epochs, history['val_accuracy'], 's-', label='Validation',
                     color=COLORS['secondary'], linewidth=2, markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        logger.info(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[Path] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names for labels
        normalize: Normalize to percentages
        save_path: Path to save figure
        show: Display figure
        figsize: Figure size
        
    Returns:
        Figure object
    """
    from sklearn.metrics import confusion_matrix
    
    set_publication_style()
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
        fmt = '.1f'
        title = 'Confusion Matrix (Normalized %)'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar_kws={'label': 'Percentage' if normalize else 'Count'}
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        logger.info(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# ROC AND PR CURVES
# =============================================================================

def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        y_true: True labels (integer encoded)
        y_pred_prob: Prediction probabilities [N, num_classes]
        class_names: Class names
        save_path: Path to save
        show: Display figure
        
    Returns:
        Figure object
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    set_publication_style()
    
    n_classes = y_pred_prob.shape[1]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        label = class_names[i] if class_names else f'Class {i}'
        ax.plot(fpr, tpr, color=colors[i], lw=1.5, alpha=0.8,
                label=f'{label} (AUC={roc_auc:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        logger.info(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """Plot PR curves for multi-class classification."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    
    set_publication_style()
    n_classes = y_pred_prob.shape[1]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_prob[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_pred_prob[:, i])
        label = class_names[i] if class_names else f'Class {i}'
        ax.plot(recall, precision, color=colors[i], lw=1.5, alpha=0.8,
                label=f'{label} (AP={ap:.2f})')
                
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    return fig


def plot_radar_chart(
    values_list: List[List[float]],
    categories: List[str],
    labels: List[str],
    title: str = "Model Comparison",
    save_path: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """Plot a radar chart comparing multiple models across categories."""
    set_publication_style()
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for i, values in enumerate(values_list):
        vals = values + values[:1]
        ax.plot(angles, vals, 'o-', linewidth=2, label=labels[i])
        ax.fill(angles, vals, alpha=0.1)
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(title, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    return fig



# =============================================================================
# ABLATION STUDY PLOTS
# =============================================================================

def plot_ablation_study(
    results_df,
    parameter: str,
    metric: str = 'test_accuracy',
    save_path: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot ablation study results for a single parameter.
    
    Args:
        results_df: DataFrame with ablation results
        parameter: Parameter column name
        metric: Metric column name
        save_path: Path to save
        show: Display figure
        
    Returns:
        Figure object
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    grouped = results_df.groupby(parameter)[metric].agg(['mean', 'std']).reset_index()
    
    ax.errorbar(
        grouped[parameter], grouped['mean'], yerr=grouped['std'],
        fmt='o-', capsize=5, linewidth=2, markersize=8,
        color=COLORS['primary'], capthick=1.5
    )
    
    ax.set_xlabel(parameter.replace('_', ' ').title())
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Effect of {parameter.replace("_", " ").title()} on {metric.replace("_", " ").title()}')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        logger.info(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_model_comparison_bar(
    models: List[str],
    accuracies: List[float],
    stds: Optional[List[float]] = None,
    save_path: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot horizontal bar chart comparing model accuracies.
    
    Args:
        models: Model names
        accuracies: Accuracy values
        stds: Standard deviations (optional)
        save_path: Path to save
        show: Display figure
        
    Returns:
        Figure object
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(models) * 0.4)))
    
    y_pos = range(len(models))
    
    bars = ax.barh(y_pos, accuracies, xerr=stds, 
                   color=COLORS['primary'], capsize=3, height=0.7, alpha=0.85)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Model Comparison')
    
    # Annotate values
    for i, (acc, bar) in enumerate(zip(accuracies, bars)):
        ax.text(acc + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{acc:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        logger.info(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_accuracy_vs_latency(
    accuracies: List[float],
    latencies: List[float],
    model_names: List[str],
    save_path: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """
    Scatter plot of accuracy vs inference latency (Pareto front).
    
    Args:
        accuracies: Accuracy values
        latencies: Latency values (ms)
        model_names: Model names for labels
        save_path: Path to save
        show: Display figure
        
    Returns:
        Figure object
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(latencies, accuracies, s=100, c=COLORS['primary'], 
                         alpha=0.7, edgecolors='white', linewidth=1)
    
    for i, name in enumerate(model_names):
        ax.annotate(name, (latencies[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Inference Latency (ms)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs Latency Trade-off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        logger.info(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig