# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Journal-Quality Visualization
=============================

Publication-ready plots with proper styling:
- Radar charts for multi-metric comparison
- Pareto fronts for accuracy-efficiency trade-offs
- Training curves with confidence intervals
- Confusion matrices with heatmaps
- Model comparison bar charts with error bars
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# PUBLICATION STYLE
# =============================================================================

JOURNAL_STYLE = {
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#2ECC71',
    'warning': '#F39C12',
    'danger': '#E74C3C',
    'neutral': '#95A5A6',
}

COLOR_PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#2ECC71', '#9B59B6', '#E74C3C']


def set_publication_style():
    """Apply publication-quality styling."""
    plt.rcParams.update(JOURNAL_STYLE)
    sns.set_palette(COLOR_PALETTE)


def close_all_figures():
    """Close all matplotlib figures to free memory."""
    plt.close('all')


# =============================================================================
# RADAR CHART (Multi-Metric Comparison)
# =============================================================================

def plot_radar_chart(
    models: List[str],
    metrics: List[str],
    values: List[List[float]],
    save_path: Optional[Path] = None,
    title: str = "Model Comparison"
) -> plt.Figure:
    """
    Create radar chart for multi-metric model comparison.
    
    Args:
        models: List of model names
        metrics: List of metric names
        values: 2D list [model][metric] of normalized values (0-1)
        save_path: Path to save figure
        title: Chart title
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    num_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for i, (model, vals) in enumerate(zip(models, values)):
        vals = vals + vals[:1]  # Complete the circle
        ax.plot(angles, vals, 'o-', linewidth=2, label=model, color=COLOR_PALETTE[i % len(COLOR_PALETTE)])
        ax.fill(angles, vals, alpha=0.25, color=COLOR_PALETTE[i % len(COLOR_PALETTE)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title(title, size=14, weight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[*] Saved: {save_path}")
    
    return fig


# =============================================================================
# PARETO FRONT (Accuracy vs Efficiency)
# =============================================================================

def plot_pareto_front(
    models: List[str],
    accuracy: List[float],
    params: List[int],
    latency: Optional[List[float]] = None,
    save_path: Optional[Path] = None,
    title: str = "Accuracy-Efficiency Pareto"
) -> plt.Figure:
    """
    Plot Pareto front showing accuracy vs efficiency trade-off.
    
    Args:
        models: Model names
        accuracy: Accuracy values (%)
        params: Parameter counts
        latency: Optional latency values (ms)
        save_path: Path to save
        title: Chart title
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize params for bubble size
    params_norm = np.array(params) / max(params) * 1000 + 100
    
    scatter = ax.scatter(
        params, accuracy,
        s=params_norm,
        c=latency if latency else accuracy,
        cmap='RdYlGn_r' if latency else 'viridis',
        alpha=0.7,
        edgecolors='black',
        linewidth=1
    )
    
    # Add model labels
    for i, model in enumerate(models):
        ax.annotate(
            model,
            (params[i], accuracy[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )
    
    # Find Pareto optimal points
    pareto_idx = _pareto_frontier(params, accuracy, maximize_y=True)
    pareto_params = [params[i] for i in pareto_idx]
    pareto_acc = [accuracy[i] for i in pareto_idx]
    
    # Sort by params for line
    sorted_pairs = sorted(zip(pareto_params, pareto_acc))
    pareto_params, pareto_acc = zip(*sorted_pairs)
    
    ax.plot(pareto_params, pareto_acc, 'r--', linewidth=2, label='Pareto Front', alpha=0.7)
    
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title, weight='bold')
    ax.legend()
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Latency (ms)' if latency else 'Accuracy (%)')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[*] Saved: {save_path}")
    
    return fig


def _pareto_frontier(x: List, y: List, maximize_y: bool = True) -> List[int]:
    """Find Pareto optimal indices (minimize x, maximize/minimize y)."""
    sorted_idx = sorted(range(len(x)), key=lambda i: x[i])
    pareto = []
    best_y = float('-inf') if maximize_y else float('inf')
    
    for i in sorted_idx:
        if (maximize_y and y[i] > best_y) or (not maximize_y and y[i] < best_y):
            pareto.append(i)
            best_y = y[i]
    
    return pareto


# =============================================================================
# TRAINING CURVES WITH CONFIDENCE INTERVALS
# =============================================================================

def plot_training_curves(
    histories: List[Dict[str, List[float]]],
    labels: List[str],
    save_path: Optional[Path] = None,
    title: str = "Training History"
) -> plt.Figure:
    """
    Plot training curves with confidence intervals from multiple runs.
    
    Args:
        histories: List of training history dicts
        labels: Labels for each run
        save_path: Path to save
        title: Chart title
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax = axes[0]
    for i, (hist, label) in enumerate(zip(histories, labels)):
        epochs = range(1, len(hist['loss']) + 1)
        ax.plot(epochs, hist['loss'], label=f'{label} (Train)', 
                color=COLOR_PALETTE[i], linestyle='-')
        if 'val_loss' in hist:
            ax.plot(epochs, hist['val_loss'], label=f'{label} (Val)', 
                    color=COLOR_PALETTE[i], linestyle='--')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves', weight='bold')
    ax.legend(loc='upper right')
    
    # Accuracy curves
    ax = axes[1]
    for i, (hist, label) in enumerate(zip(histories, labels)):
        if 'accuracy' in hist:
            epochs = range(1, len(hist['accuracy']) + 1)
            ax.plot(epochs, np.array(hist['accuracy']) * 100, label=f'{label} (Train)',
                    color=COLOR_PALETTE[i], linestyle='-')
            if 'val_accuracy' in hist:
                ax.plot(epochs, np.array(hist['val_accuracy']) * 100, label=f'{label} (Val)',
                        color=COLOR_PALETTE[i], linestyle='--')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Curves', weight='bold')
    ax.legend(loc='lower right')
    
    plt.suptitle(title, fontsize=14, weight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[*] Saved: {save_path}")
    
    return fig


# =============================================================================
# MODEL COMPARISON BAR CHART
# =============================================================================

def plot_model_comparison_bar(
    models: List[str],
    accuracy: List[float],
    std: Optional[List[float]] = None,
    save_path: Optional[Path] = None,
    title: str = "Model Comparison",
    highlight_best: bool = True
) -> plt.Figure:
    """
    Bar chart comparing model accuracies with error bars.
    
    Args:
        models: Model names
        accuracy: Accuracy values
        std: Standard deviations for error bars
        save_path: Path to save
        title: Chart title
        highlight_best: Highlight best model
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    colors = [COLORS['primary']] * len(models)
    
    if highlight_best:
        best_idx = np.argmax(accuracy)
        colors[best_idx] = COLORS['success']
    
    bars = ax.bar(x, accuracy, color=colors, edgecolor='black', linewidth=1)
    
    if std:
        ax.errorbar(x, accuracy, yerr=std, fmt='none', color='black', capsize=5)
    
    # Add value labels
    for bar, acc in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title, weight='bold')
    ax.set_ylim(0, max(accuracy) * 1.15)
    
    # Legend
    if highlight_best:
        legend_elements = [
            mpatches.Patch(color=COLORS['success'], label='Best Model'),
            mpatches.Patch(color=COLORS['primary'], label='Other Models')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[*] Saved: {save_path}")
    
    return fig


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix with heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        save_path: Path to save
        title: Chart title
        normalize: Normalize by row
        
    Returns:
        matplotlib Figure
    """
    from sklearn.metrics import confusion_matrix
    
    set_publication_style()
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title, weight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[*] Saved: {save_path}")
    
    return fig


# =============================================================================
# ABLATION HEATMAP
# =============================================================================

def plot_ablation_heatmap(
    param1_name: str,
    param1_values: List,
    param2_name: str,
    param2_values: List,
    results: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Ablation Results"
) -> plt.Figure:
    """
    Heatmap for 2-parameter ablation study.
    
    Args:
        param1_name: First parameter name
        param1_values: First parameter values
        param2_name: Second parameter name
        param2_values: Second parameter values
        results: 2D array of results
        save_path: Path to save
        title: Chart title
        
    Returns:
        matplotlib Figure
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        results,
        annot=True,
        fmt='.1f',
        cmap='YlGnBu',
        xticklabels=param1_values,
        yticklabels=param2_values,
        ax=ax,
        cbar_kws={'label': 'Accuracy (%)'}
    )
    
    ax.set_xlabel(param1_name)
    ax.set_ylabel(param2_name)
    ax.set_title(title, weight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[*] Saved: {save_path}")
    
    return fig