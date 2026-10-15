"""
Visualization Module

Comprehensive visualization tools for academic publications:
- Model comparison plots
- Performance metrics visualization
- Training curves
- Statistical comparison charts
- Multi-seed results visualization

Generates publication-quality figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Publication settings
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
FONT_SIZE = 12
TITLE_SIZE = 14
LABEL_SIZE = 11


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.titlesize': TITLE_SIZE,
        'axes.labelsize': LABEL_SIZE,
        'xtick.labelsize': LABEL_SIZE,
        'ytick.labelsize': LABEL_SIZE,
        'legend.fontsize': LABEL_SIZE - 1,
        'figure.titlesize': TITLE_SIZE,
        'figure.dpi': FIGURE_DPI,
        'savefig.dpi': FIGURE_DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def plot_model_comparison_bars(
    results: Dict[str, Dict],
    metrics: List[str] = None,
    output_path: Path = None,
    title: str = "Model Performance Comparison"
):
    """
    Create bar chart comparing multiple models.
    
    Args:
        results: Dict mapping model names to metrics dict
        metrics: List of metrics to plot
        output_path: Output file path
        title: Plot title
    """
    if metrics is None:
        metrics = ['auroc', 'auprc', 'f1', 'brier']
    
    set_publication_style()
    
    # Prepare data
    models = list(results.keys())
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        values = []
        errors = []
        for model in models:
            if metric in results[model]:
                if 'mean' in results[model][metric]:
                    values.append(results[model][metric]['mean'])
                    errors.append(results[model][metric].get('std', 0))
                else:
                    values.append(results[model][metric])
                    errors.append(0)
            else:
                values.append(0)
                errors.append(0)
        
        # Create bars
        x_pos = np.arange(len(models))
        bars = ax.bar(x_pos, values, yerr=errors, capsize=5, alpha=0.8)
        
        # Color bars
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel(metric.upper(), fontweight='bold')
        ax.set_title(f'{metric.upper()}', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (v, e) in enumerate(zip(values, errors)):
            if e > 0:
                label = f'{v:.3f}\n±{e:.3f}'
            else:
                label = f'{v:.3f}'
            ax.text(i, v + e + 0.01, label, ha='center', va='bottom', fontsize=9)
    
    fig.suptitle(title, fontsize=TITLE_SIZE + 2, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    
    plt.close()


def plot_multi_seed_boxplot(
    results: Dict[str, Dict],
    metric: str = 'auroc',
    output_path: Path = None,
    title: str = None
):
    """
    Create boxplot for multi-seed results.
    
    Args:
        results: Dict mapping model names to results with 'values' key
        metric: Metric to plot
        output_path: Output file path
        title: Plot title
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    data = []
    labels = []
    
    for model, result in results.items():
        if metric in result and 'values' in result[metric]:
            data.append(result[metric]['values'])
            labels.append(model.upper())
    
    if not data:
        logger.warning("No data available for boxplot")
        return
    
    # Create boxplot
    bp = ax.boxplot(data, labels=labels, patch_artist=True, 
                     showmeans=True, meanline=True,
                     boxprops=dict(alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     meanprops=dict(color='blue', linewidth=2, linestyle='--'))
    
    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel(f'{metric.upper()}', fontweight='bold', fontsize=LABEL_SIZE + 1)
    ax.set_xlabel('Model', fontweight='bold', fontsize=LABEL_SIZE + 1)
    
    if title is None:
        title = f'{metric.upper()} Distribution Across Multiple Seeds'
    ax.set_title(title, fontweight='bold', fontsize=TITLE_SIZE + 1)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.tick_params(axis='x', rotation=45)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Median'),
        Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    
    plt.close()


def plot_metrics_heatmap(
    results: Dict[str, Dict],
    metrics: List[str] = None,
    output_path: Path = None,
    title: str = "Model Performance Heatmap"
):
    """
    Create heatmap of metrics across models.
    
    Args:
        results: Dict mapping model names to metrics
        metrics: List of metrics to include
        output_path: Output file path
        title: Plot title
    """
    if metrics is None:
        metrics = ['auroc', 'auprc', 'f1', 'precision', 'recall', 'specificity']
    
    set_publication_style()
    
    # Prepare data matrix
    models = list(results.keys())
    data_matrix = []
    
    for model in models:
        row = []
        for metric in metrics:
            if metric in results[model]:
                if isinstance(results[model][metric], dict):
                    value = results[model][metric].get('mean', results[model][metric].get('value', 0))
                else:
                    value = results[model][metric]
                row.append(value)
            else:
                row.append(0)
        data_matrix.append(row)
    
    df = pd.DataFrame(data_matrix, index=[m.upper() for m in models], 
                     columns=[m.upper() for m in metrics])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, len(models) * 0.8 + 1))
    
    sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0, vmax=1, cbar_kws={'label': 'Score'},
                linewidths=0.5, ax=ax)
    
    ax.set_title(title, fontweight='bold', fontsize=TITLE_SIZE + 2, pad=20)
    ax.set_ylabel('Model', fontweight='bold', fontsize=LABEL_SIZE + 1)
    ax.set_xlabel('Metric', fontweight='bold', fontsize=LABEL_SIZE + 1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    
    plt.close()


def plot_roc_curves_comparison(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_path: Path = None,
    title: str = "ROC Curve Comparison"
):
    """
    Plot ROC curves for multiple models.
    
    Args:
        roc_data: Dict mapping model names to (fpr, tpr, auc) tuples
        output_path: Output file path
        title: Plot title
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))
    
    for (model, (fpr, tpr, auc)), color in zip(roc_data.items(), colors):
        ax.plot(fpr, tpr, label=f'{model.upper()} (AUC={auc:.3f})',
                linewidth=2.5, color=color)
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random (AUC=0.500)', alpha=0.5)
    
    ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=LABEL_SIZE + 1)
    ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=LABEL_SIZE + 1)
    ax.set_title(title, fontweight='bold', fontsize=TITLE_SIZE + 2)
    ax.legend(loc='lower right', frameon=True, shadow=True)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    
    plt.close()


def plot_pr_curves_comparison(
    pr_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_path: Path = None,
    title: str = "Precision-Recall Curve Comparison"
):
    """
    Plot PR curves for multiple models.
    
    Args:
        pr_data: Dict mapping model names to (recall, precision, auprc) tuples
        output_path: Output file path
        title: Plot title
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(pr_data)))
    
    for (model, (recall, precision, auprc)), color in zip(pr_data.items(), colors):
        ax.plot(recall, precision, label=f'{model.upper()} (AP={auprc:.3f})',
                linewidth=2.5, color=color)
    
    ax.set_xlabel('Recall', fontweight='bold', fontsize=LABEL_SIZE + 1)
    ax.set_ylabel('Precision', fontweight='bold', fontsize=LABEL_SIZE + 1)
    ax.set_title(title, fontweight='bold', fontsize=TITLE_SIZE + 2)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    
    plt.close()


def plot_feature_importance_comparison(
    importance_data: Dict[str, pd.DataFrame],
    top_k: int = 20,
    output_path: Path = None,
    title: str = "Top Feature Importances"
):
    """
    Compare feature importances across models.
    
    Args:
        importance_data: Dict mapping model names to importance DataFrames
        top_k: Number of top features to show
        output_path: Output file path
        title: Plot title
    """
    set_publication_style()
    
    # Get union of top features across all models
    all_features = set()
    for df in importance_data.values():
        if 'feature' in df.columns and 'importance' in df.columns:
            top_features = df.nlargest(top_k, 'importance')['feature'].values
            all_features.update(top_features)
    
    all_features = sorted(all_features)[:top_k]
    
    # Prepare data
    models = list(importance_data.keys())
    data_matrix = []
    
    for model in models:
        df = importance_data[model]
        row = []
        for feature in all_features:
            if feature in df['feature'].values:
                importance = df[df['feature'] == feature]['importance'].values[0]
                row.append(importance)
            else:
                row.append(0)
        data_matrix.append(row)
    
    # Create grouped bar chart
    x = np.arange(len(all_features))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    for i, (model, color) in enumerate(zip(models, colors)):
        offset = (i - len(models)/2 + 0.5) * width
        ax.bar(x + offset, data_matrix[i], width, label=model.upper(), 
               color=color, alpha=0.8)
    
    ax.set_xlabel('Feature', fontweight='bold', fontsize=LABEL_SIZE + 1)
    ax.set_ylabel('Importance', fontweight='bold', fontsize=LABEL_SIZE + 1)
    ax.set_title(title, fontweight='bold', fontsize=TITLE_SIZE + 2)
    ax.set_xticks(x)
    ax.set_xticklabels(all_features, rotation=45, ha='right')
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    
    plt.close()


def plot_confusion_matrices(
    confusion_data: Dict[str, np.ndarray],
    output_path: Path = None,
    title: str = "Confusion Matrices"
):
    """
    Plot confusion matrices for multiple models.
    
    Args:
        confusion_data: Dict mapping model names to confusion matrices
        output_path: Output file path
        title: Plot title
    """
    set_publication_style()
    
    n_models = len(confusion_data)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (model, cm) in enumerate(confusion_data.items()):
        ax = axes[idx]
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   cbar_kws={'label': 'Proportion'},
                   ax=ax)
        
        ax.set_title(f'{model.upper()}', fontweight='bold', fontsize=TITLE_SIZE)
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_xlabel('Predicted Label', fontweight='bold')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(title, fontsize=TITLE_SIZE + 2, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    
    plt.close()


def plot_calibration_curves(
    calibration_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path = None,
    title: str = "Calibration Curves"
):
    """
    Plot calibration curves for multiple models.
    
    Args:
        calibration_data: Dict mapping model names to (prob_true, prob_pred) tuples
        output_path: Output file path
        title: Plot title
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(calibration_data)))
    
    for (model, (prob_true, prob_pred)), color in zip(calibration_data.items(), colors):
        ax.plot(prob_pred, prob_true, marker='o', linewidth=2.5, 
                markersize=8, label=model.upper(), color=color)
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.5)
    
    ax.set_xlabel('Mean Predicted Probability', fontweight='bold', fontsize=LABEL_SIZE + 1)
    ax.set_ylabel('Fraction of Positives', fontweight='bold', fontsize=LABEL_SIZE + 1)
    ax.set_title(title, fontweight='bold', fontsize=TITLE_SIZE + 2)
    ax.legend(loc='lower right', frameon=True, shadow=True)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    
    plt.close()


def plot_ablation_results(
    ablation_df: pd.DataFrame,
    metric_col: str = 'performance_drop',
    output_path: Path = None,
    title: str = "Ablation Study Results"
):
    """
    Visualize ablation study results.
    
    Args:
        ablation_df: DataFrame with ablation results
        metric_col: Column to plot
        output_path: Output file path
        title: Plot title
    """
    set_publication_style()
    
    # Sort by performance drop (most important first)
    df = ablation_df.sort_values(metric_col, ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(df))
    colors = ['red' if x < 0 else 'green' for x in df[metric_col]]
    
    bars = ax.barh(y_pos, df[metric_col], color=colors, alpha=0.7)
    
    # Add significance markers
    if 'significant' in df.columns:
        for i, (pos, sig) in enumerate(zip(y_pos, df['significant'])):
            if sig:
                ax.text(df[metric_col].iloc[i], pos, ' ***', 
                       va='center', fontweight='bold', fontsize=12)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['component_removed'].str.replace('_', ' ').str.title())
    ax.set_xlabel('Performance Drop (AUPRC)', fontweight='bold', fontsize=LABEL_SIZE + 1)
    ax.set_title(title, fontweight='bold', fontsize=TITLE_SIZE + 2)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add text annotation
    ax.text(0.02, 0.98, 'Negative = Component is important',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    
    plt.close()


def plot_learning_curves(
    history_data: Dict[str, Dict[str, List[float]]],
    metrics: List[str] = None,
    output_path: Path = None,
    title: str = "Learning Curves"
):
    """
    Plot learning curves for models.
    
    Args:
        history_data: Dict mapping model names to history dicts
        metrics: Metrics to plot
        output_path: Output file path
        title: Plot title
    """
    if metrics is None:
        metrics = ['loss', 'auroc']
    
    set_publication_style()
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(history_data)))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for (model, history), color in zip(history_data.items(), colors):
            if f'train_{metric}' in history:
                epochs = range(1, len(history[f'train_{metric}']) + 1)
                ax.plot(epochs, history[f'train_{metric}'], 
                       label=f'{model.upper()} (Train)',
                       linewidth=2, linestyle='--', color=color, alpha=0.7)
            
            if f'val_{metric}' in history:
                epochs = range(1, len(history[f'val_{metric}']) + 1)
                ax.plot(epochs, history[f'val_{metric}'], 
                       label=f'{model.upper()} (Val)',
                       linewidth=2.5, color=color)
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel(metric.upper(), fontweight='bold')
        ax.set_title(f'{metric.upper()} vs Epoch', fontweight='bold')
        ax.legend(loc='best', frameon=True)
        ax.grid(alpha=0.3, linestyle='--')
    
    fig.suptitle(title, fontsize=TITLE_SIZE + 2, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    
    plt.close()


def create_results_summary_figure(
    results: Dict[str, Dict],
    output_path: Path = None
):
    """
    Create comprehensive summary figure with multiple subplots.
    
    Args:
        results: Model results
        output_path: Output file path
    """
    set_publication_style()
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    models = list(results.keys())
    
    # 1. AUROC comparison
    ax1 = fig.add_subplot(gs[0, 0])
    auroc_values = [results[m].get('auroc', {}).get('mean', results[m].get('auroc', 0)) 
                    for m in models]
    auroc_errors = [results[m].get('auroc', {}).get('std', 0) for m in models]
    ax1.bar(range(len(models)), auroc_values, yerr=auroc_errors, capsize=5, alpha=0.8)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([m.upper() for m in models], rotation=45, ha='right')
    ax1.set_ylabel('AUROC', fontweight='bold')
    ax1.set_title('AUROC Comparison', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. AUPRC comparison
    ax2 = fig.add_subplot(gs[0, 1])
    auprc_values = [results[m].get('auprc', {}).get('mean', results[m].get('auprc', 0)) 
                    for m in models]
    auprc_errors = [results[m].get('auprc', {}).get('std', 0) for m in models]
    ax2.bar(range(len(models)), auprc_values, yerr=auprc_errors, capsize=5, alpha=0.8, color='coral')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels([m.upper() for m in models], rotation=45, ha='right')
    ax2.set_ylabel('AUPRC', fontweight='bold')
    ax2.set_title('AUPRC Comparison', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Multiple metrics heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    metrics_to_plot = ['auroc', 'auprc', 'f1', 'precision', 'recall']
    data_matrix = []
    for model in models:
        row = []
        for metric in metrics_to_plot:
            if metric in results[model]:
                if isinstance(results[model][metric], dict):
                    value = results[model][metric].get('mean', 0)
                else:
                    value = results[model][metric]
                row.append(value)
            else:
                row.append(0)
        data_matrix.append(row)
    
    im = ax3.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax3.set_xticks(range(len(metrics_to_plot)))
    ax3.set_xticklabels([m.upper() for m in metrics_to_plot], rotation=45, ha='right')
    ax3.set_yticks(range(len(models)))
    ax3.set_yticklabels([m.upper() for m in models])
    ax3.set_title('Metrics Heatmap', fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Score')
    
    # 4-6. Additional metrics
    metric_names = ['f1', 'precision', 'recall']
    positions = [(1, 0), (1, 1), (1, 2)]
    colors_list = ['lightblue', 'lightgreen', 'lightyellow']
    
    for metric, pos, color in zip(metric_names, positions, colors_list):
        ax = fig.add_subplot(gs[pos])
        values = [results[m].get(metric, {}).get('mean', results[m].get(metric, 0)) 
                 for m in models]
        errors = [results[m].get(metric, {}).get('std', 0) for m in models]
        ax.bar(range(len(models)), values, yerr=errors, capsize=5, alpha=0.8, color=color)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.upper() for m in models], rotation=45, ha='right')
        ax.set_ylabel(metric.upper(), fontweight='bold')
        ax.set_title(f'{metric.upper()} Comparison', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Model Performance Summary', fontsize=TITLE_SIZE + 4, fontweight='bold')
    
    if output_path:
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved comprehensive summary: {output_path}")
    
    plt.close()


