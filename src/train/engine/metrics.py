"""
Binary classification metrics: AUROC, AUPRC, Brier, ECE, F1, bootstrap CI.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    balanced_accuracy_score
)
from sklearn.utils import resample
import logging

logger = logging.getLogger(__name__)


def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted probabilities
        n_bins: Number of bins
    
    Returns:
        float: ECE score
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_pred[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    prefix: str = ''
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
        prefix: Prefix for metric names
    
    Returns:
        dict: Dictionary of metrics
    """
    # Binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Compute metrics
    metrics = {}
    
    # AUROC and AUPRC
    try:
        metrics[f'{prefix}auroc'] = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        metrics[f'{prefix}auroc'] = np.nan
    
    try:
        metrics[f'{prefix}auprc'] = average_precision_score(y_true, y_pred_proba)
    except ValueError:
        metrics[f'{prefix}auprc'] = np.nan
    
    # Brier score
    metrics[f'{prefix}brier'] = brier_score_loss(y_true, y_pred_proba)
    
    # ECE
    metrics[f'{prefix}ece'] = compute_ece(y_true, y_pred_proba)
    
    # Classification metrics at threshold
    metrics[f'{prefix}f1'] = f1_score(y_true, y_pred)
    metrics[f'{prefix}precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics[f'{prefix}recall'] = recall_score(y_true, y_pred, zero_division=0)
    
    # Specificity (True Negative Rate)
    metrics[f'{prefix}specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # NPV (Negative Predictive Value)
    metrics[f'{prefix}npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    # Balanced accuracy
    metrics[f'{prefix}balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    # Confusion matrix values
    metrics[f'{prefix}tp'] = int(tp)
    metrics[f'{prefix}fp'] = int(fp)
    metrics[f'{prefix}tn'] = int(tn)
    metrics[f'{prefix}fn'] = int(fn)
    
    # Prevalence
    metrics[f'{prefix}prevalence'] = np.mean(y_true)
    
    # Threshold used
    metrics[f'{prefix}threshold'] = threshold
    
    return metrics


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric_names: List[str] = None,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
    stratified: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap confidence intervals for metrics.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric_names: List of metrics to compute CIs for
        threshold: Classification threshold
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level (e.g., 0.05 for 95% CI)
        random_state: Random seed
        stratified: Use stratified bootstrap
    
    Returns:
        dict: Dictionary with keys as metric names and values as dicts with
              'mean', 'lower', 'upper', 'std'
    """
    if metric_names is None:
        metric_names = ['auroc', 'auprc', 'f1', 'precision', 'recall', 'brier']
    
    np.random.seed(random_state)
    
    n_samples = len(y_true)
    bootstrap_metrics = {name: [] for name in metric_names}
    
    for i in range(n_bootstrap):
        if stratified:
            # Stratified sampling
            indices = resample(
                np.arange(n_samples),
                n_samples=n_samples,
                stratify=y_true,
                random_state=random_state + i
            )
        else:
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred_proba[indices]
        
        # Compute metrics for this bootstrap sample
        try:
            metrics_boot = compute_classification_metrics(
                y_true_boot, y_pred_boot, threshold=threshold
            )
            
            for name in metric_names:
                if name in metrics_boot:
                    bootstrap_metrics[name].append(metrics_boot[name])
        except Exception as e:
            logger.debug(f"Bootstrap iteration {i} failed: {e}")
            continue
    
    # Compute confidence intervals
    ci_results = {}
    for name in metric_names:
        if len(bootstrap_metrics[name]) > 0:
            values = np.array(bootstrap_metrics[name])
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_results[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'lower': np.percentile(values, lower_percentile),
                'upper': np.percentile(values, upper_percentile),
            }
        else:
            ci_results[name] = {
                'mean': np.nan,
                'std': np.nan,
                'lower': np.nan,
                'upper': np.nan,
            }
    
    return ci_results


def format_metrics_table(
    metrics: Dict[str, float],
    ci: Optional[Dict[str, Dict[str, float]]] = None
) -> str:
    """
    Format metrics as a readable table string.
    
    Args:
        metrics: Dictionary of metrics
        ci: Optional confidence intervals
    
    Returns:
        str: Formatted table
    """
    lines = []
    lines.append("=" * 60)
    lines.append("METRICS")
    lines.append("=" * 60)
    
    # Primary metrics
    primary = ['auroc', 'auprc', 'brier', 'ece']
    lines.append("\nPrimary Metrics:")
    for key in primary:
        if key in metrics:
            value = metrics[key]
            if ci and key in ci:
                ci_info = ci[key]
                lines.append(f"  {key.upper():20s}: {value:.4f} [{ci_info['lower']:.4f}, {ci_info['upper']:.4f}]")
            else:
                lines.append(f"  {key.upper():20s}: {value:.4f}")
    
    # Classification metrics
    classification = ['f1', 'precision', 'recall', 'specificity', 'npv', 'balanced_accuracy']
    lines.append("\nClassification Metrics:")
    for key in classification:
        if key in metrics:
            value = metrics[key]
            if ci and key in ci:
                ci_info = ci[key]
                lines.append(f"  {key.capitalize():20s}: {value:.4f} [{ci_info['lower']:.4f}, {ci_info['upper']:.4f}]")
            else:
                lines.append(f"  {key.capitalize():20s}: {value:.4f}")
    
    # Confusion matrix
    if all(k in metrics for k in ['tp', 'fp', 'tn', 'fn']):
        lines.append("\nConfusion Matrix:")
        lines.append(f"  TP: {metrics['tp']:6d}    FP: {metrics['fp']:6d}")
        lines.append(f"  FN: {metrics['fn']:6d}    TN: {metrics['tn']:6d}")
    
    # Threshold
    if 'threshold' in metrics:
        lines.append(f"\nThreshold: {metrics['threshold']:.4f}")
    
    lines.append("=" * 60)
    
    return '\n'.join(lines)


def select_best_threshold_youden(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_thresholds: int = 1000
) -> Tuple[float, float]:
    """
    Select best threshold using Youden's J statistic.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_thresholds: Number of thresholds to try
    
    Returns:
        Tuple of (best_threshold, best_j)
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    best_j = -1
    best_thresh = 0.5
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        j = sensitivity + specificity - 1
        
        if j > best_j:
            best_j = j
            best_thresh = thresh
    
    return best_thresh, best_j


def select_best_threshold_f1(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_thresholds: int = 1000
) -> Tuple[float, float]:
    """
    Select best threshold by maximizing F1 score.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_thresholds: Number of thresholds to try
    
    Returns:
        Tuple of (best_threshold, best_f1)
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    best_f1 = -1
    best_thresh = 0.5
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    return best_thresh, best_f1


def select_threshold_fixed_specificity(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    target_specificity: float = 0.90,
    n_thresholds: int = 1000
) -> float:
    """
    Select threshold to achieve target specificity.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        target_specificity: Target specificity value
        n_thresholds: Number of thresholds to try
    
    Returns:
        float: Selected threshold
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    best_thresh = 0.5
    min_diff = float('inf')
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        diff = abs(specificity - target_specificity)
        
        if diff < min_diff:
            min_diff = diff
            best_thresh = thresh
    
    return best_thresh


def select_threshold_fixed_sensitivity(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    target_sensitivity: float = 0.90,
    n_thresholds: int = 1000
) -> float:
    """
    Select threshold to achieve target sensitivity.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        target_sensitivity: Target sensitivity value
        n_thresholds: Number of thresholds to try
    
    Returns:
        float: Selected threshold
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    best_thresh = 0.5
    min_diff = float('inf')
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        diff = abs(sensitivity - target_sensitivity)
        
        if diff < min_diff:
            min_diff = diff
            best_thresh = thresh
    
    return best_thresh

