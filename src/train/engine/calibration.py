"""
Calibration curve computation.
"""

import numpy as np
from sklearn.calibration import calibration_curve as sklearn_calibration_curve


def get_calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform'):
    """
    Compute calibration curve for binary classification.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities for positive class
        n_bins: Number of bins for calibration curve
        strategy: Strategy for binning ('uniform' or 'quantile')
    
    Returns:
        tuple: (mean_predicted_value, fraction_of_positives, bin_edges)
            - mean_predicted_value: Mean predicted probability in each bin
            - fraction_of_positives: Fraction of positive samples in each bin
            - bin_edges: Edges of the bins used
    """
    # sklearn's calibration_curve returns (fraction_of_positives, mean_predicted_value)
    # We swap the order to match the expected output format
    frac_pos, mean_pred = sklearn_calibration_curve(
        y_true, 
        y_prob, 
        n_bins=n_bins,
        strategy=strategy
    )
    
    # Compute bin edges for reference
    if strategy == 'uniform':
        bin_edges = np.linspace(0, 1, n_bins + 1)
    else:
        # For quantile strategy, compute actual bin edges
        bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    
    return mean_pred, frac_pos, bin_edges


def compute_expected_calibration_error(y_true, y_prob, n_bins=10, strategy='uniform'):
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted probabilities and actual frequencies.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        strategy: Binning strategy
    
    Returns:
        float: Expected Calibration Error
    """
    mean_pred, frac_pos, _ = get_calibration_curve(y_true, y_prob, n_bins, strategy)
    
    # Compute sample weights for each bin
    if strategy == 'uniform':
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    else:
        bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(y_prob, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Count samples in each bin
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    
    # Compute ECE
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(len(mean_pred)):
        if bin_counts[i] > 0:
            weight = bin_counts[i] / total_samples
            ece += weight * np.abs(mean_pred[i] - frac_pos[i])
    
    return ece


def compute_maximum_calibration_error(y_true, y_prob, n_bins=10, strategy='uniform'):
    """
    Compute Maximum Calibration Error (MCE).
    
    MCE is the maximum difference between predicted and actual frequencies.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        strategy: Binning strategy
    
    Returns:
        float: Maximum Calibration Error
    """
    mean_pred, frac_pos, _ = get_calibration_curve(y_true, y_prob, n_bins, strategy)
    
    # MCE is the maximum absolute difference
    mce = np.max(np.abs(mean_pred - frac_pos))
    
    return mce
