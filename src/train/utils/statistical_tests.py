"""
Statistical significance tests: DeLong, paired t-test, McNemar, permutation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from scipy import stats
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


def delong_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray
) -> Tuple[float, float, float]:
    """
    DeLong test for comparing two ROC curves.
    
    Tests the null hypothesis that two AUROC values are equal.
    
    Args:
        y_true: True labels
        y_pred1: Predicted probabilities from model 1
        y_pred2: Predicted probabilities from model 2
    
    Returns:
        tuple: (z_statistic, p_value, auc_difference)
        
    Reference:
        DeLong et al. (1988). Comparing the areas under two or more correlated
        receiver operating characteristic curves: a nonparametric approach.
    """
    from sklearn.metrics import roc_auc_score
    
    # Compute AUROCs
    auc1 = roc_auc_score(y_true, y_pred1)
    auc2 = roc_auc_score(y_true, y_pred2)
    auc_diff = auc1 - auc2
    
    # Get positive and negative samples
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)
    
    if n_pos == 0 or n_neg == 0:
        logger.warning("DeLong test requires both positive and negative samples")
        return np.nan, np.nan, auc_diff
    
    # Structural components
    X1 = y_pred1[pos_mask]
    Y1 = y_pred1[neg_mask]
    X2 = y_pred2[pos_mask]
    Y2 = y_pred2[neg_mask]
    
    # Compute V10 and V01 for both models
    V10_1 = _compute_midrank(X1)
    V01_1 = _compute_midrank(Y1)
    V10_2 = _compute_midrank(X2)
    V01_2 = _compute_midrank(Y2)
    
    # Compute covariance matrix
    V10_diff = V10_1 - V10_2
    V01_diff = V01_1 - V01_2
    
    S10 = np.var(V10_diff, ddof=1) / n_pos
    S01 = np.var(V01_diff, ddof=1) / n_neg
    
    # Compute standard error
    se = np.sqrt(S10 + S01)
    
    if se == 0:
        logger.warning("Standard error is zero in DeLong test")
        return np.nan, np.nan, auc_diff
    
    # Compute z-statistic and p-value
    z_stat = auc_diff / se
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))  # Two-tailed test
    
    return z_stat, p_value, auc_diff


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Compute midrank for DeLong test."""
    sorted_indices = np.argsort(x)
    ranks = np.empty_like(sorted_indices, dtype=float)
    ranks[sorted_indices] = np.arange(len(x))
    
    # Handle ties by assigning average rank
    for i in range(len(x)):
        ties = np.where(x == x[i])[0]
        if len(ties) > 1:
            avg_rank = np.mean(ranks[ties])
            ranks[ties] = avg_rank
    
    return ranks


def paired_ttest(
    metric_values1: np.ndarray,
    metric_values2: np.ndarray
) -> Tuple[float, float, float]:
    """
    Paired t-test for comparing two models.
    
    Used when you have multiple runs (e.g., different seeds, cross-validation folds)
    and want to test if the difference in metrics is significant.
    
    Args:
        metric_values1: Metric values from model 1 (e.g., AUROC from 5 seeds)
        metric_values2: Metric values from model 2
    
    Returns:
        tuple: (t_statistic, p_value, mean_difference)
    """
    if len(metric_values1) != len(metric_values2):
        raise ValueError("Both arrays must have the same length")
    
    if len(metric_values1) < 2:
        logger.warning("Paired t-test requires at least 2 samples")
        return np.nan, np.nan, np.nan
    
    t_stat, p_value = stats.ttest_rel(metric_values1, metric_values2)
    mean_diff = np.mean(metric_values1) - np.mean(metric_values2)
    
    return t_stat, p_value, mean_diff


def mcnemar_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray
) -> Tuple[float, float]:
    """
    McNemar's test for comparing two classifiers.
    
    Tests if two classifiers make significantly different errors.
    
    Args:
        y_true: True labels (binary)
        y_pred1: Predictions from model 1 (binary)
        y_pred2: Predictions from model 2 (binary)
    
    Returns:
        tuple: (chi2_statistic, p_value)
    """
    # Build contingency table
    # correct1_correct2, correct1_wrong2, wrong1_correct2, wrong1_wrong2
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)
    
    n01 = np.sum(correct1 & ~correct2)  # Model 1 correct, Model 2 wrong
    n10 = np.sum(~correct1 & correct2)  # Model 1 wrong, Model 2 correct
    
    # McNemar's test statistic (with continuity correction)
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if (n01 + n10) > 0 else 0
    
    # p-value from chi-square distribution with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return chi2, p_value


def permutation_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    metric_fn: callable,
    n_permutations: int = 1000,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Permutation test for comparing two models.
    
    Non-parametric test that doesn't assume normal distribution.
    
    Args:
        y_true: True labels
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
        metric_fn: Function to compute metric (e.g., lambda y, p: roc_auc_score(y, p))
        n_permutations: Number of permutations
        random_state: Random seed
    
    Returns:
        tuple: (observed_difference, p_value)
    """
    np.random.seed(random_state)
    
    # Observed difference
    metric1 = metric_fn(y_true, y_pred1)
    metric2 = metric_fn(y_true, y_pred2)
    observed_diff = metric1 - metric2
    
    # Permutation distribution
    perm_diffs = []
    
    for i in range(n_permutations):
        # Randomly swap predictions
        swap_mask = np.random.rand(len(y_true)) > 0.5
        
        perm_pred1 = np.where(swap_mask, y_pred2, y_pred1)
        perm_pred2 = np.where(swap_mask, y_pred1, y_pred2)
        
        perm_metric1 = metric_fn(y_true, perm_pred1)
        perm_metric2 = metric_fn(y_true, perm_pred2)
        perm_diff = perm_metric1 - perm_metric2
        
        perm_diffs.append(perm_diff)
    
    perm_diffs = np.array(perm_diffs)
    
    # Two-tailed p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    
    return observed_diff, p_value


def compare_multiple_models(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    metric_name: str = 'AUROC',
    test: str = 'delong'
) -> pd.DataFrame:
    """
    Pairwise comparison of multiple models.
    
    Args:
        y_true: True labels
        predictions: Dict mapping model names to predictions
        metric_name: Metric being compared
        test: Test to use ('delong', 'mcnemar', 'permutation')
    
    Returns:
        pd.DataFrame: Pairwise comparison results
    """
    from sklearn.metrics import roc_auc_score
    
    model_names = list(predictions.keys())
    n_models = len(model_names)
    
    results = []
    
    for i in range(n_models):
        for j in range(i + 1, n_models):
            model1 = model_names[i]
            model2 = model_names[j]
            
            pred1 = predictions[model1]
            pred2 = predictions[model2]
            
            if test == 'delong':
                z_stat, p_value, diff = delong_test(y_true, pred1, pred2)
                results.append({
                    'model1': model1,
                    'model2': model2,
                    'metric': metric_name,
                    'test': 'DeLong',
                    'statistic': z_stat,
                    'p_value': p_value,
                    'difference': diff,
                    'significant': p_value < 0.05 if not np.isnan(p_value) else False
                })
            
            elif test == 'mcnemar':
                # Convert probabilities to binary predictions
                pred1_binary = (pred1 >= 0.5).astype(int)
                pred2_binary = (pred2 >= 0.5).astype(int)
                
                chi2, p_value = mcnemar_test(y_true, pred1_binary, pred2_binary)
                results.append({
                    'model1': model1,
                    'model2': model2,
                    'metric': 'Predictions',
                    'test': 'McNemar',
                    'statistic': chi2,
                    'p_value': p_value,
                    'difference': None,
                    'significant': p_value < 0.05
                })
            
            elif test == 'permutation':
                metric_fn = lambda y, p: roc_auc_score(y, p)
                diff, p_value = permutation_test(y_true, pred1, pred2, metric_fn)
                results.append({
                    'model1': model1,
                    'model2': model2,
                    'metric': metric_name,
                    'test': 'Permutation',
                    'statistic': None,
                    'p_value': p_value,
                    'difference': diff,
                    'significant': p_value < 0.05
                })
    
    return pd.DataFrame(results)


def format_comparison_table(df: pd.DataFrame) -> str:
    """
    Format comparison results as a readable table.
    
    Args:
        df: DataFrame from compare_multiple_models
    
    Returns:
        str: Formatted table
    """
    output = []
    output.append("\n" + "="*100)
    output.append("MODEL COMPARISON - STATISTICAL SIGNIFICANCE TESTS")
    output.append("="*100 + "\n")
    
    output.append(f"{'Model 1':<15} {'Model 2':<15} {'Test':<12} {'Statistic':<12} "
                  f"{'P-value':<12} {'Difference':<12} {'Significant':<12}")
    output.append("-" * 100)
    
    for _, row in df.iterrows():
        stat_str = f"{row['statistic']:.4f}" if pd.notna(row['statistic']) else "N/A"
        diff_str = f"{row['difference']:.4f}" if pd.notna(row['difference']) else "N/A"
        sig_str = "Yes*" if row['significant'] else "No"
        
        output.append(
            f"{row['model1']:<15} {row['model2']:<15} {row['test']:<12} "
            f"{stat_str:<12} {row['p_value']:<12.4f} {diff_str:<12} {sig_str:<12}"
        )
    
    output.append("-" * 100)
    output.append("\n* Significant at α = 0.05 level")
    output.append("="*100 + "\n")
    
    return '\n'.join(output)


# Import pandas at module level
try:
    import pandas as pd
except ImportError:
    logger.warning("pandas not available for statistical_tests module")

