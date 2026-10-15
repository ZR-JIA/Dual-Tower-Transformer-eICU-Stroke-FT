"""
Inference Module

Provides unified inference interface for all model types
and threshold selection strategies.
"""

import numpy as np
import torch
from typing import Any, Dict, Tuple, Optional, Literal
import logging

from engine.metrics import (
    select_best_threshold_youden,
    select_best_threshold_f1,
    select_threshold_fixed_specificity,
    select_threshold_fixed_sensitivity
)

logger = logging.getLogger(__name__)


class Predictor:
    """
    Unified predictor for all model types.
    
    Handles both tree-based models (XGBoost, RandomForest)
    and neural models (MLP, NN, Transformer).
    """
    
    def __init__(
        self,
        model: Any,
        model_type: str,
        device: Optional[torch.device] = None,
        calibrator: Optional[Any] = None
    ):
        """
        Initialize predictor.
        
        Args:
            model: Model instance
            model_type: Type of model ('xgboost', 'random_forest', 'mlp', etc.)
            device: Device for neural models
            calibrator: Optional calibrator for probability calibration
        """
        self.model = model
        self.model_type = model_type
        self.device = device
        self.calibrator = calibrator
        
        # Set model to evaluation mode if neural network
        _neural_types = [
            'mlp', 'nn', 'wide_deep', 'tab_transformer',
            'dualtower', 'dualtower_ft', 'dualtower_as',
            'dualtower_ca', 'dualtower_dcn', 'dualtower_mlp',
        ]
        if self.model_type in _neural_types:
            self.model.eval()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Input features
        
        Returns:
            np.ndarray: Predicted probabilities for positive class
        """
        if self.model_type in ['xgboost', 'random_forest']:
            # Tree-based models
            y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        elif self.model_type in [
            'mlp', 'nn', 'wide_deep', 'tab_transformer',
            'dualtower', 'dualtower_ft', 'dualtower_as',
            'dualtower_ca', 'dualtower_dcn', 'dualtower_mlp',
        ]:
            # Neural networks (including DualTower variants)
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                logits = self.model(X_tensor)
                y_pred_proba = torch.sigmoid(logits).cpu().numpy().flatten()
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Apply calibration if available
        if self.calibrator is not None:
            y_pred_proba = self.calibrator.transform(y_pred_proba)
        
        return y_pred_proba
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels.
        
        Args:
            X: Input features
            threshold: Classification threshold
        
        Returns:
            np.ndarray: Binary predictions
        """
        y_pred_proba = self.predict_proba(X)
        return (y_pred_proba >= threshold).astype(int)


def batch_predict(
    predictor: Predictor,
    X: np.ndarray,
    batch_size: int = 1024
) -> np.ndarray:
    """
    Predict in batches for large datasets.
    
    Args:
        predictor: Predictor instance
        X: Input features
        batch_size: Batch size
    
    Returns:
        np.ndarray: Predicted probabilities
    """
    n_samples = len(X)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    predictions = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        X_batch = X[start_idx:end_idx]
        
        y_pred_batch = predictor.predict_proba(X_batch)
        predictions.append(y_pred_batch)
    
    return np.concatenate(predictions)


def select_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    strategy: str = 'youden',
    **kwargs
) -> float:
    """
    Select classification threshold using specified strategy.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        strategy: Threshold selection strategy
            - 'youden': Youden's J statistic
            - 'max_f1': Maximize F1 score
            - 'fixed_specificity@X': Fix specificity at X (e.g., 'fixed_specificity@0.90')
            - 'fixed_sensitivity@X': Fix sensitivity at X
        **kwargs: Additional arguments for specific strategies
    
    Returns:
        float: Selected threshold
    """
    logger.info(f"Selecting threshold using strategy: {strategy}")
    
    if strategy == 'youden':
        threshold, j_stat = select_best_threshold_youden(y_true, y_pred_proba)
        logger.info(f"Youden threshold: {threshold:.4f} (J={j_stat:.4f})")
        return threshold
    
    elif strategy == 'max_f1':
        threshold, f1 = select_best_threshold_f1(y_true, y_pred_proba)
        logger.info(f"Max F1 threshold: {threshold:.4f} (F1={f1:.4f})")
        return threshold
    
    elif strategy.startswith('fixed_specificity@'):
        target = float(strategy.split('@')[1])
        threshold = select_threshold_fixed_specificity(y_true, y_pred_proba, target)
        logger.info(f"Fixed specificity@{target} threshold: {threshold:.4f}")
        return threshold
    
    elif strategy.startswith('fixed_sensitivity@'):
        target = float(strategy.split('@')[1])
        threshold = select_threshold_fixed_sensitivity(y_true, y_pred_proba, target)
        logger.info(f"Fixed sensitivity@{target} threshold: {threshold:.4f}")
        return threshold
    
    elif strategy == 'default':
        logger.info("Using default threshold: 0.5")
        return 0.5
    
    else:
        raise ValueError(f"Unknown threshold strategy: {strategy}")


def select_multiple_thresholds(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    strategies: list
) -> Dict[str, float]:
    """
    Select multiple thresholds using different strategies.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        strategies: List of threshold strategies
    
    Returns:
        dict: Dictionary mapping strategy name to threshold
    """
    thresholds = {}
    for strategy in strategies:
        try:
            thresholds[strategy] = select_threshold(y_true, y_pred_proba, strategy)
        except Exception as e:
            logger.warning(f"Failed to select threshold for {strategy}: {e}")
            thresholds[strategy] = 0.5
    
    return thresholds


def evaluate_at_multiple_thresholds(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model at multiple thresholds.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        thresholds: Dictionary of threshold strategies and values
    
    Returns:
        dict: Nested dictionary with metrics for each threshold
    """
    from engine.metrics import compute_classification_metrics
    
    results = {}
    for strategy, threshold in thresholds.items():
        metrics = compute_classification_metrics(
            y_true,
            y_pred_proba,
            threshold=threshold,
            prefix=''
        )
        results[strategy] = metrics
    
    return results

