"""
Probability calibration: Platt scaling, isotonic regression, beta calibration.
"""

import numpy as np
from typing import Optional, Literal
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import pickle
import logging

logger = logging.getLogger(__name__)


class CalibratorWrapper:
    """
    Wrapper for probability calibration.
    
    Supports multiple calibration methods:
    - 'isotonic': Isotonic regression (non-parametric)
    - 'platt': Platt scaling (logistic regression)
    - 'beta': Beta calibration (requires beta_calibration package)
    """
    
    def __init__(
        self,
        method: Literal['isotonic', 'platt', 'beta'] = 'isotonic'
    ):
        """
        Initialize calibrator.
        
        Args:
            method: Calibration method
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
    
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """
        Fit calibrator.
        
        Args:
            y_true: True labels (0 or 1)
            y_pred_proba: Uncalibrated predicted probabilities
        """
        logger.info(f"Fitting calibrator: {self.method}")
        
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_pred_proba, y_true)
        
        elif self.method == 'platt':
            # Platt scaling = logistic regression on predicted probabilities
            self.calibrator = LogisticRegression(max_iter=1000)
            self.calibrator.fit(y_pred_proba.reshape(-1, 1), y_true)
        
        elif self.method == 'beta':
            try:
                from betacal import BetaCalibration
                self.calibrator = BetaCalibration(parameters="abm")
                self.calibrator.fit(y_pred_proba, y_true)
            except ImportError:
                logger.warning("Beta calibration not available. Install with: pip install betacal")
                logger.info("Falling back to isotonic regression")
                self.method = 'isotonic'
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
                self.calibrator.fit(y_pred_proba, y_true)
        
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self.is_fitted = True
        logger.info("Calibrator fitted successfully")
    
    def transform(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Apply calibration to predicted probabilities.
        
        Args:
            y_pred_proba: Uncalibrated predicted probabilities
        
        Returns:
            np.ndarray: Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")
        
        if self.method == 'isotonic':
            return self.calibrator.predict(y_pred_proba)
        
        elif self.method == 'platt':
            return self.calibrator.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
        
        elif self.method == 'beta':
            return self.calibrator.predict(y_pred_proba)
        
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
    
    def fit_transform(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Fit calibrator and transform in one step.
        
        Args:
            y_true: True labels
            y_pred_proba: Uncalibrated predicted probabilities
        
        Returns:
            np.ndarray: Calibrated probabilities
        """
        self.fit(y_true, y_pred_proba)
        return self.transform(y_pred_proba)
    
    def save(self, path: str):
        """Save calibrator to file."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted calibrator")
        
        with open(path, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'calibrator': self.calibrator,
                'is_fitted': self.is_fitted
            }, f)
        logger.info(f"Calibrator saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'CalibratorWrapper':
        """Load calibrator from file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        instance = cls(method=state['method'])
        instance.calibrator = state['calibrator']
        instance.is_fitted = state['is_fitted']
        
        logger.info(f"Calibrator loaded from {path}")
        return instance


def fit_calibrator(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    method: Literal['isotonic', 'platt', 'beta'] = 'isotonic'
) -> CalibratorWrapper:
    """
    Convenience function to fit and return a calibrator.
    
    Args:
        y_true: True labels
        y_pred_proba: Uncalibrated predicted probabilities
        method: Calibration method
    
    Returns:
        CalibratorWrapper: Fitted calibrator
    """
    calibrator = CalibratorWrapper(method=method)
    calibrator.fit(y_true, y_pred_proba)
    return calibrator


def apply_calibration(
    calibrator: CalibratorWrapper,
    y_pred_proba: np.ndarray
) -> np.ndarray:
    """
    Apply calibration to predicted probabilities.
    
    Args:
        calibrator: Fitted calibrator
        y_pred_proba: Uncalibrated predicted probabilities
    
    Returns:
        np.ndarray: Calibrated probabilities
    """
    return calibrator.transform(y_pred_proba)


def evaluate_calibration(
    y_true: np.ndarray,
    y_pred_uncalibrated: np.ndarray,
    y_pred_calibrated: np.ndarray,
    n_bins: int = 10
) -> dict:
    """
    Evaluate calibration performance.
    
    Args:
        y_true: True labels
        y_pred_uncalibrated: Uncalibrated probabilities
        y_pred_calibrated: Calibrated probabilities
        n_bins: Number of bins for calibration curve
    
    Returns:
        dict: Calibration metrics before and after
    """
    from sklearn.metrics import brier_score_loss
    from engine.metrics import compute_ece
    
    results = {
        'uncalibrated': {
            'brier': brier_score_loss(y_true, y_pred_uncalibrated),
            'ece': compute_ece(y_true, y_pred_uncalibrated, n_bins=n_bins)
        },
        'calibrated': {
            'brier': brier_score_loss(y_true, y_pred_calibrated),
            'ece': compute_ece(y_true, y_pred_calibrated, n_bins=n_bins)
        }
    }
    
    # Compute improvement
    results['improvement'] = {
        'brier_reduction': results['uncalibrated']['brier'] - results['calibrated']['brier'],
        'ece_reduction': results['uncalibrated']['ece'] - results['calibrated']['ece']
    }
    
    logger.info(f"Calibration evaluation:")
    logger.info(f"  Brier before: {results['uncalibrated']['brier']:.4f}, after: {results['calibrated']['brier']:.4f}")
    logger.info(f"  ECE before: {results['uncalibrated']['ece']:.4f}, after: {results['calibrated']['ece']:.4f}")
    
    return results


def get_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    strategy: Literal['uniform', 'quantile'] = 'quantile'
) -> tuple:
    """
    Compute calibration curve for plotting.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins
        strategy: Binning strategy
    
    Returns:
        tuple: (mean_predicted_proba, fraction_of_positives, bin_counts)
    """
    if strategy == 'uniform':
        bins = np.linspace(0, 1, n_bins + 1)
    elif strategy == 'quantile':
        bins = np.percentile(y_pred_proba, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)  # Remove duplicate bin edges
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")
    
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    
    mean_predicted = []
    fraction_positive = []
    bin_counts = []
    
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            mean_predicted.append(np.mean(y_pred_proba[mask]))
            fraction_positive.append(np.mean(y_true[mask]))
            bin_counts.append(np.sum(mask))
        else:
            mean_predicted.append(np.nan)
            fraction_positive.append(np.nan)
            bin_counts.append(0)
    
    return (
        np.array(mean_predicted),
        np.array(fraction_positive),
        np.array(bin_counts)
    )

