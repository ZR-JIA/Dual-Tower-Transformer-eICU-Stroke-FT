"""
Evaluator Module

Unified evaluator for all model types.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

from .metrics import (
    compute_classification_metrics,
    compute_bootstrap_ci,
    format_metrics_table,
    select_best_threshold_youden,
    select_best_threshold_f1,
    select_threshold_fixed_specificity
)

logger = logging.getLogger(__name__)


class UnifiedEvaluator:
    """
    Unified evaluator for all model types.
    """
    
    def __init__(
        self,
        model: Any,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            config: Configuration dictionary
            device: Device (for neural models)
        """
        self.model = model
        self.config = config
        self.device = device
        self.model_type = config['model_config']['model']
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: Optional[float] = None,
        threshold_method: str = 'youden',
        compute_ci: bool = True,
        save_plots: bool = True,
        plot_dir: Optional[Path] = None,
        prefix: str = 'test'
    ) -> Tuple[Dict[str, float], float, str]:
        """
        Comprehensive evaluation.
        
        Args:
            X: Features
            y: True labels
            threshold: Classification threshold (if None, will be selected)
            threshold_method: Method to select threshold ('youden', 'max_f1', 'fixed_specificity')
            compute_ci: Compute bootstrap confidence intervals
            save_plots: Save evaluation plots
            plot_dir: Directory to save plots
            prefix: Prefix for saved files
        
        Returns:
            Tuple of (metrics_dict, threshold, threshold_method)
        """
        logger.info(f"Evaluating model on {len(y)} samples")
        
        # Get predictions (using batch processing for DL models)
        y_pred_proba = self._predict(X)
        
        # Select threshold if not provided
        if threshold is None:
            threshold, threshold_method = self._select_threshold(
                y, y_pred_proba, threshold_method
            )
            logger.info(f"Selected threshold: {threshold:.4f} (method: {threshold_method})")
        
        # Compute metrics
        metrics = compute_classification_metrics(y, y_pred_proba, threshold=threshold, prefix='')
        
        # Compute confidence intervals
        ci_results = None
        if compute_ci:
            logger.info("Computing bootstrap confidence intervals...")
            ci_results = compute_bootstrap_ci(
                y, y_pred_proba,
                threshold=threshold,
                n_bootstrap=1000,
                random_state=self.config['common']['reproducibility']['seed']
            )
            
            # Add CI to metrics with prefix
            for metric_name, ci_dict in ci_results.items():
                metrics[f'{metric_name}_ci_lower'] = ci_dict['lower']
                metrics[f'{metric_name}_ci_upper'] = ci_dict['upper']
        
        # Log metrics
        logger.info("\n" + format_metrics_table(metrics, ci_results))
        
        # Save plots
        if save_plots and plot_dir is not None:
            plot_dir = Path(plot_dir)
            plot_dir.mkdir(parents=True, exist_ok=True)
            self._save_plots(y, y_pred_proba, threshold, plot_dir, prefix)
        
        return metrics, threshold, threshold_method
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions based on model type.
        
        Supports both PyTorch nn.Module and traditional ML models (sklearn, XGBoost).
        Includes defensive data sanitization for categorical features.
        """
        # ========================================
        # PyTorch Neural Network Path
        # ========================================
        if isinstance(self.model, nn.Module):
            logger.info("Using PyTorch inference path")
            self.model.eval()
            
            # Batch processing to prevent OOM
            batch_size = 256  # Safe batch size for inference
            predictions = []
            
            # Convert full dataset to tensor first (on CPU)
            X_tensor_all = torch.FloatTensor(X)
            n_samples = len(X)
            
            # Get device from model parameters
            device = next(self.model.parameters()).device
            
            with torch.no_grad():
                for i in range(0, n_samples, batch_size):
                    # Slice batch and move to device
                    batch_X = X_tensor_all[i : i + batch_size].to(device)
                    
                    # Forward pass
                    outputs = self.model(batch_X)
                    probs = torch.sigmoid(outputs)
                    
                    # Move back to CPU immediately to free GPU memory
                    predictions.append(probs.cpu().numpy())
            
            # Concatenate all batches
            if len(predictions) > 0:
                probs = np.concatenate(predictions, axis=0)
            else:
                probs = np.array([])
            
            return probs.flatten()  # Ensure 1D array
        
        # ========================================
        # Traditional ML Model Path (sklearn, XGBoost, RandomForest)
        # ========================================
        else:
            logger.info("Using traditional ML inference path")
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(X)[:, 1]
            else:
                probs = self.model.predict(X)
            return probs
    
    def _select_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        method: str
    ) -> Tuple[float, str]:
        """Select classification threshold."""
        if method == 'youden':
            threshold, _ = select_best_threshold_youden(y_true, y_pred_proba)
            return threshold, 'youden'
        
        elif method == 'max_f1':
            threshold, _ = select_best_threshold_f1(y_true, y_pred_proba)
            return threshold, 'max_f1'
        
        elif method.startswith('fixed_specificity'):
            # Parse target specificity
            if '@' in method:
                target_spec = float(method.split('@')[1])
            else:
                target_spec = 0.90
            threshold = select_threshold_fixed_specificity(y_true, y_pred_proba, target_spec)
            return threshold, f'fixed_specificity@{target_spec}'
        
        else:
            logger.warning(f"Unknown threshold method: {method}, using 0.5")
            return 0.5, 'fixed'
    
    def _save_plots(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float,
        plot_dir: Path,
        prefix: str
    ):
        """Save evaluation plots."""
        from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
        from .calibration import get_calibration_curve
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_dir / f'{prefix}_roc_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_dir / f'{prefix}_pr_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Calibration curve
        mean_pred, frac_pos, _ = get_calibration_curve(y_true, y_pred_proba, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        mask = ~np.isnan(mean_pred)
        plt.plot(mean_pred[mask], frac_pos[mask], 'o-', label='Model')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_dir / f'{prefix}_calibration.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Threshold plot
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(y_pred_proba[y_true == 0], bins=50, alpha=0.5, label='Negative', density=True)
        plt.hist(y_pred_proba[y_true == 1], bins=50, alpha=0.5, label='Positive', density=True)
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold={threshold:.3f}')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Prediction Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = [0, 1]
        plt.xticks(tick_marks, ['Negative', 'Positive'])
        plt.yticks(tick_marks, ['Negative', 'Positive'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig(plot_dir / f'{prefix}_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to: {plot_dir}")