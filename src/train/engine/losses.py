"""
Loss functions: BCEWithLogitsLoss, Focal Loss, Label Smoothing BCE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss = -alpha * (1-p_t)^gamma * log(p_t)
    
    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor in [0, 1] for class balance
            gamma: Focusing parameter for hard examples
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Model predictions (before sigmoid)
            targets: Ground truth labels (0 or 1)
        
        Returns:
            torch.Tensor: Loss value
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingBCELoss(nn.Module):
    """
    Binary Cross Entropy Loss with Label Smoothing.
    
    Smooths hard labels (0, 1) to soft labels (epsilon, 1-epsilon)
    to prevent overconfidence.
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = 'mean',
        pos_weight: Optional[torch.Tensor] = None
    ):
        """
        Initialize Label Smoothing BCE Loss.
        
        Args:
            smoothing: Label smoothing factor
            reduction: Reduction method
            pos_weight: Weight for positive class
        """
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.pos_weight = pos_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with label smoothing.
        
        Args:
            logits: Model predictions (before sigmoid)
            targets: Ground truth labels (0 or 1)
        
        Returns:
            torch.Tensor: Loss value
        """
        # Apply label smoothing
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # Compute BCE loss
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction=self.reduction,
            pos_weight=self.pos_weight
        )
        
        return loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss.
    
    Allows per-sample weighting in addition to class weighting.
    """
    
    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        """
        Initialize Weighted BCE Loss.
        
        Args:
            pos_weight: Weight for positive class
            reduction: Reduction method
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted BCE loss.
        
        Args:
            logits: Model predictions
            targets: Ground truth labels
            sample_weights: Optional per-sample weights
        
        Returns:
            torch.Tensor: Loss value
        """
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction='none',
            pos_weight=self.pos_weight
        )
        
        if sample_weights is not None:
            loss = loss * sample_weights
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss_fn(config: dict, pos_weight: Optional[Union[float, str]] = None, 
                y_train: Optional[torch.Tensor] = None, device: Optional[torch.device] = None) -> nn.Module:
    """
    Get loss function from configuration with proper pos_weight handling.
    
    Args:
        config: Full configuration dictionary
        pos_weight: Positive class weight (float, "auto", or None)
        y_train: Training labels for auto computation (required if pos_weight="auto")
        device: Target device for tensors
    
    Returns:
        nn.Module: Loss function
    """
    from utils.config_utils import safe_cast, compute_pos_weight_from_labels
    
    # Get loss config from model_config
    model_config = config.get('model_config', {})
    loss_config = model_config.get('loss', {})
    
    if isinstance(loss_config, dict):
        loss_type = loss_config.get('name', 'bce_with_logits')
    else:
        loss_type = str(loss_config)
    
    logger.info(f"Creating loss function: {loss_type}")
    
    # Default device
    if device is None:
        device = torch.device('cpu')
    
    # Get pos_weight
    if pos_weight is None:
        pos_weight = loss_config.get('pos_weight', None)
        if pos_weight is None:
            pos_weight = config.get('common', {}).get('class_imbalance', {}).get('pos_weight', None)
    
    # Handle pos_weight
    pos_weight_tensor = None
    if pos_weight is not None:
        # Handle "auto"
        if isinstance(pos_weight, str) and pos_weight.lower() == "auto":
            if y_train is None:
                logger.warning(
                    "pos_weight='auto' requires y_train, but y_train is None. "
                    "Falling back to no class weighting."
                )
            else:
                computed_weight = compute_pos_weight_from_labels(
                    y_train, config_key="loss.pos_weight"
                )
                if computed_weight is not None:
                    pos_weight_tensor = torch.tensor([computed_weight], 
                                                     dtype=torch.float32, device=device)
                    logger.info(f"Using auto-computed pos_weight: {computed_weight:.4f}")
                else:
                    logger.warning(
                        "pos_weight auto computation failed, falling back to no weighting"
                    )
        else:
            # Try to cast to float
            try:
                pos_weight_value = safe_cast(
                    pos_weight, float, "loss.pos_weight", allow_none=False, allow_auto=False
                )
                pos_weight_tensor = torch.tensor([pos_weight_value], 
                                                 dtype=torch.float32, device=device)
                logger.info(f"Using configured pos_weight: {pos_weight_value:.4f}")
            except ValueError as e:
                logger.error(f"Failed to parse pos_weight: {e}. Falling back to no weighting.")
    
    if pos_weight_tensor is None:
        logger.info("Not using class weighting (pos_weight=None)")
    
    # Create loss function
    if loss_type == 'bce' or loss_type == 'bce_with_logits':
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    elif loss_type == 'focal':
        focal_config = loss_config.get('focal', {})
        alpha = focal_config.get('alpha', 0.25)
        gamma = focal_config.get('gamma', 2.0)
        loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
        logger.info(f"Focal Loss: alpha={alpha}, gamma={gamma}")
    
    elif loss_type == 'label_smoothing':
        smoothing_config = loss_config.get('label_smoothing', {})
        smoothing = smoothing_config.get('smoothing', 0.1)
        loss_fn = LabelSmoothingBCELoss(
            smoothing=smoothing,
            pos_weight=pos_weight_tensor
        )
        logger.info(f"Label Smoothing BCE: smoothing={smoothing}")
    
    elif loss_type == 'weighted_bce':
        loss_fn = WeightedBCELoss(pos_weight=pos_weight_tensor)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss_fn

