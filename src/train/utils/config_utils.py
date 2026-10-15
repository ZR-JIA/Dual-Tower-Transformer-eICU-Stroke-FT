"""
Safe type conversion and validation for config values.
"""

import logging
from typing import Any, Union, Optional

logger = logging.getLogger(__name__)


def safe_cast(value: Any, target_type: type, config_key: str = "unknown", 
              allow_none: bool = False, allow_auto: bool = False) -> Any:
    """
    Safely cast configuration value to target type with detailed error messages.
    
    Args:
        value: Value to cast
        target_type: Target type (int, float, str)
        config_key: Configuration key path for error messages
        allow_none: Whether to allow None values
        allow_auto: Whether to allow "auto" string (returned as-is)
    
    Returns:
        Casted value
    
    Raises:
        ValueError: If casting fails with detailed message
    
    Examples:
        >>> safe_cast("42", int, "training.epochs")
        42
        >>> safe_cast("0.001", float, "training.lr")
        0.001
        >>> safe_cast("auto", float, "loss.pos_weight", allow_auto=True)
        "auto"
    """
    # Handle None
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"Config key '{config_key}' is None but None is not allowed")
    
    # Already correct type
    if isinstance(value, target_type):
        return value
    
    # Handle "auto" keyword
    if allow_auto and isinstance(value, str) and value.lower() == "auto":
        return "auto"
    
    # Try casting
    try:
        if target_type == int:
            # Handle float strings that should be int
            if isinstance(value, str):
                # Try float first then int to handle "42.0"
                try:
                    float_val = float(value)
                    result = int(float_val)
                    if float_val != result:
                        logger.warning(
                            f"Config key '{config_key}': converting '{value}' to int, "
                            f"losing decimal part (got {result})"
                        )
                    return result
                except ValueError:
                    raise ValueError(
                        f"Config key '{config_key}': cannot convert '{value}' to int"
                    )
            else:
                return int(value)
        
        elif target_type == float:
            result = float(value)
            if isinstance(value, str):
                logger.info(
                    f"Auto-cast config.{config_key}: '{value}' -> {result}"
                )
            return result
        
        elif target_type == str:
            return str(value)
        
        else:
            raise ValueError(f"Unsupported target type: {target_type}")
    
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Config key '{config_key}': failed to cast '{value}' (type: {type(value).__name__}) "
            f"to {target_type.__name__}: {e}"
        )


def compute_pos_weight_from_labels(y_train: Any, config_key: str = "pos_weight") -> Optional[float]:
    """
    Compute pos_weight automatically from training labels.
    
    Args:
        y_train: Training labels (numpy array or tensor)
        config_key: Configuration key for logging
    
    Returns:
        float: pos_weight = n_neg / n_pos, or None if computation fails
    """
    import numpy as np
    
    try:
        # Convert to numpy if needed
        if hasattr(y_train, 'numpy'):  # PyTorch tensor
            y_np = y_train.cpu().numpy()
        elif hasattr(y_train, 'values'):  # Pandas
            y_np = y_train.values
        else:
            y_np = np.asarray(y_train)
        
        # Count classes
        n_pos = np.sum(y_np == 1)
        n_neg = np.sum(y_np == 0)
        
        if n_pos == 0:
            logger.error(
                f"{config_key} auto failed: positive class count is 0 "
                f"(total samples: {len(y_np)})"
            )
            return None
        
        if n_neg == 0:
            logger.error(
                f"{config_key} auto failed: negative class count is 0 "
                f"(total samples: {len(y_np)})"
            )
            return None
        
        pos_weight = float(n_neg) / float(n_pos)
        logger.info(
            f"{config_key} auto computed: pos_weight = {pos_weight:.4f} "
            f"(n_neg={n_neg}, n_pos={n_pos}, ratio={n_neg/n_pos:.2f}:1)"
        )
        
        return pos_weight
    
    except Exception as e:
        logger.error(f"{config_key} auto failed with exception: {e}")
        return None


def validate_binary_labels(y: Any, split_name: str = "dataset") -> tuple:
    """
    Validate that labels are binary {0, 1}.
    
    Args:
        y: Labels array
        split_name: Name of the split for error messages
    
    Returns:
        tuple: (n_samples, n_pos, n_neg)
    
    Raises:
        ValueError: If labels contain invalid values
    """
    import numpy as np
    
    # Convert to numpy
    if hasattr(y, 'numpy'):
        y_np = y.cpu().numpy()
    elif hasattr(y, 'values'):
        y_np = y.values
    else:
        y_np = np.asarray(y)
    
    # Check for non {0, 1} values
    unique_vals = np.unique(y_np)
    invalid_vals = [v for v in unique_vals if v not in [0, 1]]
    
    if invalid_vals:
        # Find indices of invalid samples
        invalid_indices = np.where(np.isin(y_np, invalid_vals))[0][:10]
        invalid_samples = [(idx, y_np[idx]) for idx in invalid_indices]
        
        raise ValueError(
            f"Invalid labels in {split_name}: found non-binary values {invalid_vals}. "
            f"Only {{0, 1}} allowed. First 10 invalid samples: {invalid_samples}"
        )
    
    n_pos = np.sum(y_np == 1)
    n_neg = np.sum(y_np == 0)
    n_samples = len(y_np)
    
    logger.info(
        f"{split_name} labels validated: {n_samples} samples "
        f"(pos={n_pos}, neg={n_neg}, ratio={n_pos/(n_pos+n_neg)*100:.1f}%)"
    )
    
    return n_samples, n_pos, n_neg

