"""
Safe type casting for YAML-loaded config values.
"""

import logging
import re
from typing import Any, Optional, Tuple, Union
from decimal import Decimal

logger = logging.getLogger(__name__)


def auto_cast_warn(keypath: str, old: Any, new: Any):
    """Log warning about automatic type conversion."""
    logger.warning(f"auto-cast {keypath}: {old!r} -> {new}")


def safe_float(x: Any, keypath: str) -> float:
    """
    Safely convert value to float.
    
    Accepts: int, float, Decimal, numpy numbers, parseable numeric strings
    (including scientific notation like "1e-3")
    
    Args:
        x: Value to convert
        keypath: Configuration key path for error messages
    
    Returns:
        float: Converted value
    
    Raises:
        ValueError: If conversion fails
    
    Examples:
        >>> safe_float(0.001, "lr")
        0.001
        >>> safe_float("0.001", "lr")
        0.001
        >>> safe_float("1e-3", "lr")
        0.001
    """
    if isinstance(x, float):
        return x
    
    if isinstance(x, (int, Decimal)):
        return float(x)
    
    # Handle numpy types if available
    if hasattr(x, 'item'):  # numpy scalar
        try:
            return float(x.item())
        except:
            pass
    
    if isinstance(x, str):
        # Try parsing as float
        try:
            result = float(x)
            auto_cast_warn(keypath, x, result)
            return result
        except ValueError:
            raise ValueError(f"Invalid float for {keypath}: {x!r}")
    
    raise ValueError(f"Invalid float for {keypath}: {x!r} (type: {type(x).__name__})")


def safe_int(x: Any, keypath: str, allow_float_strings: bool = False) -> int:
    """
    Safely convert value to int.
    
    Args:
        x: Value to convert
        keypath: Configuration key path for error messages
        allow_float_strings: If True, allow "42.0" -> 42; if False, reject it
    
    Returns:
        int: Converted value
    
    Raises:
        ValueError: If conversion fails
    
    Examples:
        >>> safe_int(42, "epochs")
        42
        >>> safe_int("42", "epochs")
        42
        >>> safe_int("42.0", "epochs", allow_float_strings=True)
        42
    """
    if isinstance(x, int) and not isinstance(x, bool):
        return x
    
    if isinstance(x, float):
        if x.is_integer():
            return int(x)
        else:
            raise ValueError(f"Invalid int for {keypath}: {x!r} (has decimal part)")
    
    if isinstance(x, str):
        try:
            # Try direct int conversion first
            result = int(x)
            auto_cast_warn(keypath, x, result)
            return result
        except ValueError:
            if allow_float_strings:
                # Try float then int
                try:
                    float_val = float(x)
                    if float_val.is_integer():
                        result = int(float_val)
                        auto_cast_warn(keypath, x, result)
                        return result
                    else:
                        raise ValueError(f"Invalid int for {keypath}: {x!r} (has decimal part)")
                except ValueError:
                    raise ValueError(f"Invalid int for {keypath}: {x!r}")
            else:
                raise ValueError(f"Invalid int for {keypath}: {x!r}")
    
    raise ValueError(f"Invalid int for {keypath}: {x!r} (type: {type(x).__name__})")


def safe_bool(x: Any, keypath: str) -> bool:
    """
    Safely convert value to bool.
    
    Accepts: True, False, "true", "false", "True", "False", "1", "0", 1, 0
    
    Args:
        x: Value to convert
        keypath: Configuration key path for error messages
    
    Returns:
        bool: Converted value
    
    Raises:
        ValueError: If conversion fails
    """
    if isinstance(x, bool):
        return x
    
    if isinstance(x, str):
        x_lower = x.lower().strip()
        if x_lower in ("true", "1", "yes"):
            if x != str(True):  # Only warn if it's a string conversion
                auto_cast_warn(keypath, x, True)
            return True
        elif x_lower in ("false", "0", "no"):
            if x != str(False):
                auto_cast_warn(keypath, x, False)
            return False
        else:
            raise ValueError(f"Invalid bool for {keypath}: {x!r}")
    
    if isinstance(x, int):
        if x == 1:
            auto_cast_warn(keypath, x, True)
            return True
        elif x == 0:
            auto_cast_warn(keypath, x, False)
            return False
    
    raise ValueError(f"Invalid bool for {keypath}: {x!r} (type: {type(x).__name__})")


def safe_tuple_floats(
    x: Any, 
    keypath: str, 
    expected_len: Optional[int] = None
) -> Tuple[float, ...]:
    """
    Safely convert value to tuple of floats.
    
    Accepts:
    - tuple/list of numbers
    - string like "0.9, 0.999" or "(0.9, 0.999)" or "0.9,0.999"
    
    Args:
        x: Value to convert
        keypath: Configuration key path for error messages
        expected_len: Expected length (None means any length)
    
    Returns:
        tuple: Tuple of floats
    
    Raises:
        ValueError: If conversion fails or length mismatch
    
    Examples:
        >>> safe_tuple_floats((0.9, 0.999), "betas")
        (0.9, 0.999)
        >>> safe_tuple_floats([0.9, 0.999], "betas")
        (0.9, 0.999)
        >>> safe_tuple_floats("0.9, 0.999", "betas")
        (0.9, 0.999)
        >>> safe_tuple_floats("(0.9, 0.999)", "betas", expected_len=2)
        (0.9, 0.999)
    """
    # Already a tuple/list of numbers
    if isinstance(x, (tuple, list)):
        try:
            result = tuple(safe_float(val, f"{keypath}[{i}]") for i, val in enumerate(x))
            if expected_len is not None and len(result) != expected_len:
                raise ValueError(
                    f"Invalid length for {keypath}: expected {expected_len}, got {len(result)}"
                )
            return result
        except Exception as e:
            raise ValueError(f"Invalid tuple for {keypath}: {x!r} - {e}")
    
    # String representation
    if isinstance(x, str):
        # Remove parentheses and whitespace
        x_clean = x.strip()
        if x_clean.startswith('(') and x_clean.endswith(')'):
            x_clean = x_clean[1:-1]
        
        # Split by comma
        parts = [p.strip() for p in x_clean.split(',')]
        
        try:
            result = tuple(safe_float(p, f"{keypath}[{i}]") for i, p in enumerate(parts))
            
            if expected_len is not None and len(result) != expected_len:
                raise ValueError(
                    f"Invalid length for {keypath}: expected {expected_len}, got {len(result)}"
                )
            
            auto_cast_warn(keypath, x, result)
            return result
        except Exception as e:
            raise ValueError(f"Invalid tuple string for {keypath}: {x!r} - {e}")
    
    raise ValueError(
        f"Invalid tuple for {keypath}: {x!r} (type: {type(x).__name__}). "
        f"Expected tuple, list, or comma-separated string."
    )


def validate_range(value: float, keypath: str, min_val: Optional[float] = None, 
                   max_val: Optional[float] = None, 
                   min_inclusive: bool = True, max_inclusive: bool = True):
    """
    Validate that a value is within a specified range.
    
    Args:
        value: Value to validate
        keypath: Configuration key path for error messages
        min_val: Minimum value (None means no minimum)
        max_val: Maximum value (None means no maximum)
        min_inclusive: Whether minimum is inclusive
        max_inclusive: Whether maximum is inclusive
    
    Raises:
        ValueError: If value is out of range
    """
    if min_val is not None:
        if min_inclusive:
            if value < min_val:
                raise ValueError(f"{keypath} must be >= {min_val}, got {value}")
        else:
            if value <= min_val:
                raise ValueError(f"{keypath} must be > {min_val}, got {value}")
    
    if max_val is not None:
        if max_inclusive:
            if value > max_val:
                raise ValueError(f"{keypath} must be <= {max_val}, got {value}")
        else:
            if value >= max_val:
                raise ValueError(f"{keypath} must be < {max_val}, got {value}")


def safe_list_ints(x: Any, keypath: str, strictly_increasing: bool = False) -> list:
    """
    Safely convert value to list of integers.
    
    Args:
        x: Value to convert (list, tuple, or comma-separated string)
        keypath: Configuration key path for error messages
        strictly_increasing: If True, validate that list is strictly increasing
    
    Returns:
        list: List of integers
    
    Raises:
        ValueError: If conversion fails or not strictly increasing
    """
    if isinstance(x, (list, tuple)):
        result = [safe_int(val, f"{keypath}[{i}]") for i, val in enumerate(x)]
    elif isinstance(x, str):
        parts = [p.strip() for p in x.split(',')]
        result = [safe_int(p, f"{keypath}[{i}]") for i, p in enumerate(parts)]
        auto_cast_warn(keypath, x, result)
    else:
        raise ValueError(f"Invalid list for {keypath}: {x!r}")
    
    if strictly_increasing:
        for i in range(1, len(result)):
            if result[i] <= result[i-1]:
                raise ValueError(
                    f"{keypath} must be strictly increasing, but {result[i-1]} >= {result[i]}"
                )
    
    return result

