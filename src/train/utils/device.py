"""
Device selection utilities for cross-platform support (CUDA / MPS / CPU).
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Auto-detect the best available device.

    Priority: cuda > mps > cpu

    Returns:
        torch.device: Selected device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    logger.info(f"Selected device: {device}")
    return device


def should_pin_memory(device: torch.device) -> bool:
    """
    Determine whether pin_memory should be enabled for DataLoaders.

    pin_memory is only beneficial when transferring data to a CUDA device.
    On MPS and CPU it provides no benefit and may cause warnings.

    Args:
        device: The training device

    Returns:
        bool: True if pin_memory should be enabled
    """
    return device.type == 'cuda'
