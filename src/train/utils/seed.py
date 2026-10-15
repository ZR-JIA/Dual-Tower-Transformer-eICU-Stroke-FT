"""
Random seed management for reproducibility.
"""

import os
import random
import numpy as np
import torch


def set_all_seeds(seed=42, deterministic=True, benchmark=False):
    """
    Set random seeds for all libraries to ensure reproducibility.
    
    Args:
        seed (int): Random seed value
        deterministic (bool): Enable deterministic mode for CUDNN
        benchmark (bool): Enable CUDNN benchmark mode (faster but non-deterministic)
    
    Returns:
        dict: Configuration applied
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # CUDNN
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
    
    # Environment variables for additional libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # XGBoost and LightGBM will use random_state parameter directly
    
    config = {
        'seed': seed,
        'deterministic': deterministic,
        'benchmark': benchmark,
        'cuda_available': torch.cuda.is_available()
    }
    
    return config


def seed_worker(worker_id):
    """
    Seed worker for PyTorch DataLoader to ensure reproducibility.
    
    Usage:
        DataLoader(..., worker_init_fn=seed_worker)
    
    Args:
        worker_id (int): Worker ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed=42):
    """
    Get PyTorch random generator with fixed seed.
    
    Args:
        seed (int): Random seed
    
    Returns:
        torch.Generator: Generator with fixed seed
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_rng_state():
    """
    Get the current RNG state for all libraries.
    
    Returns:
        dict: RNG states
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    
    return state


def set_rng_state(state):
    """
    Set the RNG state for all libraries.
    
    Args:
        state (dict): RNG states from get_rng_state()
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    
    if torch.cuda.is_available() and 'torch_cuda' in state:
        torch.cuda.set_rng_state_all(state['torch_cuda'])

