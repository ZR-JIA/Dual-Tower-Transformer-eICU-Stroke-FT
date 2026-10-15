"""
Models Package

Modular model architectures for tabular data classification.

Structure:
----------
- base.py: Shared components (NumericalFeatureTokenizer)
- mlp.py: Multi-Layer Perceptron models (MLPModel, SimpleNN)
- transformer.py: Transformer-based models (TransformerModel)
- dualtower.py: DualTower models (DualTower, TransformerRightTower)

Baseline: RF, XGB, NN, Transformer, MLP.
DualTower: dualtower (Transformer right tower) and dualtower_mlp (MLP right tower, ablation).
"""

# Base components
from .base import NumericalFeatureTokenizer

# MLP models
from .mlp import MLPModel, SimpleNN

# Transformer models
from .transformer import TransformerModel

# DualTower models
from .dualtower import DualTower, TransformerRightTower


__all__ = [
    # Base
    'NumericalFeatureTokenizer',
    
    # MLP
    'MLPModel',
    'SimpleNN',
    
    # Transformer
    'TransformerModel',
    
    # DualTower
    'DualTower',
    'TransformerRightTower',
]
