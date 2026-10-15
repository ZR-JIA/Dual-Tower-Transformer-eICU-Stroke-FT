"""
Training Engine Package

Contains trainers and evaluators for model training and evaluation.
"""

from .trainers import NeuralTrainer, TreeTrainer
from .evaluator import UnifiedEvaluator

__all__ = [
    'NeuralTrainer',
    'TreeTrainer',
    'UnifiedEvaluator',
]
