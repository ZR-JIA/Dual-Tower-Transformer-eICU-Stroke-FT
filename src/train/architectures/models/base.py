"""
Shared layers and utilities for model architectures.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class NumericalFeatureTokenizer(nn.Module):
    """
    Tokenizer for numerical features.
    
    Converts scalar numerical features into dense embeddings (tokens) that can be
    processed by Transformer layers. Each feature is projected into an embedding
    space using learned weights and biases.
    
    This is essential for Transformer-based models which require vector inputs
    rather than scalar values.
    
    Example:
        Input: [batch_size, num_features] (e.g., blood pressure = 120)
        Output: [batch_size, num_features, embed_dim] (e.g., 120 -> vector of length 64)
    """
    
    def __init__(self, num_features: int, embed_dim: int):
        """
        Initialize the tokenizer.
        
        Args:
            num_features: Number of numerical features to tokenize
            embed_dim: Dimension of the embedding space
        """
        super().__init__()
        
        # Learnable weight and bias for each feature
        self.weight = nn.Parameter(torch.Tensor(num_features, embed_dim))
        self.bias = nn.Parameter(torch.Tensor(num_features, embed_dim))
        
        # Initialize parameters using Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(num_features)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert numerical features to embeddings.
        
        Args:
            x: Input tensor [batch_size, num_features]
        
        Returns:
            Embedded features [batch_size, num_features, embed_dim]
        """
        # Reshape: [batch_size, num_features] -> [batch_size, num_features, 1]
        x = x.unsqueeze(-1)
        
        # Broadcast multiplication: feature * weight + bias
        # Output shape: [batch_size, num_features, embed_dim]
        out = x * self.weight + self.bias
        
        return out
