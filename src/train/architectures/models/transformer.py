"""
Transformer model for tabular data classification.
"""

import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    """
    Transformer-based model for tabular data.
    
    Uses self-attention over features to capture complex interactions.
    Each feature is embedded and processed through Transformer encoder layers,
    then aggregated via global average pooling for classification.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize Transformer model.
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension (embedding size)
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Feature embedding: projects each scalar feature to d_model dimensions
        self.feature_embedding = nn.Linear(1, d_model)
        
        # Learnable positional encoding for each feature position
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, input_dim]
        
        Returns:
            Logits [batch_size]
        """
        batch_size = x.size(0)
        
        # Reshape to [batch_size, input_dim, 1] for per-feature embedding
        x = x.unsqueeze(-1)
        
        # Feature embedding: [batch_size, input_dim, d_model]
        x = self.feature_embedding(x)
        
        # Add positional encoding
        x = x + self.pos_embedding
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling: [batch_size, d_model]
        x = x.mean(dim=1)
        
        # Output projection
        return self.output_head(x).squeeze(-1)
