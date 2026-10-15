"""
DualTower model with configurable right tower (MLP or Transformer).

Architecture:
    Left Tower:  Categorical embeddings (Age, Gender, Ethnicity)
    Right Tower: MLP or Transformer encoder for numerical vitals
    Fusion:      Concatenation + MLP classification head

The Transformer right tower uses NumericalFeatureTokenizer from base.py
to convert scalar features into dense token embeddings for self-attention.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, List

from .base import NumericalFeatureTokenizer

logger = logging.getLogger(__name__)


class TransformerRightTower(nn.Module):
    """
    Transformer-based right tower for processing numerical features.
    
    Pipeline: NumericalFeatureTokenizer -> TransformerEncoder -> MeanPooling
    Captures high-order interactions between physiological vitals via self-attention.
    """
    
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, dropout: float):
        """
        Args:
            input_dim: Number of numerical features
            d_model: Embedding dimension per feature token
            nhead: Number of attention heads
            num_layers: Number of Transformer encoder layers
            dropout: Dropout rate
        """
        super().__init__()
        self.tokenizer = NumericalFeatureTokenizer(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_numerical_features]
        Returns:
            [batch_size, d_model]
        """
        tokens = self.tokenizer(x)                    # [B, N, d_model]
        feat_context = self.transformer_encoder(tokens) # [B, N, d_model]
        pooled = feat_context.mean(dim=1)              # [B, d_model]
        return pooled


class DualTower(nn.Module):
    """
    Dual-Tower model for heterogeneous tabular data.
    
    Left Tower:  Learnable embeddings for categorical demographics
    Right Tower: Configurable — MLP (baseline) or Transformer (advanced)
    Fusion:      Concatenation -> MLP -> Sigmoid
    """
    
    def __init__(self, config: Dict[str, Any], input_dim: int):
        """
        Args:
            config: Model configuration dictionary (expects 'architecture' key)
            input_dim: Total input feature dimension
        """
        super().__init__()
        self.config = config
        
        arch = config.get('architecture', {})
        if not arch:
            arch = config.get('model_config', {}).get('architecture', {})
        
        # === Left Tower: Categorical Embeddings ===
        self.cat_features_config = arch['cat_features']
        
        if 'num_indices' in arch:
            self.num_indices = arch['num_indices']
        else:
            cat_idx_set = set(f['index'] for f in self.cat_features_config)
            self.num_indices = [i for i in range(input_dim) if i not in cat_idx_set]
        
        self.embeddings = nn.ModuleList()
        self.cat_indices = []
        total_emb_dim = 0
        for feat in self.cat_features_config:
            self.embeddings.append(nn.Embedding(feat['num_classes'], feat['emb_dim']))
            self.cat_indices.append(feat['index'])
            total_emb_dim += feat['emb_dim']
        
        # === Right Tower: Numerical Features (MLP or Transformer) ===
        self.right_tower_type = arch.get('right_tower_type', 'mlp')
        num_feat_count = len(self.num_indices)
        
        if self.right_tower_type == 'transformer':
            d_model = arch['trans_d_model']
            self.right_tower = TransformerRightTower(
                input_dim=num_feat_count,
                d_model=d_model,
                nhead=arch.get('trans_nhead', 4),
                num_layers=arch.get('trans_num_layers', 2),
                dropout=arch['dropout']
            )
            right_output_dim = d_model
            logger.info("DualTower: Initialized with TRANSFORMER Right Tower")
        else:
            hidden_dim = arch['right_tower_hidden_dim']
            self.right_tower = nn.Sequential(
                nn.Linear(num_feat_count, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(arch['dropout'])
            )
            right_output_dim = hidden_dim
            logger.info("DualTower: Initialized with MLP Right Tower")
        
        # === Fusion Layer ===
        fusion_in = total_emb_dim + right_output_dim
        fusion_hidden = arch['fusion_hidden_dim']
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(arch['dropout']),
            nn.Linear(fusion_hidden, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]
        Returns:
            Logits [batch_size]
        """
        # --- Left Tower ---
        emb_outs = []
        for idx, emb, feat in zip(self.cat_indices, self.embeddings, self.cat_features_config):
            col_input = torch.clamp(x[:, idx], min=0, max=feat['num_classes'] - 1).long()
            emb_outs.append(emb(col_input))
        left_out = torch.cat(emb_outs, dim=1)
        
        # --- Right Tower ---
        num_data = x[:, self.num_indices]
        right_out = self.right_tower(num_data)
        
        # --- Fusion ---
        combined = torch.cat([left_out, right_out], dim=1)
        logits = self.fusion(combined)
        
        return logits.squeeze(-1)
