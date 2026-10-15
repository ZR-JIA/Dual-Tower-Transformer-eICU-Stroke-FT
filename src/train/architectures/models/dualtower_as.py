"""
DualTower with Adaptive Safeguard Mechanism 3 (dualtower_as)

Architecture:
    Left Tower:  Categorical embeddings — identical to dualtower.py
    Right Tower: TransformerRightTowerAS
                   LearnableSoftNorm (Mechanism 3)  ← NEW
                   → NumericalFeatureTokenizer
                   → TransformerEncoder (FFN 2×, Pre-LN)
                   → MeanPooling
    Fusion:      Concatenation + MLP — identical to dualtower.py

Mechanism 3 — Learnable Per-Feature Soft Normalization (LPFSN):
    For each numerical feature i:
        x̃_i = tanh( (x_i − center_i) / exp(log_scale_i) )

    - center_i, log_scale_i: learnable per-feature parameters
    - Initialized to 0 (center) and 0 (log_scale → scale=1)
    - Since data is already StandardScaler-normalized, tanh(x) with scale=1
      provides smooth bounded suppression: |x|>2σ → softly clamped to ±0.96
    - Fully differentiable, model-integrated, end-to-end trainable
    - Parameter cost: N_num × 2 ≈ 364 params (negligible)

Ablation purpose:
    dualtower      — mean pooling, no M3          → 0.6171 ± 0.0058 (baseline)
    dualtower_as   — mean pooling + M3 (this file) → TBD
    dualtower_ft   — CLS token, no M3             → 0.6245 ± 0.0015
    dualtower_ft_as — CLS token + M3 (future)     → TBD

References:
    - Gorishniy et al. (2021). NeurIPS. DOI: 10.48550/arXiv.2106.11959
    - LeCun et al. (1998). Efficient BackProp. DOI: 10.1007/978-3-642-35289-8_3
    - Tukey (1962). Ann. Math. Stat. DOI: 10.1214/AOMS/1177704711
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any

from .base import NumericalFeatureTokenizer

logger = logging.getLogger(__name__)


class LearnableSoftNorm(nn.Module):
    """
    Learnable Per-Feature Soft Normalization (Mechanism 3).

    Applies a differentiable, per-feature tanh normalization:
        x̃_i = tanh( (x_i − center_i) / scale_i )

    Parameters are initialized to center=0, scale=1 (i.e., tanh(x)),
    appropriate for StandardScaler-normalized inputs.

    Args:
        num_features: Number of numerical features (N_num)
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.center = nn.Parameter(torch.zeros(num_features))
        self.log_scale = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N_num] — StandardScaler-normalized numerical features
        Returns:
            x̃: [B, N_num] — soft-normalized, bounded in (−1, +1)
        """
        scale = self.log_scale.exp().clamp(min=1e-6)
        return torch.tanh((x - self.center) / scale)


class TransformerRightTowerAS(nn.Module):
    """
    Transformer right tower with Adaptive Safeguard Mechanism 3.

    Identical to TransformerRightTower in dualtower.py, with LearnableSoftNorm
    inserted before NumericalFeatureTokenizer.

    Pipeline:
        x [B, N]
        → LearnableSoftNorm   → x̃ [B, N],       bounded (−1, +1)
        → NumericalTokenizer  → tokens [B, N, d_model]
        → TransformerEncoder  → context [B, N, d_model]
        → MeanPool            → pooled [B, d_model]
    """

    def __init__(self, input_dim: int, d_model: int, nhead: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.soft_norm = LearnableSoftNorm(input_dim)
        self.tokenizer = NumericalFeatureTokenizer(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N_num]
        Returns:
            pooled: [B, d_model]
        """
        x = self.soft_norm(x)                              # [B, N], in (−1, +1)
        tokens = self.tokenizer(x)                         # [B, N, d_model]
        context = self.transformer_encoder(tokens)         # [B, N, d_model]
        return context.mean(dim=1)                         # [B, d_model]


class DualTowerAS(nn.Module):
    """
    DualTower with Adaptive Safeguard Mechanism 3 (DT-AS).

    Left Tower:  Categorical embeddings — unchanged from DualTower.
    Right Tower: TransformerRightTowerAS — adds LearnableSoftNorm (M3).
    Fusion:      Concatenation + MLP — unchanged from DualTower.

    Config key: model: dualtower_as
    """

    def __init__(self, config: Dict[str, Any], input_dim: int):
        super().__init__()
        self.config = config

        arch = config.get('architecture', {})
        if not arch:
            arch = config.get('model_config', {}).get('architecture', {})

        # === Left Tower: Categorical Embeddings (identical to DualTower) ===
        self.cat_features_config = arch['cat_features']

        if 'num_indices' in arch:
            self.num_indices = arch['num_indices']
        else:
            cat_idx_set = set(f['index'] for f in self.cat_features_config)
            self.num_indices = [
                i for i in range(input_dim) if i not in cat_idx_set
            ]

        self.embeddings = nn.ModuleList()
        self.cat_indices = []
        total_emb_dim = 0
        for feat in self.cat_features_config:
            self.embeddings.append(
                nn.Embedding(feat['num_classes'], feat['emb_dim'])
            )
            self.cat_indices.append(feat['index'])
            total_emb_dim += feat['emb_dim']

        # === Right Tower: Transformer + LearnableSoftNorm (M3) ===
        d_model = arch['trans_d_model']
        num_feat_count = len(self.num_indices)

        self.right_tower = TransformerRightTowerAS(
            input_dim=num_feat_count,
            d_model=d_model,
            nhead=arch.get('trans_nhead', 4),
            num_layers=arch.get('trans_num_layers', 2),
            dropout=arch['dropout']
        )
        right_output_dim = d_model

        # === Fusion Layer (identical to DualTower) ===
        fusion_in = total_emb_dim + right_output_dim
        fusion_hidden = arch['fusion_hidden_dim']

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(arch['dropout']),
            nn.Linear(fusion_hidden, 1)
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"DualTowerAS: DT + LearnableSoftNorm (M3, mean pooling) "
            f"| {n_params:,} parameters"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim]
        Returns:
            logits: [B]
        """
        # --- Left Tower ---
        emb_outs = []
        for idx, emb, feat in zip(
            self.cat_indices, self.embeddings, self.cat_features_config
        ):
            col_input = torch.clamp(
                x[:, idx], min=0, max=feat['num_classes'] - 1
            ).long()
            emb_outs.append(emb(col_input))
        left_out = torch.cat(emb_outs, dim=1)               # [B, total_emb_dim]

        # --- Right Tower (LearnableSoftNorm → Tokenizer → Transformer → Mean) ---
        num_data = x[:, self.num_indices]
        right_out = self.right_tower(num_data)               # [B, d_model]

        # --- Fusion ---
        combined = torch.cat([left_out, right_out], dim=1)  # [B, fusion_in]
        logits = self.fusion(combined)                       # [B, 1]

        return logits.squeeze(-1)
