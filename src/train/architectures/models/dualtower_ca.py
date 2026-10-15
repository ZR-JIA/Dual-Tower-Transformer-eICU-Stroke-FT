"""
DualTower with Cross-Attention Fusion (DT-CA)

Architecture:
    Left Tower:  Categorical embeddings (Age, Gender, Ethnicity)
                 Each embedding is independently projected to d_model,
                 forming a demographic token sequence [B, 3, d_model].
    Right Tower: NumericalFeatureTokenizer -> TransformerEncoder
                 Returns full token sequence [B, N_num, d_model]
                 (no mean-pooling, preserving per-feature context).
    Fusion:      CrossAttentionFusion
                 Demographics (Left) act as Query to attend over
                 Vitals (Right) as Key/Value.
                 Gated combination of attended and pooled representations.

Key difference from dualtower.py:
    - Fusion is dynamic (cross-attention) rather than static (concatenation).
    - Right tower output is a token sequence, not a pooled vector.
    - Left tower embeddings are kept separate as individual tokens.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, List

from .base import NumericalFeatureTokenizer

logger = logging.getLogger(__name__)


class TransformerRightTowerCA(nn.Module):
    """
    Transformer encoder for numerical features.

    Same internal structure as TransformerRightTower in dualtower.py,
    but returns the full token sequence [B, N, d_model] instead of
    mean-pooling to [B, d_model], enabling downstream cross-attention.
    """

    def __init__(self, input_dim: int, d_model: int, nhead: int,
                 num_layers: int, dropout: float):
        """
        Args:
            input_dim:  Number of numerical features
            d_model:    Embedding dimension per feature token
            nhead:      Number of attention heads
            num_layers: Number of TransformerEncoder layers
            dropout:    Dropout rate
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
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N_num]
        Returns:
            tokens: [B, N_num, d_model]  — full sequence, no pooling
        """
        tokens = self.tokenizer(x)                      # [B, N, d_model]
        tokens = self.transformer_encoder(tokens)        # [B, N, d_model]
        return tokens


class CrossAttentionFusion(nn.Module):
    """
    Dynamic fusion via Cross-Attention.

    Demographic tokens (Left Tower) act as Query to attend over
    vital tokens (Right Tower) as Key and Value. This enables the
    model to learn dynamic, clinically-informed feature weighting —
    e.g., blood pressure receives higher attention weight for elderly patients.

    Pipeline:
        1. Project each categorical embedding to d_model
           → left_tokens [B, 3, d_model]
        2. Cross-Attention: Q=left_tokens, K=V=right_tokens
           → attended   [B, 3, d_model]
        3. Residual + LayerNorm
        4. Mean pool attended → demo_aware  [B, d_model]
        5. Mean pool right_tokens → vital_pooled [B, d_model]
        6. Gated combination:
               gate   = σ(Linear([demo_aware; vital_pooled]))  [B, d_model]
               fused  = gate * demo_aware + (1−gate) * vital_pooled
        7. Classification head → logit [B, 1]
    """

    def __init__(self, cat_emb_dims: List[int], d_model: int, nhead: int,
                 fusion_hidden: int, dropout: float):
        """
        Args:
            cat_emb_dims:   List of embedding dims per categorical feature
                            e.g. [16, 2, 4] for age, gender, ethnicity
            d_model:        Unified token dimension (must match right tower)
            nhead:          Number of cross-attention heads
            fusion_hidden:  Hidden dim of classification head
            dropout:        Dropout rate
        """
        super().__init__()

        # Project each categorical embedding to the shared d_model space
        self.left_projections = nn.ModuleList([
            nn.Linear(dim, d_model) for dim in cat_emb_dims
        ])

        # Cross-attention: demographics (Q) attend over vitals (K, V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

        # Gated combination of attended demographics and pooled vitals
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1)
        )

    def forward(self, emb_list: List[torch.Tensor],
                right_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb_list:     List of [B, emb_dim_i] tensors (one per categorical feature)
            right_tokens: [B, N_num, d_model] from right tower
        Returns:
            logits: [B]
        """
        # Step 1: Project categorical embeddings → demographic token sequence
        left_tokens = torch.stack(
            [proj(emb) for proj, emb in zip(self.left_projections, emb_list)],
            dim=1
        )                                                # [B, 3, d_model]

        # Step 2: Cross-attention (demographics query vitals)
        attended, _ = self.cross_attn(
            query=left_tokens,
            key=right_tokens,
            value=right_tokens
        )                                                # [B, 3, d_model]

        # Step 3: Residual + LayerNorm
        attended = self.norm(attended + left_tokens)     # [B, 3, d_model]

        # Step 4: Pool to get demographic-aware representation
        demo_aware = attended.mean(dim=1)               # [B, d_model]

        # Step 5: Pool to get vital representation
        vital_pooled = right_tokens.mean(dim=1)         # [B, d_model]

        # Step 6: Gated combination
        gate = self.gate(
            torch.cat([demo_aware, vital_pooled], dim=1)
        )                                                # [B, d_model]
        fused = gate * demo_aware + (1.0 - gate) * vital_pooled  # [B, d_model]

        # Step 7: Classify
        logits = self.classifier(fused)                 # [B, 1]
        return logits.squeeze(-1)


class DualTowerCA(nn.Module):
    """
    DualTower with Cross-Attention Fusion (DT-CA).

    Left Tower:  Learnable embeddings for categorical demographics.
                 Each feature becomes an independent token projected to d_model.
    Right Tower: Transformer encoder, returns full token sequence [B, N, d_model].
    Fusion:      Cross-Attention (demographics as Query, vitals as K/V)
                 + Gated combination → classification head.

    Config key: model: dualtower_ca
    """

    def __init__(self, config: Dict[str, Any], input_dim: int):
        """
        Args:
            config:    Model configuration dict (expects 'architecture' key)
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
            self.num_indices = [
                i for i in range(input_dim) if i not in cat_idx_set
            ]

        self.embeddings = nn.ModuleList()
        self.cat_indices = []
        cat_emb_dims = []
        for feat in self.cat_features_config:
            self.embeddings.append(
                nn.Embedding(feat['num_classes'], feat['emb_dim'])
            )
            self.cat_indices.append(feat['index'])
            cat_emb_dims.append(feat['emb_dim'])

        # === Right Tower: Returns full token sequence ===
        d_model = arch['trans_d_model']
        num_feat_count = len(self.num_indices)

        self.right_tower = TransformerRightTowerCA(
            input_dim=num_feat_count,
            d_model=d_model,
            nhead=arch.get('trans_nhead', 4),
            num_layers=arch.get('trans_num_layers', 2),
            dropout=arch['dropout']
        )

        # === Cross-Attention Fusion ===
        ca_cfg = arch.get('cross_attention', {})
        self.fusion = CrossAttentionFusion(
            cat_emb_dims=cat_emb_dims,
            d_model=d_model,
            nhead=ca_cfg.get('nhead', arch.get('trans_nhead', 4)),
            fusion_hidden=arch['fusion_hidden_dim'],
            dropout=arch['dropout']
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"DualTowerCA: Initialized with Cross-Attention Fusion "
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
            emb_outs.append(emb(col_input))              # [B, emb_dim_i]

        # --- Right Tower (full token sequence) ---
        num_data = x[:, self.num_indices]
        right_tokens = self.right_tower(num_data)        # [B, N_num, d_model]

        # --- Cross-Attention Fusion ---
        logits = self.fusion(emb_outs, right_tokens)     # [B]

        return logits
