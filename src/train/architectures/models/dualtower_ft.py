"""
DualTower with FT-Transformer Right Tower (DT-FT)

Architecture:
    Left Tower:  Categorical embeddings (Age, Gender, Ethnicity) — identical to DualTower
    Right Tower: FTRightTower
                   NumericalFeatureTokenizer
                   → prepend learnable [CLS] token
                   → TransformerEncoder (FFN dim = 4 × d_model, following FT-Transformer)
                   → [CLS] output as aggregated representation
    Fusion:      Concatenation + MLP classification head — identical to DualTower

Key difference from dualtower.py:
    - Right tower uses a learnable [CLS] token for aggregation instead of mean pooling.
    - [CLS] attends over ALL numerical feature tokens, learning to route globally
      relevant information rather than averaging all tokens equally.
    - FFN expansion ratio is 4× (FT-Transformer default) vs 2× in the original.

Reference:
    Gorishniy et al. (2021). Revisiting Deep Learning Models for Tabular Data.
    NeurIPS 2021. https://arxiv.org/abs/2106.11959
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, List

from .base import NumericalFeatureTokenizer

logger = logging.getLogger(__name__)


class FTRightTower(nn.Module):
    """
    FT-Transformer style right tower for numerical features.

    Replaces mean pooling with a learnable [CLS] token.
    The [CLS] token is prepended to the feature token sequence and
    processed jointly by the Transformer encoder. Its output serves
    as the aggregated representation of all numerical features.

    Pipeline:
        x [B, N]
        → NumericalFeatureTokenizer → tokens [B, N, d_model]
        → prepend cls_token         → tokens [B, N+1, d_model]
        → TransformerEncoder        → tokens [B, N+1, d_model]
        → tokens[:, 0, :]           → cls_out [B, d_model]
    """

    def __init__(self, input_dim: int, d_model: int, nhead: int,
                 num_layers: int, dropout: float):
        """
        Args:
            input_dim:  Number of numerical features
            d_model:    Token embedding dimension
            nhead:      Number of self-attention heads
            num_layers: Number of TransformerEncoder layers
            dropout:    Dropout rate
        """
        super().__init__()
        self.tokenizer = NumericalFeatureTokenizer(input_dim, d_model)

        # Learnable [CLS] token — initialized to zero, learned during training
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,   # FT-Transformer convention (4× vs 2×)
            dropout=dropout,
            batch_first=True,
            norm_first=True                # Pre-LN for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N_num]
        Returns:
            cls_out: [B, d_model]
        """
        B = x.size(0)
        tokens = self.tokenizer(x)                          # [B, N, d_model]

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)             # [B, 1, d_model]
        tokens = torch.cat([cls, tokens], dim=1)            # [B, N+1, d_model]

        # Self-attention over all tokens (CLS + features)
        tokens = self.transformer_encoder(tokens)           # [B, N+1, d_model]

        # Return [CLS] output as the aggregated representation
        return tokens[:, 0, :]                              # [B, d_model]


class DualTowerFT(nn.Module):
    """
    DualTower with FT-Transformer Right Tower (DT-FT).

    Left Tower:  Learnable embeddings for categorical demographics — unchanged.
    Right Tower: FTRightTower — [CLS] token aggregation instead of mean pooling.
    Fusion:      Concatenation → MLP → logit — unchanged.

    Config key: model: dualtower_ft
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

        # === Right Tower: FT-Transformer (CLS token) ===
        d_model = arch['trans_d_model']
        num_feat_count = len(self.num_indices)

        self.right_tower = FTRightTower(
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

        # ─── Adaptive Runtime Safeguard toggle ────────────────────────────
        # Default True: NaN guard + categorical clamp + numerical clamp are
        # applied in _compute_towers. Set to False at inference time to
        # bypass the inference-time safeguards (used by the stress-test
        # harness in run_safeguard_stress_test.py). See paper §3.1.2.
        self.safeguard_enabled: bool = True

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"DualTowerFT: Initialized with FT-Transformer Right Tower "
            f"(CLS aggregation, FFN 4×) | {n_params:,} parameters | "
            f"safeguard_enabled={self.safeguard_enabled}"
        )

    def _compute_towers(self, x: torch.Tensor):
        """
        Compute left and right tower outputs (shared logic).

        Includes the inference-time portion of the Adaptive Runtime Safeguard
        (three stages: NaN/Inf guard, categorical clamp, numerical clamp).
        The safeguard can be toggled via ``self.safeguard_enabled`` (default
        True). Toggling is used by the stress-test harness in
        ``run_safeguard_stress_test.py``; production callers should leave it
        enabled. See paper §3.1.2.
        """
        safeguard = getattr(self, 'safeguard_enabled', True)

        # ─── Stage A: NaN / Inf guard ──────────────────────────────────────
        # Replace NaN / ±Inf with 0 (post-scaling neutral). On clean inputs
        # this is a no-op; on downstream NaN that bypassed preprocessing this
        # prevents propagation through the network.
        if safeguard:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # ─── Left Tower (categorical) ──────────────────────────────────────
        emb_outs = []
        for idx, emb, feat in zip(
            self.cat_indices, self.embeddings, self.cat_features_config
        ):
            if safeguard:
                # Stage B: categorical clamp to embedding vocabulary range
                col_input = torch.clamp(
                    x[:, idx], min=0, max=feat['num_classes'] - 1
                ).long()
            else:
                # Bypass: direct cast. Raises IndexError if the index is
                # outside the embedding's vocabulary (e.g., age = 200).
                col_input = x[:, idx].long()
            emb_outs.append(emb(col_input))
        left_out = torch.cat(emb_outs, dim=1)                # [B, total_emb_dim]

        # ─── Right Tower (numerical, CLS aggregation) ──────────────────────
        num_data = x[:, self.num_indices]
        if safeguard:
            # Stage C: numerical clamp to ±10 in post-scaling z-score units.
            # Legitimate physiological values after StandardScaler are well
            # inside ±5; ±10 leaves a safe margin while bounding pathological
            # device errors and unit mistakes.
            num_data = torch.clamp(num_data, min=-10.0, max=10.0)
        right_out = self.right_tower(num_data)               # [B, d_model]

        return left_out, right_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim]
        Returns:
            logits: [B]
        """
        left_out, right_out = self._compute_towers(x)

        # --- Fusion ---
        combined = torch.cat([left_out, right_out], dim=1)  # [B, total_emb_dim + d_model]
        logits = self.fusion(combined)                       # [B, 1]

        return logits.squeeze(-1)
