"""
DualTower with Demographic-Conditioned Numerical Attention (DT-DC)

Architecture:
    Left Tower:  Categorical embeddings (Age, Gender, Ethnicity) — identical to DT-FT
    Right Tower: DCNRightTower
                   NumericalFeatureTokenizer
                   → prepend learnable [CLS] token
                   → DCNEncoderLayer × num_layers:
                       ① Self-attention: numerical tokens attend to each other
                       ② Cross-attention: numerical tokens attend to demographic context
                       ③ FFN (4× expansion, Pre-LN)
                   → [CLS] output as aggregated representation
    Fusion:      Concatenation + MLP — identical to DT-FT

Key difference from DT-FT (dualtower_ft.py):
    Each Transformer layer in the right tower adds cross-attention over the
    left tower's per-feature demographic embeddings (projected to d_model).
    This allows clinical numerical features (NIHSS, GCS, etc.) to be processed
    in the context of patient demographics — the cross-type conditional
    interaction that tree-based models capture through conditional splits.

Ablation chain:
    DualTower (mean pool) → DT-FT (+CLS) → DT-DC (+cross-attention)

References:
    Vaswani et al. (2017). Attention Is All You Need. NeurIPS.
        DOI: 10.48550/arXiv.1706.03762
    Gorishniy et al. (2021). Revisiting Deep Learning Models for Tabular Data.
        DOI: 10.48550/arXiv.2106.11959
    Huang et al. (2020). TabTransformer: Tabular Data Modeling Using
        Contextual Embeddings. DOI: 10.48550/arXiv.2012.06678
    Grinsztajn et al. (2022). Why do tree-based models still outperform deep
        learning on tabular data? NeurIPS. DOI: 10.48550/arXiv.2207.08815
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, List

from .base import NumericalFeatureTokenizer

logger = logging.getLogger(__name__)


class DemographicProjector(nn.Module):
    """
    Projects per-feature categorical embeddings to d_model.

    Each categorical embedding is independently projected via a Linear layer,
    preserving per-feature identity. Results are stacked into a sequence of
    demographic context tokens used as keys/values in cross-attention.

    Args:
        emb_dims: List of embedding dimensions, one per categorical feature
        d_model:  Target dimension matching the right tower's d_model
    """

    def __init__(self, emb_dims: List[int], d_model: int):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(dim, d_model) for dim in emb_dims
        ])

    def forward(self, emb_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            emb_list: List of [B, emb_dim_i] tensors, one per categorical feature
        Returns:
            dem_context: [B, n_cat, d_model]
        """
        projected = [proj(emb) for proj, emb in zip(self.projections, emb_list)]
        return torch.stack(projected, dim=1)                 # [B, n_cat, d_model]


class DCNEncoderLayer(nn.Module):
    """
    Transformer encoder layer augmented with demographic cross-attention.

    Sub-layer order (Pre-LN throughout, consistent with DT-FT):
        ① LayerNorm → Self-Attention (num tokens attend to each other) → residual
        ② LayerNorm → Cross-Attention (num tokens → demographic context) → residual
        ③ LayerNorm → FFN (GELU, 4× expansion) → residual
    """

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        ffn_dim = d_model * 4

        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, dem_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:           [B, N+1, d_model]  (CLS + numerical tokens)
            dem_context: [B, n_cat, d_model] (projected demographic embeddings)
        Returns:
            x:           [B, N+1, d_model]
        """
        # ① Self-attention (Pre-LN)
        x_norm = self.norm1(x)
        sa_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + sa_out

        # ② Cross-attention (Pre-LN): numerical tokens query demographic context
        x_norm = self.norm2(x)
        ca_out, _ = self.cross_attn(x_norm, dem_context, dem_context)
        x = x + ca_out

        # ③ FFN (Pre-LN)
        x = x + self.ffn(self.norm3(x))

        return x


class DCNRightTower(nn.Module):
    """
    FT-Transformer right tower augmented with demographic cross-attention.

    Identical to FTRightTower (dualtower_ft.py) except each encoder layer
    is a DCNEncoderLayer that includes cross-attention over demographic context.

    Pipeline:
        x [B, N]  +  dem_context [B, n_cat, d_model]
        → NumericalFeatureTokenizer → tokens [B, N, d_model]
        → prepend cls_token         → tokens [B, N+1, d_model]
        → DCNEncoderLayer × L       → tokens [B, N+1, d_model]
        → LayerNorm
        → tokens[:, 0, :]           → cls_out [B, d_model]
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.tokenizer = NumericalFeatureTokenizer(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layers = nn.ModuleList([
            DCNEncoderLayer(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, dem_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:           [B, N_num]
            dem_context: [B, n_cat, d_model]
        Returns:
            cls_out: [B, d_model]
        """
        B = x.size(0)
        tokens = self.tokenizer(x)                           # [B, N, d_model]

        cls = self.cls_token.expand(B, -1, -1)              # [B, 1, d_model]
        tokens = torch.cat([cls, tokens], dim=1)             # [B, N+1, d_model]

        for layer in self.layers:
            tokens = layer(tokens, dem_context)

        tokens = self.final_norm(tokens)
        return tokens[:, 0, :]                               # [B, d_model]


class DualTowerDCN(nn.Module):
    """
    DualTower with Demographic-Conditioned Numerical Attention (DT-DC).

    Left Tower:  Categorical embeddings — identical to DT-FT.
    Right Tower: DCNRightTower — CLS aggregation + per-layer cross-attention
                 from numerical tokens to projected demographic embeddings.
    Fusion:      Concatenation + MLP — identical to DT-FT.

    Config key: model: dualtower_dcn
    """

    def __init__(self, config: Dict[str, Any], input_dim: int):
        super().__init__()
        self.config = config

        arch = config.get('architecture', {})
        if not arch:
            arch = config.get('model_config', {}).get('architecture', {})

        # === Left Tower: Categorical Embeddings (identical to DT-FT) ===
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
        emb_dims = []
        total_emb_dim = 0
        for feat in self.cat_features_config:
            self.embeddings.append(
                nn.Embedding(feat['num_classes'], feat['emb_dim'])
            )
            self.cat_indices.append(feat['index'])
            emb_dims.append(feat['emb_dim'])
            total_emb_dim += feat['emb_dim']

        # === Demographic Projector: each embedding → d_model ===
        d_model = arch['trans_d_model']
        self.dem_proj = DemographicProjector(emb_dims, d_model)

        # === Right Tower: CLS + cross-attention ===
        num_feat_count = len(self.num_indices)
        self.right_tower = DCNRightTower(
            input_dim=num_feat_count,
            d_model=d_model,
            nhead=arch.get('trans_nhead', 4),
            num_layers=arch.get('trans_num_layers', 2),
            dropout=arch['dropout'],
        )

        # === Fusion Layer (identical to DT-FT) ===
        fusion_in = total_emb_dim + d_model
        fusion_hidden = arch['fusion_hidden_dim']
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(arch['dropout']),
            nn.Linear(fusion_hidden, 1),
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"DualTowerDCN: DT-FT + Demographic Cross-Attention "
            f"(CLS, FFN 4×, cross-attn per layer) | {n_params:,} parameters"
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
        left_out = torch.cat(emb_outs, dim=1)                # [B, total_emb_dim]

        # --- Demographic Context for Cross-Attention ---
        dem_context = self.dem_proj(emb_outs)                 # [B, n_cat, d_model]

        # --- Right Tower (CLS + cross-attention conditioning) ---
        num_data = x[:, self.num_indices]
        right_out = self.right_tower(num_data, dem_context)   # [B, d_model]

        # --- Fusion ---
        combined = torch.cat([left_out, right_out], dim=1)    # [B, total_emb_dim + d_model]
        logits = self.fusion(combined)                         # [B, 1]

        return logits.squeeze(-1)
