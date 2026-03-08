"""
row_transformer_v4.py — MQGoraTransformerV4

Adds one new flag vs v3 MQGoraTransformer:
  use_label_ctx_layernorm (bool): applies LayerNorm to label_ctx_vec (CA_FIX_2).

All v1/v2 model classes (GoraTransformer, StandardTransformer, SingleViewTransformer)
are re-exported unchanged from v3.
"""
import sys, os

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3_SRC = os.path.normpath(os.path.join(_HERE, '..', '..', '..', 'gora_tabular', 'src'))
if _V3_SRC not in sys.path:
    sys.path.insert(0, _V3_SRC)

import torch
import torch.nn as nn
from typing import Dict, Optional

from moe_router import RichMoERouter
from geometry_attention import SparseGeomLayer
from row_transformer import (
    GoraTransformer, StandardTransformer, SingleViewTransformer, ColumnEmbedder,
)
from meta_learner import ViewSpecificEmbedder, ManifoldReader, AlphaGate, alpha_gate_loss
from meta_learner_v4 import LabelContextEncoderV4

__all__ = [
    'GoraTransformer', 'StandardTransformer', 'SingleViewTransformer',
    'MQGoraTransformerV4',
]


class MQGoraTransformerV4(nn.Module):
    """
    Manifold-Query GoRA v4.

    Identical to v3 MQGoraTransformer except:
      use_label_ctx_layernorm: adds LayerNorm(d_z) to label_ctx_vec (CA_FIX_2).

    Ablation flags (same as v3):
      use_label_ctx   (G8+):  LabelContextEncoderV4 active
      use_teacher_query (G9+): z_anc used as ManifoldReader cross-attn query
      use_alpha_gate  (G10):  AlphaGate prediction fusion

    Returns: (pred_final, pi, tau, aux_losses_dict)
    """

    def __init__(
        self,
        n_features: int,
        obs_dim: int,
        n_views: int,
        out_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
        d_z: int = 64,
        n_classes: int = 1,
        use_label_ctx: bool = False,
        use_teacher_query: bool = False,
        use_alpha_gate: bool = False,
        use_label_ctx_layernorm: bool = False,   # ← V4 addition (CA_FIX_2)
        lam_alpha: float = 0.05,
    ):
        super().__init__()
        self.n_views = n_views
        self.d_model = d_model
        self.d_z = d_z
        self.use_label_ctx = use_label_ctx
        self.use_teacher_query = use_teacher_query
        self.use_alpha_gate = use_alpha_gate
        self.lam_alpha = lam_alpha

        # Stage A
        self.anchor_embed = ColumnEmbedder(n_features, d_model)
        self.view_embedder = ViewSpecificEmbedder(n_views, n_features, d_model)

        # Stage B — V4: use LabelContextEncoderV4 with optional LayerNorm
        self.label_enc: Optional[LabelContextEncoderV4] = None
        if use_label_ctx:
            self.label_enc = LabelContextEncoderV4(
                n_views, d_model, d_z,
                n_out=max(1, n_classes),
                use_layernorm=use_label_ctx_layernorm,
            )

        # Stage C
        self.manifold_reader = ManifoldReader(
            n_views=n_views, d_z=d_z, d_model=d_model,
            use_query_z=use_teacher_query, dropout=dropout,
        )

        # Stage D
        self.router = RichMoERouter(
            obs_dim=obs_dim, n_heads=n_heads, n_views=n_views,
            d_z=d_z, d_model=d_model,
            has_z=use_teacher_query,
            has_label=use_label_ctx,
            has_ctx=True, hidden=64,
        )

        # Stage E
        self.layers = nn.ModuleList([
            SparseGeomLayer(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Linear(d_model // 2, out_dim),
        )

        # Stage F
        self.alpha_gate: Optional[AlphaGate] = None
        if use_alpha_gate:
            self.alpha_gate = AlphaGate(d_z, d_model, out_dim)

    def forward(
        self,
        x_anc: torch.Tensor,
        g_anc: torch.Tensor,
        x_nei: torch.Tensor,
        ew_anc: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
        z_anc: Optional[torch.Tensor] = None,
        lbl_nei: Optional[torch.Tensor] = None,
        agree_score: Optional[torch.Tensor] = None,
        **_,
    ):
        B, K, _ = x_nei.shape
        M = self.n_views

        if view_mask is None:
            view_mask = (ew_anc > 0).float()

        # Stage A
        h_anc = self.anchor_embed(x_anc)
        x_nei_per_view = [x_nei for _ in range(M)]
        h_nei_list = self.view_embedder(x_nei_per_view)
        h_nei_shared = h_nei_list[0]

        # Stage B
        label_ctx_vec, lbl_delta, lbl_weight = None, None, None
        if self.label_enc is not None:
            label_ctx_vec, lbl_delta, lbl_weight = self.label_enc(
                lbl_nei, ew_anc, view_mask)

        # Stage C
        ctx_vec = self.manifold_reader(
            h_nei_list=h_nei_list, ew_anc=ew_anc, view_mask=view_mask,
            z_anc=z_anc if self.use_teacher_query else None,
            lbl_delta=lbl_delta, lbl_weight=lbl_weight,
        )

        # Stage D
        pi, tau = self.router(
            g=g_anc,
            z_anc=z_anc if self.use_teacher_query else None,
            label_ctx_vec=label_ctx_vec,
            ctx_vec=ctx_vec,
        )

        # Stage E
        for layer in self.layers:
            h_anc = layer(h_anc, h_nei_shared, pi, tau, ew_anc)

        pred_base = self.pred_head(h_anc)

        # Stage F
        aux_losses: Dict[str, torch.Tensor] = {}
        if self.alpha_gate is not None and z_anc is not None:
            pred_final, alpha, pred_local = self.alpha_gate(z_anc, ctx_vec, pred_base)
            if agree_score is not None:
                aux_losses["alpha"] = alpha_gate_loss(alpha, agree_score, self.lam_alpha)
        else:
            pred_final = pred_base

        return pred_final, pi, tau, aux_losses
