"""
row_transformer_v4.py — MQGoraTransformerV4 with explicit split-track mode routing.

This extends the v3 MQ-GoRA scaffold with:
  - explicit observer-driven `beta` gating for isolation vs interaction
  - a split-track router that returns `(pi, beta, tau)`
  - a faithful isolation branch built from per-view value summaries

The interaction branch keeps the existing sparse geometry-attention path.
The final representation is blended before prediction:

  final_h = mean_h [ (1 - beta_{i,h}) * iso_{i,h} + beta_{i,h} * inter_i ]

This keeps the implementation close to the repo's existing v3 code while
making the routing semantics explicit.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.gora_tabular.src.geometry_attention import SparseGeomLayer
from experiments.gora_tabular.src.meta_learner import (
    AlphaGate,
    ViewSpecificEmbedder,
    alpha_gate_loss,
)
from experiments.gora_tabular.src.row_transformer import (
    ColumnEmbedder,
    GoraTransformer,
    SingleViewTransformer,
    StandardTransformer,
)

from experiments.mq_gora_v4.shared.src.meta_learner_v4 import (
    LabelContextEncoderV4,
    ManifoldReaderV4,
)

EPS = 1e-6

__all__ = [
    "GoraTransformer",
    "StandardTransformer",
    "SingleViewTransformer",
    "SplitTrackRouter",
    "blend_iso_interaction",
    "MQGoraTransformerV4",
]


class SplitTrackRouter(nn.Module):
    """
    Rich router for v4 that outputs:
      pi   [B, H, M]  per-head view trust
      beta [B, H]     per-head isolation-vs-interaction gate
      tau  [H]        per-head attention temperature
    """

    def __init__(
        self,
        obs_dim: int,
        n_heads: int,
        n_views: int,
        d_z: int = 64,
        d_model: int = 64,
        has_z: bool = False,
        has_label: bool = False,
        has_ctx: bool = True,
        hidden: int = 64,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_views = n_views
        self.has_z = has_z
        self.has_label = has_label
        self.has_ctx = has_ctx
        self.d_z = d_z
        self.d_model = d_model

        in_dim = obs_dim
        if has_z:
            in_dim += d_z
        if has_label:
            in_dim += d_z
        if has_ctx:
            in_dim += d_model

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.backbone = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.view_heads = nn.ModuleList(
            [nn.Linear(hidden, n_views) for _ in range(n_heads)]
        )
        self.mode_heads = nn.ModuleList(
            [nn.Linear(hidden, 1) for _ in range(n_heads)]
        )
        self.log_tau = nn.Parameter(torch.zeros(n_heads))

    def forward(
        self,
        g: torch.Tensor,
        z_anc: Optional[torch.Tensor] = None,
        label_ctx_vec: Optional[torch.Tensor] = None,
        ctx_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = g.shape[0]
        parts = [g]
        if self.has_z:
            parts.append(
                z_anc
                if z_anc is not None
                else torch.zeros(batch_size, self.d_z, device=g.device)
            )
        if self.has_label:
            parts.append(
                label_ctx_vec
                if label_ctx_vec is not None
                else torch.zeros(batch_size, self.d_z, device=g.device)
            )
        if self.has_ctx:
            parts.append(
                ctx_vec
                if ctx_vec is not None
                else torch.zeros(batch_size, self.d_model, device=g.device)
            )

        hidden = self.backbone(self.input_proj(torch.cat(parts, dim=-1)))
        pi = torch.stack(
            [torch.softmax(view_head(hidden), dim=-1) for view_head in self.view_heads],
            dim=1,
        )
        beta = torch.stack(
            [torch.sigmoid(mode_head(hidden)).squeeze(-1) for mode_head in self.mode_heads],
            dim=1,
        )
        tau = torch.exp(self.log_tau)
        return pi, beta, tau


def blend_iso_interaction(
    iso_heads: torch.Tensor,
    inter_rep: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """
    Blend head-specific isolation reps with a shared interaction rep.

    iso_heads: [B, H, D]
    inter_rep: [B, D]
    beta: [B, H]
    returns: [B, D]
    """

    inter_heads = inter_rep.unsqueeze(1).expand(-1, iso_heads.shape[1], -1)
    blended = (1.0 - beta.unsqueeze(-1)) * iso_heads + beta.unsqueeze(-1) * inter_heads
    return blended.mean(dim=1)


class MQGoraTransformerV4(nn.Module):
    """
    Manifold-Query GoRA v4.

    Relative to v3 MQ-GoRA this adds:
      - explicit `beta` mode routing
      - an explicit isolation branch from per-view summaries
      - optional LayerNorm on label_ctx_vec (CA_FIX_2)
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
        use_label_ctx_layernorm: bool = False,
        lam_alpha: float = 0.05,
    ):
        super().__init__()
        self.n_views = n_views
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_z = d_z
        self.use_label_ctx = use_label_ctx
        self.use_teacher_query = use_teacher_query
        self.use_alpha_gate = use_alpha_gate
        self.lam_alpha = lam_alpha

        self.anchor_embed = ColumnEmbedder(n_features, d_model)
        self.view_embedder = ViewSpecificEmbedder(n_views, n_features, d_model)

        self.label_enc: Optional[LabelContextEncoderV4] = None
        if use_label_ctx:
            self.label_enc = LabelContextEncoderV4(
                n_views=n_views,
                d_model=d_model,
                d_z=d_z,
                n_out=max(1, n_classes),
                use_layernorm=use_label_ctx_layernorm,
            )

        self.manifold_reader = ManifoldReaderV4(
            n_views=n_views,
            d_z=d_z,
            d_model=d_model,
            use_query_z=use_teacher_query,
            dropout=dropout,
        )

        self.router = SplitTrackRouter(
            obs_dim=obs_dim,
            n_heads=n_heads,
            n_views=n_views,
            d_z=d_z,
            d_model=d_model,
            has_z=use_teacher_query,
            has_label=use_label_ctx,
            has_ctx=True,
            hidden=64,
        )

        self.layers = nn.ModuleList(
            [SparseGeomLayer(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)]
        )
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, out_dim),
        )

        self.alpha_gate: Optional[AlphaGate] = None
        if use_alpha_gate:
            self.alpha_gate = AlphaGate(d_z, d_model, out_dim)

    def _compute_iso_view_contexts(
        self,
        h_nei_list: list[torch.Tensor],
        ew_anc: torch.Tensor,
        view_mask: torch.Tensor,
        lbl_delta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        view_ctxs = []
        for view_idx in range(self.n_views):
            h_view = h_nei_list[view_idx]
            value_view = self.manifold_reader.W_V[view_idx](h_view)
            if lbl_delta is not None:
                value_view = value_view + lbl_delta[:, :, view_idx, :]
            weights = ew_anc[:, :, view_idx] * view_mask[:, :, view_idx]
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=EPS)
            view_ctx = (weights.unsqueeze(-1) * value_view).sum(dim=1)
            view_ctxs.append(view_ctx)
        return torch.stack(view_ctxs, dim=1)

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
        _, _, _ = x_nei.shape
        n_views = self.n_views

        if view_mask is None:
            view_mask = (ew_anc > 0).float()

        h_anc = self.anchor_embed(x_anc)
        x_nei_per_view = [x_nei for _ in range(n_views)]
        h_nei_list = self.view_embedder(x_nei_per_view)
        h_nei_shared = h_nei_list[0]

        label_ctx_vec, lbl_delta, lbl_weight = None, None, None
        if self.label_enc is not None:
            label_ctx_vec, lbl_delta, lbl_weight = self.label_enc(lbl_nei, ew_anc, view_mask)

        ctx_vec, per_view_ctx = self.manifold_reader(
            h_nei_list=h_nei_list,
            ew_anc=ew_anc,
            view_mask=view_mask,
            z_anc=z_anc if self.use_teacher_query else None,
            lbl_delta=lbl_delta,
            lbl_weight=lbl_weight,
            return_per_view=True,
        )

        pi, beta, tau = self.router(
            g=g_anc,
            z_anc=z_anc if self.use_teacher_query else None,
            label_ctx_vec=label_ctx_vec,
            ctx_vec=ctx_vec,
        )

        iso_view_ctxs = self._compute_iso_view_contexts(
            h_nei_list=h_nei_list,
            ew_anc=ew_anc,
            view_mask=view_mask,
            lbl_delta=lbl_delta,
        )
        iso_heads = torch.einsum("bhm,bmd->bhd", pi, iso_view_ctxs)

        for layer in self.layers:
            h_anc = layer(h_anc, h_nei_shared, pi, tau, ew_anc)
        inter_rep = h_anc
        final_rep = blend_iso_interaction(iso_heads, inter_rep, beta)
        pred_base = self.pred_head(final_rep)

        aux_losses: Dict[str, torch.Tensor] = {}
        debug = {
            "view_ctxs": iso_view_ctxs,
            "manifold_view_ctxs": per_view_ctx,
            "iso_heads": iso_heads,
            "inter_rep": inter_rep,
            "final_rep": final_rep,
        }

        if self.alpha_gate is not None and z_anc is not None:
            pred_final, alpha, pred_local = self.alpha_gate(z_anc, ctx_vec, pred_base)
            debug["alpha"] = alpha
            debug["pred_local"] = pred_local
            if agree_score is not None:
                aux_losses["alpha"] = alpha_gate_loss(alpha, agree_score, self.lam_alpha)
        else:
            pred_final = pred_base

        return pred_final, pi, beta, tau, aux_losses, debug
