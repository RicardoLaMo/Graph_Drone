"""
meta_learner_v4.py — v4 extensions to meta_learner.py

New components:
  - LabelContextEncoderV4: optional LayerNorm on label_ctx_vec output
  - ManifoldReaderV4: v3-compatible reader with optional per-view context return
"""

from __future__ import annotations

import math
import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.gora_tabular.src.meta_learner import (
    AlphaGate,
    LabelContextEncoder as _LabelContextEncoderV3,
    ManifoldReader as _ManifoldReaderV3,
    ViewSpecificEmbedder,
    alpha_gate_loss,
)

__all__ = [
    "ViewSpecificEmbedder",
    "AlphaGate",
    "alpha_gate_loss",
    "LabelContextEncoderV4",
    "ManifoldReaderV4",
]

EPS = 1e-8


class LabelContextEncoderV4(_LabelContextEncoderV3):
    """V4 extension: optional LayerNorm on the label context vector."""

    def __init__(
        self,
        n_views: int,
        d_model: int,
        d_z: int,
        n_out: int,
        use_layernorm: bool = False,
    ):
        super().__init__(n_views, d_model, d_z, n_out)
        self.label_ctx_ln: nn.Module = nn.LayerNorm(d_z) if use_layernorm else nn.Identity()

    def forward(
        self,
        lbl_nei: Optional[torch.Tensor],
        ew_anc: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        label_ctx_vec, lbl_delta, lbl_weight = super().forward(lbl_nei, ew_anc, view_mask)
        return self.label_ctx_ln(label_ctx_vec), lbl_delta, lbl_weight


class ManifoldReaderV4(_ManifoldReaderV3):
    """
    v4 extension of the v3 ManifoldReader.

    The forward path is unchanged, but `return_per_view=True` exposes the
    unweighted per-view context vectors for diagnostics and the isolation path.
    """

    def forward(
        self,
        h_nei_list: List[torch.Tensor],
        ew_anc: torch.Tensor,
        view_mask: torch.Tensor,
        z_anc: Optional[torch.Tensor] = None,
        lbl_delta: Optional[torch.Tensor] = None,
        lbl_weight: Optional[torch.Tensor] = None,
        return_per_view: bool = False,
    ):
        batch_size, _, n_views = ew_anc.shape
        _ = batch_size

        if self.use_query_z and z_anc is not None:
            query = self.W_Q(z_anc)
        else:
            query = None

        view_weights = torch.softmax(self.log_view_weight, dim=0)
        weighted_parts = []
        raw_parts = []

        for view_idx in range(n_views):
            h_view = h_nei_list[view_idx]
            value_view = self.W_V[view_idx](h_view)
            if lbl_delta is not None:
                value_view = value_view + lbl_delta[:, :, view_idx, :]

            if query is not None:
                key_view = self.W_K[view_idx](h_view)
                scores = (query.unsqueeze(1) * key_view).sum(-1) / math.sqrt(self.d_model)
                scores = scores + torch.log(ew_anc[:, :, view_idx] + EPS)
                if lbl_weight is not None:
                    scores = scores + lbl_weight[:, :, view_idx]
                absent = view_mask[:, :, view_idx] == 0
                scores = scores.masked_fill(absent, -1e9)
                attn = self.dropout(torch.softmax(scores, dim=-1))
            else:
                weights = ew_anc[:, :, view_idx] * view_mask[:, :, view_idx]
                weights = weights / weights.sum(-1, keepdim=True).clamp(min=EPS)
                attn = weights

            ctx_view = (attn.unsqueeze(-1) * value_view).sum(1)
            raw_parts.append(ctx_view)
            weighted_parts.append(view_weights[view_idx] * ctx_view)

        ctx_vec = torch.stack(weighted_parts, dim=0).sum(0)
        per_view_ctx = torch.stack(raw_parts, dim=1)
        if return_per_view:
            return ctx_vec, per_view_ctx
        return ctx_vec
