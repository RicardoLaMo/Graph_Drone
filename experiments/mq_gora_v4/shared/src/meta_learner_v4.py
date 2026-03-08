"""
meta_learner_v4.py — V4 extensions to meta_learner.py

New components:
  LabelContextEncoderV4 — adds optional LayerNorm on label_ctx_vec output (CA_FIX_2)

Unchanged components are imported directly from v3 meta_learner.
"""
import sys, os

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3_SRC = os.path.normpath(os.path.join(_HERE, '..', '..', '..', 'gora_tabular', 'src'))
if _V3_SRC not in sys.path:
    sys.path.insert(0, _V3_SRC)

import torch
import torch.nn as nn
from typing import Optional, Tuple

from meta_learner import (                          # re-export unchanged
    ViewSpecificEmbedder, ManifoldReader,
    AlphaGate, alpha_gate_loss,
    LabelContextEncoder as _LabelContextEncoderV3,
)

__all__ = [
    'ViewSpecificEmbedder', 'ManifoldReader',
    'AlphaGate', 'alpha_gate_loss',
    'LabelContextEncoderV4',
]

EPS = 1e-8


class LabelContextEncoderV4(_LabelContextEncoderV3):
    """
    V4 extension: adds optional LayerNorm on the label_ctx_vec output.

    CA_FIX_2 rationale:
      Raw regression labels enter the MLP with high variance and unknown scale.
      Even with y-normalisation in the precompute step the MLP output can still
      saturate.  A LayerNorm on label_ctx_vec re-centres and scales the router
      signal before it concatenates with [g_anc; z_anc; ctx_vec].

    Args:
      use_layernorm (bool): if True, apply LayerNorm(d_z) to label_ctx_vec.
        Recommended: True for regression, optional for classification.
    """

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
        label_ctx_vec = self.label_ctx_ln(label_ctx_vec)
        return label_ctx_vec, lbl_delta, lbl_weight
