"""
router.py — Exact ObserverRouter implementation per routing_contract.md.

Contract: g_i → pi_i (view weights, softmax) + beta_i (mode gate, sigmoid)
  beta→0 = isolation, beta→1 = interaction (non-reversible)
  First-pass: per-row routing. Per-head extension: add head axis to view_head.

DO NOT modify these semantics without updating the contract.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ObserverRouter(nn.Module):
    """
    [Contract §Required routing variables]
    g: [B, obs_dim]
    returns:
      pi:   [B, n_views]   view weights, softmax, sums to 1
      beta: [B, 1]         mode gate, sigmoid ∈ [0,1]
    """
    def __init__(self, obs_dim: int, n_views: int, hidden_dim: int = 32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.view_head = nn.Linear(hidden_dim, n_views)   # → pi
        self.mode_head = nn.Linear(hidden_dim, 1)          # → beta

        # Per-head extension point: replace view_head with
        # nn.Linear(hidden_dim, n_heads * n_views) and reshape output.

    def forward(self, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(g)
        pi = torch.softmax(self.view_head(h), dim=-1)    # [B, V]
        beta = torch.sigmoid(self.mode_head(h))           # [B, 1]
        return pi, beta
