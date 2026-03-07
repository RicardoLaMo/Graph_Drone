"""
moe_router.py — Mixture-of-Experts router for GoRA-Tabular.

Produces π_{i,h,m} per row i, per attention head h, per view m.
Shape: [B, H, M]

These weights form the head-specific effective adjacency:
  Ã_{ij}^{i,h} = Σ_m π_{i,h,m} · A^(m)_{ij}

Design: shared backbone (captures geometry), then per-head softmax heads.
This allows different heads to specialise to different views based on geometry.

Temperature τ_h per head is learned — allows heads to sharpen or broaden.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MoERouter(nn.Module):
    """
    Observer-driven Mixture-of-Experts router.

    g: [B, obs_dim]  — geometry signals (kappa, LID, LOF, per-view density)
    Returns:
      pi:  [B, H, M]   view routing weights per head (softmax, sums to 1 over M)
      tau: [H]          per-head temperature (learned)
    """

    def __init__(self, obs_dim: int, n_heads: int, n_views: int, hidden: int = 32):
        super().__init__()
        self.n_heads = n_heads
        self.n_views = n_views

        # Shared geometry backbone
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )

        # Per-head routing logits
        # Each head gets its own linear projection from the shared backbone
        self.head_projections = nn.ModuleList([
            nn.Linear(hidden, n_views) for _ in range(n_heads)
        ])

        # Per-head learned temperature (log-space for stability)
        # τ_h > 0 always; larger τ sharpens the adjacency log-bias
        self.log_tau = nn.Parameter(torch.zeros(n_heads))

    def forward(self, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        g: [B, obs_dim]
        returns:
          pi:  [B, H, M]  — per-head view weights
          tau: [H]         — per-head temperature (positive)
        """
        h = self.backbone(g)                                  # [B, hidden]
        pi_per_head = []
        for head_proj in self.head_projections:
            logits = head_proj(h)                             # [B, M]
            pi_per_head.append(torch.softmax(logits, dim=-1))
        pi = torch.stack(pi_per_head, dim=1)                  # [B, H, M]
        tau = torch.exp(self.log_tau)                         # [H]  positive
        return pi, tau


class UniformRouter(nn.Module):
    """Ablation G3: uniform pi = 1/M for all rows and heads. No geometry used."""

    def __init__(self, n_heads: int, n_views: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_views = n_views
        # tau still learned per head in the full model — here fix to 1
        self.log_tau = nn.Parameter(torch.zeros(n_heads))

    def forward(self, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = g.shape[0]
        pi = torch.ones(B, self.n_heads, self.n_views, device=g.device) / self.n_views
        tau = torch.exp(self.log_tau)
        return pi, tau


class RandomRouter(nn.Module):
    """Ablation G4: shuffled pi from shuffled g — tests if geometry signal matters."""

    def __init__(self, obs_dim: int, n_heads: int, n_views: int, hidden: int = 32):
        super().__init__()
        self.real_router = MoERouter(obs_dim, n_heads, n_views, hidden)

    def forward(self, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shuffle g along batch dim to destroy row-level geometric coordination
        perm = torch.randperm(g.shape[0], device=g.device)
        g_shuffled = g[perm]
        return self.real_router(g_shuffled)
