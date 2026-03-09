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
from typing import Optional, Tuple


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

    def __init__(self, n_heads: int, n_views: int, **kwargs):
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

    def __init__(self, obs_dim: int = 1, n_heads: int = 4, n_views: int = 4, hidden: int = 32, **kwargs):
        super().__init__()
        self.real_router = MoERouter(obs_dim, n_heads, n_views, hidden)

    def forward(self, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shuffle g along batch dim to destroy row-level geometric coordination
        perm = torch.randperm(g.shape[0], device=g.device)
        g_shuffled = g[perm]
        return self.real_router(g_shuffled)


class RichMoERouter(nn.Module):
    """
    v3 router for MQ-GoRA (G7-G10).

    Accepts the full manifold-aware input bundle:
      g_anc         [B, obs_dim]  — geometry scalars (always present)
      z_anc         [B, d_z]      — frozen teacher embedding (G9/G10)
      label_ctx_vec [B, d_z]      — per-view label context (G8/G9/G10)
      ctx_vec       [B, d_model]  — neighbourhood context from ManifoldReader (G7-G10)

    An input_proj fuses the active components → hidden, then the same
    backbone + per-head projection structure as MoERouter runs unchanged.

    Absent components are zeroed (controlled by has_* flags), preserving
    backward-compatibility for ablation experiments G7/G8/G9.
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
        if has_z:     in_dim += d_z
        if has_label: in_dim += d_z
        if has_ctx:   in_dim += d_model

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.backbone = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.head_projections = nn.ModuleList([
            nn.Linear(hidden, n_views) for _ in range(n_heads)
        ])
        self.log_tau = nn.Parameter(torch.zeros(n_heads))

    def forward(
        self,
        g: torch.Tensor,
        z_anc: Optional[torch.Tensor] = None,
        label_ctx_vec: Optional[torch.Tensor] = None,
        ctx_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        g:             [B, obs_dim]
        z_anc:         [B, d_z]   or None
        label_ctx_vec: [B, d_z]   or None
        ctx_vec:       [B, d_model] or None
        Returns pi [B, H, M], tau [H]
        """
        B = g.shape[0]
        parts = [g]
        if self.has_z:
            parts.append(z_anc if z_anc is not None
                         else torch.zeros(B, self.d_z, device=g.device))
        if self.has_label:
            parts.append(label_ctx_vec if label_ctx_vec is not None
                         else torch.zeros(B, self.d_z, device=g.device))
        if self.has_ctx:
            parts.append(ctx_vec if ctx_vec is not None
                         else torch.zeros(B, self.d_model, device=g.device))

        x = torch.cat(parts, dim=-1)
        h = self.backbone(self.input_proj(x))
        pi = torch.stack([torch.softmax(hp(h), dim=-1)
                          for hp in self.head_projections], dim=1)  # [B, H, M]
        tau = torch.exp(self.log_tau)                               # [H]
        return pi, tau
