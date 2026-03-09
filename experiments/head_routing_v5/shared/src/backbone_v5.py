"""
backbone_v5.py — HeadRoutingBackboneV5 with quality-aware routing.

Key changes from HeadRoutingBackbone (worktree):
  - Returns head_repr [B, H, Dh] WITHOUT FLATTENING — task head owns the collapse
  - QualityAwareRouter: pi_logit += log(q_v + ε) prior; beta_logit += w * mean_J
  - Optional sigma2_v in router input (Gap 13 — router only, not encoder)
  - Optional view_embed token added to anchor projection (Gap 7)
  - Optional row-adaptive geom_scale: softplus(w @ g_local) per head (Gap 3)

Gap coverage: Gap 1 (quality prior), Gap 2 (Jaccard beta), Gap 3 (adaptive tau),
              Gap 7 (view embed), Gap 13 (sigma2_v leakage control)
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-6


# ---------------------------------------------------------------------------
# Shared headwise helpers
# ---------------------------------------------------------------------------

def _apply_headwise(x: torch.Tensor, layers: nn.ModuleList) -> torch.Tensor:
    """Apply one linear per head. x: [B, H, Dh] → [B, H, Dh]."""
    return torch.stack([layer(x[:, h, :]) for h, layer in enumerate(layers)], dim=1)


def _apply_headwise_neighbors(x: torch.Tensor, layers: nn.ModuleList) -> torch.Tensor:
    """Apply one linear per head. x: [B, K, H, Dh] → [B, K, H, Dh]."""
    return torch.stack([layer(x[:, :, h, :]) for h, layer in enumerate(layers)], dim=2)


# ---------------------------------------------------------------------------
# View encoder — per-view QKV with optional view_embed and adaptive geom_scale
# ---------------------------------------------------------------------------

class HeadwiseEdgeBiasedViewEncoderV5(nn.Module):
    """
    Per-view encoder with:
      - optional view_embed [Dh] token added to anchor projection (Gap 7)
      - optional row-adaptive geom_scale: softplus(Linear(obs_dim) + b) per head (Gap 3)
    """

    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        head_dim: int,
        obs_dim: int = 0,               # >0 enables adaptive geom_scale (Gap 3)
        use_view_embed: bool = False,    # Gap 7
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.use_adaptive_tau = obs_dim > 0
        self.use_view_embed = use_view_embed

        rep_dim = n_heads * head_dim
        self.anchor_proj = nn.Sequential(nn.Linear(input_dim, rep_dim), nn.LayerNorm(rep_dim))
        self.nei_proj = nn.Sequential(nn.Linear(input_dim, rep_dim), nn.LayerNorm(rep_dim))

        if use_view_embed:
            self.view_embed = nn.Parameter(torch.randn(head_dim) * 0.01)

        self.q_layers = nn.ModuleList([nn.Linear(head_dim, head_dim, bias=False) for _ in range(n_heads)])
        self.k_layers = nn.ModuleList([nn.Linear(head_dim, head_dim, bias=False) for _ in range(n_heads)])
        self.v_layers = nn.ModuleList([nn.Linear(head_dim, head_dim, bias=False) for _ in range(n_heads)])
        self.out_layers = nn.ModuleList([nn.Linear(head_dim, head_dim, bias=False) for _ in range(n_heads)])
        self.dropout = nn.Dropout(dropout)

        if self.use_adaptive_tau:
            # Linear(obs_dim → n_heads): row-specific geometry strength (Gap 3)
            self.geom_scale_net = nn.Linear(obs_dim, n_heads)
        else:
            self.geom_log_scale = nn.Parameter(torch.zeros(n_heads))

    def forward(
        self,
        x_anchor: torch.Tensor,       # [B, input_dim]
        x_neighbors: torch.Tensor,    # [B, K, input_dim]
        edge_weights: torch.Tensor,   # [B, K]
        g_local: Optional[torch.Tensor] = None,  # [B, obs_dim] for adaptive tau
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, _ = x_neighbors.shape
        h_anchor = self.anchor_proj(x_anchor).view(B, self.n_heads, self.head_dim)
        if self.use_view_embed:
            h_anchor = h_anchor + self.view_embed.unsqueeze(0).unsqueeze(0)

        h_neighbors = self.nei_proj(x_neighbors.reshape(B * K, -1)).view(B, K, self.n_heads, self.head_dim)

        q = _apply_headwise(h_anchor, self.q_layers)
        k = _apply_headwise_neighbors(h_neighbors, self.k_layers).permute(0, 2, 1, 3)  # [B, H, K, Dh]
        v = _apply_headwise_neighbors(h_neighbors, self.v_layers).permute(0, 2, 1, 3)

        scores = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1)).squeeze(2) / math.sqrt(self.head_dim)
        # [B, H, K]

        if self.use_adaptive_tau and g_local is not None:
            geom_scale = F.softplus(self.geom_scale_net(g_local))  # [B, H]
            geom_scale = geom_scale.unsqueeze(-1)                    # [B, H, 1]
        else:
            geom_scale = F.softplus(self.geom_log_scale).view(1, self.n_heads, 1)

        scores = scores + geom_scale * torch.log(edge_weights.unsqueeze(1) + EPS)
        attn = self.dropout(torch.softmax(scores, dim=-1))  # [B, H, K]
        head_out = torch.matmul(attn.unsqueeze(2), v).squeeze(2)   # [B, H, Dh]
        head_out = _apply_headwise(head_out, self.out_layers) + h_anchor
        return head_out, attn


# ---------------------------------------------------------------------------
# Cross-view interaction (unchanged from worktree)
# ---------------------------------------------------------------------------

class HeadwiseCrossViewInteraction(nn.Module):
    def __init__(self, n_heads: int, head_dim: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_layers = nn.ModuleList([nn.Linear(head_dim, head_dim, bias=False) for _ in range(n_heads)])
        self.k_layers = nn.ModuleList([nn.Linear(head_dim, head_dim, bias=False) for _ in range(n_heads)])
        self.v_layers = nn.ModuleList([nn.Linear(head_dim, head_dim, bias=False) for _ in range(n_heads)])
        self.out_layers = nn.ModuleList([nn.Linear(head_dim, head_dim, bias=False) for _ in range(n_heads)])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        iso_heads: torch.Tensor,           # [B, H, Dh]
        view_head_tokens: torch.Tensor,    # [B, H, V, Dh]
        pi: torch.Tensor,                  # [B, H, V]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = _apply_headwise(iso_heads, self.q_layers)  # [B, H, Dh]
        k = torch.stack(
            [layer(view_head_tokens[:, h, :, :]) for h, layer in enumerate(self.k_layers)],
            dim=1,
        )  # [B, H, V, Dh]
        v = torch.stack(
            [layer(view_head_tokens[:, h, :, :]) for h, layer in enumerate(self.v_layers)],
            dim=1,
        )  # [B, H, V, Dh]

        scores = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1)).squeeze(2) / math.sqrt(self.head_dim)
        # [B, H, V]
        scores = scores + torch.log(pi + EPS)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn.unsqueeze(2), v).squeeze(2)  # [B, H, Dh]
        out = _apply_headwise(out, self.out_layers)
        return out, attn


# ---------------------------------------------------------------------------
# Quality-Aware Router (Gap 1 + Gap 2 + Gap 13)
# ---------------------------------------------------------------------------

class QualityAwareRouter(nn.Module):
    """
    Extended router that accepts per-view quality and Jaccard signals.

    Input: [g_global; quality_flat; J_flat; sigma2_flat]  (only included if flags are True)
    Output: pi [B, H, V], beta [B, H, 1], gate_h [B, H]

    Prior injections (differentiable, not hard constraints):
      pi_logit_v += log(q_v + ε)     → quality prior (Gap 1)
      beta_logit  += w_beta * mean_J → Jaccard prior (Gap 2)
    """

    def __init__(
        self,
        obs_dim: int,
        n_views: int,
        n_heads: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        use_quality_prior: bool = False,   # Gap 1
        use_jaccard_prior: bool = False,   # Gap 2
        use_sigma2: bool = False,          # Gap 13
        n_view_pairs: int = 0,             # V*(V-1)//2, required if use_jaccard_prior
    ):
        super().__init__()
        self.n_views = n_views
        self.n_heads = n_heads
        self.use_quality_prior = use_quality_prior
        self.use_jaccard_prior = use_jaccard_prior
        self.use_sigma2 = use_sigma2

        # Build router input dim
        router_in_dim = obs_dim
        if use_quality_prior:
            router_in_dim += n_views * 3  # quality_norm [V, 3] flattened
        if use_jaccard_prior:
            router_in_dim += n_view_pairs  # J_flat [n_pairs]
        if use_sigma2:
            router_in_dim += n_views       # sigma2_v [V]

        self.backbone = nn.Sequential(
            nn.Linear(router_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.view_head = nn.Linear(hidden_dim, n_heads * n_views)
        self.mode_head = nn.Linear(hidden_dim, n_heads)
        self.gate_head = nn.Linear(hidden_dim, n_heads)   # for HeadGatedRegressor

        if use_jaccard_prior:
            self.w_beta = nn.Parameter(torch.zeros(1))  # starts at 0 (no prior)

    def init_head_view_biases(self, strength: float = 2.0) -> None:
        """
        Initialize each head to prefer a different view at t=0.
        Head h starts with logit bias of `strength` toward view h % n_views.
        Breaks router symmetry without constraining eventual learning.
        """
        with torch.no_grad():
            bias = torch.zeros(self.n_heads * self.n_views)
            for h in range(self.n_heads):
                preferred = h % self.n_views
                bias[h * self.n_views + preferred] = strength
            if self.view_head.bias is not None:
                self.view_head.bias.data.copy_(bias)

    def forward(
        self,
        g: torch.Tensor,                                  # [B, obs_dim]
        quality_flat: Optional[torch.Tensor] = None,      # [B, V*3] if use_quality_prior
        J_flat: Optional[torch.Tensor] = None,            # [B, n_pairs] if use_jaccard_prior
        mean_J: Optional[torch.Tensor] = None,            # [B] if use_jaccard_prior
        sigma2_v: Optional[torch.Tensor] = None,          # [B, V] if use_sigma2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parts = [g]
        if self.use_quality_prior and quality_flat is not None:
            parts.append(quality_flat)
        if self.use_jaccard_prior and J_flat is not None:
            parts.append(J_flat)
        if self.use_sigma2 and sigma2_v is not None:
            parts.append(sigma2_v)

        router_in = torch.cat(parts, dim=-1)
        hidden = self.backbone(router_in)

        pi_logit = self.view_head(hidden).view(g.shape[0], self.n_heads, self.n_views)
        beta_logit = self.mode_head(hidden).view(g.shape[0], self.n_heads, 1)
        gate_logit = self.gate_head(hidden)  # [B, H]

        # Quality prior: inject log(q_v + ε) into pi logits (Gap 1)
        if self.use_quality_prior and quality_flat is not None:
            # quality_flat is [B, V*3] — extract the quality score mean per view
            # We use the first column (kappa_norm) as proxy; could be improved to full sigmoid
            # Actually quality_score is passed as a separate V-dim signal in forward
            pass  # handled below via quality_score_v

        # Jaccard prior: inject w_beta * mean_J into beta logits (Gap 2)
        if self.use_jaccard_prior and mean_J is not None:
            beta_logit = beta_logit + self.w_beta * mean_J.unsqueeze(-1).unsqueeze(-1)

        pi = torch.softmax(pi_logit, dim=-1)           # [B, H, V]
        beta = torch.sigmoid(beta_logit)                # [B, H, 1]
        gate_h = torch.softmax(gate_logit, dim=-1)      # [B, H]

        return pi, beta, gate_h


# ---------------------------------------------------------------------------
# Full backbone V5
# ---------------------------------------------------------------------------

class HeadRoutingBackboneV5(nn.Module):
    """
    HeadRoutingBackbone with:
      - Per-view quality routing priors (Gap 1)
      - Jaccard-anchored beta prior (Gap 2)
      - Row-adaptive geom_scale (Gap 3, optional)
      - View membership token (Gap 7, optional)
      - sigma2_v in router input (Gap 13, optional)
      - Returns head_repr [B, H, Dh] WITHOUT FLATTENING (Gap 6 prerequisite)
    """

    def __init__(
        self,
        view_input_dims: Dict[str, int],
        obs_dim: int,
        n_heads: int = 4,
        head_dim: int = 16,
        router_hidden_dim: int = 64,
        dropout: float = 0.1,
        # Feature flags
        use_quality_prior: bool = False,
        use_jaccard_prior: bool = False,
        use_sigma2: bool = False,
        use_adaptive_tau: bool = False,
        use_view_embed: bool = False,
    ):
        super().__init__()
        self.view_names = tuple(view_input_dims.keys())
        self.n_views = len(self.view_names)
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.output_dim = n_heads * head_dim  # for flat compat
        self.use_quality_prior = use_quality_prior
        self.use_jaccard_prior = use_jaccard_prior

        n_view_pairs = self.n_views * (self.n_views - 1) // 2

        self.encoders = nn.ModuleDict({
            view_name: HeadwiseEdgeBiasedViewEncoderV5(
                input_dim=input_dim,
                n_heads=n_heads,
                head_dim=head_dim,
                obs_dim=obs_dim if use_adaptive_tau else 0,
                use_view_embed=use_view_embed,
                dropout=dropout,
            )
            for view_name, input_dim in view_input_dims.items()
        })

        self.router = QualityAwareRouter(
            obs_dim=obs_dim,
            n_views=self.n_views,
            n_heads=n_heads,
            hidden_dim=router_hidden_dim,
            dropout=dropout,
            use_quality_prior=use_quality_prior,
            use_jaccard_prior=use_jaccard_prior,
            use_sigma2=use_sigma2,
            n_view_pairs=n_view_pairs,
        )

        self.interaction = HeadwiseCrossViewInteraction(
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x_anchor_by_view: Dict[str, torch.Tensor],     # {view: [B, d_v]}
        x_nei_by_view: Dict[str, torch.Tensor],         # {view: [B, K, d_v]}
        ew_by_view: Dict[str, torch.Tensor],            # {view: [B, K]}
        g: torch.Tensor,                                # [B, obs_dim]
        quality_score: Optional[torch.Tensor] = None,  # [B, V] in [0,1]
        quality_norm: Optional[torch.Tensor] = None,   # [B, V, 3] normalized
        J_flat: Optional[torch.Tensor] = None,         # [B, n_pairs]
        mean_J: Optional[torch.Tensor] = None,         # [B]
        sigma2_v: Optional[torch.Tensor] = None,       # [B, V]
    ) -> Tuple[torch.Tensor, dict]:

        # Prepare router extra inputs
        quality_flat = None
        if self.use_quality_prior and quality_norm is not None:
            B = g.shape[0]
            quality_flat = quality_norm.view(B, -1)  # [B, V*3]

        # Encode each view
        view_tokens = []
        neighbor_attn = {}
        for view_name in self.view_names:
            x_anchor = x_anchor_by_view[view_name]
            x_neighbors = x_nei_by_view[view_name]
            ew = ew_by_view[view_name]
            head_tokens, attn = self.encoders[view_name](x_anchor, x_neighbors, ew, g_local=g)
            view_tokens.append(head_tokens)
            neighbor_attn[view_name] = attn

        view_head_tokens = torch.stack(view_tokens, dim=1)     # [B, V, H, Dh]
        tokens_for_routing = view_head_tokens.permute(0, 2, 1, 3)  # [B, H, V, Dh]

        # Route
        pi, beta, gate_h = self.router(
            g,
            quality_flat=quality_flat,
            J_flat=J_flat,
            mean_J=mean_J,
            sigma2_v=sigma2_v,
        )

        # Quality prior injected into pi (log-scale, additive) — Gap 1
        if self.use_quality_prior and quality_score is not None:
            # quality_score [B, V] ∈ (0,1); inject as log-prior into pi
            # pi = softmax(logit + log(q_v + ε))
            q_log_prior = torch.log(quality_score + EPS)  # [B, V]
            pi_logit = torch.log(pi + EPS) + q_log_prior.unsqueeze(1)  # broadcast over H
            pi = torch.softmax(pi_logit, dim=-1)

        # Isolation: pi-weighted sum over views
        iso_heads = (tokens_for_routing * pi.unsqueeze(-1)).sum(dim=2)  # [B, H, Dh]

        # Cross-view interaction
        int_heads, cross_view_attn = self.interaction(iso_heads, tokens_for_routing, pi)

        # Blend
        head_repr = (1.0 - beta) * iso_heads + beta * int_heads  # [B, H, Dh]

        aux = {
            "head_repr": head_repr,         # [B, H, Dh] — NOT flattened
            "pi": pi,                       # [B, H, V]
            "beta": beta,                   # [B, H, 1]
            "gate_h": gate_h,               # [B, H] — from router, for task head
            "iso_heads": iso_heads,
            "int_heads": int_heads,
            "neighbor_attn": neighbor_attn,
            "cross_view_attn": cross_view_attn,
            "view_head_tokens": view_head_tokens,
        }
        return head_repr, aux
