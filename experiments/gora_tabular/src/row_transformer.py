"""
row_transformer.py — Full TabularRowTransformer (GoRA-Tabular).

Rows are treated as sequence tokens. Each row's representation is formed
through geometry-aware multi-head attention where each head's attention
softmax is structurally shaped by the MoE-routed adjacency.

Variants implemented here:
  GoraTransformer   — full G2: pi_{i,h,m} from MoERouter
  UniformTransformer — G3: pi = 1/M (uniform, no geometry)
  RandomTransformer  — G4: pi from shuffled g (geometry destroyed)
  StandardTransformer — G0: no graph bias at all
  SingleViewTransformer — G1: graph bias from one fixed view (GEO)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

from .moe_router import MoERouter, UniformRouter, RandomRouter
from .geometry_attention import (
    GeometryAwareTransformerLayer, StandardTransformerLayer
)


class ColumnEmbedder(nn.Module):
    """Projects each scalar feature column to d_model/n_cols dimension, then concatenates."""
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x):
        return self.bn(self.proj(x))   # [N, d_model]


# ─── GoRA-Tabular (G2: full routing) ─────────────────────────────────────────

class GoraTransformer(nn.Module):
    """
    Full GoRA-Tabular model.
    Routing geometry g_i → π_{i,h,m} → logit bias inside attention.
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
        router_cls=None,        # MoERouter by default
        router_kwargs: dict = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_views = n_views

        self.embed = ColumnEmbedder(n_features, d_model)

        RouterCls = router_cls or MoERouter
        rk = router_kwargs or {}
        self.router = RouterCls(obs_dim=obs_dim, n_heads=n_heads, n_views=n_views, **rk)

        self.layers = nn.ModuleList([
            GeometryAwareTransformerLayer(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Linear(d_model // 2, out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,       # [N, n_features]
        g: torch.Tensor,       # [N, obs_dim]
        adjs: List[torch.Tensor],  # M × [N, N]
        return_attn: bool = False,
    ):
        h = self.embed(x)                        # [N, d_model]
        pi, tau = self.router(g)                 # [N, H, M], [H]

        all_attn = {}
        for li, layer in enumerate(self.layers):
            h, attn_maps = layer(h, pi, tau, adjs, return_attn=(return_attn and li == 0))
            if return_attn and attn_maps:
                all_attn = attn_maps             # only first layer for analysis

        out = self.pred_head(h)                  # [N, out_dim]
        return out, pi, tau, all_attn


# ─── G0: Standard (no graph bias) ────────────────────────────────────────────

class StandardTransformer(nn.Module):
    def __init__(self, n_features, out_dim, d_model=64, n_heads=4, n_layers=2,
                 ff_dim=128, dropout=0.1, **kwargs):
        super().__init__()
        self.embed = ColumnEmbedder(n_features, d_model)
        self.layers = nn.ModuleList([
            StandardTransformerLayer(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Linear(d_model // 2, out_dim),
        )

    def forward(self, x, g=None, adjs=None, return_attn=False):
        h = self.embed(x)
        for layer in self.layers:
            h, _ = layer(h)
        out = self.pred_head(h)
        return out, None, None, {}


# ─── G1: Single-view fixed bias ───────────────────────────────────────────────

class SingleViewTransformer(nn.Module):
    """G1: Graph bias from one fixed view (GEO for CA, PCA for MNIST). No routing."""
    def __init__(self, n_features, obs_dim, n_views, out_dim, d_model=64, n_heads=4,
                 n_layers=2, ff_dim=128, dropout=0.1, fixed_view_idx=1, **kwargs):
        super().__init__()
        self.fixed_view = fixed_view_idx
        self.embed = ColumnEmbedder(n_features, d_model)
        from .geometry_attention import GeometryAwareTransformerLayer
        self.layers = nn.ModuleList([
            GeometryAwareTransformerLayer(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.log_tau = nn.Parameter(torch.zeros(n_heads))
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Linear(d_model // 2, out_dim),
        )
        self.n_heads = n_heads
        self.n_views = n_views

    def forward(self, x, g=None, adjs=None, return_attn=False):
        N = x.shape[0]
        # pi peaked at fixed_view for all rows/heads
        pi = torch.zeros(N, self.n_heads, self.n_views, device=x.device)
        pi[:, :, self.fixed_view] = 1.0
        tau = torch.exp(self.log_tau)
        h = self.embed(x)
        attn_out = {}
        for li, layer in enumerate(self.layers):
            h, am = layer(h, pi, tau, adjs, return_attn=(return_attn and li == 0))
            if return_attn and am: attn_out = am
        out = self.pred_head(h)
        return out, pi, tau, attn_out
