"""
row_transformer.py — GoRA-Tabular model variants (sparse neighbourhood version).

API change from dense version:
  All models now accept precomputed neighbourhood tensors:
    neigh_idx: [N, K] — neighbour indices
    edge_wts:  [N, K, M] — Gaussian edge weights per view

The training loop fetches x_nei = X_embed[neigh_idx[batch]]  per mini-batch.
Routing then shapes which view's edges each head trusts.
"""
import torch
import torch.nn as nn
from typing import List

from .moe_router import MoERouter, UniformRouter, RandomRouter
from .geometry_attention import SparseGeomLayer, NoGraphLayer


class ColumnEmbedder(nn.Module):
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x):
        return self.bn(self.proj(x))


class GoraTransformer(nn.Module):
    """
    G2: Full GoRA-Tabular.
    Sparse neighbourhood attention with geometry-routing bias per head.
    """
    def __init__(self, n_features, obs_dim, n_views, out_dim,
                 d_model=64, n_heads=4, n_layers=2, ff_dim=128,
                 dropout=0.1, router_cls=None, router_kwargs=None):
        super().__init__()
        self.d_model = d_model
        self.embed = ColumnEmbedder(n_features, d_model)
        RouterCls = router_cls or MoERouter
        rk = router_kwargs or {}
        self.router = RouterCls(obs_dim=obs_dim, n_heads=n_heads, n_views=n_views, **rk)
        self.layers = nn.ModuleList([
            SparseGeomLayer(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Linear(d_model // 2, out_dim),
        )

    def forward(self, x_anc, g_anc, x_nei, ew_anc):
        """
        x_anc:  [B, n_features]   anchor row raw features
        g_anc:  [B, obs_dim]      observer vector for anchor rows
        x_nei:  [B, K, n_features] neighbour raw features (fetched by train loop)
        ew_anc: [B, K, M]         edge weights per view
        """
        B, K, _ = x_nei.shape
        h_anc = self.embed(x_anc)                              # [B, d_model]
        h_nei = self.embed(x_nei.reshape(B * K, -1)).view(B, K, -1)  # [B, K, d_model]

        pi, tau = self.router(g_anc)                           # [B, H, M], [H]

        for layer in self.layers:
            h_anc = layer(h_anc, h_nei, pi, tau, ew_anc)

        out = self.pred_head(h_anc)                            # [B, out_dim]
        return out, pi, tau


class StandardTransformer(nn.Module):
    """G0: no graph bias — neighbourhood context only via standard attention."""
    def __init__(self, n_features, obs_dim, n_views, out_dim,
                 d_model=64, n_heads=4, n_layers=2, ff_dim=128, dropout=0.1,
                 **kwargs):
        super().__init__()
        self.embed = ColumnEmbedder(n_features, d_model)
        self.layers = nn.ModuleList([
            NoGraphLayer(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)
        ])
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Linear(d_model // 2, out_dim),
        )

    def forward(self, x_anc, g_anc, x_nei, ew_anc):
        B, K, _ = x_nei.shape
        h_anc = self.embed(x_anc)
        h_nei = self.embed(x_nei.reshape(B * K, -1)).view(B, K, -1)
        for layer in self.layers:
            h_anc = layer(h_anc, h_nei)
        return self.pred_head(h_anc), None, None


class SingleViewTransformer(nn.Module):
    """G1: fixed view routing — pi peaked at one fixed view, tau learned."""
    def __init__(self, n_features, obs_dim, n_views, out_dim,
                 d_model=64, n_heads=4, n_layers=2, ff_dim=128, dropout=0.1,
                 fixed_view_idx=1, **kwargs):
        super().__init__()
        self.fixed_view = fixed_view_idx
        self.n_heads = n_heads
        self.n_views = n_views
        self.embed = ColumnEmbedder(n_features, d_model)
        self.layers = nn.ModuleList([
            SparseGeomLayer(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)
        ])
        self.log_tau = nn.Parameter(torch.zeros(n_heads))
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Linear(d_model // 2, out_dim),
        )

    def forward(self, x_anc, g_anc, x_nei, ew_anc):
        B, K, _ = x_nei.shape
        pi = torch.zeros(B, self.n_heads, self.n_views, device=x_anc.device)
        pi[:, :, self.fixed_view] = 1.0
        tau = torch.exp(self.log_tau)
        h_anc = self.embed(x_anc)
        h_nei = self.embed(x_nei.reshape(B * K, -1)).view(B, K, -1)
        for layer in self.layers:
            h_anc = layer(h_anc, h_nei, pi, tau, ew_anc)
        return self.pred_head(h_anc), pi, tau
