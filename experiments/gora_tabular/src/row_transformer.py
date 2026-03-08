"""
row_transformer.py — GoRA-Tabular model variants (sparse neighbourhood version).

API change from dense version:
  All models now accept precomputed neighbourhood tensors:
    neigh_idx: [N, K] — neighbour indices
    edge_wts:  [N, K, M] — Gaussian edge weights per view

The training loop fetches x_nei = X_embed[neigh_idx[batch]]  per mini-batch.
Routing then shapes which view's edges each head trusts.

v3 addition: MQGoraTransformer (Manifold-Query GoRA)
  G7: ViewSpecificEmbedder + avg-pool ManifoldReader + RichMoERouter(ctx only)
  G8: + LabelContextEncoder (router + value augmentation)
  G9: + ManifoldReader with z_anc cross-attn query + z_anc in router
  G10: + AlphaGate prediction fusion
"""
import torch
import torch.nn as nn
from typing import List, Optional

from .moe_router import MoERouter, UniformRouter, RandomRouter, RichMoERouter
from .geometry_attention import SparseGeomLayer, NoGraphLayer
from .meta_learner import (
    ViewSpecificEmbedder, LabelContextEncoder,
    ManifoldReader, AlphaGate, alpha_gate_loss
)


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


# ─── v3: MQGoraTransformer ────────────────────────────────────────────────────

class MQGoraTransformer(nn.Module):
    """
    Manifold-Query GoRA (v3).

    Implements the G7→G8→G9→G10 ablation ladder via constructor flags:
      use_label_ctx   (G8+): LabelContextEncoder active
      use_teacher_query (G9+): z_anc used as ManifoldReader cross-attn query
      use_alpha_gate  (G10): AlphaGate prediction fusion active

    ViewSpecificEmbedder is always active (required for ManifoldReader).

    Forward signature extends GoraTransformer:
      x_anc      [B, d_x]
      g_anc      [B, obs_dim]
      x_nei      [B, K, d_x]      (union pool, all views mixed)
      ew_anc     [B, K, M]
      view_mask  [B, K, M]        (required; marks valid slots per view)
      z_anc      [B, d_z]         (required if use_teacher_query or use_alpha_gate)
      lbl_nei    [B, K, M] or None  (None at inference)
      agree_score [B] or None      (for alpha gate auxiliary loss)

    Returns: (pred_final, pi, tau, aux_losses_dict)
      pred_final:      [B, out_dim]
      pi:              [B, H, M]
      tau:             [H]
      aux_losses_dict: {"alpha": scalar} or {}
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

        # STAGE A: view-specific anchor embedding + per-view neighbour embeddings
        self.anchor_embed = ColumnEmbedder(n_features, d_model)
        self.view_embedder = ViewSpecificEmbedder(n_views, n_features, d_model)

        # STAGE B: label context (G8+)
        self.label_enc: Optional[LabelContextEncoder] = None
        if use_label_ctx:
            self.label_enc = LabelContextEncoder(n_views, d_model, d_z, n_out=max(1, n_classes))

        # STAGE C: manifold reader
        self.manifold_reader = ManifoldReader(
            n_views=n_views,
            d_z=d_z,
            d_model=d_model,
            use_query_z=use_teacher_query,
            dropout=dropout,
        )

        # STAGE D: rich router
        self.router = RichMoERouter(
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

        # STAGE E: GoRA transformer layers (unchanged from v2)
        self.layers = nn.ModuleList([
            SparseGeomLayer(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Linear(d_model // 2, out_dim),
        )

        # STAGE F: alpha gate (G10 only)
        self.alpha_gate: Optional[AlphaGate] = None
        if use_alpha_gate:
            self.alpha_gate = AlphaGate(d_z, d_model, out_dim)

    def forward(
        self,
        x_anc: torch.Tensor,                         # [B, d_x]
        g_anc: torch.Tensor,                         # [B, obs_dim]
        x_nei: torch.Tensor,                         # [B, K, d_x]
        ew_anc: torch.Tensor,                        # [B, K, M]
        view_mask: Optional[torch.Tensor] = None,    # [B, K, M]
        z_anc: Optional[torch.Tensor] = None,        # [B, d_z]
        lbl_nei: Optional[torch.Tensor] = None,      # [B, K, M] or None
        agree_score: Optional[torch.Tensor] = None,  # [B] or None
    ):
        B, K, _ = x_nei.shape
        M = self.n_views

        if view_mask is None:
            view_mask = (ew_anc > 0).float()

        # STAGE A — Embeddings
        h_anc = self.anchor_embed(x_anc)                          # [B, d_model]
        # Split x_nei by view for ViewSpecificEmbedder:
        # x_nei [B, K, d_x] is the union pool (view-agnostic slots).
        # Each view's embedder sees the same x_nei but projects differently.
        x_nei_per_view = [x_nei for _ in range(M)]               # M x [B, K, d_x]
        h_nei_list = self.view_embedder(x_nei_per_view)           # M x [B, K, d_model]

        # h_nei for SparseGeomLayer (use view-0 embedding as shared repr)
        h_nei_shared = h_nei_list[0]                              # [B, K, d_model]

        # STAGE B — Label context
        label_ctx_vec, lbl_delta, lbl_weight = None, None, None
        if self.label_enc is not None:
            label_ctx_vec, lbl_delta, lbl_weight = self.label_enc(
                lbl_nei, ew_anc, view_mask)

        # STAGE C — Manifold-queried neighbourhood reading
        ctx_vec = self.manifold_reader(
            h_nei_list=h_nei_list,
            ew_anc=ew_anc,
            view_mask=view_mask,
            z_anc=z_anc if self.use_teacher_query else None,
            lbl_delta=lbl_delta,
            lbl_weight=lbl_weight,
        )                                                          # [B, d_model]

        # STAGE D — Rich routing
        pi, tau = self.router(
            g=g_anc,
            z_anc=z_anc if self.use_teacher_query else None,
            label_ctx_vec=label_ctx_vec,
            ctx_vec=ctx_vec,
        )                                                          # [B, H, M], [H]

        # STAGE E — GoRA transformer layers
        for layer in self.layers:
            h_anc = layer(h_anc, h_nei_shared, pi, tau, ew_anc)

        pred_base = self.pred_head(h_anc)                         # [B, out_dim]

        # STAGE F — Alpha gate
        aux_losses = {}
        if self.alpha_gate is not None and z_anc is not None:
            pred_final, alpha, pred_local = self.alpha_gate(z_anc, ctx_vec, pred_base)
            if agree_score is not None:
                aux_losses["alpha"] = alpha_gate_loss(alpha, agree_score, self.lam_alpha)
        else:
            pred_final = pred_base

        return pred_final, pi, tau, aux_losses
