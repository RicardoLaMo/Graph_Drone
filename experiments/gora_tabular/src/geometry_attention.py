"""
geometry_attention.py — GoRA-Tabular sparse neighbourhood attention.

Redesigned for scalability: instead of full [N,N] dense attention,
each row i attends only to its pre-fetched k-hop neighbourhood.

For anchor row i with neighbourhood N(i) = [j_1, ..., j_k]:

  logit_{i,t}^h = <q_i^h, k_{j_t}^h> / sqrt(d_k)
                + log(τ_h · Ã_{i,j_t}^{i,h} + ε)

where:
  Ã_{i,j_t}^{i,h} = Σ_m π_{i,h,m} · w^(m)_{i,j_t}   (scalar, just one edge)

This reduces memory from O(N²) to O(B × K) per forward pass.
The routing still shapes which view's neighbour edges are trusted per head.

Shapes:
  x_anchors:    [B, d_model]   — anchor row embeddings
  x_neigh:      [B, K, d_model] — neighbour embeddings per anchor
  pi:           [B, H, M]     — routing weights (from MoERouter)
  tau:          [H]            — per-head temperatures
  edge_weights: [B, K, M]     — Gaussian edge weights per view
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List


EPS = 1e-6


class SparseGeomAttention(nn.Module):
    """
    Sparse neighbourhood attention with geometry-routing bias.
    Each anchor row attends to exactly K neighbours.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_anc: torch.Tensor,       # [B, d_model]  anchor embeddings
        x_nei: torch.Tensor,       # [B, K, d_model] neighbour embeddings
        pi: torch.Tensor,           # [B, H, M]
        tau: torch.Tensor,          # [H]
        ew: torch.Tensor,           # [B, K, M]  per-view edge weights (Gaussian)
    ) -> torch.Tensor:
        B, K, _ = x_nei.shape
        H, dk = self.n_heads, self.d_k

        # Project queries from anchors, keys/values from neighbours
        Q = self.W_q(x_anc).view(B, H, dk)                # [B, H, dk]
        K_ = self.W_k(x_nei.reshape(B * K, -1)).view(B, K, H, dk)  # [B, K, H, dk]
        V_ = self.W_v(x_nei.reshape(B * K, -1)).view(B, K, H, dk)  # [B, K, H, dk]

        # Effective edge weight per head:
        # Ã_{i,j,h} = Σ_m π_{i,h,m} · w^(m)_{i,j}
        # pi: [B, H, M], ew: [B, K, M] → [B, H, K]
        A_eff = torch.einsum('bhm,bkm->bhk', pi, ew)       # [B, H, K]  ∈ [0,1]

        # Graph bias: log(τ_h · Ã + ε)
        # tau: [H] → broadcast to [1, H, 1]
        tau_exp = tau.view(1, H, 1)
        graph_bias = torch.log(tau_exp * A_eff + EPS)       # [B, H, K]

        # Content scores: Q [B, H, dk] × K [B, K, H, dk] → [B, H, K]
        # Permute K to [B, H, K, dk]
        K_t = K_.permute(0, 2, 1, 3)                        # [B, H, K, dk]
        Q_exp = Q.unsqueeze(2)                               # [B, H, 1, dk]
        scores = (Q_exp @ K_t.transpose(-2, -1)).squeeze(2) / math.sqrt(dk)  # [B, H, K]

        logits = scores + graph_bias                         # [B, H, K]
        attn = self.dropout(torch.softmax(logits, dim=-1))  # [B, H, K]

        # Aggregate values: [B, H, K] × [B, H, K, dk] → [B, H, dk]
        V_t = V_.permute(0, 2, 1, 3)                        # [B, H, K, dk]
        out = (attn.unsqueeze(2) @ V_t).squeeze(2)          # [B, H, dk]
        out = out.reshape(B, H * dk)                         # [B, d_model]
        return self.W_o(out)


class SparseGeomLayer(nn.Module):
    """Single GoRA transformer layer using sparse neighbourhood attention."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        ff_dim = ff_dim or d_model * 4
        self.attn = SparseGeomAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x_anc: torch.Tensor,    # [B, d_model]
        x_nei: torch.Tensor,    # [B, K, d_model]
        pi: torch.Tensor,       # [B, H, M]
        tau: torch.Tensor,      # [H]
        ew: torch.Tensor,       # [B, K, M]
    ) -> torch.Tensor:
        # Residual on anchors only
        attn_out = self.attn(self.norm1(x_anc), x_nei, pi, tau, ew)
        x_anc = x_anc + self.drop(attn_out)
        x_anc = x_anc + self.drop(self.ff(self.norm2(x_anc)))
        return x_anc


class NoGraphLayer(nn.Module):
    """G0 baseline: standard self-attention, no graph bias. Anchors only."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        ff_dim = ff_dim or d_model * 4
        # Use standard attention (without bias)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.n_heads = n_heads; self.d_k = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_anc, x_nei, pi=None, tau=None, ew=None):
        B, K, _ = x_nei.shape; H, dk = self.n_heads, self.d_k
        Q = self.W_q(self.norm1(x_anc)).view(B, H, dk)
        Kv = self.W_k(x_nei.reshape(B*K, -1)).view(B, K, H, dk)
        Vv = self.W_v(x_nei.reshape(B*K, -1)).view(B, K, H, dk)
        Kt = Kv.permute(0, 2, 1, 3); Vt = Vv.permute(0, 2, 1, 3)
        scores = (Q.unsqueeze(2) @ Kt.transpose(-2,-1)).squeeze(2) / math.sqrt(dk)
        attn = torch.softmax(scores, dim=-1)
        out = (attn.unsqueeze(2) @ Vt).squeeze(2).reshape(B, -1)
        x_anc = x_anc + self.drop(self.W_o(out))
        x_anc = x_anc + self.drop(self.ff(self.norm2(x_anc)))
        return x_anc
