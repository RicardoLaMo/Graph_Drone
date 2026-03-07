"""
geometry_attention.py — GoRA-style geometry-inside-attention for tabular rows.

Core equation per head h for row i attending to row j:
  logit_{ij}^h = <q_i^h, k_j^h> / sqrt(d_k)
               + log(τ_h · Ã_{ij}^{i,h} + ε)

where:
  Ã_{ij}^{i,h} = Σ_m π_{i,h,m} · A^(m)_{ij}     (geo-weighted effective adjacency)
  τ_h           = per-head learned temperature (sharpens/broadens graph bias)
  ε             = 1e-6  (log-barrier stability)

Rows NOT in the adjacency of row i get:
  log(τ_h · 0 + ε) = log(ε) ≈ -13.8  → near-zero attention

This is the mechanism that makes routing structural:
  A head routed to GEO (π_{h,GEO}≈1) can only attend to geographically near rows.
  A head routed to FULL can attend to feature-similar rows.

Mode (isolation vs interaction) is structural here:
  isolation:  π is peaked → head can only see one view's neighbourhood
  interaction: π is flat → head sees a convex mix of all view neighbourhoods

Note on shapes:
  N = total rows in dataset
  B_idx = selected row indices for a batch (e.g. train_idx)
  We compute attention over ALL N rows (for full neighbourhood context),
  but only compute loss on B_idx rows.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple


EPS = 1e-6


class GeometryAwareAttention(nn.Module):
    """
    Multi-head self-attention where each head's logits are biased by
    the geometry-routed effective adjacency log(τ_h · Ã_{ij}^{i,h} + ε).

    Inputs:
      x:          [N, d_model]        row feature embeddings
      pi:         [N, H, M]           per-row per-head routing weights
      tau:        [H]                 per-head temperature
      adjs:       List of M tensors, each [N, N]  row-normalised adjacency per view
                  (dense, precomputed). For large N use edge_index variant.
    Output:
      out:        [N, d_model]
      attn_map:   {head_idx: [N, N]} (optional, for head specialisation analysis)
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

    def _effective_adj(
        self,
        pi: torch.Tensor,   # [N, H, M]
        adjs: List[torch.Tensor],  # M × [N, N]
        head: int,
    ) -> torch.Tensor:
        """
        Computes Ã^{i,h} = Σ_m π_{i,h,m} · A^(m)
        Shape: [N, N]
        """
        pi_h = pi[:, head, :]  # [N, M]
        # Stack adjacencies: [M, N, N]
        A_stack = torch.stack(adjs, dim=0).to(pi.device)           # [M, N, N]
        # Ã_{ij}^{i,h} = Σ_m π_{i,h,m} · A^(m)_{ij}
        # Einsum: pi_h [N, M], A_stack [M, N, N] → [N, N]
        # For each row i: Ã[i, j] = Σ_m pi_h[i, m] * A_stack[m, i, j]
        A_eff = torch.einsum('im,mij->ij', pi_h, A_stack)  # [N, N]
        return A_eff

    def forward(
        self,
        x: torch.Tensor,         # [N, d_model]
        pi: torch.Tensor,         # [N, H, M]
        tau: torch.Tensor,        # [H]
        adjs: List[torch.Tensor], # M × [N, N]
        return_attn: bool = False,
    ):
        N, D = x.shape
        Q = self.W_q(x).view(N, self.n_heads, self.d_k)  # [N, H, dk]
        K = self.W_k(x).view(N, self.n_heads, self.d_k)
        V = self.W_v(x).view(N, self.n_heads, self.d_k)

        head_outputs = []
        attn_maps = {}

        for h in range(self.n_heads):
            # Content scores: <q^h, k^h> / sqrt(d_k)  [N, N]
            q_h = Q[:, h, :]   # [N, dk]
            k_h = K[:, h, :]
            v_h = V[:, h, :]

            scores = torch.matmul(q_h, k_h.t()) / math.sqrt(self.d_k)  # [N, N]

            # Geometry bias: log(τ_h · Ã_{ij}^{i,h} + ε)
            # Ã: [N, N]
            A_eff = self._effective_adj(pi, adjs, h)                    # [N, N]
            tau_h = tau[h]
            graph_bias = torch.log(tau_h * A_eff + EPS)                 # [N, N]

            # Combine: content + geometry
            logits = scores + graph_bias                                 # [N, N]

            # Structural mode:
            #   Rows where A_eff is peaked → attention also peaked (isolation)
            #   Rows where A_eff is flat → attention also broader (interaction)
            # This emerges naturally from the log-barrier term.

            attn = self.dropout(torch.softmax(logits, dim=-1))          # [N, N]
            out_h = torch.matmul(attn, v_h)                             # [N, dk]
            head_outputs.append(out_h)
            if return_attn:
                attn_maps[h] = attn.detach()

        # Concat and project
        out = torch.cat(head_outputs, dim=-1)    # [N, d_model]
        out = self.W_o(out)
        return (out, attn_maps) if return_attn else (out, None)


class GeometryAwareTransformerLayer(nn.Module):
    """Single transformer layer with geometry-aware attention."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        ff_dim = ff_dim or d_model * 4
        self.attn = GeometryAwareAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pi, tau, adjs, return_attn=False):
        attn_out, attn_maps = self.attn(self.norm1(x), pi, tau, adjs, return_attn)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, attn_maps


class StandardTransformerLayer(nn.Module):
    """Baseline G0: standard multi-head attention, no graph bias."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        ff_dim = ff_dim or d_model * 4
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        # x: [N, D] → treat as sequence of length N, batch=1
        x_seq = x.unsqueeze(0)    # [1, N, D]
        attn_out, _ = self.attn(x_seq, x_seq, x_seq, need_weights=False)
        x = x + self.drop(attn_out.squeeze(0))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, None
