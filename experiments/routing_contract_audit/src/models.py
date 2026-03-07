"""
models.py — A_posthoc_combiner and B_intended_router per routing_contract.md.

Model A: post-hoc combination. No explicit pi/beta. observers optionally concatenated.
Model B: explicit pi + beta + iso_rep + inter_rep + final_rep blend before prediction.

Fairness constraints (contract §Fairness constraints):
- A and B share the same ViewEncoder architecture.
- A and B share the same observer vector definition.
- The substantive difference is routing semantics only.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from .router import ObserverRouter


# ─── Shared view encoder (GraphSAGE) ─────────────────────────────────────────

class ViewEncoder(nn.Module):
    """
    [Contract §Required code interfaces / ViewEncoder]
    GraphSAGE 2-layer encoder shared by both models.
    Returns: rep [B, rep_dim]
    """
    def __init__(self, in_dim: int, rep_dim: int = 64):
        super().__init__()
        self.s1 = SAGEConv(in_dim, rep_dim * 2)
        self.bn1 = nn.BatchNorm1d(rep_dim * 2)
        self.s2 = SAGEConv(rep_dim * 2, rep_dim)
        self.bn2 = nn.BatchNorm1d(rep_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.s1(x, edge_index)))
        x = F.dropout(x, 0.1, self.training)
        return F.relu(self.bn2(self.s2(x, edge_index)))  # [N, rep_dim]


# ─── Model A: A_posthoc_combiner ─────────────────────────────────────────────

class A_posthoc_combiner(nn.Module):
    """
    [Contract §Model A]
    Post-hoc combination. No explicit pi/beta routing semantics.
    Concatenates all view reps (+ optionally g) → MLP → prediction.
    """
    def __init__(self, n_views: int, rep_dim: int, obs_dim: int, out_dim: int):
        super().__init__()
        in_dim = n_views * rep_dim + obs_dim   # g always concatenated in A
        self.net = nn.Sequential(
            nn.Linear(in_dim, rep_dim * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(rep_dim * 2, rep_dim), nn.GELU(),
            nn.Linear(rep_dim, out_dim),
        )

    def forward(self, reps: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        reps: [B, V, D]
        g:    [B, G]  (observer features, concatenated but NOT routing)
        returns: prediction [B, out_dim]
        """
        B, V, D = reps.shape
        flat = reps.reshape(B, V * D)          # flatten all views
        inp = torch.cat([flat, g], dim=-1)     # observers appended, not routing
        return self.net(inp)


# ─── Model B: B_intended_router ──────────────────────────────────────────────

class B_intended_router(nn.Module):
    """
    [Contract §Model B]
    Explicit observer-driven routing:
      pi:       [B, V]   view trust weights
      beta:     [B, 1]   isolation (0) vs interaction (1) gate
      iso_rep:  [B, D]   weighted sum of view reps
      inter_rep:[B, D]   jointly transformed fused rep
      final_rep:[B, D]   (1-beta)*iso_rep + beta*inter_rep
    """
    def __init__(self, obs_dim: int, n_views: int, rep_dim: int, out_dim: int):
        super().__init__()
        self.router = ObserverRouter(obs_dim, n_views, hidden_dim=32)
        self.interaction_mlp = nn.Sequential(
            nn.Linear(rep_dim, rep_dim), nn.GELU(),
            nn.Linear(rep_dim, rep_dim),
        )
        self.pred_head = nn.Linear(rep_dim, out_dim)

    def forward(self, reps: torch.Tensor, g: torch.Tensor):
        """
        reps: [B, V, D]  — same per-view encoders as Model A
        g:    [B, G]     — observer vector (routing prior ONLY, not prediction feature)
        returns: (prediction, pi, beta, iso_rep, inter_rep, final_rep)
        """
        pi, beta = self.router(g)                         # [B,V], [B,1]
        weighted = reps * pi.unsqueeze(-1)                # [B, V, D]
        iso_rep = weighted.sum(dim=1)                     # [B, D]
        inter_rep = self.interaction_mlp(iso_rep)         # [B, D]
        # beta→0 = isolation, beta→1 = interaction (contract §Required final blend)
        final_rep = (1.0 - beta) * iso_rep + beta * inter_rep
        return self.pred_head(final_rep), pi, beta, iso_rep, inter_rep, final_rep
