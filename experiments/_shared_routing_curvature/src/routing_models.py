"""
routing_models.py — Core routing architecture (shared across datasets).

ObserverRouter:  g_i → (π_i view weights, β_i isolation/interaction gate)
ViewCombiner:    (reps, π, β) → blended output
MultiViewRoutingModel: end-to-end wrapper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObserverRouter(nn.Module):
    """g_i → per-row view weights + isolation/interaction gate."""
    def __init__(self, obs_dim: int, n_views: int, hidden_dim: int = 32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.view_head = nn.Linear(hidden_dim, n_views)
        self.mode_head = nn.Linear(hidden_dim, 1)

    def forward(self, g):
        h = self.backbone(g)
        pi = torch.softmax(self.view_head(h), dim=-1)   # [B, n_views]
        beta = torch.sigmoid(self.mode_head(h))          # [B, 1]
        return pi, beta


class ViewCombiner(nn.Module):
    """Combines per-view representations using isolation/interaction blend."""
    def __init__(self, rep_dim: int, n_views: int, out_dim: int):
        super().__init__()
        self.interaction_mlp = nn.Sequential(
            nn.Linear(rep_dim, rep_dim), nn.GELU(),
            nn.Linear(rep_dim, rep_dim),
        )
        self.pred_head = nn.Linear(rep_dim, out_dim)

    def forward(self, reps, pi, beta):
        # reps: [B, n_views, rep_dim], pi: [B, n_views], beta: [B, 1]
        weighted = reps * pi.unsqueeze(-1)
        iso_rep = weighted.sum(dim=1)                    # isolation
        inter_rep = self.interaction_mlp(iso_rep)        # interaction
        final_rep = (1.0 - beta) * iso_rep + beta * inter_rep
        return self.pred_head(final_rep)


class NoRouterCombiner(nn.Module):
    """Learned view combiner WITHOUT observer routing (ablation control)."""
    def __init__(self, n_views: int, rep_dim: int, out_dim: int):
        super().__init__()
        self.logw = nn.Parameter(torch.zeros(n_views))
        self.head = nn.Linear(rep_dim, out_dim)

    def forward(self, reps):
        w = torch.softmax(self.logw, dim=0)
        combined = (reps * w.view(1, -1, 1)).sum(dim=1)
        return self.head(combined)
