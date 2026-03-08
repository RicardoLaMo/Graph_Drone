"""
task_heads_v5.py — Head-Gated Regressor and Classifier.

Gap 6 (primary): HeadGatedRegressor
  - pred_h = Linear_h(head_repr_h)     [B, H]   — H separate, independent predictions
  - gate_h  = softmax(GateNet(g, pi, beta, quality))  [B, H]  — routing-aware gate
  - pred    = sum_h gate_h * pred_h    [B]

Gap 11 (optional): GlobalResidual
  - pred += w_global * GlobalResidual(mean_h(head_repr))

Gap 9 (optional): heteroscedastic uncertainty head
  - mu_h, logvar_h = UncertaintyHead_h(head_repr_h)
  - L_nll = sum_h gate_h * [(mu_h - y)^2 / exp(logvar_h) + logvar_h]
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Flat regression head (A0 reference — same as worktree RegressionHead)
# ---------------------------------------------------------------------------

class FlatRegressionHead(nn.Module):
    """
    Legacy-style: reshape [B, H, Dh] → [B, H*Dh] → MLP → scalar.
    Used for A0 reference to reproduce existing behaviour exactly.
    """

    def __init__(self, n_heads: int, head_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        input_dim = n_heads * head_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, head_repr: torch.Tensor, aux: dict) -> Tuple[torch.Tensor, dict]:
        B = head_repr.shape[0]
        flat = head_repr.reshape(B, -1)
        pred = self.net(flat).squeeze(-1)
        return pred, {"pred_h": None, "gate_h": None}


# ---------------------------------------------------------------------------
# Gate network: sees routing variables + observer to produce head weights
# ---------------------------------------------------------------------------

class GateNet(nn.Module):
    """
    Computes gate_h = softmax(Linear → [B, H]).
    Input is the concatenation of: g [B, obs_dim], pi_flat [B, H*V], beta_flat [B, H],
    and optionally quality_flat [B, V].
    """

    def __init__(
        self,
        n_heads: int,
        obs_dim: int,
        n_views: int,
        use_quality_in_gate: bool = True,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.use_quality_in_gate = use_quality_in_gate
        gate_in = obs_dim + n_heads * n_views + n_heads  # g + pi_flat + beta_flat
        if use_quality_in_gate:
            gate_in += n_views  # quality_score [B, V]
        self.net = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_heads),
        )

    def forward(
        self,
        g: torch.Tensor,                              # [B, obs_dim]
        pi: torch.Tensor,                             # [B, H, V]
        beta: torch.Tensor,                           # [B, H, 1]
        quality_score: Optional[torch.Tensor] = None, # [B, V]
    ) -> torch.Tensor:
        B = g.shape[0]
        pi_flat = pi.reshape(B, -1)
        beta_flat = beta.reshape(B, -1)
        parts = [g, pi_flat, beta_flat]
        if self.use_quality_in_gate and quality_score is not None:
            parts.append(quality_score)
        gate_in = torch.cat(parts, dim=-1)
        return torch.softmax(self.net(gate_in), dim=-1)  # [B, H]


# ---------------------------------------------------------------------------
# GlobalResidual (Gap 11)
# ---------------------------------------------------------------------------

class GlobalResidual(nn.Module):
    """
    Safety net: a small MLP on mean-pooled head representations.
    w_global starts at 0 — the path is zero at init and grows only if needed.
    """

    def __init__(self, head_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(head_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.w_global = nn.Parameter(torch.zeros(1))

    def forward(self, head_repr: torch.Tensor) -> torch.Tensor:
        # head_repr: [B, H, Dh]
        mean_repr = head_repr.mean(dim=1)  # [B, Dh]
        return (self.net(mean_repr).squeeze(-1) * F.softplus(self.w_global)).squeeze(-1)  # [B]


# ---------------------------------------------------------------------------
# HeadGatedRegressor (Gap 6 — primary fix)
# ---------------------------------------------------------------------------

class HeadGatedRegressor(nn.Module):
    """
    Head-Gated Regressor implementing Gap 6.

    Each head h has its own independent linear prediction:
        pred_h = W_h @ head_repr_h + b_h     [B]

    The gate is computed from routing variables:
        gate_h = softmax(GateNet(g, pi, beta, quality_score))   [B, H]

    Final prediction:
        pred = sum_h gate_h * pred_h + (optional) GlobalResidual

    The gradient of MSE flows back through gate_h to g, pi, beta — this is the key
    signal-amplification: the router learns what to route based on prediction outcomes,
    not just representation quality.
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        obs_dim: int,
        n_views: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        use_quality_in_gate: bool = True,
        use_global_residual: bool = False,   # Gap 11
        use_uncertainty: bool = False,        # Gap 9
        use_router_gate: bool = True,         # if False: use backbone's gate_h directly
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.use_global_residual = use_global_residual
        self.use_uncertainty = use_uncertainty
        self.use_router_gate = use_router_gate

        # H separate per-head prediction networks
        if use_uncertainty:
            # Output mu_h and logvar_h for each head
            self.head_nets = nn.ModuleList([
                nn.Linear(head_dim, 2)  # [mu, logvar]
                for _ in range(n_heads)
            ])
        else:
            self.head_nets = nn.ModuleList([
                nn.Linear(head_dim, 1)
                for _ in range(n_heads)
            ])

        if use_router_gate:
            self.gate_net = GateNet(
                n_heads=n_heads,
                obs_dim=obs_dim,
                n_views=n_views,
                use_quality_in_gate=use_quality_in_gate,
                hidden_dim=hidden_dim,
            )

        if use_global_residual:
            self.global_residual = GlobalResidual(head_dim)

    def forward(
        self,
        head_repr: torch.Tensor,                       # [B, H, Dh]
        aux: dict,                                      # from backbone forward
        g: torch.Tensor,                               # [B, obs_dim]
        quality_score: Optional[torch.Tensor] = None,  # [B, V]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns
        -------
        pred     : [B]     final prediction (denormalized outside)
        head_out : dict    pred_h [B, H], gate_h [B, H], optionally logvar_h [B, H]
        """
        B, H, Dh = head_repr.shape
        pi = aux["pi"]      # [B, H, V]
        beta = aux["beta"]  # [B, H, 1]

        # Per-head predictions
        if self.use_uncertainty:
            head_outputs = [self.head_nets[h](head_repr[:, h, :]) for h in range(H)]  # list of [B, 2]
            mu_h = torch.stack([o[:, 0] for o in head_outputs], dim=1)      # [B, H]
            logvar_h = torch.stack([o[:, 1] for o in head_outputs], dim=1)  # [B, H]
            pred_h = mu_h
        else:
            pred_h = torch.stack([
                self.head_nets[h](head_repr[:, h, :]).squeeze(-1)
                for h in range(H)
            ], dim=1)  # [B, H]
            logvar_h = None

        # Gate
        if self.use_router_gate:
            gate_h = self.gate_net(g, pi, beta, quality_score)  # [B, H]
        else:
            # Use the gate from the backbone router directly
            gate_h = aux["gate_h"]  # [B, H]

        # Gated sum
        pred = (gate_h * pred_h).sum(dim=1)  # [B]

        # Global residual safety net
        if self.use_global_residual:
            pred = pred + self.global_residual(head_repr)

        head_out = {
            "pred_h": pred_h,   # [B, H]
            "gate_h": gate_h,   # [B, H]
            "logvar_h": logvar_h,
        }
        return pred, head_out


# ---------------------------------------------------------------------------
# HeadGatedClassifier (Gap 6 for classification)
# ---------------------------------------------------------------------------

class HeadGatedClassifier(nn.Module):
    """
    Classification equivalent of HeadGatedRegressor.
        logit_h = Linear_h(head_repr_h)           [B, H, C]
        gate_h  = softmax(GateNet(...))            [B, H]
        logit   = sum_h gate_h.unsqueeze(-1) * logit_h  [B, C]
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        n_classes: int,
        obs_dim: int,
        n_views: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        use_quality_in_gate: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_nets = nn.ModuleList([
            nn.Linear(head_dim, n_classes)
            for _ in range(n_heads)
        ])
        self.gate_net = GateNet(
            n_heads=n_heads,
            obs_dim=obs_dim,
            n_views=n_views,
            use_quality_in_gate=use_quality_in_gate,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        head_repr: torch.Tensor,
        aux: dict,
        g: torch.Tensor,
        quality_score: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        B, H, Dh = head_repr.shape
        pi = aux["pi"]
        beta = aux["beta"]

        logit_h = torch.stack([
            self.head_nets[h](head_repr[:, h, :])
            for h in range(H)
        ], dim=1)  # [B, H, C]

        gate_h = self.gate_net(g, pi, beta, quality_score)  # [B, H]
        logit = (gate_h.unsqueeze(-1) * logit_h).sum(dim=1)  # [B, C]

        head_out = {"logit_h": logit_h, "gate_h": gate_h}
        return logit, head_out


# ---------------------------------------------------------------------------
# FlatClassificationHead (A0 reference for classification)
# ---------------------------------------------------------------------------

class FlatClassificationHead(nn.Module):
    def __init__(self, n_heads: int, head_dim: int, n_classes: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_heads * head_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, head_repr: torch.Tensor, aux: dict) -> Tuple[torch.Tensor, dict]:
        B = head_repr.shape[0]
        flat = head_repr.reshape(B, -1)
        logit = self.net(flat)
        return logit, {"logit_h": None, "gate_h": None}
