"""
losses_v5.py — Composite loss for v5 experiments.

Implements:
  L_task    — MSE / Huber / CrossEntropy (primary)
  L_pdiv    — prediction diversity: -Var_h(pred_h) (Gap 4)
  L_lb      — load balancing: -entropy(mean_B pi_h) (Gap 4)
  L_orth    — head disagreement (cos-sim penalty, from worktree)
  L_cons    — routing-gate consistency KL(mean_v pi_h || gate_h) (Gap 10)
  L_nll     — heteroscedastic NLL: gate-weighted (mu-y)²/exp(logvar)+logvar (Gap 9)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


EPS = 1e-8


# ---------------------------------------------------------------------------
# Task losses
# ---------------------------------------------------------------------------

def regression_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str = "huber") -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    if loss_type == "huber":
        return F.huber_loss(pred, target, delta=1.0)
    raise ValueError(f"Unknown loss_type: {loss_type}")


def classification_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logit, target)


# ---------------------------------------------------------------------------
# Prediction diversity (Gap 4)
# ---------------------------------------------------------------------------

def prediction_diversity_loss(pred_h: torch.Tensor) -> torch.Tensor:
    """
    L_pdiv = -mean_B Var_h(pred_h)
    Penalizes prediction collapse: forces heads to predict differently.

    pred_h: [B, H]
    """
    if pred_h is None or pred_h.shape[1] <= 1:
        return pred_h.new_tensor(0.0) if pred_h is not None else torch.tensor(0.0)
    var_h = pred_h.var(dim=1, unbiased=False)  # [B]
    return -var_h.mean()


# ---------------------------------------------------------------------------
# Load balance (Gap 4)
# ---------------------------------------------------------------------------

def load_balance_loss(pi: torch.Tensor) -> torch.Tensor:
    """
    L_lb = -sum_h entropy(mean_B pi_h)
    Penalizes one-view routing: forces routing mass to be spread across views.

    pi: [B, H, V]
    """
    mean_pi = pi.mean(dim=0)  # [H, V]
    # Entropy per head: -sum_v mean_pi_h_v * log(mean_pi_h_v)
    entropy = -(mean_pi * torch.log(mean_pi + EPS)).sum(dim=-1)  # [H]
    return -entropy.mean()  # minimize → maximize entropy


# ---------------------------------------------------------------------------
# Head disagreement (from worktree losses.py — representation diversity)
# ---------------------------------------------------------------------------

def _mean_pairwise_cosine(attn: torch.Tensor) -> torch.Tensor:
    """attn: [B, H, T]."""
    if attn.shape[1] <= 1:
        return attn.new_tensor(0.0)
    normed = F.normalize(attn, p=2, dim=-1)
    sims = torch.matmul(normed, normed.transpose(1, 2))
    H = attn.shape[1]
    mask = torch.triu(torch.ones(H, H, device=attn.device, dtype=torch.bool), diagonal=1)
    return sims[:, mask].mean()


def head_disagreement_loss(
    neighbor_attn: dict[str, torch.Tensor],
    cross_view_attn: torch.Tensor,
) -> torch.Tensor:
    """Penalizes cosine similarity of head attention maps (representation collapse)."""
    losses = [_mean_pairwise_cosine(attn) for attn in neighbor_attn.values()]
    losses.append(_mean_pairwise_cosine(cross_view_attn))
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Routing-gate consistency (Gap 10)
# ---------------------------------------------------------------------------

def routing_gate_consistency_loss(pi: torch.Tensor, gate_h: torch.Tensor) -> torch.Tensor:
    """
    L_cons = mean_B KL( softmax(mean_v pi_h) || gate_h )

    pi: [B, H, V] — routing weights over views per head
    gate_h: [B, H] — decoder gate

    Penalizes systematic divergence between routing mass and decoder trust.
    Weight with λ_cons ≈ 0.01.
    """
    # Mean over views to get a head-level importance: [B, H]
    mean_pi_h = pi.mean(dim=-1)  # [B, H]
    # KL(p || q) = sum p * log(p/q)
    p = mean_pi_h + EPS
    q = gate_h + EPS
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    kl = (p * (torch.log(p) - torch.log(q))).sum(dim=-1)
    return kl.mean()


# ---------------------------------------------------------------------------
# Heteroscedastic NLL (Gap 9)
# ---------------------------------------------------------------------------

def heteroscedastic_nll_loss(
    mu_h: torch.Tensor,
    logvar_h: torch.Tensor,
    gate_h: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    L_nll = mean_B sum_h gate_h * [(mu_h - y)^2 / exp(logvar_h) + logvar_h]

    mu_h:    [B, H]
    logvar_h: [B, H]
    gate_h:  [B, H]
    target:  [B]
    """
    sq_err = (mu_h - target.unsqueeze(1)) ** 2  # [B, H]
    precision = torch.exp(-logvar_h)             # [B, H]
    nll_per_head = precision * sq_err + logvar_h  # [B, H]
    return (gate_h * nll_per_head).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Composite loss builder
# ---------------------------------------------------------------------------

def composite_loss(
    # Task
    pred: torch.Tensor,
    target: torch.Tensor,
    task: str = "regression",           # "regression" | "classification"
    loss_type: str = "huber",
    # Head outputs
    pred_h: Optional[torch.Tensor] = None,
    logvar_h: Optional[torch.Tensor] = None,
    gate_h: Optional[torch.Tensor] = None,
    # Routing
    pi: Optional[torch.Tensor] = None,
    neighbor_attn: Optional[dict] = None,
    cross_view_attn: Optional[torch.Tensor] = None,
    # Weights
    lambda_orth: float = 0.0,
    lambda_pdiv: float = 0.0,
    lambda_lb: float = 0.0,
    lambda_cons: float = 0.0,
    lambda_nll: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    Compute composite loss and return breakdown dict for logging.
    """
    breakdown = {}

    if task == "regression":
        L_task = regression_loss(pred, target, loss_type)
    else:
        L_task = classification_loss(pred, target.long())
    breakdown["L_task"] = L_task.detach().item()
    total = L_task

    if lambda_orth > 0 and neighbor_attn is not None and cross_view_attn is not None:
        L_orth = head_disagreement_loss(neighbor_attn, cross_view_attn)
        breakdown["L_orth"] = L_orth.detach().item()
        total = total + lambda_orth * L_orth

    if lambda_pdiv > 0 and pred_h is not None:
        L_pdiv = prediction_diversity_loss(pred_h)
        breakdown["L_pdiv"] = L_pdiv.detach().item()
        total = total + lambda_pdiv * L_pdiv

    if lambda_lb > 0 and pi is not None:
        L_lb = load_balance_loss(pi)
        breakdown["L_lb"] = L_lb.detach().item()
        total = total + lambda_lb * L_lb

    if lambda_cons > 0 and pi is not None and gate_h is not None:
        L_cons = routing_gate_consistency_loss(pi, gate_h)
        breakdown["L_cons"] = L_cons.detach().item()
        total = total + lambda_cons * L_cons

    if lambda_nll > 0 and logvar_h is not None and gate_h is not None:
        L_nll = heteroscedastic_nll_loss(pred_h, logvar_h, gate_h, target)
        breakdown["L_nll"] = L_nll.detach().item()
        total = total + lambda_nll * L_nll

    breakdown["total"] = total.detach().item()
    return total, breakdown
