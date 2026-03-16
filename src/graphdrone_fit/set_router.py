from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .config import SetRouterConfig

@dataclass(frozen=True)
class RouterOutputs:
    specialist_weights: torch.Tensor
    defer_prob: torch.Tensor
    full_index: int
    router_kind: str

class BootstrapFullRouter(nn.Module):
    """Static router — always defers to the anchor (FULL) expert, no training."""
    def forward(self, tokens: torch.Tensor, *, full_index: int, task_token: torch.Tensor = None) -> RouterOutputs:
        n_rows, n_experts, _ = tokens.shape
        specialist_weights = torch.zeros((n_rows, n_experts), dtype=tokens.dtype, device=tokens.device)
        defer_prob = torch.zeros((n_rows, 1), dtype=tokens.dtype, device=tokens.device)
        return RouterOutputs(specialist_weights, defer_prob, full_index, "bootstrap_full_only")

class HyperSetRouter(nn.Module):
    """
    HyperGraph-inspired Router.
    Uses a TaskToken (global Bayesian prior) combined with cross-attention over expert
    tokens to compute task-conditioned specialist weights.  The TaskToken allows the
    router to modulate *which* specialists to defer to based on dataset-level statistics
    (mean, std, sparsity, dimensionality), improving SNR on high-dimensional / multi-class tasks.
    """
    def __init__(self, token_dim: int, task_dim: int = 4, n_heads: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.token_dim = token_dim
        self.task_dim = task_dim

        # Project expert tokens and task signal into shared hidden space
        self.expert_proj = nn.Linear(token_dim, hidden_dim)
        self.task_proj = nn.Linear(task_dim, hidden_dim)

        # Cross-attention: task-conditioned anchor queries over all expert keys/values
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)

        # Defer head: given attention output, predict how much to defer from anchor to specialists
        self.defer_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 1)
        )
        
        # Specialist preference: Learnable bias to nudge exploration
        self.exploration_bias = nn.Parameter(torch.zeros(1))

    def forward(self, tokens: torch.Tensor, *, full_index: int, task_token: torch.Tensor = None) -> RouterOutputs:
        # tokens: [B, E, D],  task_token: [B, 1, D_task] or None
        B, E, D = tokens.shape
        device = tokens.device

        e_hidden = self.expert_proj(tokens)                          # [B, E, H]
        anchor_hidden = e_hidden[:, full_index:full_index + 1, :]   # [B, 1, H]

        # Build task-conditioned query: anchor + global task signal
        if task_token is not None:
            t_hidden = self.task_proj(task_token.to(device))        # [B, 1, H]
            # More aggressive integration: concatenate or multiply? Let's try additive query refinement.
            query = anchor_hidden + t_hidden
        else:
            query = anchor_hidden

        # Cross-attention: query attends over all expert hidden states
        attn_out, _ = self.attn(query, e_hidden, e_hidden)          # [B, 1, H]

        # Specialist weights: alignment between the task-refined query and specialist tokens
        # Increased sensitivity: Scaling dot product by sqrt(H)
        expert_scores = (e_hidden * attn_out).sum(dim=-1) / (attn_out.shape[-1] ** 0.5) # [B, E]
        specialist_weights = F.softmax(expert_scores, dim=-1)       # [B, E]

        # Defer probability: how much to blend specialists in vs. anchor alone.
        # We add the exploration_bias to encourage non-zero deferral during early training.
        defer_logits = self.defer_head(attn_out.squeeze(1)) + self.exploration_bias
        defer_prob = torch.sigmoid(defer_logits)  # [B, 1]

        return RouterOutputs(specialist_weights, defer_prob, full_index, "hyper_set_router")


def build_set_router(config: SetRouterConfig, token_dim: int = 14, task_dim: int = 4) -> torch.nn.Module:
    config.validate()
    if config.kind == "bootstrap_full_only":
        return BootstrapFullRouter()
    # All trainable router kinds use HyperSetRouter (ContextualTransformerRouter superseded)
    return HyperSetRouter(token_dim=token_dim, task_dim=task_dim)
