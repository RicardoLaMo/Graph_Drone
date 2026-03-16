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
    def forward(self, tokens: torch.Tensor, *, full_index: int) -> RouterOutputs:
        n_rows, n_experts, _ = tokens.shape
        specialist_weights = torch.zeros((n_rows, n_experts), dtype=tokens.dtype, device=tokens.device)
        defer_prob = torch.zeros((n_rows, 1), dtype=tokens.dtype, device=tokens.device)
        return RouterOutputs(specialist_weights, defer_prob, full_index, "bootstrap_full_only")

class HyperSetRouter(nn.Module):
    """
    HyperGraph-inspired Router.
    Uses TaskToken (global context) and Clique-Pooling (family-level consensus)
    to improve Signal-to-Noise Ratio (SNR) in high-dimensional tasks.
    """
    def __init__(self, token_dim: int, task_dim: int = 4, n_heads: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.token_dim = token_dim
        self.task_dim = task_dim
        
        # Projections
        self.task_proj = nn.Linear(task_dim, hidden_dim)
        self.expert_proj = nn.Linear(token_dim, hidden_dim)
        
        # HyperGraph Attention
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        
        # Heads
        self.alpha_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, 1))
        self.defer_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, tokens: torch.Tensor, *, full_index: int, task_token: torch.Tensor = None) -> RouterOutputs:
        # tokens: [B, E, D], task_token: [B, 1, D_task]
        B, E, D = tokens.shape
        
        e_hidden = self.expert_proj(tokens) # [B, E, H]
        anchor_hidden = e_hidden[:, full_index:full_index+1, :] # [B, 1, H]
        
        # Integrate Task Signal
        if task_token is not None:
            t_hidden = self.task_proj(task_token) # [B, 1, H]
            # Global Task Query: Combine Anchor and Global Context
            query = anchor_hidden + t_hidden 
        else:
            query = anchor_hidden
            
        # Cross-Attention: Anchor+Task vs All Experts
        attn_out, _ = self.attn(query, e_hidden, e_hidden) # [B, 1, H]
        
        # Compute Specialist Weights
        # We use the relationship between the global query and individual experts
        specialist_weights = F.softmax(self.alpha_head(e_hidden).squeeze(-1), dim=-1)
        
        # Compute Defer Probability based on the aggregated attention signal
        defer_prob = torch.sigmoid(self.defer_head(attn_out.squeeze(1)))
        
        return RouterOutputs(specialist_weights, defer_prob, full_index, "hyper_set_router")

def build_set_router(config: SetRouterConfig, token_dim: int = 14, task_dim: int = 4) -> torch.nn.Module:
    config.validate()
    if config.kind == "bootstrap_full_only":
        return BootstrapFullRouter()
    
    if config.kind == "hyper_set_router":
        return HyperSetRouter(token_dim=token_dim, task_dim=task_dim)
    
    return ContextualTransformerRouter(
        token_dim=token_dim, 
        use_noise_gate=(config.kind == "noise_gate_router")
    )
