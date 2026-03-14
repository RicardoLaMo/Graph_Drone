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

class ContextualTransformerRouter(nn.Module):
    """
    Consolidated Router for GraphDrone.
    Includes Noise-Gate Pruning and Cross-Attention specialist weighting.
    """
    def __init__(self, token_dim: int, n_heads: int = 4, hidden_dim: int = 64, use_noise_gate: bool = True):
        super().__init__()
        self.token_dim = token_dim
        self.use_noise_gate = use_noise_gate
        
        if use_noise_gate:
            self.noise_gate = nn.Sequential(
                nn.Linear(token_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        self.q_proj = nn.Linear(token_dim, hidden_dim)
        self.k_proj = nn.Linear(token_dim, hidden_dim)
        self.v_proj = nn.Linear(token_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        
        self.defer_head = nn.Sequential(nn.Linear(token_dim, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, tokens: torch.Tensor, *, full_index: int) -> RouterOutputs:
        if self.use_noise_gate:
            validity = self.noise_gate(tokens)
            tokens = tokens * validity
            
        anchor_token = tokens[:, full_index:full_index+1, :]
        q = self.q_proj(anchor_token)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)
        
        # need_weights=True → attn_weights shape [B, 1, E] (anchor attends to all E experts)
        # average_attn_weights averages over heads → clean [B, 1, E] distribution
        _, attn_weights = self.attn(q, k, v, need_weights=True, average_attn_weights=True)
        # squeeze(1) → [B, E]: cross-attention distribution = learned relevance per expert
        specialist_weights = attn_weights.squeeze(1)
        defer_prob = torch.sigmoid(self.defer_head(anchor_token.squeeze(1)))
        
        return RouterOutputs(specialist_weights, defer_prob, full_index, "contextual_transformer_router")

def build_set_router(config: SetRouterConfig, token_dim: int = 14) -> torch.nn.Module:
    config.validate()
    if config.kind == "bootstrap_full_only":
        return BootstrapFullRouter()
    
    return ContextualTransformerRouter(
        token_dim=token_dim, 
        use_noise_gate=(config.kind == "noise_gate_router")
    )
