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

class ContextualTransformerRouter(nn.Module):
    """
    Consolidated Router for GraphDrone.
    Includes Noise-Gate Pruning and Cross-Attention specialist weighting.
    """
    def __init__(self, token_dim: int, n_heads: int = 4, hidden_dim: int = 64, use_noise_gate: bool = True):
        super().__init__()
        self.token_dim = token_dim
        self.use_noise_gate = use_noise_gate
        
        # 1. Noise Gate (SNR-based pruning)
        if use_noise_gate:
            self.noise_gate = nn.Sequential(
                nn.Linear(token_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        # 2. Cross-Attention Core
        self.q_proj = nn.Linear(token_dim, hidden_dim)
        self.k_proj = nn.Linear(token_dim, hidden_dim)
        self.v_proj = nn.Linear(token_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        
        # 3. Specialist Trust Head
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # 4. Defer Head
        self.defer_head = nn.Sequential(
            nn.Linear(token_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, tokens: torch.Tensor, *, full_index: int) -> RouterOutputs:
        # tokens: [N, E, D]
        
        # Step A: Noise-Gating
        if self.use_noise_gate:
            # Assume SNR is a field in the token (last field or specific slice)
            # For simplicity in this consolidated version, we gate based on full token context
            validity = self.noise_gate(tokens)
            tokens = tokens * validity
            
        anchor_token = tokens[:, full_index:full_index+1, :]
        
        # Step B: Cross-Attention
        q = self.q_proj(anchor_token)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)
        
        attn_out, _ = self.attn(q, k, v)
        
        # Step C: Weights & Defer
        specialist_weights = F.softmax(self.alpha_head(v).squeeze(-1), dim=-1)
        defer_prob = torch.sigmoid(self.defer_head(anchor_token.squeeze(1)))
        
        return RouterOutputs(
            specialist_weights=specialist_weights,
            defer_prob=defer_prob,
            full_index=full_index,
            router_kind="contextual_transformer_router"
        )

def build_set_router(config: SetRouterConfig, token_dim: int = 14) -> torch.nn.Module:
    config.validate()
    if config.kind == "bootstrap_full_only":
        # Legacy placeholder
        return ContextualTransformerRouter(token_dim=token_dim, use_noise_gate=False)
    
    return ContextualTransformerRouter(
        token_dim=token_dim, 
        use_noise_gate=(config.kind == "noise_gate_router")
    )
