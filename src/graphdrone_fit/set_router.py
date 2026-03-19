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
    Uses explicit scaled dot-product cross-attention (anchor queries all experts)
    to avoid nn.MultiheadAttention version incompatibilities (4D attn_weights bug).
    Includes optional Noise-Gate Pruning.
    """
    def __init__(self, token_dim: int, n_heads: int = 4, hidden_dim: int = 64, use_noise_gate: bool = True):
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.scale = hidden_dim ** -0.5
        self.use_noise_gate = use_noise_gate

        if use_noise_gate:
            self.noise_gate = nn.Sequential(
                nn.Linear(token_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        # Anchor queries experts — single-query cross-attention (no MHA needed)
        self.q_proj = nn.Linear(token_dim, hidden_dim)
        self.k_proj = nn.Linear(token_dim, hidden_dim)

        self.defer_head = nn.Sequential(nn.Linear(token_dim, 32), nn.GELU(), nn.Linear(32, 1))
        # Anchor-first prior: initialize defer toward zero so the router must earn blending.
        # sigmoid(-3) ≈ 0.047 — specialists prove themselves before being trusted.
        nn.init.constant_(self.defer_head[-1].bias, -3.0)

    def forward(self, tokens: torch.Tensor, *, full_index: int) -> RouterOutputs:
        if self.use_noise_gate:
            validity = self.noise_gate(tokens)
            tokens = tokens * validity

        anchor_token = tokens[:, full_index, :]         # [B, token_dim]
        q = self.q_proj(anchor_token)                   # [B, hidden_dim]
        k = self.k_proj(tokens)                         # [B, E, hidden_dim]

        # Scaled dot-product: anchor attends to all experts → [B, E]
        attn_scores = torch.einsum("bh,beh->be", q, k) * self.scale
        specialist_weights = torch.softmax(attn_scores, dim=-1)

        defer_prob = torch.sigmoid(self.defer_head(anchor_token))  # [B, 1]

        return RouterOutputs(specialist_weights, defer_prob, full_index, "contextual_transformer_router")

def build_set_router(config: SetRouterConfig, token_dim: int = 14) -> torch.nn.Module:
    config.validate()
    if config.kind == "bootstrap_full_only":
        return BootstrapFullRouter()
    
    return ContextualTransformerRouter(
        token_dim=token_dim, 
        use_noise_gate=(config.kind == "noise_gate_router")
    )
