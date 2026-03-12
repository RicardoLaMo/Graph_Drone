import torch
import torch.nn as nn
import torch.nn.functional as F
from graphdrone_fit.set_router import RouterOutputs

class CrossAttentionSetRouter(nn.Module):
    """
    Advanced Cross-Attention Router:
    Uses the 'FULL' anchor as the QUERY to attend to the specialists (KEYS/VALUES).
    
    Why: The 'FULL' generalist represents the global context. Specialists should
    be selected only if they are relevant relative to what 'FULL' already knows.
    """
    def __init__(self, token_dim: int, n_heads: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.token_dim = token_dim
        self.n_heads = n_heads
        
        # Projection for Multi-Head Attention
        self.q_proj = nn.Linear(token_dim, hidden_dim)
        self.k_proj = nn.Linear(token_dim, hidden_dim)
        self.v_proj = nn.Linear(token_dim, hidden_dim)
        
        # Weight prediction (alpha) from attention output
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # Defer head (delta) based on anchor context
        self.defer_head = nn.Sequential(
            nn.Linear(token_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, tokens: torch.Tensor, *, full_index: int) -> RouterOutputs:
        # tokens: [N, E, D]
        n_rows, n_experts, _ = tokens.shape
        
        # Extract the 'FULL' anchor as the query
        anchor_token = tokens[:, full_index:full_index+1, :] # [N, 1, D]
        
        # 1. Multi-Head Cross Attention
        # Query: FULL token
        # Key/Value: All specialist tokens
        q = self.q_proj(anchor_token) # [N, 1, H]
        k = self.k_proj(tokens)       # [N, E, H]
        v = self.v_proj(tokens)       # [N, E, H]
        
        # Compute Attention Scores (contextual similarity)
        # score: [N, 1, E]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1]**0.5)
        
        # 2. Specialist Weights (Alpha)
        # Instead of raw attention, we pass the contextual representation back to a head
        # to predict the 'trust' in each expert.
        alpha_logits = self.alpha_head(v).squeeze(-1) # [N, E]
        # We apply the attention weights as a "gate" on the logits
        gated_logits = alpha_logits * attn_scores.squeeze(1)
        specialist_weights = F.softmax(gated_logits, dim=-1)
        
        # 3. Defer Probability (Delta)
        defer_logit = self.defer_head(anchor_token.squeeze(1)) # [N, 1]
        defer_prob = torch.sigmoid(defer_logit)
        
        return RouterOutputs(
            specialist_weights=specialist_weights,
            defer_prob=defer_prob,
            full_index=full_index,
            router_kind="cross_attention_set_router",
        )
