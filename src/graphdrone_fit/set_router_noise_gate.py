import torch
import torch.nn as nn
import torch.nn.functional as F
from .set_router_contextual import CrossAttentionSetRouter

class NoiseGateRouter(CrossAttentionSetRouter):
    """
    Key Component: Noise-Gate Pruning Router.
    Learns to 'mask out' experts that have low SNR for a specific query point.
    """
    def __init__(self, token_dim: int, n_heads: int = 4, hidden_dim: int = 64):
        super().__init__(token_dim, n_heads, hidden_dim)
        
        # The Noise Gate: learns a threshold for SNR-based pruning
        self.snr_threshold_head = nn.Sequential(
            nn.Linear(token_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, tokens, *, full_index: int, snr_slice=( -1, None)):
        """
        tokens: [B, E, D]
        snr_slice: indices where the SNR features are stored
        """
        # 1. Extract SNR scores from tokens
        # We assume SNR is the last feature (added by NoiseAwareTokenBuilder)
        snr_scores = tokens[:, :, -1:] # [B, E, 1]
        
        # 2. Compute per-expert 'Signal Validity' mask
        # Higher SNR -> higher validity
        validity_logits = self.snr_threshold_head(tokens) # [B, E, 1]
        
        # Hard mask (Straight-Through Estimator or Soft approximation)
        # We use a soft mask here: gate = sigmoid(snr - learned_threshold)
        gate = torch.sigmoid(snr_scores - validity_logits)
        
        # 3. Apply Gate to tokens before Attention
        # Pruned experts will have near-zero embeddings
        gated_tokens = tokens * gate
        
        # 4. Standard Cross-Attention on gated tokens
        out = super().forward(gated_tokens, full_index=full_index)
        
        # Mechanism Audit: Record how many experts were 'pruned'
        pruning_rate = (gate < 0.5).float().mean()
        
        return out, pruning_rate
