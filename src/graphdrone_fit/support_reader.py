import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerSupportReader(nn.Module):
    """
    Transformer-based Query Contextual Support Reader.
    
    Architecture:
    1. Query: Current query row embedding (x_q).
    2. Context: Set of K neighbors (x_k, y_k) from the support set.
    3. Transformation: Cross-Attention where x_q attends to the support set.
    
    Why this works (Theory):
    - TabPFN / TabR research: Tabular data performance is highly dependent on 
      local context. Instead of global weights, we compute a local posterior
      representation.
    - Contextual Mapping: x_q defines the 'where' in the manifold, and the 
      Transformer computes the 'how much to trust' each support neighbor.
    """
    def __init__(self, d_x: int, d_model: int = 64, n_heads: int = 4, d_y: int = 1):
        super().__init__()
        self.d_model = d_model
        
        # Project raw features into embedding space
        self.x_proj = nn.Linear(d_x, d_model)
        self.y_proj = nn.Linear(d_y, d_model)
        
        # Cross-Attention: Query attends to (Neighbor_X + Neighbor_Y)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x_q: torch.Tensor, x_support: torch.Tensor, y_support: torch.Tensor):
        """
        x_q: [B, d_x] - Query point
        x_support: [B, K, d_x] - K neighbors
        y_support: [B, K, d_y] - K neighbor labels
        """
        B, K, _ = x_support.shape
        
        # 1. Embed query and support
        q = self.x_proj(x_q).unsqueeze(1) # [B, 1, d_model]
        
        # Context is (X_k, Y_k) combined
        # In a more advanced version, we could use a specialized encoder for (x,y) pairs
        k_emb = self.x_proj(x_support) + self.y_proj(y_support) # [B, K, d_model]
        v_emb = k_emb # simplified
        
        # 2. Cross-Attention
        # Query: x_q
        # Key/Value: Support set
        attn_out, _ = self.attn(q, k_emb, v_emb) # [B, 1, d_model]
        
        # 3. Residual and FFN
        out = self.norm(q + attn_out)
        out = self.norm(out + self.ffn(out))
        
        return out.squeeze(1) # [B, d_model] - This is the "Query Context Token"
