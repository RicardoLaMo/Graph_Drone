import torch
import torch.nn as nn
from src.graphdrone_fit.set_router_contextual import CrossAttentionSetRouter
from support_reader import TransformerSupportReader

class SupportAwareGraphDroneRouter(nn.Module):
    """
    Key Component: Support-Aware Contextual Router.
    
    This merges the Transformer Support Reader with the Cross-Attention Set Router.
    It works well with the new Token Builder by replacing static moments 
    with learned context embeddings.
    """
    def __init__(self, d_x: int, n_experts: int, d_model: int = 64):
        super().__init__()
        
        # 1. Per-expert Support Readers
        # (Could share weights or have independent readers per subspace)
        self.support_readers = nn.ModuleList([
            TransformerSupportReader(d_x=d_x, d_model=d_model)
            for _ in range(n_experts)
        ])
        
        # 2. Contextual Set Router (the one we just built)
        # token_dim: includes prediction, quality, and now the support embedding
        # Let's assume prediction(3) + quality(0) + support(d_model) + descriptor(6)
        token_dim = 3 + 0 + d_model + 6 
        self.set_router = CrossAttentionSetRouter(token_dim=token_dim)

    def forward(self, query_features, support_features_list, support_labels_list, 
                base_tokens, full_index: int):
        """
        query_features: [B, d_x]
        support_features_list: list of [B, K, d_x_v] for each expert
        support_labels_list: list of [B, K, 1] for each expert
        base_tokens: [B, E, D_base] - partial tokens without support
        """
        B, E, _ = base_tokens.shape
        
        # 1. Generate Contextual Support Embeddings
        support_embeddings = []
        for i in range(E):
            emb = self.support_readers[i](
                query_features, # or query_features_v
                support_features_list[i],
                support_labels_list[i]
            )
            support_embeddings.append(emb) # [B, d_model]
            
        support_tokens = torch.stack(support_embeddings, dim=1) # [B, E, d_model]
        
        # 2. Build Full Tokens
        # Concatenate support embeddings into the token
        full_tokens = torch.cat([base_tokens, support_tokens], dim=-1)
        
        # 3. Contextual Routing
        return self.set_router(full_tokens, full_index=full_index)
