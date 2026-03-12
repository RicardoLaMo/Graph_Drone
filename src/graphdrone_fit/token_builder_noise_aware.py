import torch
import numpy as np
from .token_builder_enhanced import EnhancedTokenBuilder

class NoiseAwareTokenBuilder(EnhancedTokenBuilder):
    """
    Key Component: SNR-Aware Token Builder.
    Explicitly calculates the Signal-to-Noise ratio for each expert view.
    """
    def build_snr_tokens(self, predictions, support_moments):
        """
        predictions: [B, E]
        support_moments: [B, E, 4] (idx 0 is mean, idx 1 is variance)
        """
        device = support_moments.device
        pred_t = torch.as_tensor(predictions, dtype=torch.float32).to(device)
        mean_neighbors = support_moments[:, :, 0]
        var_neighbors = support_moments[:, :, 1]
        
        # Signal: Agreement with local neighbors
        signal = torch.abs(pred_t - mean_neighbors)
        # Noise: Local label variance
        noise = torch.sqrt(var_neighbors + 1e-6)
        
        # SNR: Higher is better (more signal, less noise)
        snr = signal / noise
        
        # Log-space SNR often more stable for neural networks
        log_snr = torch.log1p(snr)
        
        return log_snr.unsqueeze(-1) # [B, E, 1]

    def build(self, *, predictions, descriptors, full_expert_id, 
              quality_features=None, support_encoding=None):
        
        # 1. Base Enhanced Tokens
        base_batch = super().build(
            predictions=predictions,
            descriptors=descriptors,
            full_expert_id=full_expert_id,
            quality_features=quality_features,
            support_encoding=support_encoding
        )
        
        # 2. Add SNR Field
        if support_encoding is not None:
            snr_tokens = self.build_snr_tokens(predictions, support_encoding.tensor)
            
            # Ensure device parity for concatenation
            snr_tokens = snr_tokens.to(base_batch.tokens.device)
            new_tokens = torch.cat([base_batch.tokens, snr_tokens], dim=-1)
            
            # Update field names and slices
            new_slices = base_batch.field_slices.copy()
            new_names = base_batch.field_names.copy()
            
            cursor = base_batch.tokens.shape[-1]
            new_slices["snr"] = (cursor, cursor + 1)
            new_names["snr"] = ("log_snr",)
            
            from graphdrone_fit.token_builder import TokenBatch
            return TokenBatch(
                tokens=new_tokens,
                expert_ids=base_batch.expert_ids,
                field_slices=new_slices,
                field_names=new_names
            )
        return base_batch
