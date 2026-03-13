import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_prior_alignment_tokens(X_train_views, X_query_views):
    """
    Computes 'Prior-Alignment' tokens based on local manifold density.
    Rationale: If a view's subspace is densely populated around the query,
    the 'prior' for that view is likely more valid for this specific row.
    """
    n_experts = len(X_train_views)
    alignments = []
    
    for v_idx in range(n_experts):
        # We use distance to neighbors as a proxy for 'Alignment'
        # Smaller distance -> Higher alignment with the view's manifold
        knn = NearestNeighbors(n_neighbors=15, n_jobs=-1)
        knn.fit(X_train_views[v_idx])
        distances, _ = knn.kneighbors(X_query_views[v_idx])
        
        # Mean distance to neighbors
        mean_dist = distances.mean(axis=1)
        
        # Convert to alignment score (Inverse distance, normalized)
        # We use a soft-min style normalization across rows
        alignment = 1.0 / (mean_dist + 1e-6)
        alignments.append(alignment)
        
    # Stack and normalize across experts for each row
    # So the router sees which expert is 'best aligned' relatively
    alignment_tensor = torch.tensor(np.stack(alignments, axis=1), dtype=torch.float32)
    normalized_alignment = torch.softmax(alignment_tensor, dim=1)
    
    return normalized_alignment.unsqueeze(-1) # [B, E, 1]
