import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

def calculate_kappa(X: np.ndarray, indices: np.ndarray, r: int = 2) -> np.ndarray:
    """
    GoRA Principle: Local Non-Flatness Proxy (PCA residual ratio).
    X: Training set manifold.
    indices: [N_query, K] indices of neighbors in X.
    r: Number of principal components to consider 'flat'.
    """
    n_query = indices.shape[0]
    out = np.zeros(n_query, np.float32)
    for i in range(n_query):
        # Local neighborhood centered at zero
        nb = X[indices[i]]
        c = nb - nb.mean(0)
        # SVD for local variance structure
        try:
            _, s, _ = np.linalg.svd(c, full_matrices=False)
            tv = (s ** 2).sum()
            # Ratio of variance NOT captured by top-r components
            out[i] = 0.0 if tv < 1e-12 else 1.0 - (s[:r] ** 2).sum() / tv
        except:
            out[i] = 0.0
    return out

def calculate_lid(dists: np.ndarray) -> np.ndarray:
    """
    GoRA Principle: Local Intrinsic Dimensionality.
    Uses the growth rate of distances to estimate latent dimension.
    """
    k = dists.shape[1]
    ks = np.arange(1, k + 1, dtype=np.float64)
    out = np.zeros(dists.shape[0], np.float32)
    for i in range(dists.shape[0]):
        d = np.maximum(dists[i], 1e-12)
        # Fit log(k) vs log(dist)
        try:
            out[i] = max(float(np.polyfit(np.log(ks), np.log(d), 1)[0]), 0.0)
        except:
            out[i] = 0.0
    return out
