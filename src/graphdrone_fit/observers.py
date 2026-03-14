import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

def calculate_kappa(X: np.ndarray, indices: np.ndarray, r: int = 2) -> np.ndarray:
    """
    GoRA Principle: Local Non-Flatness Proxy (PCA residual ratio).
    X: Training set manifold.
    indices: [N_query, K] indices of neighbors in X.
    r: Number of principal components to consider 'flat'.

    Vectorised via batched SVD: gather all neighbourhoods into a
    [N, K, D] array and call np.linalg.svd once.  Falls back to the
    scalar loop only for degenerate rows where the batch SVD fails.
    """
    n_query, k = indices.shape
    nb = X[indices]          # [N, K, D]
    c  = nb - nb.mean(axis=1, keepdims=True)   # centre each neighbourhood

    try:
        # np.linalg.svd on a stacked array: [N, K, D] → s: [N, min(K,D)]
        _, s, _ = np.linalg.svd(c, full_matrices=False)
        tv = (s ** 2).sum(axis=1)              # [N] total variance per query
        top_r = (s[:, :r] ** 2).sum(axis=1)   # [N] variance in top-r PCs
        out = np.where(tv < 1e-12, 0.0, 1.0 - top_r / tv)
        return out.astype(np.float32)
    except Exception:
        # Scalar fallback for any edge-case (e.g. rank-deficient batches)
        out = np.zeros(n_query, np.float32)
        for i in range(n_query):
            try:
                _, s_i, _ = np.linalg.svd(c[i], full_matrices=False)
                tv_i = (s_i ** 2).sum()
                out[i] = 0.0 if tv_i < 1e-12 else 1.0 - (s_i[:r] ** 2).sum() / tv_i
            except Exception:
                out[i] = 0.0
        return out


def calculate_lid(dists: np.ndarray) -> np.ndarray:
    """
    GoRA Principle: Local Intrinsic Dimensionality.
    Uses the growth rate of distances to estimate latent dimension.

    Vectorised via least-squares normal equations on the full [N, K]
    distance matrix at once, avoiding the per-row polyfit loop.
    """
    n, k = dists.shape
    d = np.maximum(dists, 1e-12).astype(np.float64)
    ks = np.arange(1, k + 1, dtype=np.float64)      # [K]
    log_k  = np.log(ks)                               # [K]
    log_d  = np.log(d)                                # [N, K]

    # Fit log_k vs log_d[i] for each i with a single closed-form solve.
    # slope = (K*sum(log_k*log_d) - sum(log_k)*sum(log_d)) /
    #         (K*sum(log_k^2) - sum(log_k)^2)
    k_f   = float(k)
    slk   = log_k.sum()
    slk2  = (log_k ** 2).sum()
    denom = k_f * slk2 - slk ** 2

    if abs(denom) < 1e-12:
        return np.zeros(n, np.float32)

    sld   = log_d.sum(axis=1)                          # [N]
    slkld = (log_k * log_d).sum(axis=1)                # [N]
    slopes = (k_f * slkld - slk * sld) / denom        # [N]
    return np.maximum(slopes, 0.0).astype(np.float32)
