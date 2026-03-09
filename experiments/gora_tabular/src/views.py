"""
views.py — Sparse adjacency matrices A^(m) per view for GoRA-Tabular.

Each A^(m) is a symmetric kNN adjacency stored as a dense [N, N] float32
for small N (≤ 20k), or as edge_index + edge_weight for large N.

We store as torch sparse tensors for efficient batched mixing:
  Ã_{ij}^{i,h} = Σ_m π_{i,h,m} · A^(m)_{ij}

Design choice: row-normalised (each row sums to 1) so that mixing
adjacencies via π produces a valid convex combination per row.
"""
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from typing import Dict, Tuple


def _row_normalised_knn(X_topo: np.ndarray, k: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (src, dst, weight) for a symmetric kNN graph, row-normalised per src."""
    nb = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X_topo)
    dists, idx = nb.kneighbors(X_topo)
    dists, idx = dists[:, 1:], idx[:, 1:]    # drop self-loop
    N = X_topo.shape[0]
    # Gaussian weights: w_ij = exp(-d²/σ²), σ = median distance
    sigma = np.median(dists) + 1e-8
    w = np.exp(-dists ** 2 / sigma ** 2)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-8)   # row-normalise
    src = np.repeat(np.arange(N), k)
    dst = idx.flatten()
    wts = w.flatten()
    return src.astype(np.int64), dst.astype(np.int64), wts.astype(np.float32)


def build_sparse_adj(X_topo: np.ndarray, N: int, k: int = 15) -> torch.Tensor:
    """Returns dense [N, N] row-normalised adjacency. Only feasible for N ≤ 25k."""
    src, dst, wts = _row_normalised_knn(X_topo, k)
    A = torch.zeros(N, N, dtype=torch.float32)
    A[src, dst] = torch.tensor(wts)
    return A


def build_edge_index(X_topo: np.ndarray, k: int = 15):
    """Returns (edge_index [2, E], edge_weight [E]) for memory-efficient forward."""
    src, dst, wts = _row_normalised_knn(X_topo, k)
    ei = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    ew = torch.tensor(wts, dtype=torch.float32)
    return ei, ew


# ─── California Housing ───────────────────────────────────────────────────────

GEO_COLS    = [6, 7]
SOCIO_COLS  = [0, 1, 2, 3, 4]

def california_view_features(X: np.ndarray, pca_d: int = 4) -> Dict[str, np.ndarray]:
    pca = PCA(n_components=pca_d, random_state=42)
    Xp = pca.fit_transform(X).astype(np.float32)
    return {
        "FULL":    X,
        "GEO":     X[:, GEO_COLS],
        "SOCIO":   X[:, SOCIO_COLS],
        "LOWRANK": Xp,
    }


def build_california_views(X: np.ndarray, k: int = 15) -> Dict[str, Tuple]:
    """Returns {view_name: (edge_index, edge_weight)} per view."""
    vfeats = california_view_features(X)
    views = {}
    for name, Xv in vfeats.items():
        ei, ew = build_edge_index(Xv, k)
        views[name] = (ei, ew)
        print(f"  [view {name}] {ei.shape[1]} edges, N={X.shape[0]}")
    return views, vfeats


# ─── MNIST-784 ────────────────────────────────────────────────────────────────

def _block_feats(X: np.ndarray, n_blocks: int = 16) -> np.ndarray:
    imgs = X.reshape(-1, 28, 28); s = 7
    return np.stack(
        [imgs[:, r * s:(r + 1) * s, c * s:(c + 1) * s].mean((1, 2))
         for r in range(4) for c in range(4)], axis=1
    ).astype(np.float32)


def mnist_view_features(X: np.ndarray, pca_d: int = 50) -> Dict[str, np.ndarray]:
    pca = PCA(n_components=pca_d, random_state=42)
    Xp = pca.fit_transform(X).astype(np.float32)
    Xb = _block_feats(X)
    return {
        "FULL":  X,
        "BLOCK": Xb,
        "PCA":   Xp,
    }


def build_mnist_views(X: np.ndarray, k: int = 15) -> Dict[str, Tuple]:
    vfeats = mnist_view_features(X)
    views = {}
    for name, Xv in vfeats.items():
        ei, ew = build_edge_index(Xv, k)
        views[name] = (ei, ew)
        print(f"  [view {name}] {ei.shape[1]} edges, N={X.shape[0]}")
    return views, vfeats
