"""
views.py — Per-dataset view construction (kNN graph per view).
"""
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from torch_geometric.data import Data


def _knn_graph(X_topo, X_node, y, k=15, is_clf=True):
    nb = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X_topo)
    _, idx = nb.kneighbors(X_topo)
    idx = idx[:, 1:]  # drop self
    src = np.repeat(np.arange(X_topo.shape[0]), k); dst = idx.flatten()
    ei = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    yt = torch.tensor(y, dtype=torch.long if is_clf else torch.float32)
    return Data(x=torch.tensor(X_node, dtype=torch.float32), edge_index=ei, y=yt)


# ─── California views ─────────────────────────────────────────────────────────
GEO = [6, 7]; SOCIO = [0, 1, 2, 3, 4]

def california_views(X, y, k=15):
    """Returns dict {tag: PyG Data}. Topology varies per view; node features always X."""
    pca = PCA(n_components=4, random_state=42); Xp = pca.fit_transform(X).astype(np.float32)
    return {
        "FULL":    _knn_graph(X,         X, y, k, is_clf=False),
        "GEO":     _knn_graph(X[:,GEO],  X, y, k, is_clf=False),
        "SOCIO":   _knn_graph(X[:,SOCIO],X, y, k, is_clf=False),
        "LOWRANK": _knn_graph(Xp,        X, y, k, is_clf=False),
    }, Xp


# ─── MNIST views ──────────────────────────────────────────────────────────────

def _block_feats(X, n=16):
    imgs = X.reshape(-1, 28, 28); s = 7
    return np.stack([imgs[:, r*s:(r+1)*s, c*s:(c+1)*s].mean((1,2)) for r in range(4) for c in range(4)], 1).astype(np.float32)


def mnist_views(X, y, k=15, pca_d=50):
    pca = PCA(n_components=pca_d, random_state=42); Xp = pca.fit_transform(X).astype(np.float32)
    Xb = _block_feats(X)
    return {
        "FULL":  _knn_graph(X,  X, y, k, is_clf=True),
        "BLOCK": _knn_graph(Xb, X, y, k, is_clf=True),
        "PCA":   _knn_graph(Xp, X, y, k, is_clf=True),
    }, Xp
