"""
s01_graphs.py — Build 3 row-graph views: FULL, SPATIAL-BLOCK, PCA-50.
Node features = full 784-d scaled, topology changes per view.
"""

import numpy as np
import torch
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from torch_geometric.data import Data

EXP_ROOT = Path(__file__).parent.parent
ARTIFACTS = EXP_ROOT / "artifacts"
K = 15
PCA_DIMS = 50
N_BLOCKS = 16  # 4x4 spatial blocks on 28x28 image


def knn_edge_index(X_sub: np.ndarray, k: int):
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1).fit(X_sub)
    _, indices = nbrs.kneighbors(X_sub)
    indices = indices[:, 1:]
    src = np.repeat(np.arange(X_sub.shape[0]), k)
    dst = indices.flatten()
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


def spatial_block_features(X_raw: np.ndarray, n_blocks=N_BLOCKS) -> np.ndarray:
    """
    Aggregate 28x28 MNIST pixel rows into N_BLOCKS mean-pooled features.
    For 4x4 blocks: 4 row-strips × 4 col-strips = 16 features per sample.
    """
    n = X_raw.shape[0]
    imgs = X_raw.reshape(n, 28, 28)
    strip_rows = 28 // 4  # 4 row-strips of height 7
    strip_cols = 28 // 4  # 4 col-strips of width 7
    blocks = []
    for r in range(4):
        for c in range(4):
            block = imgs[:, r*strip_rows:(r+1)*strip_rows, c*strip_cols:(c+1)*strip_cols]
            blocks.append(block.mean(axis=(1, 2)))
    return np.stack(blocks, axis=1).astype(np.float32)


def build_graphs():
    print("[s01_graphs] Loading preprocessed data...")
    X_all = np.load(ARTIFACTS / "X_all.npy")
    y_all = np.load(ARTIFACTS / "y_all.npy")

    x_tensor = torch.tensor(X_all, dtype=torch.float32)
    y_tensor = torch.tensor(y_all, dtype=torch.long)

    # ── FULL view ─────────────────────────────────────────────────────
    print(f"[s01_graphs] FULL: kNN(k={K}) on 784-d...")
    ei_full = knn_edge_index(X_all, K)
    torch.save(Data(x=x_tensor, edge_index=ei_full, y=y_tensor), ARTIFACTS / "graph_FULL.pt")
    print(f"[s01_graphs] FULL: {ei_full.shape[1]} edges")

    # ── SPATIAL-BLOCK view ────────────────────────────────────────────
    print(f"[s01_graphs] SPATIAL-BLOCK: {N_BLOCKS} block features from 28x28 layout...")
    X_blocks = spatial_block_features(X_all)
    ei_block = knn_edge_index(X_blocks, K)
    torch.save(Data(x=x_tensor, edge_index=ei_block, y=y_tensor), ARTIFACTS / "graph_BLOCK.pt")
    print(f"[s01_graphs] BLOCK: {ei_block.shape[1]} edges, block features shape: {X_blocks.shape}")

    # ── PCA-50 view ───────────────────────────────────────────────────
    print(f"[s01_graphs] PCA-{PCA_DIMS}: reducing 784d → {PCA_DIMS}d for topology...")
    pca = PCA(n_components=PCA_DIMS, random_state=42)
    X_pca = pca.fit_transform(X_all).astype(np.float32)
    np.save(ARTIFACTS / "X_pca.npy", X_pca)
    explained = float(pca.explained_variance_ratio_.sum())
    print(f"[s01_graphs] PCA variance explained: {explained:.3f}")
    ei_pca = knn_edge_index(X_pca, K)
    torch.save(Data(x=x_tensor, edge_index=ei_pca, y=y_tensor), ARTIFACTS / "graph_PCA.pt")
    print(f"[s01_graphs] PCA: {ei_pca.shape[1]} edges")

    print("[s01_graphs] Done. Graphs: FULL, BLOCK, PCA saved.")


if __name__ == "__main__":
    build_graphs()
