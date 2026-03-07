"""
01_graphs.py — Build 4 separate kNN graph views on all rows.
Uses sklearn NearestNeighbors. Saves torch_geometric Data objects.
"""

import json
import numpy as np
import torch
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

ARTIFACTS = Path("artifacts")
K = 15
SEED = 42

# Feature indices (must match 00_data.py order)
# MedInc=0, HouseAge=1, AveRooms=2, AveBedrms=3, Population=4, AveOccup=5, Lat=6, Long=7
GEO_COLS = [6, 7]
SOCIO_COLS = [0, 1, 2, 3, 5]  # MedInc, HouseAge, AveRooms, AveBedrms, AveOccup


def build_knn_edge_index(X_sub: np.ndarray, k: int):
    """Build directed kNN edge_index from feature submatrix. Returns (2, N*k) LongTensor."""
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1).fit(X_sub)
    _, indices = nbrs.kneighbors(X_sub)
    indices = indices[:, 1:]  # exclude self

    src = np.repeat(np.arange(X_sub.shape[0]), k)
    dst = indices.flatten()
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return edge_index


def build_graphs():
    print("[01_graphs] Loading preprocessed data...")
    X_all = np.load(ARTIFACTS / "X_all.npy")
    y_all = np.load(ARTIFACTS / "y_all.npy")

    with open(ARTIFACTS / "feature_names.json") as f:
        feature_names = json.load(f)

    x_tensor = torch.tensor(X_all, dtype=torch.float32)
    y_tensor = torch.tensor(y_all, dtype=torch.float32)

    views = {
        "FULL": X_all,
        "GEO": X_all[:, GEO_COLS],
        "SOCIO": X_all[:, SOCIO_COLS],
    }

    for name, X_sub in views.items():
        print(f"[01_graphs]  Building {name} graph (k={K}, n_features={X_sub.shape[1]})...")
        edge_index = build_knn_edge_index(X_sub, K)
        data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
        torch.save(data, ARTIFACTS / f"graph_{name}.pt")
        print(f"[01_graphs]  {name}: {edge_index.shape[1]} edges")

    # CURVATURE view built after 02_curvature.py runs (curvature-weighted edges)
    # Placeholder saved here — 01b_curvature_graph.py will overwrite
    print("[01_graphs] FULL, GEO, SOCIO graphs saved. CURVATURE graph to be built after 02_curvature.py.")
    print("[01_graphs] Done.")


if __name__ == "__main__":
    build_graphs()
