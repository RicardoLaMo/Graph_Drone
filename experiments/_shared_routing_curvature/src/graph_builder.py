"""
graph_builder.py — Shared kNN graph construction.
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data


def build_knn_graph(X_topo, X_node, y, k=15):
    """Build kNN graph: topology from X_topo, node features from X_node."""
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1).fit(X_topo)
    _, indices = nbrs.kneighbors(X_topo)
    indices = indices[:, 1:]
    src = np.repeat(np.arange(X_topo.shape[0]), k)
    dst = indices.flatten()
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    data = Data(
        x=torch.tensor(X_node, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long if y.dtype in [np.int64, np.int32] else torch.float32),
    )
    return data
