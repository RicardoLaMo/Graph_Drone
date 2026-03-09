"""
views_v5.py — CA view features and per-view kNN.
Reuses same logic as the worktree views.py but self-contained in v5.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


GEO_COLS = [6, 7]
SOCIO_COLS = [0, 1, 2, 3, 4]


def build_california_views(X: np.ndarray, train_idx: np.ndarray | None = None) -> dict[str, np.ndarray]:
    fit_idx = train_idx if train_idx is not None else np.arange(X.shape[0])
    lowrank = PCA(n_components=4, random_state=42).fit(X[fit_idx]).transform(X).astype(np.float32)
    return {
        "FULL": X.astype(np.float32),
        "GEO": X[:, GEO_COLS].astype(np.float32),
        "SOCIO": X[:, SOCIO_COLS].astype(np.float32),
        "LOWRANK": lowrank,
    }


def _pad_rows(values: np.ndarray, k: int) -> np.ndarray:
    if values.shape[1] >= k:
        return values[:, :k]
    pad_count = k - values.shape[1]
    pad = np.repeat(values[:, -1:], pad_count, axis=1)
    return np.concatenate([values, pad], axis=1)


def _weights_from_dists(dists: np.ndarray) -> np.ndarray:
    sigma = float(np.median(dists) + 1e-8)
    weights = np.exp(-(dists ** 2) / (sigma ** 2)).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True) + 1e-8
    return weights


def _query_reference_knn(
    X_query: np.ndarray,
    X_ref: np.ndarray,
    ref_idx: np.ndarray,
    k: int,
    query_idx: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    extra_neighbor = 1 if query_idx is not None else 0
    n_neighbors = min(len(ref_idx), k + extra_neighbor)
    nb = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1).fit(X_ref)
    dists, local_idx = nb.kneighbors(X_query)
    global_idx = ref_idx[local_idx].astype(np.int64)

    if query_idx is None:
        dists = _pad_rows(dists.astype(np.float32), k)
        global_idx = _pad_rows(global_idx, k)
        return global_idx, _weights_from_dists(dists)

    trimmed_idx = np.empty((len(query_idx), k), dtype=np.int64)
    trimmed_dists = np.empty((len(query_idx), k), dtype=np.float32)
    for row, q_idx in enumerate(query_idx):
        keep_mask = global_idx[row] != q_idx
        row_idx = global_idx[row][keep_mask]
        row_dists = dists[row][keep_mask]
        if row_idx.size == 0:
            row_idx = global_idx[row][:1]
            row_dists = dists[row][:1]
        if row_idx.size < k:
            pad_count = k - row_idx.size
            row_idx = np.concatenate([row_idx, np.repeat(row_idx[-1:], pad_count)])
            row_dists = np.concatenate([row_dists, np.repeat(row_dists[-1:], pad_count)])
        trimmed_idx[row] = row_idx[:k]
        trimmed_dists[row] = row_dists[:k]
    return trimmed_idx, _weights_from_dists(trimmed_dists)


def build_per_view_knn(
    view_feats: Dict[str, np.ndarray],
    k: int,
    train_idx: np.ndarray | None = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if train_idx is None:
        train_idx = np.arange(next(iter(view_feats.values())).shape[0], dtype=np.int64)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    all_idx = np.arange(next(iter(view_feats.values())).shape[0], dtype=np.int64)
    eval_idx = np.setdiff1d(all_idx, train_idx, assume_unique=False)

    per_view = {}
    for name, X_view in view_feats.items():
        ref_view = X_view[train_idx]
        neighbors = np.empty((X_view.shape[0], k), dtype=np.int64)
        weights = np.empty((X_view.shape[0], k), dtype=np.float32)

        train_neighbors, train_weights = _query_reference_knn(
            X_view[train_idx], ref_view, train_idx, k=k, query_idx=train_idx,
        )
        neighbors[train_idx] = train_neighbors
        weights[train_idx] = train_weights

        if len(eval_idx):
            eval_neighbors, eval_weights = _query_reference_knn(
                X_view[eval_idx], ref_view, train_idx, k=k, query_idx=None,
            )
            neighbors[eval_idx] = eval_neighbors
            weights[eval_idx] = eval_weights

        per_view[name] = (neighbors, weights)
    return per_view
