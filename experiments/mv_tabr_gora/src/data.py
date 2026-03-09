"""
data.py — MV-TabR-GoRA dataset builder.

Wraps the existing California v5 data/views/observers pipeline and produces
the complete Bundle needed for training and inference:

  X           [N, F]        raw features (RobustScaler'd, log1p on cols 2,4)
  y           [N]           raw targets
  y_norm      [N]           standardised targets
  train_idx   [n_train]     global training indices
  val_idx     [n_val]       global validation indices
  test_idx    [n_test]      global test indices
  view_feats  {name: [N, d_v]}   raw view feature matrices
  per_view_knn {name: ([N,K], [N,K])}  (global_indices, distance_weights)
  sigma2_v    [N, V]        per-view label variance (log-robust-normalised)
  J_flat      [N, n_pairs]  pairwise Jaccard overlap between views
  mean_J      [N]           mean Jaccard (view agreement signal)
  target_stats {mean, std}
  view_names  list[str]     ordered list of view names
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.head_routing_v5.california.src.data_v5 import (
    build_california_dataset,
    TargetStats,
)
from experiments.head_routing_v5.california.src.views_v5 import (
    build_california_views,
    build_per_view_knn,
)
from experiments.head_routing_v5.california.src.train_v5 import (
    maybe_truncate_splits,
    standardise_observers,
)
from experiments.head_routing_v5.shared.src.observers_v5 import build_v5_observers


EPS = 1e-8


@dataclass
class MVDataBundle:
    # Raw / normalised data
    X: np.ndarray              # [N, F]  float32
    y: np.ndarray              # [N]     float32 (raw)
    y_norm: np.ndarray         # [N]     float32 (standardised)
    train_idx: np.ndarray      # [n_train]
    val_idx: np.ndarray        # [n_val]
    test_idx: np.ndarray       # [n_test]
    target_stats: dict         # {mean, std}
    # View structure
    view_feats: Dict[str, np.ndarray]          # {name → [N, d_v]}
    view_names: List[str]                      # ordered
    view_dims: Dict[str, int]                  # {name → d_v}
    # Per-view retrieval (train-referenced kNN; global indices)
    per_view_knn: Dict[str, Tuple[np.ndarray, np.ndarray]]  # {name → (idx[N,K], wt[N,K])}
    K: int                                     # neighbours per view
    # Quality signals
    sigma2_v: np.ndarray       # [N, V]  log-robust-normalised per-view label variance
    mean_J: np.ndarray         # [N]     mean Jaccard across view pairs
    J_flat: np.ndarray         # [N, n_pairs]  all pairwise Jaccard values
    raw_geo: Optional[np.ndarray] = None      # [N, 2] raw lat/lon from original dataset
    geo_segment_ids: Optional[np.ndarray] = None  # [N] train-fit pseudo-community ids


def _compute_sigma2_v(
    y: np.ndarray,
    per_view_knn: Dict[str, Tuple[np.ndarray, np.ndarray]],
    train_idx: np.ndarray,
    view_names: List[str],
) -> np.ndarray:
    """
    Per-view label variance sigma2_v[i,v] = Var(y[ kNN_v(i) ]).
    Uses raw (un-normalised) y to compute variance, then log-robust-normalises.
    Neighbours are always training points, so no test-label leakage.
    """
    N = y.shape[0]
    V = len(view_names)
    sigma2_raw = np.zeros((N, V), dtype=np.float32)

    for v_idx, name in enumerate(view_names):
        nei_idx, _ = per_view_knn[name]   # [N, K]
        y_nei = y[nei_idx]                 # [N, K]  — nei_idx contains GLOBAL indices
        sigma2_raw[:, v_idx] = y_nei.var(axis=1)

    # log-robust-normalise on train rows (matches observers_v5 convention)
    log_s2 = np.log1p(np.maximum(sigma2_raw, 0.0)).astype(np.float64)
    train_vals = log_s2[train_idx]
    median = np.median(train_vals, axis=0)        # [V]
    q25 = np.percentile(train_vals, 25, axis=0)
    q75 = np.percentile(train_vals, 75, axis=0)
    iqr = np.maximum(q75 - q25, EPS)
    sigma2_norm = ((log_s2 - median) / iqr).astype(np.float32)
    return sigma2_norm


def _jaccard_pair_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Vectorised Jaccard between rows of a [N, K] and b [N, K].
    Uses sorted membership test: for each row, sort a and b, then
    count |intersection| = count of elements in a that appear in b via searchsorted.
    This avoids a Python loop over N rows.
    """
    N, K = a.shape
    # For each row: |intersection| via broadcast comparison [N, K_a, K_b]
    # is too large, so use sorting trick instead.
    a_sorted = np.sort(a, axis=1)    # [N, K]
    b_sorted = np.sort(b, axis=1)    # [N, K]
    # Count matches using searchsorted per row
    inter = np.zeros(N, dtype=np.float32)
    for n in range(N):
        # Count elements of a_sorted[n] that appear in b_sorted[n]
        idx = np.searchsorted(b_sorted[n], a_sorted[n])
        idx_clipped = np.clip(idx, 0, K - 1)
        inter[n] = float(np.sum(b_sorted[n][idx_clipped] == a_sorted[n]))
    union = 2 * K - inter   # |A∪B| = |A| + |B| - |A∩B|
    return inter / np.maximum(union, 1.0)


def _compute_jaccard(
    per_view_knn: Dict[str, Tuple[np.ndarray, np.ndarray]],
    view_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pairwise Jaccard overlap between view kNN sets.
    Returns:
        J_flat : [N, n_pairs]  each column = |A∩B|/|A∪B| for a pair of views
        mean_J : [N]
    Uses vectorised searchsorted instead of Python set loop.
    """
    n_views = len(view_names)
    knn_sets = [per_view_knn[v][0] for v in view_names]   # list of [N, K]

    pairs = [(i, j) for i in range(n_views) for j in range(i + 1, n_views)]
    J_cols = [
        _jaccard_pair_vectorized(knn_sets[i], knn_sets[j])
        for i, j in pairs
    ]

    J_flat = np.stack(J_cols, axis=1).astype(np.float32)   # [N, n_pairs]
    mean_J = J_flat.mean(axis=1)                            # [N]
    return J_flat, mean_J


def _weights_from_row_dists(dists: np.ndarray) -> np.ndarray:
    d = np.asarray(dists, dtype=np.float32)
    out = np.zeros_like(d, dtype=np.float32)
    for i in range(d.shape[0]):
        row = d[i]
        pos = row[row > 0]
        sigma = float(np.median(pos) if pos.size else 1.0) + EPS
        w = np.exp(-(row ** 2) / (sigma ** 2)).astype(np.float32)
        w /= w.sum() + EPS
        out[i] = w
    return out


def _build_geo_segment_ids(
    raw_geo: np.ndarray,
    train_idx: np.ndarray,
    n_clusters: int = 96,
    seed: int = 42,
) -> np.ndarray:
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    km.fit(raw_geo[train_idx])
    return km.predict(raw_geo).astype(np.int64)


def _apply_same_segment_weight_bias(
    base_knn: Dict[str, Tuple[np.ndarray, np.ndarray]],
    segment_ids: np.ndarray,
    anchor_ids: Optional[np.ndarray] = None,
    target_views: Tuple[str, ...] = ("FULL", "GEO"),
    bonus: float = 2.0,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if anchor_ids is None:
        anchor_ids = np.arange(next(iter(base_knn.values()))[0].shape[0], dtype=np.int64)
    for name, (idx, wt) in base_knn.items():
        if name not in target_views:
            out[name] = (idx.copy(), wt.copy())
            continue
        same_seg = (segment_ids[idx] == segment_ids[anchor_ids][:, None]).astype(np.float32)
        wt_new = wt * (1.0 + bonus * same_seg)
        wt_new /= wt_new.sum(axis=1, keepdims=True) + EPS
        out[name] = (idx.copy(), wt_new.astype(np.float32))
    return out


def _build_segment_poolmix_for_view(
    X_view: np.ndarray,
    base_idx: np.ndarray,
    train_idx: np.ndarray,
    segment_ids: np.ndarray,
    k: int,
    k_seg: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    N = X_view.shape[0]
    merged_idx = np.empty((N, k), dtype=np.int64)
    merged_dists = np.empty((N, k), dtype=np.float32)

    segment_train_rows: Dict[int, np.ndarray] = {}
    segment_models: Dict[int, NearestNeighbors] = {}
    for seg in np.unique(segment_ids[train_idx]):
        seg_train = train_idx[segment_ids[train_idx] == seg]
        segment_train_rows[int(seg)] = seg_train
        n_neighbors = min(len(seg_train), k_seg + 1)
        segment_models[int(seg)] = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1).fit(X_view[seg_train])

    same_seg_candidates: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for seg, seg_rows in segment_train_rows.items():
        q_idx = np.where(segment_ids == seg)[0]
        if len(q_idx) == 0:
            continue
        model = segment_models[seg]
        dists, local_idx = model.kneighbors(X_view[q_idx])
        global_idx = seg_rows[local_idx].astype(np.int64)
        same_seg_candidates[seg] = (q_idx.astype(np.int64), global_idx, dists.astype(np.float32))

    seg_lookup: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for seg, payload in same_seg_candidates.items():
        q_idx, global_idx, dists = payload
        for row_pos, row_idx in enumerate(q_idx):
            seg_lookup[int(row_idx)] = (global_idx[row_pos], dists[row_pos])

    for row_idx in range(N):
        base_row = base_idx[row_idx]
        seg_row, seg_d = seg_lookup.get(row_idx, (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)))

        chosen: list[int] = []
        chosen_dists: list[float] = []

        for cand, dist in zip(seg_row, seg_d):
            if cand == row_idx:
                continue
            if cand not in chosen:
                chosen.append(int(cand))
                chosen_dists.append(float(dist))
            if len(chosen) >= k_seg:
                break

        for cand in base_row:
            if cand == row_idx:
                continue
            if int(cand) not in chosen:
                chosen.append(int(cand))
                dist = float(np.linalg.norm(X_view[row_idx] - X_view[int(cand)]))
                chosen_dists.append(dist)
            if len(chosen) >= k:
                break

        if not chosen:
            chosen = [int(base_row[0])]
            chosen_dists = [float(np.linalg.norm(X_view[row_idx] - X_view[chosen[0]]))]

        while len(chosen) < k:
            chosen.append(chosen[-1])
            chosen_dists.append(chosen_dists[-1])

        merged_idx[row_idx] = np.asarray(chosen[:k], dtype=np.int64)
        merged_dists[row_idx] = np.asarray(chosen_dists[:k], dtype=np.float32)

    merged_wt = _weights_from_row_dists(merged_dists)
    return merged_idx, merged_wt


def _build_random_poolmix_for_view(
    X_view: np.ndarray,
    base_idx: np.ndarray,
    train_idx: np.ndarray,
    k: int,
    k_rand: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    N = X_view.shape[0]
    merged_idx = np.empty((N, k), dtype=np.int64)
    merged_dists = np.empty((N, k), dtype=np.float32)
    train_pool = np.asarray(train_idx, dtype=np.int64)

    for row_idx in range(N):
        chosen: list[int] = []
        chosen_dists: list[float] = []

        perm = rng.permutation(len(train_pool))
        for pos in perm:
            cand = int(train_pool[pos])
            if cand == row_idx:
                continue
            if cand not in chosen:
                chosen.append(cand)
                chosen_dists.append(float(np.linalg.norm(X_view[row_idx] - X_view[cand])))
            if len(chosen) >= k_rand:
                break

        for cand in base_idx[row_idx]:
            cand_i = int(cand)
            if cand_i == row_idx:
                continue
            if cand_i not in chosen:
                chosen.append(cand_i)
                chosen_dists.append(float(np.linalg.norm(X_view[row_idx] - X_view[cand_i])))
            if len(chosen) >= k:
                break

        if not chosen:
            fallback = int(base_idx[row_idx][0])
            chosen = [fallback]
            chosen_dists = [float(np.linalg.norm(X_view[row_idx] - X_view[fallback]))]

        while len(chosen) < k:
            chosen.append(chosen[-1])
            chosen_dists.append(chosen_dists[-1])

        merged_idx[row_idx] = np.asarray(chosen[:k], dtype=np.int64)
        merged_dists[row_idx] = np.asarray(chosen_dists[:k], dtype=np.float32)

    merged_wt = _weights_from_row_dists(merged_dists)
    return merged_idx, merged_wt


def build_geo_segmented_bundle(
    K: int = 24,
    smoke: bool = False,
    smoke_train: int = 500,
    smoke_val: int = 200,
    smoke_test: int = 200,
    seed: int = 42,
    mode: str = "weight_bias",
    n_clusters: int = 96,
    target_views: Tuple[str, ...] = ("FULL", "GEO"),
    same_segment_bonus: float = 2.0,
    same_segment_frac: float = 0.5,
) -> MVDataBundle:
    base = build_mv_data_bundle(
        K=K,
        smoke=smoke,
        smoke_train=smoke_train,
        smoke_val=smoke_val,
        smoke_test=smoke_test,
        seed=seed,
    )

    raw_ds = build_california_dataset(seed=seed)
    from sklearn.datasets import fetch_california_housing
    raw_geo = fetch_california_housing().data[:, [6, 7]].astype(np.float32)
    segment_ids = _build_geo_segment_ids(raw_geo, base.train_idx, n_clusters=n_clusters, seed=seed)

    if mode == "weight_bias":
        per_view_knn = _apply_same_segment_weight_bias(
            base.per_view_knn,
            segment_ids,
            anchor_ids=np.arange(base.X.shape[0], dtype=np.int64),
            target_views=target_views,
            bonus=same_segment_bonus,
        )
    elif mode == "poolmix":
        per_view_knn = {
            name: (idx.copy(), wt.copy())
            for name, (idx, wt) in base.per_view_knn.items()
        }
        k_seg = max(1, int(round(K * same_segment_frac)))
        for name in target_views:
            idx_new, wt_new = _build_segment_poolmix_for_view(
                X_view=base.view_feats[name],
                base_idx=base.per_view_knn[name][0],
                train_idx=base.train_idx,
                segment_ids=segment_ids,
                k=K,
                k_seg=k_seg,
                seed=seed,
            )
            per_view_knn[name] = (idx_new, wt_new)
    elif mode == "random_poolmix":
        per_view_knn = {
            name: (idx.copy(), wt.copy())
            for name, (idx, wt) in base.per_view_knn.items()
        }
        k_rand = max(1, int(round(K * same_segment_frac)))
        for name in target_views:
            idx_new, wt_new = _build_random_poolmix_for_view(
                X_view=base.view_feats[name],
                base_idx=base.per_view_knn[name][0],
                train_idx=base.train_idx,
                k=K,
                k_rand=k_rand,
                seed=seed,
            )
            per_view_knn[name] = (idx_new, wt_new)
    else:
        raise ValueError(f"Unknown geo retrieval mode: {mode}")

    sigma2_v = _compute_sigma2_v(base.y, per_view_knn, base.train_idx, base.view_names)
    J_flat, mean_J = _compute_jaccard(per_view_knn, base.view_names)

    return MVDataBundle(
        X=base.X,
        y=base.y,
        y_norm=base.y_norm,
        train_idx=base.train_idx,
        val_idx=base.val_idx,
        test_idx=base.test_idx,
        target_stats=base.target_stats,
        view_feats=base.view_feats,
        view_names=base.view_names,
        view_dims=base.view_dims,
        per_view_knn=per_view_knn,
        K=base.K,
        sigma2_v=sigma2_v,
        mean_J=mean_J,
        J_flat=J_flat,
        raw_geo=raw_geo,
        geo_segment_ids=segment_ids,
    )


def build_mv_data_bundle(
    K: int = 24,
    smoke: bool = False,
    smoke_train: int = 500,
    smoke_val: int = 200,
    smoke_test: int = 200,
    seed: int = 42,
) -> MVDataBundle:
    """
    Build the complete MV-TabR-GoRA data bundle for California Housing.

    K : neighbours per view (total context = K * V, defaults to 24 * 4 = 96 ≈ TabR)
    """
    # ---- Raw data --------------------------------------------------------
    ds = build_california_dataset(seed=seed)
    X, y = ds["X"], ds["y"]
    train_idx = ds["train_idx"]
    val_idx = ds["val_idx"]
    test_idx = ds["test_idx"]
    target_stats = ds["target_stats"]

    # ---- Normalise target ------------------------------------------------
    mean_t = float(target_stats["mean"])
    std_t = float(target_stats["std"])
    y_norm = ((y - mean_t) / std_t).astype(np.float32)

    # ---- Optional smoke subsetting ---------------------------------------
    train_idx, val_idx, test_idx = maybe_truncate_splits(
        train_idx, val_idx, test_idx,
        smoke, smoke_train, smoke_val, smoke_test,
    )

    # ---- Views -----------------------------------------------------------
    view_feats = build_california_views(X, train_idx=train_idx)
    view_names = list(view_feats.keys())
    view_dims = {name: view_feats[name].shape[1] for name in view_names}

    # ---- Per-view kNN (train-referenced, global indices) ----------------
    per_view_knn = build_per_view_knn(view_feats, k=K, train_idx=train_idx)

    # ---- Quality signals -------------------------------------------------
    sigma2_v = _compute_sigma2_v(y, per_view_knn, train_idx, view_names)
    J_flat, mean_J = _compute_jaccard(per_view_knn, view_names)

    from sklearn.datasets import fetch_california_housing
    raw_geo = fetch_california_housing().data[:, [6, 7]].astype(np.float32)

    return MVDataBundle(
        X=X,
        y=y,
        y_norm=y_norm,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        target_stats=target_stats,
        view_feats=view_feats,
        view_names=view_names,
        view_dims=view_dims,
        per_view_knn=per_view_knn,
        K=K,
        sigma2_v=sigma2_v,
        mean_J=mean_J,
        J_flat=J_flat,
        raw_geo=raw_geo,
        geo_segment_ids=None,
    )
