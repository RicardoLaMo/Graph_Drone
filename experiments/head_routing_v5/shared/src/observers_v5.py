"""
observers_v5.py — Per-view geometry observers with robust normalization.

New vs. observers.py in the worktree:
  - kappa_v, LID_v, LOF_v computed independently in each view's own feature space
  - Jaccard overlap J[N, V, V] between per-view kNN sets
  - sigma2_v = Var(y_train[N_v(i)]) per view (log-normalized, leakage-free)
  - Robust normalization: log-transform for LID/LOF/sigma2, median/IQR for kappa

Gap coverage: Gap 1, Gap 2, Gap 5, Gap 12, Gap 13 (PRD 2026-03-08)
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors


# ---------------------------------------------------------------------------
# Low-level helpers (shared with old observers.py logic)
# ---------------------------------------------------------------------------

def _pad_rows(values: np.ndarray, k: int) -> np.ndarray:
    if values.shape[1] >= k:
        return values[:, :k]
    pad_count = k - values.shape[1]
    pad = np.repeat(values[:, -1:], pad_count, axis=1)
    return np.concatenate([values, pad], axis=1)


def _query_knn(
    X_query: np.ndarray,
    X_ref: np.ndarray,
    ref_idx: np.ndarray,
    k: int,
    query_idx: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    extra = 1 if query_idx is not None else 0
    n_neighbors = min(len(ref_idx), k + extra)
    nb = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1).fit(X_ref)
    dists, local_idx = nb.kneighbors(X_query)
    global_idx = ref_idx[local_idx].astype(np.int64)

    if query_idx is None:
        return _pad_rows(dists.astype(np.float32), k), _pad_rows(global_idx, k)

    trimmed_dists = np.empty((len(query_idx), k), dtype=np.float32)
    trimmed_idx = np.empty((len(query_idx), k), dtype=np.int64)
    for row, q_idx in enumerate(query_idx):
        keep = global_idx[row] != q_idx
        ri = global_idx[row][keep]
        rd = dists[row][keep]
        if ri.size == 0:
            ri = global_idx[row][:1]
            rd = dists[row][:1]
        if ri.size < k:
            pad = k - ri.size
            ri = np.concatenate([ri, np.repeat(ri[-1:], pad)])
            rd = np.concatenate([rd, np.repeat(rd[-1:], pad)])
        trimmed_idx[row] = ri[:k]
        trimmed_dists[row] = rd[:k]
    return trimmed_dists, trimmed_idx


# ---------------------------------------------------------------------------
# Curvature: 1 - fraction of variance explained by top-rank singular values
# ---------------------------------------------------------------------------

def _kappa(X: np.ndarray, indices: np.ndarray, rank: int = 2) -> np.ndarray:
    out = np.zeros(X.shape[0], dtype=np.float32)
    for i in range(X.shape[0]):
        nbhd = X[indices[i]]
        centered = nbhd - nbhd.mean(axis=0, keepdims=True)
        _, sv, _ = np.linalg.svd(centered, full_matrices=False)
        total = float((sv ** 2).sum())
        if total < 1e-12:
            out[i] = 0.0
        else:
            out[i] = 1.0 - float((sv[:rank] ** 2).sum()) / total
    return out


# ---------------------------------------------------------------------------
# LID: slope of log-log rank-distance curve
# ---------------------------------------------------------------------------

def _lid(dists: np.ndarray) -> np.ndarray:
    ks = np.arange(1, dists.shape[1] + 1, dtype=np.float64)
    out = np.zeros(dists.shape[0], dtype=np.float32)
    for i in range(dists.shape[0]):
        d = np.maximum(dists[i], 1e-12).astype(np.float64)
        out[i] = max(float(np.polyfit(np.log(ks), np.log(d), 1)[0]), 0.0)
    return out


# ---------------------------------------------------------------------------
# Per-view kNN query (returns both dists and neighbor indices)
# ---------------------------------------------------------------------------

def _compute_view_knn(
    X_view: np.ndarray,
    train_idx: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (dists [N, k], indices [N, k]) using train-referenced kNN."""
    all_idx = np.arange(X_view.shape[0], dtype=np.int64)
    eval_idx = np.setdiff1d(all_idx, train_idx, assume_unique=False)

    dists_all = np.empty((X_view.shape[0], k), dtype=np.float32)
    idx_all = np.empty((X_view.shape[0], k), dtype=np.int64)

    tr_d, tr_i = _query_knn(X_view[train_idx], X_view[train_idx], train_idx, k, query_idx=train_idx)
    dists_all[train_idx] = tr_d
    idx_all[train_idx] = tr_i

    if len(eval_idx):
        ev_d, ev_i = _query_knn(X_view[eval_idx], X_view[train_idx], train_idx, k, query_idx=None)
        dists_all[eval_idx] = ev_d
        idx_all[eval_idx] = ev_i

    return dists_all, idx_all


# ---------------------------------------------------------------------------
# Robust normalization helpers
# ---------------------------------------------------------------------------

def _robust_norm_params(arr: np.ndarray, train_idx: np.ndarray) -> Tuple[float, float]:
    """Compute median and IQR on train rows."""
    v = arr[train_idx]
    median = float(np.median(v))
    q25, q75 = float(np.percentile(v, 25)), float(np.percentile(v, 75))
    iqr = q75 - q25 + 1e-8
    return median, iqr


def _log_robust_normalize(
    arr: np.ndarray,
    train_idx: np.ndarray,
) -> Tuple[np.ndarray, dict]:
    """log1p → standardize with train median/IQR. For heavy-tailed quantities."""
    log_arr = np.log1p(np.maximum(arr, 0.0)).astype(np.float64)
    median, iqr = _robust_norm_params(log_arr, train_idx)
    normed = ((log_arr - median) / iqr).astype(np.float32)
    return normed, {"median": median, "iqr": iqr, "log_transform": True}


def _standardize(arr: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Mean/std standardization on train set."""
    v = arr[train_idx]
    mean = float(v.mean())
    std = float(v.std()) + 1e-8
    return ((arr - mean) / std).astype(np.float32), {"mean": mean, "std": std, "log_transform": False}


# ---------------------------------------------------------------------------
# Per-view quality observers (Gap 1)
# ---------------------------------------------------------------------------

def compute_per_view_observers(
    view_feats: Dict[str, np.ndarray],
    train_idx: np.ndarray,
    k: int = 15,
    rank: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Compute kappa_v, LID_v, LOF_v in each view's own feature space.

    Returns
    -------
    quality_raw : [N, V, 3]   kappa, LID, LOF per view (raw, before normalization)
    quality_norm : [N, V, 3]  same, after robust normalization
    view_knn_idx : [N, V, k]  neighbour indices for Jaccard + sigma2_v
    norm_stats   : dict with per-view normalization parameters
    """
    train_idx = np.asarray(train_idx, dtype=np.int64)
    view_names = list(view_feats.keys())
    V = len(view_names)
    N = next(iter(view_feats.values())).shape[0]

    quality_raw = np.zeros((N, V, 3), dtype=np.float32)   # [kappa, LID, LOF]
    quality_norm = np.zeros((N, V, 3), dtype=np.float32)
    view_knn_idx = np.zeros((N, V, k), dtype=np.int64)
    norm_stats: dict = {}

    for v_idx, view_name in enumerate(view_names):
        X_v = view_feats[view_name]

        # kNN in this view's own feature space
        dists_v, idx_v = _compute_view_knn(X_v, train_idx, k)
        view_knn_idx[:, v_idx, :] = idx_v

        # Curvature (kappa)
        kappa_v = _kappa(X_v, idx_v, rank=rank)
        quality_raw[:, v_idx, 0] = kappa_v
        kappa_norm, kappa_stats = _standardize(kappa_v, train_idx)
        quality_norm[:, v_idx, 0] = kappa_norm

        # LID (log-robust)
        lid_v = _lid(dists_v)
        quality_raw[:, v_idx, 1] = lid_v
        lid_norm, lid_stats = _log_robust_normalize(lid_v, train_idx)
        quality_norm[:, v_idx, 1] = lid_norm

        # LOF (fit on train, score all)
        lof_model = LocalOutlierFactor(n_neighbors=k, novelty=True, n_jobs=-1)
        lof_model.fit(X_v[train_idx])
        all_idx = np.arange(N, dtype=np.int64)
        eval_idx = np.setdiff1d(all_idx, train_idx)
        lof_v = np.empty(N, dtype=np.float32)
        lof_v[train_idx] = (-lof_model.negative_outlier_factor_).astype(np.float32)
        if len(eval_idx):
            lof_v[eval_idx] = (-lof_model.score_samples(X_v[eval_idx])).astype(np.float32)
        # Clip at 99th percentile then log-robust normalize
        clip_val = float(np.percentile(lof_v[train_idx], 99))
        lof_clipped = np.minimum(lof_v, clip_val)
        quality_raw[:, v_idx, 2] = lof_v
        lof_norm, lof_stats = _log_robust_normalize(lof_clipped, train_idx)
        quality_norm[:, v_idx, 2] = lof_norm

        norm_stats[view_name] = {
            "kappa": kappa_stats,
            "lid": lid_stats,
            "lof": {**lof_stats, "clip_val": clip_val},
        }

    return quality_raw, quality_norm, view_knn_idx, norm_stats


def build_quality_score(quality_norm: np.ndarray) -> np.ndarray:
    """
    Build a scalar quality q_v [N, V] from normalized [kappa, LID, LOF].
    q_v = sigmoid( -kappa_norm - LID_norm - LOF_norm )
    Lower curvature, lower LID, lower outlier score = higher quality.
    """
    # All three: high = bad quality → negate for quality
    raw = -quality_norm[:, :, 0] - quality_norm[:, :, 1] - quality_norm[:, :, 2]
    # Sigmoid to [0, 1]
    quality = 1.0 / (1.0 + np.exp(-raw.astype(np.float64)))
    return quality.astype(np.float32)


# ---------------------------------------------------------------------------
# Jaccard overlap matrix (Gap 2 + Gap 5)
# ---------------------------------------------------------------------------

def compute_jaccard(view_knn_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute J[N, V, V]: pairwise Jaccard overlap between view neighborhoods.
    Also returns J_flat [N, V*(V-1)/2]: upper triangle, row-major.

    Parameters
    ----------
    view_knn_idx : [N, V, k] neighbor indices per view
    """
    N, V, k = view_knn_idx.shape
    J = np.zeros((N, V, V), dtype=np.float32)
    n_pairs = V * (V - 1) // 2

    for i in range(N):
        for v1 in range(V):
            J[i, v1, v1] = 1.0
            for v2 in range(v1 + 1, V):
                s1 = set(view_knn_idx[i, v1].tolist())
                s2 = set(view_knn_idx[i, v2].tolist())
                inter = len(s1 & s2)
                union = len(s1 | s2)
                j_val = inter / union if union > 0 else 0.0
                J[i, v1, v2] = j_val
                J[i, v2, v1] = j_val

    # Upper triangle (no diagonal) → flat
    pairs = [(v1, v2) for v1 in range(V) for v2 in range(v1 + 1, V)]
    J_flat = np.stack([J[:, v1, v2] for v1, v2 in pairs], axis=1)  # [N, n_pairs]
    mean_J = J_flat.mean(axis=1, keepdims=True)  # [N, 1]

    return J_flat, mean_J.squeeze(1)


# ---------------------------------------------------------------------------
# sigma2_v — neighbourhood label variance per view (Gap 13)
# ---------------------------------------------------------------------------

def compute_sigma2_v(
    view_knn_idx: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute sigma2_v [N, V] = Var(y_train[N_v(i)]).

    Leakage controls:
      - Train rows: self is excluded by view_knn_idx (built with query_idx self-exclusion)
      - Val/test rows: N_v(i) is drawn from train set only (guaranteed by _compute_view_knn)
      - sigma2_v never enters encoder or head — router only (enforced in backbone_v5.py)

    Returns sigma2_v_norm [N, V], mean_sigma2_v [N], norm_stats
    """
    train_idx = np.asarray(train_idx, dtype=np.int64)
    train_set = set(train_idx.tolist())
    N, V, k = view_knn_idx.shape
    sigma2_v_raw = np.zeros((N, V), dtype=np.float32)

    for v in range(V):
        for i in range(N):
            nei = view_knn_idx[i, v]
            # Only use training labels
            train_nei = [idx for idx in nei if idx in train_set]
            if not train_nei:
                sigma2_v_raw[i, v] = 0.0
            else:
                labels = y[np.array(train_nei, dtype=np.int64)]
                sigma2_v_raw[i, v] = float(labels.var())

    # Log-robust normalize per view
    sigma2_norm = np.zeros_like(sigma2_v_raw)
    norm_stats = {}
    for v in range(V):
        normed, stats = _log_robust_normalize(sigma2_v_raw[:, v], train_idx)
        sigma2_norm[:, v] = normed
        norm_stats[v] = stats

    mean_sigma2 = sigma2_norm.mean(axis=1)

    # Sanity log (range check)
    for v in range(V):
        vmin = float(sigma2_v_raw[train_idx, v].min())
        vmax = float(sigma2_v_raw[train_idx, v].max())
        vmean = float(sigma2_v_raw[train_idx, v].mean())

    return sigma2_norm, mean_sigma2, norm_stats


# ---------------------------------------------------------------------------
# Global observers (same as legacy observers.py — for backward compat)
# ---------------------------------------------------------------------------

def compute_global_observers(
    X_lowrank: np.ndarray,
    view_feats: Dict[str, np.ndarray],
    k: int = 15,
    train_idx: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legacy-compatible observer: kappa, LID, LOF in LOWRANK space + density per view.
    Returns observers [N, 3+V], kappa [N].
    """
    if train_idx is None:
        train_idx = np.arange(X_lowrank.shape[0], dtype=np.int64)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    all_idx = np.arange(X_lowrank.shape[0], dtype=np.int64)
    eval_idx = np.setdiff1d(all_idx, train_idx)

    dists = np.empty((X_lowrank.shape[0], k), dtype=np.float32)
    indices = np.empty((X_lowrank.shape[0], k), dtype=np.int64)

    tr_d, tr_i = _query_knn(X_lowrank[train_idx], X_lowrank[train_idx], train_idx, k, query_idx=train_idx)
    dists[train_idx] = tr_d
    indices[train_idx] = tr_i
    if len(eval_idx):
        ev_d, ev_i = _query_knn(X_lowrank[eval_idx], X_lowrank[train_idx], train_idx, k, query_idx=None)
        dists[eval_idx] = ev_d
        indices[eval_idx] = ev_i

    kappa = _kappa(X_lowrank, indices, rank=2)
    lid = _lid(dists)

    lof_model = LocalOutlierFactor(n_neighbors=k, novelty=True, n_jobs=-1)
    lof_model.fit(X_lowrank[train_idx])
    lof = np.empty(X_lowrank.shape[0], dtype=np.float32)
    lof[train_idx] = (-lof_model.negative_outlier_factor_).astype(np.float32)
    if len(eval_idx):
        lof[eval_idx] = (-lof_model.score_samples(X_lowrank[eval_idx])).astype(np.float32)

    # Per-view density (normalised mean kNN distance in each view's feature space)
    extras = []
    for view_name in sorted(view_feats.keys()):
        X_v = view_feats[view_name]
        train_idx_a = np.asarray(train_idx, dtype=np.int64)
        all_idx_a = np.arange(X_v.shape[0], dtype=np.int64)
        eval_idx_a = np.setdiff1d(all_idx_a, train_idx_a)
        md = np.empty(X_v.shape[0], dtype=np.float32)
        td, _ = _query_knn(X_v[train_idx_a], X_v[train_idx_a], train_idx_a, k, query_idx=train_idx_a)
        md[train_idx_a] = td.mean(axis=1)
        if len(eval_idx_a):
            ed, _ = _query_knn(X_v[eval_idx_a], X_v[train_idx_a], train_idx_a, k, query_idx=None)
            md[eval_idx_a] = ed.mean(axis=1)
        tr_md = md[train_idx_a]
        mn, rng = float(tr_md.min()), float(tr_md.max() - tr_md.min())
        extras.append(((md - mn) / (rng + 1e-8)).astype(np.float32))

    observers = np.concatenate(
        [kappa[:, None], lid[:, None], lof[:, None]] + [e[:, None] for e in extras],
        axis=1,
    ).astype(np.float32)
    return observers, kappa


# ---------------------------------------------------------------------------
# Main entry point: build_v5_observers
# ---------------------------------------------------------------------------

def build_v5_observers(
    view_feats: Dict[str, np.ndarray],
    per_view_knn: Dict[str, Tuple[np.ndarray, np.ndarray]],
    y: np.ndarray | None,
    train_idx: np.ndarray,
    k: int = 15,
) -> dict:
    """
    Build all v5 observer arrays.

    Returns a dict with keys:
      g_global      [N, obs_dim]   global observers (kappa, LID, LOF, density×V)
      quality_norm  [N, V, 3]      per-view kappa/LID/LOF, normalized
      quality_score [N, V]         scalar quality q_v ∈ (0,1)
      J_flat        [N, V*(V-1)/2] Jaccard overlap (upper triangle)
      mean_J        [N]            mean Jaccard per row
      sigma2_v      [N, V] or None log-normalized label variance per view (None if y is None)
      view_knn_idx  [N, V, k]      neighbour indices used for Jaccard/sigma2_v
      view_names    [V]            ordered list of view names
      norm_stats    dict           normalization parameters (apply to val/test)
    """
    train_idx = np.asarray(train_idx, dtype=np.int64)
    view_names = list(view_feats.keys())
    lowrank_key = "LOWRANK" if "LOWRANK" in view_feats else list(view_feats.keys())[-1]

    # Global observers (legacy compat)
    g_global, kappa_global = compute_global_observers(
        view_feats[lowrank_key], view_feats, k=k, train_idx=train_idx
    )

    # Per-view observers
    quality_raw, quality_norm, view_knn_idx, per_view_norm_stats = compute_per_view_observers(
        view_feats, train_idx, k=k
    )
    quality_score = build_quality_score(quality_norm)

    # Jaccard
    J_flat, mean_J = compute_jaccard(view_knn_idx)

    # sigma2_v
    sigma2_v = None
    if y is not None:
        sigma2_v, _, sigma2_stats = compute_sigma2_v(view_knn_idx, y, train_idx)
    else:
        sigma2_stats = {}

    return {
        "g_global": g_global.astype(np.float32),
        "quality_norm": quality_norm,
        "quality_score": quality_score,
        "J_flat": J_flat.astype(np.float32),
        "mean_J": mean_J.astype(np.float32),
        "sigma2_v": sigma2_v,
        "view_knn_idx": view_knn_idx,
        "view_names": view_names,
        "kappa_global": kappa_global,
        "norm_stats": {
            "per_view": per_view_norm_stats,
            "sigma2": sigma2_stats,
        },
    }
