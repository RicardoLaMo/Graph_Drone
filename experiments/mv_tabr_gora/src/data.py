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
    )
