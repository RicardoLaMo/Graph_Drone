"""
observers.py — Geometric observer per row.

g_i = [kappa_i, lid_i, lof_i, degree_per_view_i...]
These are routing priors only — never appended to X for prediction.

kappa: local non-flatness proxy (PCA residual ratio)
LID:   local intrinsic dimensionality
LOF:   local outlier density ratio
degree_per_view: normalised kNN degree per view adjacency
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from scipy.stats import spearmanr
from typing import Dict


def _knn(X: np.ndarray, k: int):
    nb = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
    d, idx = nb.kneighbors(X)
    return d[:, 1:], idx[:, 1:]   # drop self


def _kappa(X: np.ndarray, indices: np.ndarray, r: int = 2) -> np.ndarray:
    n = X.shape[0]; out = np.zeros(n, np.float32)
    for i in range(n):
        nb = X[indices[i]]; c = nb - nb.mean(0)
        _, s, _ = np.linalg.svd(c, full_matrices=False); tv = (s ** 2).sum()
        out[i] = 0.0 if tv < 1e-12 else 1.0 - (s[:r] ** 2).sum() / tv
    return out


def _lid(dists: np.ndarray) -> np.ndarray:
    k = dists.shape[1]; ks = np.arange(1, k + 1, dtype=np.float64)
    out = np.zeros(dists.shape[0], np.float32)
    for i in range(dists.shape[0]):
        d = np.maximum(dists[i], 1e-12)
        out[i] = max(float(np.polyfit(np.log(ks), np.log(d), 1)[0]), 0.0)
    return out


def compute_observers(X_pca: np.ndarray, X_views: Dict[str, np.ndarray], k: int = 15) -> np.ndarray:
    """
    Returns obs [N, obs_dim] where obs_dim = 3 + len(views)
    Columns: [kappa, lid, lof, degree_view0, degree_view1, ...]
    """
    print(f"  [obs] kNN k={k} n={X_pca.shape[0]} d_pca={X_pca.shape[1]}")
    dists, indices = _knn(X_pca, k)
    kp = _kappa(X_pca, indices, r=2)
    print(f"  [obs] kappa: mean={kp.mean():.4f} std={kp.std():.4f}")
    ld = _lid(dists)

    print("  [obs] LOF...")
    lof_m = LocalOutlierFactor(n_neighbors=k, novelty=False, n_jobs=-1)
    lof_m.fit(X_pca)
    lof = (-lof_m.negative_outlier_factor_).astype(np.float32)

    # Per-view degree (normalised): how many edges exist for this row in each view
    degree_cols = []
    for vname, Xv in X_views.items():
        _, vidx = _knn(Xv, k)
        deg = np.ones(X_pca.shape[0], np.float32) * k   # kNN is regular — all degree=k
        # Use mean distance as proxy for local density (richer signal than binary degree)
        vdists, _ = _knn(Xv, k); mean_dist = vdists.mean(1)
        # normalise to [0,1]
        md_min = mean_dist.min(); md_range = mean_dist.max() - mean_dist.min()
        norm_dist = ((mean_dist - md_min) / (md_range + 1e-8)).astype(np.float32)
        degree_cols.append(norm_dist)
        print(f"  [obs] view '{vname}' mean_dist: {mean_dist.mean():.4f}")

    obs = np.concatenate(
        [kp[:, None], ld[:, None], lof[:, None]] + [d[:, None] for d in degree_cols],
        axis=1
    ).astype(np.float32)
    print(f"  [obs] Final shape: {obs.shape}")
    return obs, kp
