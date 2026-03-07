"""
observer.py — Shared observer feature computation.
Computes: kappa (local PCA residual), LID, LOF, local density.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from scipy.stats import spearmanr


def knn_dists_indices(X, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
    d, idx = nbrs.kneighbors(X)
    return d[:, 1:], idx[:, 1:]


def local_pca_residual(X, indices, r=2):
    n = X.shape[0]
    kappa = np.zeros(n, dtype=np.float32)
    for i in range(n):
        nbr = X[indices[i]]
        centered = nbr - nbr.mean(0)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        tv = (s ** 2).sum()
        kappa[i] = 0.0 if tv < 1e-12 else float(1.0 - (s[:r] ** 2).sum() / tv)
    return kappa


def lid_proxy(dists):
    k = dists.shape[1]
    ks = np.arange(1, k + 1, dtype=np.float64)
    lid = np.zeros(dists.shape[0], dtype=np.float32)
    for i in range(dists.shape[0]):
        d = np.maximum(dists[i], 1e-12)
        lid[i] = float(max(np.polyfit(np.log(ks), np.log(d), 1)[0], 0.0))
    return lid


def local_density(dists):
    return (1.0 / np.maximum(dists.mean(1), 1e-12)).astype(np.float32)


def compute_observers(X, k=15, r=2):
    """Returns (observer_array, kappa, stats_dict)."""
    print(f"  [observer] kNN k={k}, n={X.shape[0]}, d={X.shape[1]}")
    dists, indices = knn_dists_indices(X, k)
    kappa = local_pca_residual(X, indices, r)
    print(f"  [observer] kappa: mean={kappa.mean():.4f} std={kappa.std():.4f}")
    lid = lid_proxy(dists)
    print("  [observer] LOF...")
    lof_m = LocalOutlierFactor(n_neighbors=k, novelty=False, n_jobs=-1)
    lof_m.fit(X)
    lof = (-lof_m.negative_outlier_factor_).astype(np.float32)
    density = local_density(dists)
    obs = np.stack([kappa, lid, lof, density], axis=1).astype(np.float32)
    return obs, kappa


def multiscale_stability(X, k_scales=(10, 20, 30), r=2):
    kappas = {}
    for k in k_scales:
        _, idx = knn_dists_indices(X, k)
        kappas[k] = local_pca_residual(X, idx, r)
    results = {}
    n = X.shape[0]
    top_n = max(1, n // 5)
    top_sets = {k: set(np.argsort(kappas[k])[-top_n:]) for k in k_scales}
    ks = list(k_scales)
    for i, k1 in enumerate(ks):
        for k2 in ks[i + 1:]:
            rho, pval = spearmanr(kappas[k1], kappas[k2])
            results[f"spearman_k{k1}_k{k2}"] = float(rho)
            results[f"pval_k{k1}_k{k2}"] = float(pval)
            overlap = len(top_sets[k1] & top_sets[k2]) / top_n
            results[f"top20pct_overlap_k{k1}_k{k2}"] = float(overlap)
    return results


def bin_curvature(kappa, y, extra_cols=None):
    low_t, high_t = np.percentile(kappa, 33), np.percentile(kappa, 67)
    bins = np.where(kappa <= low_t, "low", np.where(kappa <= high_t, "medium", "high"))
    df = pd.DataFrame({"row_idx": np.arange(len(kappa)), "kappa": kappa, "curvature_bin": bins})
    if extra_cols:
        for k, v in extra_cols.items():
            df[k] = v
    return df
