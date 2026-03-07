"""
observers.py — Shared observer feature computation.
g_i = [kappa, LID, LOF, density] — routing priors, NOT predictive features.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from scipy.stats import spearmanr


def _knn(X, k):
    nb = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
    d, idx = nb.kneighbors(X)
    return d[:, 1:], idx[:, 1:]


def kappa(X, indices, r=2):
    n = X.shape[0]; out = np.zeros(n, np.float32)
    for i in range(n):
        nb = X[indices[i]]; c = nb - nb.mean(0)
        _, s, _ = np.linalg.svd(c, full_matrices=False); tv = (s**2).sum()
        out[i] = 0. if tv < 1e-12 else 1. - (s[:r]**2).sum() / tv
    return out


def lid(dists):
    k = dists.shape[1]; ks = np.arange(1, k + 1, dtype=np.float64)
    out = np.zeros(dists.shape[0], np.float32)
    for i in range(dists.shape[0]):
        d = np.maximum(dists[i], 1e-12)
        out[i] = max(float(np.polyfit(np.log(ks), np.log(d), 1)[0]), 0.)
    return out


def compute_observers(X, k=15, r=2):
    """Returns obs [N,4]: kappa, LID, LOF, density — all routing priors only."""
    print(f"  [obs] kNN k={k} n={X.shape[0]} d={X.shape[1]}")
    dists, indices = _knn(X, k)
    kp = kappa(X, indices, r)
    print(f"  [obs] kappa mean={kp.mean():.4f} std={kp.std():.4f}")
    ld = lid(dists)
    print("  [obs] LOF...")
    lof_m = LocalOutlierFactor(n_neighbors=k, novelty=False, n_jobs=-1); lof_m.fit(X)
    lof = (-lof_m.negative_outlier_factor_).astype(np.float32)
    density = (1. / np.maximum(dists.mean(1), 1e-12)).astype(np.float32)
    obs = np.stack([kp, ld, lof, density], axis=1).astype(np.float32)
    return obs, kp


def stability(X, scales=(10, 20, 30), r=2):
    kaps = {}
    for k in scales:
        _, idx = _knn(X, k); kaps[k] = kappa(X, idx, r)
    res = {}; n = X.shape[0]; top_n = max(1, n // 5)
    tops = {k: set(np.argsort(kaps[k])[-top_n:]) for k in scales}
    ks = list(scales)
    for i, k1 in enumerate(ks):
        for k2 in ks[i + 1:]:
            rho, _ = spearmanr(kaps[k1], kaps[k2])
            res[f"spearman_k{k1}_k{k2}"] = float(rho)
            res[f"overlap_k{k1}_k{k2}"] = len(tops[k1] & tops[k2]) / top_n
    return res


def bin_kappa(kp):
    lo, hi = np.percentile(kp, 33), np.percentile(kp, 67)
    return np.where(kp <= lo, "low", np.where(kp <= hi, "medium", "high"))
